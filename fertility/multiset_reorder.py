import sys
from typing import Dict, Optional, List, Tuple, Union

from collections import Counter

import allennlp.models
import numpy as np
import torch
from allennlp.common.checks import ConfigurationError

from allennlp.data import Vocabulary, TextFieldTensors, Token
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import util

from allennlp.nn.util import sequence_cross_entropy_with_logits, get_lengths_from_binary_sequence_mask, \
    get_mask_from_sequence_lengths

from fertility.eval.acc_by_length import MetricByCat
from fertility.eval.ds import DSEval
from fertility.eval.exclusion_tracker import InclusionMetric
from fertility.eval.identity_counter import IdentityCounter, EntropyCounter
from fertility.eval.lev import LevenstheinMetric, MyMetric
from fertility.reordering.jump_perm import AbstractVariationalPermutationModel


import torch.nn.functional as F

import random

from fertility.test_mode import TestMode
from fertility.constants import COGS_VAR, COPY_SYMBOL

import numba


@numba.njit
def linearize_np(counts, tgt_length):
    # shape (batch, isl, vocab)
    monotonic_output = np.zeros((counts.shape[0], tgt_length), dtype=np.int64)
    alignment = np.zeros((counts.shape[0], counts.shape[1], tgt_length), dtype=np.float32)
    pointer = np.zeros(counts.shape[0], dtype=np.int32)
    tgt_mask = np.zeros((counts.shape[0], tgt_length),dtype=np.int32)
    tgt_copy = np.zeros((counts.shape[0], tgt_length),dtype=np.int64)
    for b in range(counts.shape[0]):
        for i in range(counts.shape[1]):
            for v in range(counts.shape[2]):
                for a in range(counts[b,i,v]):
                    if pointer[b] < tgt_length:
                        monotonic_output[b, pointer[b]] = v
                        alignment[b, i, pointer[b]] = True
                        tgt_mask[b, pointer[b]] = True
                        tgt_copy[b, pointer[b]] = a+1
                        pointer[b] += 1

    return monotonic_output, alignment, tgt_mask, tgt_copy


def linearize_multisets(predicted_multisets: torch.Tensor, max_l):
    # shape (batch, input_seq_len, vocab)
    o, a, m, c = linearize_np(predicted_multisets.detach().cpu().numpy(), max_l)

    o = torch.from_numpy(o).to(predicted_multisets.device)
    a = torch.from_numpy(a).to(predicted_multisets.device)
    m = torch.from_numpy(m).to(predicted_multisets.device).bool()
    c = torch.from_numpy(c).to(predicted_multisets.device)
    return o, a, m, c


@Model.register("multiset_reorder")
class MultisetReorder(Model):

    def __init__(self, vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 target_embedder: TextFieldEmbedder,
                 permutation_model: AbstractVariationalPermutationModel,
                 multiset_model_path: str,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 target_namespace: str = "target_tokens",
                 rho: float = 0.05,
                 metrics: List[MyMetric] = None,
                 concat_reprs: bool = True,
                 positional_embedding_dim: Optional[int] = None,
                 ignore_case_for_possible_matchings: bool = False,
                 token_reindexer: SingleIdTokenIndexer = None
                 ):
        super().__init__(vocab)

        self.ds_eval = DSEval()

        self.target_embedder = target_embedder

        self.source_embedder = source_embedder

        self.target_namespace = target_namespace

        self.encoder = encoder
        self.rho = rho

        self.permutation_model = permutation_model

        self.permutation_model.is_stage_2 = True #TODO

        self.metrics = metrics or []

        self.token_reindexer = token_reindexer

        self.seq_acc = LevenstheinMetric()

        self.track_metric_by_cat = MetricByCat(self.metrics)

        self.parser_perm_entropy = EntropyCounter()
        self.posterior_perm_entropy = EntropyCounter()

        self.concat_reprs = concat_reprs

        self.ignore_case_for_possible_matchings = ignore_case_for_possible_matchings

        self.inclusion_metric = InclusionMetric("multiset_was_correct")


        # self.track_metric_by_cat = MetricByCat([self.freq_acc])

        # self.conv_mask = _conv_create_mask(self.max_n+1, self.max_l+1)

        self._copy_id = self.vocab.get_token_index(COPY_SYMBOL, self.target_namespace)

        if COGS_VAR in self.vocab.get_token_to_index_vocabulary(self.target_namespace):
            self._cogs_var_id = self.vocab.get_token_index(COGS_VAR, self.target_namespace)
        else:
            self._cogs_var_id = None


        self.multiset_model = allennlp.models.load_archive(multiset_model_path).model
        for param in self.multiset_model.parameters():
            param.requires_grad_(False)
            # param.requires_grad_(True)

        # Check that we recognize every item in the multiset target vocab, so we can embed it
        for token in self.multiset_model.vocab.get_token_to_index_vocabulary(namespace=target_namespace):
            if token not in self.vocab.get_token_to_index_vocabulary(self.target_namespace):
                print(f"Warning: missing key in our vocab: '{token}' - will treat this as UNK.", file=sys.stderr)


        self.positional_embedding = torch.nn.Parameter(
            torch.randn([1, 1, self.multiset_model.max_n+1, self.source_embedder.get_output_dim() if positional_embedding_dim is None else positional_embedding_dim]),
            requires_grad=True)


    def compute_repr(self, encoder_outputs, alignment, linearized_multisets_output, copy_info):
        repr = torch.einsum("bid, bij -> bjd", encoder_outputs, alignment)

        positional_info = (self.positional_embedding * F.one_hot(copy_info, num_classes=self.multiset_model.max_n + 1).unsqueeze(-1)).sum(2)
        embedded_target = self.target_embedder({"tokens": {"tokens": linearized_multisets_output}})

        if self.concat_reprs:
            repr = torch.cat([repr, positional_info, embedded_target], dim=-1)
        else:
            repr = repr + positional_info + embedded_target
        return repr


    def reindex(self, linearized_multisets_output):
        # Re-index the linearized_multisets_output according to OUR vocabulary:
        reindex_linearized_multiset_outputs = np.array([[self.vocab.get_token_index(self.multiset_model.vocab.get_token_from_index(elem, self.target_namespace),
                                                                                       namespace=self.target_namespace) for elem in seq]
                                                         for seq in linearized_multisets_output.cpu().numpy()])
        return torch.from_numpy(reindex_linearized_multiset_outputs).to(linearized_multisets_output.device)

    def compute_permitted_alignments(self, readable_linearized_outputs: List[List[str]], readable_targets: List[List[str]], shape):
        permitted = np.ones(shape, dtype=bool)
        for b in range(permitted.shape[0]):
            for i in range(len(readable_linearized_outputs[b])):
                for j in range(len(readable_targets[b])):
                    if self.ignore_case_for_possible_matchings:
                        permitted[b, i, j] = readable_linearized_outputs[b][i].lower() == readable_targets[b][j].lower()
                    else:
                        permitted[b, i, j] = readable_linearized_outputs[b][i] == readable_targets[b][j]
        return permitted

    def multisets_correct(self, readable_linearized_outputs: List[List[str]], readable_targets: List[List[str]]) -> np.array:
        d = []
        for pred, target in zip(readable_linearized_outputs, readable_targets):
            pred_counter = Counter(pred)
            del pred_counter[self.multiset_model.vocab._padding_token]
            if self.ignore_case_for_possible_matchings:
                pred_counter = {k.lower(): v for k, v in pred_counter.items()}

            target_counter = Counter(target)
            if self.ignore_case_for_possible_matchings:
                target_counter = {k.lower(): v for k, v in pred_counter.items()}

            d.append(pred_counter == target_counter)

        return np.array(d)


    def compute_linearized_and_repr(self, predicted_multisets, max_length, readable_source, encoder_outputs):
        linearized_multisets_output, alignment, pred_target_mask, copy_info = linearize_multisets(
            predicted_multisets, max_length)

        readable_linearized_outputs = self.multiset_model.make_sequence_readable(linearized_multisets_output, alignment,
                                                                                 readable_source)

        if self.token_reindexer is not None:
            linearized_multisets_output_cpu = linearized_multisets_output.cpu().numpy()
            reidx = np.zeros(linearized_multisets_output.shape, dtype=np.int64)
            for b in range(reidx.shape[0]):
                tokens = [Token(t) for i,t in enumerate(readable_linearized_outputs[b])]
                idx = self.token_reindexer.tokens_to_indices(tokens, self.vocab)["tokens"]
                # Overwrite embedding of variables (i.e. numbers) with the generic cogs_var_id if applicable
                if self._cogs_var_id is not None:
                    idx = [self._cogs_var_id if linearized_multisets_output_cpu[b, i] == self.multiset_model._cogs_var_id else t for i,t in enumerate(idx)]

                for i,x in enumerate(idx):
                    if x == self.vocab.get_token_index("@@UNKNOWN@@", self.token_reindexer.namespace):
                        print(f"Warning, could not re-embed token '{readable_linearized_outputs[b][i]}', replaced with UNK", file=sys.stderr)

                reidx[b, 0:len(readable_linearized_outputs[b])] = idx

            reindex_linearized_multisets_output = torch.from_numpy(reidx).to(encoder_outputs.device)
        else:
            reindex_linearized_multisets_output = self.reindex(linearized_multisets_output)

        repr = self.compute_repr(encoder_outputs, alignment, reindex_linearized_multisets_output, copy_info)

        return readable_linearized_outputs, repr, alignment, pred_target_mask

    def forward(self, source_tokens: TextFieldTensors,
                metadata: List[Dict],
                target_tokens: Optional[TextFieldTensors] = None,
                real_target_tokens: Optional[TextFieldTensors] = None,
                source_to_copy_mask: Optional[torch.Tensor] = None,
                alignment: Optional[torch.Tensor] = None,
                emphasis: Optional[torch.Tensor] = None,
                copyable_inputs: Optional[torch.Tensor] = None) -> Dict[
        str, torch.Tensor]:

        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_source = self.source_embedder(source_tokens)

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        if self.training and not hasattr(self, "is_in_test_mode"):
            raise ConfigurationError("Please add the test_mode callback")

        if self.encoder is not None:
            if self.rho > 0.0:
                encoder_outputs = self.encoder(embedded_source, source_mask)
                encoder_outputs = self.rho * encoder_outputs + embedded_source
            else:
                encoder_outputs = embedded_source
        else:
            encoder_outputs = embedded_source

        readable_source = [m["source_tokens"] for m in metadata]

        ret = {"source_lengths": get_lengths_from_binary_sequence_mask(source_mask),
               "readable_source": readable_source}
        ids = [m["#"] for m in metadata]  # identify the instances by arbitrary unique numbers.
        categories = [m["category"] if "category" in m else None for m in metadata]

        ret["categories"] = categories

        d = {"source_tokens": source_tokens, "metadata": metadata,
             "alignment": alignment, "emphasis": emphasis, "copyable_inputs": copyable_inputs}
        multiset_output = self.multiset_model(**d)

        predicted_multisets = multiset_output["predicted_multisets"] #shape (batch, input_seq_len, vocab)

        readable_target = None
        if target_tokens:
            targets = target_tokens["tokens"]["tokens"]  # shape (batch_size, output seq length)
            target_mask = util.get_text_field_mask(target_tokens)  # shape (batch_size, output seq length)
            # target_lengths = get_lengths_from_binary_sequence_mask(target_mask)
            ret["targets"] = targets
            ret["gold_target_lengths"] = get_lengths_from_binary_sequence_mask(target_mask)

            readable_target = [m["target_tokens"] for m in metadata]
            ret["readable_targets"] = readable_target

            if hasattr(self, "is_in_test_mode") and not self.is_in_test_mode:
                # We don't compute this at test time, to save time.

                readable_linearized_outputs, repr, alignment, pred_target_mask \
                    = self.compute_linearized_and_repr(predicted_multisets, targets.shape[1], readable_source, encoder_outputs)

                # Compute the permitted alignments
                permitted_alignments = self.compute_permitted_alignments(readable_linearized_outputs, readable_target, (targets.shape[0], targets.shape[1], targets.shape[1]))

                permitted_alignments = torch.from_numpy(permitted_alignments).to(repr.device)

                multiset_correct = self.multisets_correct(readable_linearized_outputs, readable_target)
                self.inclusion_metric.add_instances(multiset_correct)

                multiset_correct = torch.from_numpy(multiset_correct).to(repr.device)
                # print(multiset_correct)

                if real_target_tokens is None:
                    real_target_tokens = target_tokens
                # permitted_alignments = None
                x, loss = self.permutation_model.forward(repr, target_mask, real_target_tokens, target_mask, ids,
                                               permitted_alignments=permitted_alignments, loss_mask = multiset_correct)

                self.posterior_perm_entropy.add_matrix(x, target_mask)

                ret["loss"] = loss

        if not self.training:
            max_length = int(predicted_multisets.sum([1,2]).max().cpu().numpy())

            readable_linearized_outputs, repr, alignment, pred_target_mask \
                = self.compute_linearized_and_repr(predicted_multisets, max_length, readable_source, encoder_outputs)

            ret["marginal_alignment"] = alignment.transpose(1,2)

            x, _ = self.permutation_model.forward(repr, pred_target_mask.bool(), target_mask=pred_target_mask.bool())

            ret["reorder_after_fertility"] = x

            self.parser_perm_entropy.add_matrix(x, pred_target_mask)

            ret["preds_before_permutation"] = readable_linearized_outputs

            readable_prediction = self.make_readable(readable_linearized_outputs, torch.argmax(x, dim=2), pred_target_mask)

            ret["readable_predictions"] = readable_prediction

            if readable_target is not None:
                self.seq_acc.add_instances(readable_prediction, readable_target)

                for m in self.metrics:
                    m.add_instances(readable_prediction, readable_target)

                if any(categories):
                    self.track_metric_by_cat.add_instances(readable_prediction, readable_target, categories)

            # torch.set_printoptions(3, sci_mode=False, linewidth=200)
            # print(x[0])
            # print(readable_source[0])
            # print(readable_linearized_outputs[0])
            # print(readable_prediction[0])
            # print("Target")
            # print(readable_target[0])
            # print("---")

        return ret


    def make_readable(self, readable_linearized: List[List[str]], permutation, target_mask):
        """

        :param readable_linearized:
        :param permutation: shape (batch, output_seq_len)
        :param target_mask: shape (batch, output_seq_len)
        :return:
        """
        lengths = target_mask.sum(dim=-1).detach().cpu().numpy()
        permutation = permutation.detach().cpu().numpy()
        res = []
        for b in range(lengths.shape[0]):
            d = []
            for i in range(lengths[b]):
                if permutation[b, i] < len(readable_linearized[b]):
                    d.append(readable_linearized[b][permutation[b, i]])
                else:
                    print("Warning: illegal accesss in make_readable", file=sys.stderr)
                    d.append("ILLEGAL_ACCESS_IN_MAKE_READABLE")
            res.append(d)
        return res


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        d = dict()
        d.update(self.seq_acc.get_metric(reset))
        for m in self.metrics:
            d.update(m.get_metric(reset))

        d.update(self.inclusion_metric.get_metric(reset))

        d["posterior_perm_entropy"] = self.posterior_perm_entropy.get_metrics(reset)["perm_entropy"]
        d["parser_perm_entropy"] = self.parser_perm_entropy.get_metrics(reset)["perm_entropy"]

        if not self.training:
            d.update(self.track_metric_by_cat.get_metrics(reset))

        return d


