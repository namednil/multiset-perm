import copy
import math
from typing import Dict, Optional, List, Tuple, Union

from collections import Counter

import numpy as np
import torch
from allennlp.common import Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, ConditionalRandomField, FeedForward, Embedding
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder, PretrainedTransformerEmbedder
from allennlp.nn import util, Activation

from allennlp.nn.util import sequence_cross_entropy_with_logits, get_lengths_from_binary_sequence_mask, \
    get_mask_from_sequence_lengths
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM

from fertility.conv_utils import my_sequence_cross_entropy_with_logits, multi_set_pred_dyn, _conv_create_mask
from fertility.dataset_readers.lexicon_learning import Lexicon
from fertility.eval.acc_by_length import MetricByCat
from fertility.eval.ds import DSEval
from fertility.eval.lev import LevenstheinMetric, MyMetric
from fertility.pretrained_transformer.my_mismatched import PretrainedTransformerMismatchedEmbedderRho
from fertility.pretrained_transformer.my_pretrained_transformer_embedder import PretrainedTransformerEmbedderWithRho
from fertility.translation_model import LexicalTranslationModel, LSTMTranslationModel, TranslationModel

import torch.nn.functional as F

import random

from fertility.constants import COGS_VAR, COPY_SYMBOL

@MyMetric.register("my_acc")
class FreqAcc(MyMetric):

    def __init__(self, prefix:str):
        self.correct = 0
        self.total = 0
        self.prefix = prefix

    def get_metric(self, reset: bool) -> Dict[str, float]:
        if self.total == 0:
            return {self.prefix+"_acc" : 0.0}
        r = self.correct / self.total
        if reset:
            self.reset()
        return {self.prefix+"_acc" : r}

    def reset(self):
        self.correct = 0
        self.total = 0

    def add_instances(self, predictions, gold) -> None:
        assert len(predictions) == len(gold)
        self.total += len(predictions)
        for p,g in zip(predictions, gold):
            self.correct += (p == g)
            # if p != g:
            #     print("Incorrect")
            #     print("Pred")
            #     print(p)
            #     print("Gold")
            #     print(g)
            #     print("----")


class TranslationModelWithNonContextualized(TranslationModel):

    def forward(self, embedded_input: torch.Tensor, input_mask: torch.Tensor, non_contextetualized_input: Optional[torch.Tensor] = None, fertilities: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

@TranslationModel.register("multiset_lex_translation")
class LexicalTranslationModel(TranslationModelWithNonContextualized):

    def __init__(self, vocab: Vocabulary, maximum_fertility: int, target_namespace: str,
                 bias: bool = True, mlp: Optional[FeedForward] = None, input_dim: Optional[int] = None,
                 include_non_contextual: bool = False):
        super().__init__(vocab, maximum_fertility, target_namespace)

        self.maximum_fertility = maximum_fertility
        # self.positional_dim = positional_dim
        self.target_namespace = target_namespace

        # self.positional_embedding = torch.nn.Parameter(
        #     torch.randn([1, self.maximum_fertility + 1, self.positional_dim]),
        #     requires_grad=True)
        self.mlp = mlp

        self.include_non_contextual = include_non_contextual
        if include_non_contextual:
            self.mlp_non_contextual = copy.deepcopy(mlp)
            self.non_contextual_output_layer = torch.nn.Linear(self.mlp.get_output_dim(), (self.maximum_fertility+1) * self.vocab_size, bias=bias)

        if self.mlp is not None:
            self.output_layer = torch.nn.Linear(self.mlp.get_output_dim(), (self.maximum_fertility+1) * self.vocab_size, bias=bias)
        else:
            self.output_layer = torch.nn.Linear(input_dim, (self.maximum_fertility+1) * self.vocab_size, bias=bias)


    def get_input_dim(self) -> int:
        return self.mlp.get_input_dim()

    def forward(self, embedded_input: torch.Tensor, input_mask: torch.Tensor,
                non_contextetualized_input: Optional[torch.Tensor] = None,
                fertilities: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        compute log probs for output
        :param input_mask: shape (batch_size, input_seq_len)
        :param embedded_input: shape (batch_size, input_seq_len, embedding_dim)
        :return: shape (batch_size, input_seq_len,  vocab size, maximum fertility + 1) NOT NORMALIZED
        """

        batch_size, input_seq_len, embedding_dim = embedded_input.shape

        if self.mlp is not None:
            o = self.output_layer(self.mlp(embedded_input)) #shape (batch_size, input_seq_len, self.maximum fertility * vocab size)
        else:
            o = self.output_layer(embedded_input) #shape (batch_size, input_seq_len, self.maximum fertility * vocab size)
        o = o.reshape([batch_size, input_seq_len, self.vocab.get_vocab_size(self.target_namespace), self.maximum_fertility+1])
        # I hope the above doesn't break for specific dimensionalities...

        if self.include_non_contextual:
            r = self.non_contextual_output_layer(self.mlp_non_contextual(non_contextetualized_input)).reshape([batch_size, input_seq_len, self.vocab.get_vocab_size(self.target_namespace), self.maximum_fertility+1])
            o = o + r

        return o

@Model.register("multiset_pred")
class MultisetPred(Model):

    def __init__(self, vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 max_n: int,
                 mlp: Optional[FeedForward] = None,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 target_namespace: str = "target_tokens",
                 pretrain_epochs: int = 0,
                 rho: float = 1.0,
                 alignment_threshold: float = 0.8,
                 alignment_loss_weight: float = 0.1,
                 lower_case_eval: bool = False,
                 translation_bias: bool = True,
                 lexicon: Optional[Lexicon] = None
                 ):
        super().__init__(vocab)

        self.ds_eval = DSEval()

        self.rho = rho

        self.source_embedder = source_embedder

        self.pretrain_epochs = pretrain_epochs

        self.target_namespace = target_namespace

        self.encoder = encoder
        self.max_n = max_n
        self.lower_case_eval = lower_case_eval

        self.alignment_threshold = alignment_threshold
        self.alignment_loss_weight = alignment_loss_weight
        self.lexicon = lexicon

        # self.metrics = [FreqAcc("freq")]
        self.freq_acc = FreqAcc("freq")

        self.track_metric_by_cat = MetricByCat([self.freq_acc])

        self.translation_model = LexicalTranslationModel(vocab, maximum_fertility=max_n, target_namespace=self.target_namespace, mlp=mlp,
                                                             bias=translation_bias)

        # self.conv_mask = _conv_create_mask(self.max_n+1, self.max_l+1)

        self.mask_out_items = []
        if COPY_SYMBOL in self.vocab.get_token_to_index_vocabulary(self.target_namespace):
            self._copy_id = self.vocab.get_token_index(COPY_SYMBOL, self.target_namespace)
            self.mask_out_items.append(self._copy_id)
        else:
            self._copy_id = None

        if COGS_VAR in self.vocab.get_token_to_index_vocabulary(self.target_namespace):
            self._cogs_var_id = self.vocab.get_token_index(COGS_VAR, self.target_namespace)
            self.mask_out_items.append(self._cogs_var_id)
        else:
            self._cogs_var_id = None


        if self._copy_id is not None and self.lexicon is None:
            raise ConfigurationError("Copy token present but no lexicon given.")


    def compute_dist(self, logits, source_mask = None, max_l = None):

        if max_l is None:
            #Find longest input sequence in batch
            max_length = max(source_mask.sum(dim=-1).cpu().numpy())
            max_l = self.max_n*max_length+1

        conv_mask = torch.from_numpy(_conv_create_mask(self.max_n+1, max_l)).to(logits.device)
        # if every input token can create at most self.max_n tokens of a certain type
        # and there are max_length input tokens, there can be at most self.max_n * max_length
        # tokens of a certain type when considering the entire sequence

        dist = multi_set_pred_dyn(logits, conv_mask) #shape (batch, vocab, self.max_n*max_length+1)

        return dist


    def compute_dist_for_dynamic(self, logits: torch.Tensor, rule_mask: torch.Tensor, index: int, max_l: int):
        """

        :param logits: shape (batch, input_seq_len, vocab, max_n+1)
        :param rule_mask: shape (batch, input_seq_len, num items)
        :return: shape (batch, num items, max_l)
        """

        interesting_logits = logits[:, :, index, :] #shape (batch, input_seq_len, max_n + 1)
        # interesting_logits = interesting_logits.unsqueeze(2) - 10_000 * ~rule_mask.unsqueeze(-1)
        # mask = torch.((batch, input_seq_len, items, max_n+1))
        mask = torch.zeros(rule_mask.shape + (self.max_n+1,), device=logits.device)
        mask[:, :, :, 0] = 1_000 * ~rule_mask

        interesting_logits = interesting_logits.unsqueeze(2) + mask
        interesting_logits = torch.log_softmax(interesting_logits, dim=-1) # just make sure it stays normalized after this intervention
        # this doesn't affect scores that were unmodified by the mask

        return self.compute_dist(interesting_logits, max_l = max_l)

    def compute_dynamic_loss(self, dist: torch.Tensor, counts: torch.Tensor):
        """

        :param dist: shape (batch, num items, max_l)
        :param counts: shape (batch, num items)
        :return: shape (1,)
        """

        one_hot_counts = F.one_hot(counts, num_classes=dist.shape[-1])
        return -(one_hot_counts * dist).sum([1,2]).mean()

    def compute_dynamic_loss_for(self, multiset_logits, rule_mask, rule_freq, index):
        if rule_freq.numel() == 0:
            return  0.0
        max_count = int(rule_freq.max().detach().cpu().numpy()) + 1
        copy_dist = self.compute_dist_for_dynamic(multiset_logits, rule_mask, index,
                                                  max_l=max_count)
        loss = self.compute_dynamic_loss(copy_dist, rule_freq)

        # Many tokens must not copy at all, compute_dynamic_loss forced them to contribute 0 tokens
        # but here we compute the corresponding loss
        no_copy_at_all = torch.all(~rule_mask,
                                   dim=-1)  # shape (batch_size, input_seq_len) where no copying should be made
        loss += -(no_copy_at_all * multiset_logits[:, :, index, 0]).sum() / multiset_logits.shape[0]

        return loss

    def compute_loss(self, dist, gold_freq, mask_out_indices):
        target_vals = F.one_hot(gold_freq, num_classes=dist.shape[-1])  # shape (batch, vocab_size, max_l+1)

        vocab_mask = torch.ones((1, dist.shape[-2], 1), dtype=torch.bool) #shape (1, vocab_size, 1)
        for index in mask_out_indices:
            vocab_mask[0, index, 0] = False
        vocab_mask = vocab_mask.to(dist.device)

        loss = -(target_vals * vocab_mask * dist).sum(dim=[1, 2]).mean()
        return loss

    def compute_logits(self, encoder_outputs: torch.Tensor, source_mask: torch.Tensor, readable_source: List[List[str]], non_contextualized_repr: torch.Tensor) -> torch.Tensor:

        pad_mask = torch.ones((source_mask.shape[0], source_mask.shape[1], 1, self.max_n+1), dtype=torch.bool, device=encoder_outputs.device)
        pad_mask[:, :, :, 0] = False

        logits = self.translation_model(encoder_outputs, source_mask, non_contextetualized_input=non_contextualized_repr)
        logits = logits -1000 * (~source_mask).unsqueeze(-1).unsqueeze(-1) * pad_mask

        logits = torch.log_softmax(logits, dim=-1) #shape (batch, input_seq_len, vocab, max_n+1)

        return logits


    def distill_alignment(self, logits, source_mask, targets, target_mask, alignment: torch.Tensor):

        hard_alignment = alignment >= self.alignment_threshold #shape (batch, input, output)

        one_hot_targets = F.one_hot(targets, num_classes = self.vocab.get_vocab_size(self.target_namespace))

        # counts[b, i, v] = sum_o hard_alignment[b,i,o] * targets[b,o,v]
        # counts = torch.einsum("bio, bov, bo -> biv", hard_alignment.long(), one_hot_targets, target_mask.long()) #shape (batch, input, vocab)
        # the above doesn't work on GPUs because einsum isn't implemented for non-floats, so use the following equivalent but hard-to-read code:
        # first expand everything to match the shape biov, then sum out dimension o
        counts = (hard_alignment.unsqueeze(-1) * one_hot_targets.unsqueeze(1) * target_mask.unsqueeze(1).unsqueeze(-1)).sum(2) #shape (batch, input, vocab)

        # We don't supervise alignments to copy/lexicon mappings
        if self._copy_id is not None:
            counts[:, :, self._copy_id] = 0
        if self._cogs_var_id is not None:
            counts[:, :, self._cogs_var_id] = 0

        allowed = torch.cumsum(F.one_hot(counts, num_classes=self.max_n+1), dim=-1) #shape (batch, input, vocab, max_n + 1)
        # allowed[b, i, v, k] = 0 if the alignment contradicts that sequence position i in batch b produces k tokens of type v
        #convert to log space
        allowed = -100_000 * (1-allowed)

        loss = source_mask.unsqueeze(-1) * (allowed + logits).logsumexp(dim=-1)
        loss = -loss.sum(dim=[-1, -2]).mean()

        return loss


    def compute_frequency(self, targets):
        f = torch.zeros((targets.shape[0], self.vocab.get_vocab_size(self.target_namespace)),
                        dtype=torch.long, device=targets.device)

        br = torch.arange(targets.shape[0], device=targets.device)
        for i in range(targets.shape[1]):
            f[br, targets[:, i]] += 1

        f[:, 0] = 0 # Padding with 0 should not be reproduced.

        return f


    def get_non_contextualized_repr(self, source_tokens: TextFieldTensors):
        # token_ids =
        assert len(list(source_tokens.keys())) == 1
        if "token_ids" in source_tokens[list(source_tokens.keys())[0]]:
            token_ids = source_tokens[list(source_tokens.keys())[0]]["token_ids"]
        else:
            token_ids = None

        if isinstance(self.source_embedder, PretrainedTransformerMismatchedEmbedderRho) or isinstance(self.source_embedder, PretrainedTransformerMismatchedEmbedder):
            repr = self.source_embedder._matched_embedder.transformer_model.get_input_embeddings()(token_ids)
        elif isinstance(self.source_embedder, PretrainedTransformerEmbedderWithRho) or isinstance(self.source_embedder, PretrainedTransformerEmbedder):
            repr = self.source_embedder.transformer_model.get_input_embeddings()(token_ids)
        else:
            repr = self.source_embedder(source_tokens)

        return repr

    def get_copy_mask(self, source_tokens: List[List[str]], source_mask):
        """
        Create mask which tokens may copy/use the lexicon.
        :param source_tokens:
        :param source_mask:
        :return:
        """
        x = np.ones(source_mask.shape + (self.vocab.get_vocab_size(self.target_namespace), self.max_n+1), dtype=bool)
        for b, toks in enumerate(source_tokens):
            for i, tok in enumerate(toks):
                x[b, i, self._copy_id, 1:] = tok in self.lexicon

        return torch.from_numpy(x).to(source_mask.device)

    def forward(self, source_tokens: TextFieldTensors,
                metadata: List[Dict],
                target_tokens: Optional[TextFieldTensors] = None,
                rule_mask_copy: Optional[torch.Tensor] = None,
                rule_mask_cogs_var: Optional[torch.Tensor] = None,
                rule_freq_copy: Optional[torch.Tensor] = None,
                rule_freq_cogs_var: Optional[torch.Tensor] = None,
                alignment: Optional[torch.Tensor] = None,
                emphasis: Optional[torch.Tensor] = None,
                copyable_inputs: Optional[torch.Tensor] = None # DON'T NEED THIS ANYMORE
                ) -> Dict[
        str, torch.Tensor]:

        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_source = self.source_embedder(source_tokens)

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        if self.encoder:
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

        non_contextualized_repr = self.get_non_contextualized_repr(source_tokens)

        multiset_logits = self.compute_logits(encoder_outputs, source_mask, readable_source, non_contextualized_repr)
        if self.lexicon and self._copy_id is not None:
            mask = self.get_copy_mask(readable_source, source_mask) #shape (batch, input_seq_len)
            multiset_logits = multiset_logits - 1000 * ~mask
            # set the logits to -1000 for input tokens without lexicon entry to use the lexicon/copying.

        predicted_multisets = torch.argmax(multiset_logits, dim=-1)  # shape (batch, input_seq_len, vocab)

        ret["predicted_multisets"] = predicted_multisets

        if target_tokens:
            targets = target_tokens["tokens"]["tokens"]  # shape (batch_size, output seq length)
            target_mask = util.get_text_field_mask(target_tokens)  # shape (batch_size, output seq length)
            target_lengths = get_lengths_from_binary_sequence_mask(target_mask)
            ret["targets"] = targets
            ret["gold_target_lengths"] = get_lengths_from_binary_sequence_mask(target_mask)

            readable_target = [m["target_tokens"] for m in metadata]
            ret["readable_targets"] = readable_target

            # Compute frequency of target tokens
            freqs = self.compute_frequency(targets) #shape (batch, vocab_size)

            dist = self.compute_dist(multiset_logits, source_mask, max_l=torch.max(freqs).cpu().numpy()+1)

            loss = self.compute_loss(dist, freqs, self.mask_out_items)

            if rule_mask_copy is not None:
                loss += self.compute_dynamic_loss_for(multiset_logits, rule_mask_copy, rule_freq_copy, self._copy_id)

            if rule_freq_cogs_var is not None:
                loss += self.compute_dynamic_loss_for(multiset_logits, rule_mask_cogs_var, rule_freq_cogs_var, self._cogs_var_id)

            if alignment is not None and hasattr(self, "epoch") and self.epoch < self.pretrain_epochs:
                loss += self.alignment_loss_weight * self.distill_alignment(multiset_logits, source_mask, targets, target_mask, alignment)

            # if not self.training:
            #     # Compute dist over all possible values
                  # The non-optimal performance of the dyn. programming starts to bite here!
                  # Actually, we don't need to compute dist!
                  # We need: predict most likely frequencies per input token AND find most likely explanation of a multiset (create input for permutations training data)
            #     dist = self.compute_dist(multiset_logits, source_mask)

            if not self.training:

                readable_multisets = self.make_readable(predicted_multisets, source_mask, readable_source)

                ret["readable_multisets"] = readable_multisets

                total_readable_multisets = self.assignment_to_freq_count(readable_multisets)
                gold_readable_freqs = [Counter(f) for f in readable_target]

                if self.lower_case_eval:
                    total_readable_multisets = [{k.lower(): v for k, v in ms.items()} for ms in total_readable_multisets]
                    gold_readable_freqs = [{k.lower(): v for k, v in ms.items()} for ms in gold_readable_freqs]

                self.freq_acc.add_instances(total_readable_multisets,
                                            gold_readable_freqs)

                if all(x is not None for x in categories):
                    self.track_metric_by_cat.add_instances(total_readable_multisets, gold_readable_freqs, categories)

            #TODO: new problem: "image file" in Okapi -> format = jpeg AND format = png

            if False: # Debug mode
                for o in range(len(readable_multisets)):
                    if total_readable_multisets[o] != Counter(readable_target[o]):
                        print(readable_source[o])
                        print(readable_target[o])
                        for i, x in enumerate(readable_multisets[o]):
                            print(readable_source[o][i], dict(x))
                        print("===")


            ret["loss"] = loss

        return ret


    def readable_freq(self, freqs: torch.Tensor):
        counts = Counter()
        for v in range(self.vocab.get_vocab_size(self.target_namespace)):
            if freqs[v] > 0:
                tok = self.vocab.get_token_from_index(v, self.target_namespace)
                counts[tok] = freqs[v]
        return counts


    def make_sequence_readable(self, seq: torch.Tensor, alignment: torch.Tensor, readable_source: List[List[str]]) -> List[List[str]]:
        """
        This turns a linearized sequence into a sequence of string tokens.
        :param seq: shape (batch, output, seq_len)
        :param alignment: shape (batch, input_seq_len, output_seq_len)
        :param readable_source:
        :return:
        """
        alignment_mapping = torch.argmax(alignment, dim=1).detach().cpu().numpy() #shape (batch, output_seq_len)
        seq = seq.detach().cpu().numpy()
        res = []
        for b in range(seq.shape[0]):
            sent = []
            for i in range(seq.shape[1]):
                if seq[b, i] == self._copy_id:
                    sent.append(self.lexicon[readable_source[b][alignment_mapping[b, i]]])
                elif seq[b, i] == self._cogs_var_id:
                    sent.append(str(alignment_mapping[b, i]))
                elif seq[b,i] > 0: #not PADDING
                    sent.append(self.vocab.get_token_from_index(seq[b,i], self.target_namespace))
            res.append(sent)
        return res


    def make_readable(self, choice: torch.Tensor, source_mask: torch.Tensor, readable_source: List[List[str]]) -> List[List[Dict[str, int]]]:
        """

        :param choice: shape (batch, input_seq_len, vocab size) containing how many tokens of that type an input token generates
        :return:
        """
        choice = choice.cpu().numpy()
        source_lengths = source_mask.sum(dim=-1).cpu().numpy()
        batch, input_seq_len, vocab_size = choice.shape

        r = []
        for b in range(batch):
            batch_elem = []
            for i in range(source_lengths[b]):
                c = self.readable_freq(choice[b, i])
                if COPY_SYMBOL in c:
                    copy_count = c[COPY_SYMBOL]
                    del c[COPY_SYMBOL]
                    if self.lexicon:
                        c[self.lexicon[readable_source[b][i]]] = copy_count
                    else:
                        c[readable_source[b][i]] = copy_count
                if COGS_VAR in c:
                    cogs_var_count = c[COGS_VAR]
                    del c[COGS_VAR]
                    c[str(i)] = cogs_var_count

                batch_elem.append(c)
            r.append(batch_elem)

        # print(r[0])
        return r

    def assignment_to_freq_count(self, readable_assignments: List[List[Dict[str, int]]]) -> List[Dict[str, int]]:
        return [sum(r, start=Counter()) for r in readable_assignments]




    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        d = dict()
        d.update(self.freq_acc.get_metric(reset))
        d.update(self.track_metric_by_cat.get_metrics(reset))
        return d

