from collections import Counter
from os import PathLike
from typing import Optional, Dict, Any, Union, List, Tuple

import allennlp
import numpy as np
import torch
import transformers

from allennlp.common.lazy import Lazy
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models.model import Model
from allennlp.modules.transformer.t5 import T5 as T5Module, T5Output, IntT, BoolT
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.checkpoint import CheckpointWrapper
from allennlp.training.metrics import ROUGE, BLEU

from fertility.eval.exclusion_tracker import InclusionMetric
from fertility.eval.lev import MyMetric
from fertility.multiset_reorder import linearize_multisets


@Model.register("perm-t5")
class PermT5(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        multiset_model_path: str,
        metrics: Optional[List[MyMetric]] = None,
        beam_search: Lazy[BeamSearch] = Lazy(BeamSearch, beam_size=3, max_steps=50),
        checkpoint_wrapper: Optional[CheckpointWrapper] = None,
        weights_path: Optional[Union[str, PathLike]] = None,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._model_name = model_name
        # We only instantiate this when we need it.
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.t5 = T5Module.from_pretrained_module(
            model_name,
            beam_search=beam_search,
            ddp_accelerator=self.ddp_accelerator,
            checkpoint_wrapper=checkpoint_wrapper,
            weights_path=weights_path,
        )
        self.sep = "</s>"

        self.multiset_model = allennlp.models.load_archive(multiset_model_path).model

        from fertility.baseline.allennlp_bart import Acc
        self._metrics = [Acc()] + ([] if not metrics else metrics)
        self.inclusion_metric = InclusionMetric("multiset_was_correct")
        self.is_permutation = InclusionMetric("predicted_essentially_permutation")

    def _post_load_state_dict(
        self, missing_keys: List[str], unexpected_keys: List[str]
    ) -> Tuple[List[str], List[str]]:
        missing_keys_to_ignore = [
            "t5.encoder.token_embeddings.weight",
            "t5.decoder.token_embeddings.weight",
        ]
        if self.t5._tie_word_embeddings:
            missing_keys_to_ignore.append("t5.lm_head.weight")
        for key in missing_keys_to_ignore:
            if key in missing_keys:
                missing_keys.remove(key)
        return missing_keys, unexpected_keys

    def compute_linearized_and_repr(self, predicted_multisets, max_length, readable_source):
        linearized_multisets_output, alignment, pred_target_mask, copy_info = linearize_multisets(
            predicted_multisets, max_length)

        readable_linearized_outputs = self.multiset_model.make_sequence_readable(linearized_multisets_output, alignment,
                                                                                 readable_source)

        return readable_linearized_outputs

    def forward(self, source_tokens: TextFieldTensors,
                metadata: List[Dict],
                target_tokens: Optional[TextFieldTensors] = None,
                source_to_copy_mask: Optional[torch.Tensor] = None,
                alignment: Optional[torch.Tensor] = None,
                copyable_inputs: Optional[torch.Tensor] = None,
                emphasis: Optional[torch.Tensor] = None) -> Dict[
        str, torch.Tensor]:


        d = {"source_tokens": source_tokens, "metadata": metadata,
             "alignment": alignment, "emphasis": emphasis, "copyable_inputs": copyable_inputs}
        multiset_output = self.multiset_model(**d)

        predicted_multisets = multiset_output["predicted_multisets"] #shape (batch, input_seq_len, vocab)

        readable_source = [m["source_tokens"] for m in metadata]

        decoder_attention_mask: Optional[torch.Tensor] = None
        targets = None

        if target_tokens:
            # targets, decoder_attention_mask = target_tokens["tokens"]["token_ids"], target_tokens["tokens"]["mask"]  # shape (batch_size, output seq length)
            t5_targets = [" ".join(x["target_tokens"]) for x in metadata]
            t5_tok_targets = self._tokenizer(t5_targets, return_tensors="pt", padding=True)
            targets = t5_tok_targets["input_ids"].to(predicted_multisets.device)
            decoder_attention_mask = t5_tok_targets["attention_mask"].to(predicted_multisets.device)

            linearized_strs = self.compute_linearized_and_repr(predicted_multisets, targets.shape[1], readable_source)
        else:
            if self.training:
                raise ValueError("'target_tokens' required during training")
            max_length = int(predicted_multisets.sum([1,2]).max().cpu().numpy())
            linearized_strs = self.compute_linearized_and_repr(predicted_multisets, max_length, readable_source)

        # Tokenize linearized strs and batch them
        t5_input_strs = [" ".join(s) + " " + self.sep + " " + " ".join(l) for s,l in zip(readable_source, linearized_strs)]

        t5_tokenized = self._tokenizer(t5_input_strs, return_tensors="pt", padding=True)
        input_ids = t5_tokenized["input_ids"].to(predicted_multisets.device)
        attention_mask = t5_tokenized["attention_mask"].to(predicted_multisets.device)

        linearized_freqs = [Counter(self._tokenizer(" ".join(s))["input_ids"]) for s in linearized_strs]
        if target_tokens:
            gold_freqs = [Counter(x) for x in self._tokenizer(t5_targets, padding=False)["input_ids"]]
            multiset_was_correct = np.array([x == y for x,y in zip(linearized_freqs, gold_freqs)])
            self.inclusion_metric.add_instances(multiset_was_correct)

            #If training: discard instances with incorrect multisets predicted in first stage.
            if self.training:
                multiset_was_correct = torch.from_numpy(multiset_was_correct).to(input_ids.device)
                input_ids = input_ids[multiset_was_correct]
                attention_mask = attention_mask[multiset_was_correct]
                targets = targets[multiset_was_correct]
                decoder_attention_mask = decoder_attention_mask[multiset_was_correct]

        output: T5Output = self.t5(
            input_ids,
            attention_mask=attention_mask,
            labels=targets,
            decoder_attention_mask=decoder_attention_mask,
        )
        output_dict: Dict[str, torch.Tensor] = {}

        if self.training:
            assert output.loss is not None
            output_dict["loss"] = output.loss
        else:
            # Shape: (batch_size, beam_size, num_tokens)
            assert output.predictions is not None
            # Shape: (batch_size, beam_size)
            assert output.predicted_log_probs is not None
            # Shape: (batch_size, num_tokens)
            output_dict["predictions"] = output.predictions[:, 0, :]
            # Shape: (batch_size, )
            output_dict["predicted_log_probs"] = output.predicted_log_probs[:, 0]

            output_dict["predicted_text"] = self._tokenizer.batch_decode(
                output_dict["predictions"], skip_special_tokens=True  # type: ignore[attr-defined]
            )

            self.count_perms(linearized_freqs, output_dict["predicted_text"])

            for metric in self._metrics:
                metric.add_instances(output_dict["predicted_text"], self._tokenizer.batch_decode(targets, skip_special_tokens=True))

            if targets is not None:
                assert output.loss is not None
                output_dict["loss"] = output.loss

        return output_dict

    def count_perms(self, linearized_freqs, predicted_text):
        predicted_perms = []  # [Counter(self._tokenizer(x)["input_ids"]) == linearized_freqs[i] for i,x in enumerate(output_dict["predicted_text"])]
        for i, x in enumerate(predicted_text):
            pred_f = Counter(self._tokenizer(x)["input_ids"])
            lf = Counter(linearized_freqs[i])
            # The T5 tokenizer sometimes adds weird token 3s (which corresponds to an empty string)
            # see https://discuss.huggingface.co/t/2-tokens-for-one-character-in-t5/14960
            # it's not quite clear why and when that happens.
            # For the purpose of this metric, let's just delete all instances of that character.
            del pred_f[3]
            del lf[3]
            predicted_perms.append(pred_f == lf)
        self.is_permutation.add_instances(predicted_perms)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.inclusion_metric.get_metric(reset)
        metrics.update(self.is_permutation.get_metric(reset))
        if not self.training:
            for metric in self._metrics:
                metrics.update(metric.get_metric(reset=reset))
        return metrics
