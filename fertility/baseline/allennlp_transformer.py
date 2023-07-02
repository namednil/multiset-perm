from typing import Optional, Dict, Union, List

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.nn import util
from allennlp.nn.util import get_lengths_from_binary_sequence_mask

from .relative_trafo.layers import Transformer
from .relative_trafo.layers.transformer import RelativeTransformer, UniversalTransformer, UniversalRelativeTransformer
from .relative_trafo.trafo_models import TransformerEncDecModel
from ..eval.lev import MyMetric

END_TOKEN = "@end@"

@Model.register("csordas_transformer")
class AllennlpRelTrafo(Model):

    transformer_type2class = {"vanilla": Transformer,
                              "relative": RelativeTransformer, "universal": UniversalTransformer,
                              "relative_universal": UniversalRelativeTransformer,
                              "universal_relative": UniversalRelativeTransformer}

    def __init__(self, vocab: Vocabulary, transformer_type: str,
                 target_namespace: str = "target_tokens",
                 state_size: int = 512, dim_feedforward: int = 2048,
                 nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 decode_on_dev_after_epoch: Optional[int] = 0,
                 scale_mode: Optional[str] = "none",
                 embedding_init: Optional[str] = "pytorch",
                 metrics: Optional[List[MyMetric]] = None):
        super().__init__(vocab)

        self.decode_on_dev_after_epoch = decode_on_dev_after_epoch
        assert transformer_type in self.transformer_type2class

        self.metrics = metrics or []

        self.target_namespace = target_namespace
        self.enc_dec = TransformerEncDecModel(self.vocab.get_vocab_size(), self.vocab.get_vocab_size(target_namespace)-1,
                                              self.vocab.get_token_index(END_TOKEN, target_namespace),
                                              state_size,
        max_len, transformer=self.transformer_type2class[transformer_type], nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                              dropout=dropout, scale_mode=scale_mode, dim_feedforward=dim_feedforward,
                                              embedding_init=embedding_init)#-1 because the code adds 1 under the hood.
                                              # scale_mode="down")
        self.max_len = max_len



    def make_readable(self, batch_of_tokens: torch.Tensor, lengths: torch.Tensor) -> List[List[str]]:
        r = []
        for batch, length_of_seq in zip(batch_of_tokens.cpu().numpy(), lengths.cpu().numpy()):
            pred = [self.vocab.get_token_from_index(tok, namespace=self.target_namespace) for tok in batch]
            if END_TOKEN in pred:
                pred = pred[: pred.index(END_TOKEN)]
            r.append(pred)
        return r

    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        if "targets" in output_dict:
            output_dict["readable_targets"] = self.make_readable(output_dict["targets"], output_dict["gold_target_lengths"])

        if "predictions" in output_dict:
            output_dict["readable_predictions"] = self.make_readable(output_dict["predictions"], output_dict["predicted_target_lengths"])

        if "predictions_given_length" in output_dict:
            output_dict["readable_predictions_given_length"] = self.make_readable(output_dict["predictions_given_length"], output_dict["gold_target_lengths"])
        return output_dict


    def forward(self, source_tokens: TextFieldTensors, target_tokens: Optional[TextFieldTensors],
                metadata: List[Dict]) -> Dict[
        str, torch.Tensor]:

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        src_len = get_lengths_from_binary_sequence_mask(source_mask)

        source_token_ids = source_tokens["tokens"]["tokens"]

        ret = dict()

        if target_tokens:
            targets = target_tokens["tokens"]["tokens"]  # shape (batch_size, output seq length)
            target_mask = util.get_text_field_mask(target_tokens)  # shape (batch_size, output seq length)
            tgt_len = get_lengths_from_binary_sequence_mask(target_mask)

            # result = self.enc_dec(source_token_ids, src_len, targets[:, :-1], tgt_len, True) #shape (batch_size, target_seq_length, vocab size)
            result = self.enc_dec(source_token_ids, src_len, targets[:, 1:], tgt_len, True) #shape (batch_size, target_seq_length, vocab size)
            # result.data = result.data[:, :-1, :].contiguous()
            targets_for_loss = targets[:, 1:].contiguous()
            target_mask_for_loss = target_mask[:, 1:].contiguous()
            # ret["loss"] = my_sequence_cross_entropy_with_logits(result.data, targets_for_loss, target_mask_for_loss)
            # ret["loss"] = my_sequence_cross_entropy_with_logits(result.data, targets_for_loss, torch.ones_like(target_mask_for_loss))
            ret["loss"] = my_sequence_cross_entropy_with_logits(result.data, targets_for_loss, target_mask_for_loss)
            # ret["loss"] = my_sequence_cross_entropy_with_logits(result.data, targets, target_mask)

        if not self.training and (not hasattr(self, "epoch") or self.epoch >= self.decode_on_dev_after_epoch):
            predictions = self.enc_dec(source_token_ids, src_len, None, None, False, self.max_len-2)
            predictions.data = torch.argmax(predictions.data, dim=-1)
            if target_tokens:
                readable_pred = self.make_readable(predictions.data, predictions.length)
                # readable_targets = [x[1:] for x in self.make_readable(targets, tgt_len)] #remove @start@
                readable_targets = [m["target_tokens"] for m in metadata]
                # self.lev_metric.add_instances(readable_pred, readable_targets)
                for m in self.metrics:
                    m.add_instances(readable_pred, readable_targets)

                ret["readable_pred"] = readable_pred
                ret["readable_targets"] = readable_targets

        return ret

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        d = dict()
        if not self.training:
            for m in self.metrics:
                d.update(m.get_metric(reset))
        return d



###########################


def my_sequence_cross_entropy_with_logits(
        logits: torch.FloatTensor,
        targets: torch.LongTensor,
        weights: Union[torch.FloatTensor, torch.BoolTensor],
        average: str = "batch",
        label_smoothing: float = None,
        gamma: float = None,
        alpha: Union[float, List[float], torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the `torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for trafo_models which use padding.

    # Parameters

    logits : `torch.FloatTensor`, required.
        A `torch.FloatTensor` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : `torch.LongTensor`, required.
        A `torch.LongTensor` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : `Union[torch.FloatTensor, torch.BoolTensor]`, required.
        A `torch.FloatTensor` of size (batch, sequence_length)
    average: `str`, optional (default = `"batch"`)
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If `None`, return a vector
        of losses per batch element.
    label_smoothing : `float`, optional (default = `None`)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like `[0.05, 0.05, 0.85, 0.05]` if the 3rd class was
        the correct label.
    gamma : `float`, optional (default = `None`)
        Focal loss[*] focusing parameter `gamma` to reduces the relative loss for
        well-classified examples and put more focus on hard. The greater value
        `gamma` is, the more focus on hard examples.
    alpha : `Union[float, List[float]]`, optional (default = `None`)
        Focal loss[*] weighting factor `alpha` to balance between classes. Can be
        used independently with `gamma`. If a single `float` is provided, it
        is assumed binary case using `alpha` and `1 - alpha` for positive and
        negative respectively. If a list of `float` is provided, with the same
        length as the number of classes, the weights will match the classes.
        [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, 2017, pp. 2999-3007.

    # Returns

    `torch.FloatTensor`
        A torch.FloatTensor representing the cross entropy loss.
        If `average=="batch"`, the returned loss is a scalar.
        If `average is None`, the returned loss is a vector of shape (batch_size, seq_len).

    """
    if average not in {None, "batch"}:
        raise ValueError("Got average f{average}, expected one of None or 'batch'")

    # make sure weights are float
    weights = weights.to(logits.dtype)
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1.0 - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor

    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):

            # shape : (2,)
            alpha_factor = torch.tensor(
                [1.0 - float(alpha), float(alpha)], dtype=weights.dtype, device=weights.device
            )

        elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):

            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)

            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(
                ("alpha must be float, list of float, or torch.FloatTensor, {} provided.").format(
                    type(alpha)
                )
            )
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(
            *targets.size()
        )
        weights = weights * alpha_factor

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        smoothed_targets = torch.full_like(log_probs_flat, smoothing_value).scatter_(
            -1, targets_flat, 1.0 - label_smoothing + smoothing_value
        )
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims)
        num_non_empty_sequences = (weights_batch_sum > 0).sum()
        return per_batch_loss.sum() / num_non_empty_sequences
    else:
        return negative_log_likelihood
