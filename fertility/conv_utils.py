from typing import Union, List

import numpy
import numpy as np
import torch
import torch.nn.functional as F


#torch.backends.cudnn.benchmark = False

# torch.use_deterministic_algorithms(True)


def cumulative_sum(t: torch.Tensor, l: int):
    """
    t: (batch_size, number of random vars, support (0.... n-1))
    l: maximum sum

    returns tensor of shape (batch_size, number of random vars, l)
    """
    status = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    batch_size, num_rv, n = t.shape
    s = torch.zeros(batch_size, l, device=t.device)
    s[:, 0] = 1.0
    tensors = []
    t_flipped = t.flip(2)  # only flip it once.
    for i in range(t.shape[1]):
        s = add_rand_var(s, t_flipped[:, i, :], l, True)
        tensors.append(s)
    torch.use_deterministic_algorithms(status)

    return torch.stack(tensors, dim=1)


def sum_of_rand_vars(t: torch.Tensor, l: int):
    """
    t: (batch_size, number of random vars, support (0.... n-1))
    l: maximum sum

    returns tensor of shape (batch_size, l)
    """
    status = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    batch_size, num_rv, n = t.shape
    s = torch.zeros(batch_size, l, device=t.device)
    s[:, 0] = 1.0
    t_flipped = t.flip(2)  # only flip it once.
    for i in range(t.shape[1]):
        s = add_rand_var(s, t_flipped[:, i, :], l, True)

    torch.use_deterministic_algorithms(status)
    return s


def add_rand_var(null, eins, max_l, eins_flipped_already: bool = False):
    """
    Adding two (batched) discrete random variables
    null: shape (batch_size, n) representing the probabilities for 0 ... n-1
    eins: shape (batch_size, k) representing the probabilities for 0 ... k-1
    max_l: int

    returns a tensor of shape (batch_size, max_l)
    """
    if eins_flipped_already:
        eins_flip = eins
    else:
        eins_flip = eins.flip(1)  # have to flip it because pytorch implements cross-correlation, not convolution!

    # Full convolution that computes the probabilities of 0 ... n +k :
    # null_padded = F.pad(null, (eins.shape[1]-1, eins.shape[1]-1))
    # PAD PAD PAD X X X X PAD PAD PAD
    # output_length = (null.shape[1] + (eins.shape[1]-1) + (eins.shape[1]-1)) - (eins.shape[1]) + 1
    # solve for padding on the right:
    # padding right = output_length - null.shape[1] - (eins.shape[1]-1) + eins.shape[1] - 1 = output_length - null.shape[1]
    # ~ print("output length", output_length)

    null_padded = F.pad(null, (eins.shape[1] - 1, max_l - null.shape[1]))
    # ~ print("Null padded shape", null_padded.shape)

    return F.conv1d(null_padded.unsqueeze(0), eins_flip.unsqueeze(1), padding=0,
                    groups=null_padded.shape[0]).squeeze(0)


import numba


@numba.njit
def _conv_create_mask(n, m):
    a = np.zeros((1, 1, m, n, m), dtype=np.float32) - 100_000  # we add two numbers c = a + b, where a < m, b < n and c < m
    for i in range(m):
        for j in range(m):
            if 0 <= i - j < n:
                a[0, 0, j, i - j, i] = 0

    return a

def multi_set_pred_dyn(x:torch.Tensor, conv_mask):
    """
    Returns a tensor of shape (batch, vocab, max_n)
    This implementation is not asymptotically optimal (in max_n) but it operates in logspace.

    conv_mask: shape (1, 1, max_l, max_n, max_l)
    """
    # x, shape: (batch, seq_len, vocab, max_l)
    max_l = conv_mask.shape[-3]
    assert max_l == conv_mask.shape[-1]
    assert conv_mask.shape[3] == x.shape[-1]

    state = torch.zeros((x.shape[0], x.shape[2], conv_mask.shape[-1]), device=x.device) - 100_000
    state[:, :, 0] = 0.0

    unsq_x = x.unsqueeze(3).unsqueeze(-1)  # shape (batch, seq_len, vocab, 1, max_n, 1)

    # mask = torch.from_numpy(_conv_create_mask(x.shape[-1], max_l)).to(x.device)

    for i in range(x.shape[1]):
        a = state.unsqueeze(-1).unsqueeze(-1)  # shape (batch, vocab, max_n, 1, 1)
        j = (a + unsq_x[:, i] + conv_mask)  # shape (batch, vocab, max_n, max_n, max_n)
        state = j.logsumexp(dim=[2, 3])
    return state




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
    in the sequence. This allows loss computations for models which use padding.

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
