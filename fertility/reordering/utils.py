import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
from allennlp.nn import Activation
from allennlp.nn.util import get_device_of
from torch.nn import Module

def scale_grad(t: torch.Tensor, s: float):
    detached = t.detach()
    zero_with_grad = t - detached
    return s*zero_with_grad + detached

def sinusoidal_pos_embedding(d_model: int, max_len: int = 5000, pos_offset: int = 0, f: int = 1,
                             device: Optional[torch.device] = None):
    pe = torch.zeros(max_len, d_model, device=device)
    position = f * torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1) + pos_offset
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def create_intersection_mask(mask1, mask2):
    """

    :param mask1: shape (batch_size, seq_len1)
    :param mask2: shape (batch_size, seq_len2)
    :return: shape (batch_size, seq_len1, seq_len2)
    """
    return mask1.unsqueeze(2) & mask2.unsqueeze(1)

def concat_sum(seq1 : torch.Tensor, seq2 : torch.Tensor, mask1 : Optional[torch.Tensor] = None, mask2 : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    @type seq1: first sequence, shape (batch_size, seq_len1, input_dim1)
    @type seq2: second sequence, shape (batch_size, seq_len2, input_dim2)
    @type mask1: mask of first sequence, shape (batch_size, seq_len1)
    @type mask2: mask of second sequence, shape (batch_size, seq_len2)

    @return a tuple with the concatenation of every pair passed through the FF + corresponding mask.
        Shapes: (batch_size, seq_len1, seq_len2, hidden_size)
                (batch_size, seq_len1, seq_len2)
                An element is masked if at least one of the corresponding vectors was masked.
    """

    batch_size, seq_len1, hidden_dim = seq1.shape
    _, seq_len2, _ = seq2.shape

    concatenated = seq1.unsqueeze(2) + seq2.unsqueeze(1)

    mask = None
    if mask1 is not None and mask2 is not None:
        mask = create_intersection_mask(mask1, mask2)

    return concatenated, mask

class ConcatMLP(Module):
    """
    Intuitively, takes two list of (batched) vectors and creates an output tensor
    that contains the result of concatenating every pair and feeding it through a feed-forward neural network.
    If you set the activation to be linear and bias=False, you simply get a trainable matrix multiplication instead of fully-fledged feed-forward pass.
    """

    def __init__(self, hidden_size : int, input_dim1 : int, input_dim2 : int, activation : Activation, bias : bool = True):
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_size = hidden_size
        self.activation = activation

        self.W1 = nn.Linear(input_dim1, hidden_size, bias = bias)
        self.W2 = nn.Linear(input_dim2, hidden_size, bias = False)

    def forward(self, seq1 : torch.Tensor, seq2 : torch.Tensor, mask1 : Optional[torch.Tensor] = None, mask2 : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        @type seq1: first sequence, shape (batch_size, seq_len1, input_dim1)
        @type seq2: second sequence, shape (batch_size, seq_len2, input_dim2)
        @type mask1: mask of first sequence, shape (batch_size, seq_len1)
        @type mask2: mask of second sequence, shape (batch_size, seq_len2)

        @return a tuple with the concatenation of every pair passed through the FF + corresponding mask.
            Shapes: (batch_size, seq_len1, seq_len2, hidden_size)
                    (batch_size, seq_len1, seq_len2)
                    An element is masked if at least one of the corresponding vectors was masked.
        """

        concatenated, mask = concat_sum(self.W1(seq1), self.W2(seq2), mask1, mask2)
        return self.activation(concatenated), mask



# import numba

# import numpy as np

# @numba.njit()
# def simple_sinusoid(frac, num_dim):
#     vec = np.zeros(num_dim, dtype=np.float32)
#     for i in range(num_dim):
#         # vec[i] = np.sin(2 * np.pi * frac + i) #without times 2???
#         # vec[i] = np.sin(np.pi * frac + i) #without times 2???
#         vec[i] = np.sin(0.5 * np.pi * frac + i) #without times 2???
#     return vec
#
# @numba.njit()
# def simple_sinusoids(lengths, num_entries, num_dim):
#     batch_size = lengths.shape[0]
#     z = np.zeros((batch_size, num_entries, num_entries, num_dim), dtype=np.float32)
#
#     # for b in range(batch_size):
#     #     for i in range(num_entries):
#     #         for j in range(num_entries):
#     #             repr = simple_sinusoid((i-j)/lengths[b], num_dim) # don't use abs?
#     #             z[b, i, j] = repr
#
#     for b in range(batch_size):
#         for i in range(num_entries):
#             for j in range(i, num_entries):
#                 repr = simple_sinusoid(abs(i-j)/lengths[b], num_dim) # don't use abs?
#                 # repr = simple_sinusoid((i-j)/lengths[b], num_dim) # don't use abs?
#                 # z[b, i, j] = repr
#                 z[b, i, j] = repr
#                 z[b, j, i] = repr
#     return z


class RelativeConcatMLP(Module):
    """
    Intuitively, takes two list of (batched) vectors and creates an output tensor
    that contains the result of concatenating every pair and feeding it through a feed-forward neural network.
    If you set the activation to be linear and bias=False, you simply get a trainable matrix multiplication instead of fully-fledged feed-forward pass.
    """

    def __init__(self, hidden_size : int, input_dim1 : int, input_dim2 : int, activation : Activation, bias : bool = True, max_len: int = 300, add_relative_bias: bool = True):
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_size = hidden_size
        self.activation = activation

        self.W1 = nn.Linear(input_dim1, hidden_size, bias = bias)
        self.W2 = nn.Linear(input_dim2, hidden_size, bias = False)
        self.W3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.05)

        assert hidden_size % 2 == 0

        self.add_relative_bias = add_relative_bias

        if add_relative_bias:
            relative_distance_encoding = torch.zeros((1, max_len, max_len, hidden_size))
            pos_end = sinusoidal_pos_embedding(hidden_size, max_len)
            for i in range(max_len):
                for j in range(max_len):
                    relative_distance_encoding[0, i, j, :] = pos_end[abs(j-i)] #+ (pos_end[0] if i < j else 0) #TODO: remove the last bit.

            self.register_buffer("relative_distance_encoding", relative_distance_encoding, persistent=False)

    def forward(self, seq1 : torch.Tensor, seq2 : torch.Tensor, mask1 : Optional[torch.Tensor] = None, mask2 : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        @type seq1: first sequence, shape (batch_size, seq_len1, input_dim1)
        @type seq2: second sequence, shape (batch_size, seq_len2, input_dim2)
        @type mask1: mask of first sequence, shape (batch_size, seq_len1)
        @type mask2: mask of second sequence, shape (batch_size, seq_len2)

        @return a tuple with the concatenation of every pair passed through the FF + corresponding mask.
            Shapes: (batch_size, seq_len1, seq_len2, hidden_size)
                    (batch_size, seq_len1, seq_len2)
                    An element is masked if at least one of the corresponding vectors was masked.
        """

        concatenated, mask = concat_sum(self.W1(seq1), self.W2(seq2), mask1, mask2)
        l1 = seq1.shape[1]
        l2 = seq2.shape[1]
        if self.add_relative_bias:
            concatenated = self.activation(concatenated) + self.relative_distance_encoding[:, :l1, :l2]
        else:
            concatenated = self.activation(concatenated)

        return self.activation(self.W3(concatenated)), mask