# Copyright 2023 Matthias Lindemann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import Optional, Tuple, Union

import numpy as np
import torch

def gen_kl(log_x, log_y):
    return (log_x.exp() * (log_x-log_y)).sum() - torch.logsumexp(log_x, dim=tuple(range(len(log_x.shape)))).exp() + torch.logsumexp(log_y, dim=tuple(range(len(log_y.shape)))).exp()

def gen_kl_with_mask(log_x, log_y, mask):
    log_mask = -1000 * ~mask
    return (mask * log_x.exp() * (log_x-log_y)).sum() - torch.logsumexp(log_x + log_mask, dim=tuple(range(len(log_x.shape)))).exp() + torch.logsumexp(log_y + log_mask, dim=tuple(range(len(log_y.shape)))).exp()


def kl(log_x, log_y):
    return (log_x.exp() * (log_x-log_y)).sum()

def kl_with_mask(log_x, log_y, mask):
    return (mask * log_x.exp() * (log_x-log_y)).sum()


def proj3(log_z: torch.Tensor, log_w: torch.Tensor, in_place: bool = False, log_red_w_buf: Optional[torch.Tensor] = None, log_z_red_buffer: Optional[torch.Tensor] = None):
    """
    Returns log(x) and log(y) that are solutions to
    min KL(x|log_z) + KL(w|log_y)
    with the following constraints (not written in logspace)
        sum_i x_ij = 1 for all j
        sum_k y_ijk = x_ij for all i, j

    This corresponds to Prop. 2 in our paper.

    log_z: shape (batch, n, m)
    log_w: shape (batch, n, m, n)
    """
    batch, n, m = log_z.shape
    assert log_w.shape == (batch, n, m, n)

    if in_place:
        assert log_red_w_buf is not None
        assert log_z_red_buffer is not None
        torch.logsumexp(log_w, dim=-1, keepdim=True, out=log_red_w_buf)  # shape (batch, n, m, 1)

        log_z += log_red_w_buf.squeeze(3)
        log_z /= 2

        torch.logsumexp(log_z, dim=1, keepdim=True, out=log_z_red_buffer)  # sum over first non-batch dimension!
        log_z -= log_z_red_buffer

        log_w += log_z.unsqueeze(3)
        log_w -= log_red_w_buf
        log_x = log_z
        log_y = log_w
    else:

        log_reduced_w = torch.logsumexp(log_w, dim=-1,
                                        keepdim=True)
        # shape (batch, n, n, 1), summed out last dimension (sum_{k'} B_{i,j,k'} in paper)

        log_q = (log_z + log_reduced_w.squeeze(3)) / 2 # log T in Prop. 2

        log_x = log_q - torch.logsumexp(log_q, dim=1, keepdim=True)  # sum over first non-batch dimension!

        log_y = log_x.unsqueeze(3) + log_w - log_reduced_w

    return log_x, log_y


_last_solve_steps = None

def stabilised_bregman(log_z: torch.Tensor, log_w: torch.Tensor, iter=100, tol=0.0001, omega=1.0):
    """
    Use shift-invariance for better numerical properties. Maybe doesn't matter much in practice.
    :param log_z:
    :param log_w:
    :param iter:
    :param tol:
    :param omega:
    :return:
    """
    n = log_w.shape[1]
    mask = torch.ones((1, 1, n, 1), dtype=torch.bool, device=log_w.device)
    mask[0] = False
    log_w_mod = log_w - mask * torch.amax(log_w, dim=[1, 3], keepdim=True)
    log_z_mod = log_z - torch.amax(log_z, dim=1, keepdim=True)

    return bregman(log_z_mod, log_w_mod, iter, tol, omega)

def in_place_omega_update(old, new, omega):
    """
    Computes new = (1-omega) * old + omega * new, changing both "old" and "new". Returns "new"
    :param old:
    :param new:
    :param omega:
    :return:
    """
    if omega == 1.0:
        return new
    torch.mul(new, omega, out=new)
    torch.mul(old, (1-omega), out=old)
    torch.add(new, old, out=new)
    return new

def omega_update(old, new, omega):
    if omega == 1.0:
        return new
    return (1-omega) * old + omega * new


def bregman(log_z: torch.Tensor, log_w: torch.Tensor, iter=100, tol=0.0001, omega=1.0, in_place: bool = False):
    """
    Runs Bregman's method. Corresponds to Algorithm 1 in our paper (https://arxiv.org/abs/2305.16954).
    Omega > 1.0 is an overrelaxation, which can speed up convergence when carefully chosen
    (but we might lose convergence guarantees).

    in_place saves memory but cannot be used for automatic differentiation (would need implicit differentiation for that)

    :param log_z: shape (batch, n, n), corresponds to log U in the paper, log_z[b,i,j] is the score for input i being mapped to output j for batch element b.
    :param log_w: shape (batch, n, n, n), corresponds to log W in the paper,
        log_w[b,i,j,k] is the score for input i being mapped to output j and input k being mapped to output j-1 in batch element b.
    :param iter:
    :return:
    """
    global _last_solve_steps
    _last_solve_steps = iter

    batch, n, _ = log_z.shape

    log_z_compare_buffer = torch.empty((batch, n, n), device=log_w.device)

    part_w_a = log_w[:, :, 0, :].clone()
    part_w = log_w[:, :, 1:, :]

    if in_place:
        log_w_red_buffer = torch.empty((batch, n, n-1, 1), device=log_w.device)
        log_z_red_buffer = torch.empty((batch, 1, n-1), device=log_w.device)
        log_z_red_buffer_c = torch.empty((batch, n, 1), device=log_w.device)
        copy_log_z = torch.empty_like(log_z)
        copy_part_w = torch.empty_like(part_w)
    else:
        log_w_red_buffer, log_z_red_buffer = None, None

    for i in range(iter):
        if in_place:
            old_log_z = log_z.clone()
        else:
            old_log_z = log_z

        if in_place and omega != 1.0:
            # We need to make a copy of the old points, otherwise we can't interpolate.
            copy_log_z.copy_(log_z)
            copy_part_w.copy_(part_w)

        # PART (i) takes care of
        # sum_i Z_ij = 1
        # sum_k W_ijk = x_ij
        part_z_a = log_z[:, :, 0]
        if in_place:
            part_z_a -= torch.logsumexp(part_z_a, 1, keepdim=True)
        else:
            part_z_a = torch.log_softmax(part_z_a, dim=1)

        part_z = log_z[:, :, 1:]
        part_z, part_w_new = proj3(part_z, part_w, in_place=in_place, log_red_w_buf=log_w_red_buffer, log_z_red_buffer=log_z_red_buffer)

        log_z_new = torch.cat([part_z_a.unsqueeze(2), part_z], dim=2)
        # log_w_new = torch.cat([part_w_a.unsqueeze(2), part_w], dim=2)

        if in_place:
            log_z = in_place_omega_update(copy_log_z, log_z_new, omega)
            part_w = in_place_omega_update(copy_part_w, part_w_new, omega)
        else:
            log_z = omega_update(log_z, log_z_new, omega)
            part_w = omega_update(part_w, part_w_new, omega)

        # END OF PART (i)

        # PART (ii) takes care of
        # sum_i Z_ij = 1
        # sum_i W_i,(j+1), k = Z_kj

        if in_place and omega != 1.0:
            # We need to make a copy of the old points, otherwise we can't interpolate.
            copy_log_z.copy_(log_z)
            copy_part_w.copy_(part_w)

        # disregard first j for w and last j for z.
        # this is because we have the offset (j+1) for y and j for x
        part_z = log_z[:, :, :-1]

        part_z_a = log_z[:, :, -1]
        # part_z_a = part_z_a - torch.logsumexp(part_z_a, 1) #technically belongs to part (ii), but already ensure this in part (i)

        #Permute so we normalise over i (we always normalise over the last dimension of log_w), using Prop. 2 again
        permuted_part_w = torch.einsum("bijk -> bkji", part_w)
        part_z, part_w_new = proj3(part_z, permuted_part_w, in_place=in_place, log_red_w_buf=log_w_red_buffer, log_z_red_buffer=log_z_red_buffer)
        # Undo permutation of indices
        part_w_new = torch.einsum("bkji -> bijk", part_w_new)

        log_z_new = torch.cat([part_z, part_z_a.unsqueeze(2)], dim=2)

        if in_place:
            log_z = in_place_omega_update(copy_log_z, log_z_new, omega)
            part_w = in_place_omega_update(copy_part_w, part_w_new, omega)
        else:
            log_z = omega_update(log_z, log_z_new, omega)
            part_w = omega_update(part_w, part_w_new, omega)

        # END OF PART (ii)

        # PART (iii), takes care of
        # sum_j Z_ij = 1

        if in_place and omega != 1.0:
            # We need to make a copy of the old points, otherwise we can't interpolate.
            copy_log_z.copy_(log_z)

        if in_place:
            torch.logsumexp(log_z, dim=-1, keepdim=True, out=log_z_red_buffer_c)
            torch.sub(log_z, log_z_red_buffer_c, out=log_z_new)
        else:
            log_z_new = torch.log_softmax(log_z, dim=-1)

        if in_place:
            log_z = in_place_omega_update(copy_log_z, log_z_new, omega)
        else:
            log_z = omega_update(log_z, log_z_new, omega)

        ## END OF PART (iii)
        with torch.no_grad():
            torch.sub(old_log_z.exp(), log_z.exp(), out=log_z_compare_buffer)
            torch.abs(log_z_compare_buffer, out=log_z_compare_buffer)

            if torch.all(log_z_compare_buffer.max() < tol):
                _last_solve_steps = i
                break

    # Normalize again over the first axis, this is important if it's not fully converged.
    # If we don't do this, it's not a probability distribution, and we incentivize the model
    # to increase values beyond 1 to minimize loss (into negative numbers)
    # and therefore we would incentivize it to violate the constraints!

    log_w = torch.cat([part_w_a.unsqueeze(2), part_w], dim=2)

    log_z = torch.log_softmax(log_z, dim=1)

    return log_z, log_w


def prepare_scores(initial_scores, transition_scores, final_scores):
    """
    Takes scores for jumps and prepares the initial log U and log W from the paper.
    :param initial_scores: shape (batch, n)
    :param transition_scores: shape (batch, n, n) where transition_scores[b, k, i] means jump from k to i.
    :param final_scores: shape (batch, n)
    :return:
    """
    batch, n = initial_scores.shape
    log_w = transition_scores.unsqueeze(1) * torch.ones((1, n, 1, 1))
    log_w = torch.einsum("bjki -> bijk", log_w)

    # set the scores for jumps to position 0 to 0 because there is no previous position on which we can condition.
    mask = torch.ones((1, 1, n, 1), device=transition_scores.device)
    mask[:, :, 0, :] = 0.0
    log_w = log_w * mask

    mask_first = torch.zeros((1, 1, n))
    mask_first[0, 0, 0] = 1.0
    mask_last = torch.zeros((1, 1, n))
    mask_last[0, 0, -1] = 1.0

    middle_mask = torch.ones((1, 1, n))
    middle_mask[0, 0, 0] = 0.0
    middle_mask[0, 0, -1] = 0.0

    log_z = torch.zeros(batch, n, n) + mask_first * initial_scores.unsqueeze(2) + mask_last * final_scores.unsqueeze(2)

    return log_z, log_w

def prepare_scores_with_mask(initial_scores, transition_scores, final_scores, source_mask, square_mask):
    """
    Takes scores for jumps and prepares the initial log U and log W from the paper but also takes mask into account.
    :param initial_scores: shape (batch, n)
    :param transition_scores: shape (batch, n, n) where transition_scores[b, k, i] means jump from k to i.
    :param final_scores: shape (batch, n)
    :return:
    """
    batch, n = initial_scores.shape

    m2 = torch.all(~square_mask, dim=1, keepdim=True) * torch.all(~square_mask, dim=2, keepdim=True)
    #m2[b, i, j] = 1 iff i and j are both in the padded area
    lengths = source_mask.sum(dim=-1)-1 #(batch_size,)

    #Ensure that we don't go into padding region with scores 0 for transitions within the padding region.
    # transition_scores = transition_scores * square_mask
    # transition_scores = (transition_scores - 10e12 * ~square_mask) * ~m2 #this doesn't give the desired result in the final permutation matrix.

    log_w = transition_scores.unsqueeze(1) * torch.ones((1, n, 1, 1), device=transition_scores.device)
    log_w = torch.einsum("bjki -> bijk", log_w)

    # set the scores for jumps to position 0 to 0 because there is no previous output position on which we can condition.
    m = torch.ones((1, 1, n, 1), device=transition_scores.device)
    m[:, :, 0, :] = 0.0
    log_w = log_w * m

    mask_first = torch.zeros((1, 1, n), device=final_scores.device)
    mask_first[0, 0, 0] = 1.0

    mask_last = torch.zeros((batch, n, n), device=final_scores.device)
    mask_last[torch.arange(batch, device=mask_last.device), :, lengths] = 1.0

    # We do like to go from padding to padding, but we don't like to go from non-padding to padding or vice-versa.
    mask_for_padding = m2*1e12 - 1e12 * torch.logical_xor(~square_mask, m2)

    log_z = torch.zeros(batch, n, n, device=initial_scores.device) + mask_first * initial_scores.unsqueeze(2) + mask_last * final_scores.unsqueeze(2) + mask_for_padding
    # log_z = torch.zeros(batch, n, n) + mask_first * initial_scores.unsqueeze(2) + mask_last * final_scores.unsqueeze(2)

    return log_z, log_w


import torch.nn.functional as F

def simple_rand_perm(n, seed):
    import random
    random.seed(seed)
    l = list(range(n))
    random.shuffle(l)
    t = torch.zeros(1, n, n)
    for i, x in enumerate(l):
        t[0, x, i] = 1.0
    return t

def make_random_permutation(n, seed, max_val, min_val=-50):
    import random
    random.seed(seed)

    l = list(range(n))
    random.shuffle(l)

    initial_scores = torch.zeros((1, n)) + min_val
    initial_scores[0, l[0]] = max_val
    final_scores = torch.zeros((1, n)) + min_val
    final_scores[0, l[-1]] = max_val

    transition_scores = torch.zeros((1, n, n)) + min_val
    for i, j in zip(l[:-1], l[1:]):
        transition_scores[0, i, j] = max_val

    return initial_scores, transition_scores, final_scores


def get_matrix_plot(matrix):
    import seaborn as sns

    import matplotlib.pyplot as plt

    if isinstance(matrix, torch.Tensor):
        matrix = matrix.squeeze()
        matrix = matrix.detach().cpu().numpy()
    myfig = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="mako_r")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(45)
    n = 5
    # Generate a random permutation
    ini, tr, fin = make_random_permutation(n, 14, max_val=5, min_val=-5)
    # ini, tr, fin = make_random_permutation(n, 14, max_val=1, min_val=-1)
    # ini are scores for starting the output with a particular position,
    # analogously for fin
    # tr are transition scores (scores for jumps)
    # the more extreme you set max_val and min_val,
    # the strong the noise needs to be to change the solution

    # if you set max_val and min_val to small values (e.g. 1 and -1)
    # this also results in an "even"/high entropy relaxed permutation.


    # add some noise to the transition scores to make it more interesting
    tr += 3*torch.randn_like(tr)

    #Put the scores into two tensors that fit with the expected format
    #(see doc string of the function "bregman")
    log_z, log_w = prepare_scores(ini, tr, fin)
    # we can backprogpate through this, why not turn on gradients?
    log_z.requires_grad = True
    log_w.requires_grad = True
    beta = np.log(n)

    log_x, log_y = bregman(beta*log_z, beta*log_w, iter=50, tol=0.001, omega=1.0)

    print(log_x.exp())

    # visualize with the seaborn library:
    # get_matrix_plot(log_x.exp())
