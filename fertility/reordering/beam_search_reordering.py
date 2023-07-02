from typing import Dict

import torch
from allennlp.nn.beam_search import BeamSearch, FinalSequenceScorer



import numba
import numpy as np


def jl_mask(n):
    """
    jl_mask[i,j] is 1 iff i + 1 = j, i.e. j is one bigger than one
    :param n:
    :return:
    """
    return torch.eye(n+1, n)[1:, :]
def make_instance(n, max_val = 10):
    initial_scores = torch.zeros((1, n))
    initial_scores[0, 0] = max_val
    final_scores = torch.zeros((1, n))
    final_scores[0, -1] = max_val

    transition_scores = jl_mask(n).unsqueeze(0) * max_val

    return initial_scores, transition_scores, final_scores
@numba.njit
def _fill_binary_mask(max_preds: np.array) -> np.array:
    batch, n = max_preds.shape
    m = np.zeros((batch, n, n), dtype=np.float32) - 10000
    y = np.zeros((batch, n, n, n), dtype=np.float32) - 10000
    for b in range(batch):
        for i in range(n):
            m[b, max_preds[b, i], i] = 0.0
            if i > 0:
                y[b, max_preds[b, i], i, max_preds[b, i-1]] = 0.0
    return m, y


def beam_search_perm(log_x, log_y, beam_size = 10):
    batch, n, n = log_x.shape
    assert log_y.shape == (batch, n, n, n)

    beam_search = BeamSearch(end_index=-1, max_steps=n, beam_size=min(beam_size, n))
    # set an impossible end index that will never be found.

    batch_range = torch.arange(batch, device=log_x.device)

    def step(indices_last_predicted: torch.Tensor, state: Dict[str, torch.Tensor], timestep: int):
        used = state["used"].clone() #shape (group_size, n)

        group_range = torch.arange(used.shape[0], device=used.device)

        next_scores = log_x[state["batch_index"], :, timestep]
        if timestep > 0:
            # Mask out things that we used in the last timestep:
            used[group_range, indices_last_predicted] = True
            next_scores += log_y[state["batch_index"], :, timestep, indices_last_predicted] #shape (group_size, n)
            next_scores -= 100_000 * used #don't use anything again that we've used already

        return next_scores, {"used": used, "batch_index": state["batch_index"]}

    preds, values = beam_search.search(start_predictions=torch.zeros(batch, device=log_x.device),
                              start_state={"used": torch.zeros(batch, n, dtype=torch.bool, device=log_x.device),
                                           "batch_index": batch_range},
                                                   step=step)

    max_pred = preds[:, 0, :] #shape (batch_size, n) with highest scoring permutation per batch element.

    # matrix[b, i, max_pred[b, i]] = True
    out_x, out_y = _fill_binary_mask(max_pred.cpu().numpy())

    return torch.from_numpy(out_x).to(log_x.device), torch.from_numpy(out_y).to(log_x.device)


if __name__ == "__main__":
    from fertility.reordering.bregman_for_perm import make_random_permutation, prepare_scores, bregman

    import torch
    torch.manual_seed(17)
    torch.set_printoptions(2, sci_mode=False)
    n = 3
    batch = 1
    ini, tr, fin = make_random_permutation(n, seed=43, max_val=20)

    ini2, tr2, fin2 = make_instance(n, max_val=20)

    ini = torch.cat([ini, ini2])
    tr = torch.cat([tr, tr2])
    fin = torch.cat([fin, fin2])

    # log_x, log_y = prepare_scores(ini, tr, fin)

    log_x = torch.randn((batch, n, n)) * 50
    log_y = torch.randn((batch, n, n, n)) * 10

    ba, bb = beam_search_perm(log_x, log_y, beam_size=4)
    print(ba.exp())
    print(bb.exp())

    a, b = bregman(log_x, log_y, iter=400)
    print(a.exp())
    print(b.exp())





