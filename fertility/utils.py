import pickle
from typing import Tuple, List, Dict

import numpy as np
import torch

import torch.nn.functional as F


def viterbi_to_array(list: List[Tuple[List[int], float]]) -> np.array:
    batch_size = len(list)
    max_l = max(len(x[0]) for x in list)
    arr = np.zeros((batch_size, max_l), dtype=np.int)
    for i, (sub_l, _) in enumerate(list):
        for j, x in enumerate(sub_l):
            arr[i, j] = x
    return arr


def make_same_length(m1: torch.Tensor, m2: torch.Tensor, m2_mask: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    :param m1: shape (batch_size, s1)
    :param m2: shape (batch_size, s2)
    :param m2_mask: (batch_size, s2)
    :return: a tuple of shape (batch_size, max(s1, s2)),
                (batch_size, max(s1, s2)),
                 (batch_size, max(s1, s2))
    where the first one is m1 with optional zero padding
    where the second one is m2 with optional zero padding
    and the third one is the m2_mask, with optional zero padding at the end
    """

    b, s1 = m1.shape
    b, s2 = m2.shape
    b, s3 = m2_mask.shape
    assert s2 == s3
    m = max(s1, s2)
    if s1 < m:
        # pad only m1
        return F.pad(m1, (0, m - s1)), m2, m2_mask
    else:
        # pad m2 and its mask
        return m1, F.pad(m2, (0, m - s2)), F.pad(m2_mask, (0, m - s2))


def tensor_dict_to_numpy(d):
    r = dict()
    for k,v in d.items():
        if isinstance(v, torch.Tensor):
            r[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            r[k] = tensor_dict_to_numpy(v)
        else:
            r[k] = v
    return r

def dump_tensor_dict(d, fname):
    with open(fname, "wb") as f:
        arr = tensor_dict_to_numpy(d)
        pickle.dump(arr, f)