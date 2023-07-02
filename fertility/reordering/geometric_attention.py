import torch
import numpy as np

import numba
from allennlp.modules.matrix_attention import DotProductMatrixAttention


@numba.njit()
def _gen_matrix(a):
    # a = np.zeros((n, n, n), dtype=bool)
    n = a.shape[0]
    assert a.shape == (n,n,n)
    for i in range(n):
        for j in range(n):
            if i < j:
                a[i, j, (i + 1):j] = True
                a[i, j, max(0, i - (j - i - 1)):i] = True
            elif i == j:
                pass
            else:
                a[i, j, (j + 1):i] = True
                a[i, j, i + 1:min(n, i + (i - j) + 1)] = True

    return a

def gen_matrix(n):
    return torch.from_numpy(_gen_matrix(np.zeros((n, n, n), dtype=bool)))


def geom_attention_inefficient(x, masking_matrix):
    n = x.shape[1]
    a = masking_matrix[:n, :n, :n].unsqueeze(0) * x[:, :, :, 0].unsqueeze(2)
    a = a.sum(-1) + x[:, :, :, 1]
    # mask out diagonal
    a = a - 10_000 * torch.eye(x.shape[1], device=x.device).unsqueeze(0)
    return a


def log_sigmoid(x):
    return x - torch.nn.functional.softplus(x)

def log_1m(x: torch.Tensor) -> torch.Tensor:
    """
    Computes log(1 - exp(x)), for x < 0.
    See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    :param x:
    :return:
    """
    #assert torch.all(x < 0)
    return torch.log(1-torch.exp(x))
    # return torch.log(-x.expm1())


class GeometricAttention(torch.nn.Module):
    """
    A simple but not asymptotically optimal implementation of geometric attention (see https://arxiv.org/abs/2110.07732)

    Takes O(n^3) time and space instead of O(n^2) but doesn't require a CUDA compiler.
    """

    def __init__(self, input_dim, d_model, max_len: int = 200):
        super().__init__()
        self.wq = torch.nn.Linear(input_dim, d_model, bias=True)
        self.wk = torch.nn.Linear(input_dim, d_model, bias=False)
        self.wlr = torch.nn.Linear(input_dim, 1, bias=True)
        self.wrl = torch.nn.Linear(input_dim, 1, bias=True)

        self.alpha = torch.nn.Parameter(torch.tensor([1.0 / np.sqrt(d_model)]), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        self.register_buffer("masking_matrix", gen_matrix(max_len), persistent=False)

        self.att = DotProductMatrixAttention()
        self.d_model = d_model

    def forward(self, h: torch.Tensor, mask, normalize: bool = False):
        """

        :param h: shape (batch, seq_len, input_dim)
        :return: shape (batch, seq_len, seq_len) with log attention scores (sums to 1 over last dimension if normalize=True, otherwise sum to < 1)
        """

        queries = self.wq(h)
        keys = self.wk(h)

        # both shape (batch, seq_len,1)
        left = self.wlr(h)
        right = self.wrl(h)

        n = left.shape[1]
        ones = torch.ones(1, n, n, device=h.device)
        d_left = left * torch.triu(ones)
        d_right = right * torch.tril(ones, -1)
        d = d_left + d_right

        raw_scores = self.alpha * self.att(queries, keys) + self.beta * d + self.gamma

        raw_scores = torch.clip(raw_scores, -16, 14)  # ADDED this, for numerical stability

        align_scores = log_sigmoid(raw_scores)
        align_scores = torch.stack([log_1m(align_scores), align_scores], dim=-1) #shape (batch, seq_len, seq_len, 2)

        r = geom_attention_inefficient(align_scores, self.masking_matrix)

        # print(align_scores[0])

        alignment_mask = mask.unsqueeze(1) * mask.unsqueeze(
            2)  # alignment_mask[b, i, j] = True iff the element is not in the padding.

        if normalize:
            r = torch.log_softmax(r, dim=-1)
        return r, alignment_mask




if __name__ == "__main__":

    # print(gen_matrix(3))
    torch.manual_seed(42)

    m = GeometricAttention(3, 4, max_len=10)
    x = torch.randn(2, 5, 3)

    print(m(x).exp())
    print(m(x).exp().sum(dim=-1))