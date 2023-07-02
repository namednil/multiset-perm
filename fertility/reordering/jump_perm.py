import copy
import itertools
from typing import Tuple, Optional, Union, List

import numpy as np
import torch
from allennlp.common import Registrable
#import torch_semiring_einsum
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import Activation
from torch.nn import Module

from fertility.reordering.beam_search_reordering import beam_search_perm
from fertility.reordering.geometric_attention import GeometricAttention
from fertility.reordering.gumbel import sample_gumbel
from fertility.reordering.bregman_for_perm import bregman, prepare_scores_with_mask, gen_kl, kl, kl_with_mask
from fertility.reordering.mip_solving import or_tools_solve
from fertility.reordering.utils import RelativeConcatMLP, sinusoidal_pos_embedding
from fertility.scheduler import RateScheduler

import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment


def apply_linear_sum_assignment(log_x: torch.Tensor):
    log_x_np = (-log_x).detach().cpu().numpy()

    r = []
    for matrix in log_x_np:
        z = np.zeros_like(matrix) - 1000
        row_ind, col_ind = linear_sum_assignment(matrix)
        z[row_ind, col_ind] = 0.0
        r.append(torch.from_numpy(z))

    return torch.stack(r, dim=0).to(log_x.device)

class JumpDotProduct(torch.nn.Module):

    def __init__(self, d_model: int, source_dim: int, dropout: float = 0.1, max_len: int = 400):
        super().__init__()
        from allennlp.modules.matrix_attention import DotProductMatrixAttention
        self.att = DotProductMatrixAttention()
        self.source_dim = source_dim

        self.d_model = d_model

        self.w_q_x = torch.nn.Linear(source_dim, d_model, bias=False)
        self.w_k_x = torch.nn.Linear(source_dim, d_model, bias=False)
        self.w_k2_x = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_q2_x = torch.nn.Linear(source_dim, d_model, bias=False)

        self.dropout = torch.nn.Dropout(dropout)

        pos_relative_encoding = sinusoidal_pos_embedding(d_model, max_len=max_len).numpy()
        neg_relative_encoding = sinusoidal_pos_embedding(d_model, max_len=max_len, f=-1).numpy()

        relative_encoding = np.zeros((1, max_len, max_len, d_model), dtype=pos_relative_encoding.dtype)
        for i in range(max_len):
            for j in range(max_len):
                if i <= j:
                    relative_encoding[0, i, j] = pos_relative_encoding[j-i]
                else:
                    relative_encoding[0, i, j] = neg_relative_encoding[i-j]
        self.register_buffer("relative_sinusoid", torch.from_numpy(relative_encoding))


        """
        
        A_{i,j} = W_q(H_i) dot product with W_k,E(H_j) + 
          W_q(H_i) dot product with W_{k,P}(P_{i-j})

        """

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """

        :param x: shape (batch_size, seq_len, dim1)
        :param mask: shape (batch_size, seq_len)
        :return:
        """
        batch, seq_len, _ = x.shape

        #all of shape (batch_size, seq_len, d_model)
        q_x = self.w_q_x(x)
        q2_x = self.w_q2_x(x)
        k_x = self.w_k_x(x)

        edge_scores = self.att(q_x, k_x) #shape (batch, seq_len, seq_len)

        rel = self.w_k2_x(self.relative_sinusoid[:, :seq_len, :seq_len]) #shape (1, seq_len, seq_len, d_model)

        rel_scores = torch.einsum("bid, bijd -> bij", q2_x, rel)

        edge_scores = (edge_scores + rel_scores) / np.sqrt(self.d_model)

        alignment_mask = mask.unsqueeze(1) * mask.unsqueeze(
            2)  # alignment_mask[b, i, j] = True iff the element is not in the padding.

        return edge_scores, alignment_mask


class AbstractVariationalPermutationModel(Model):


    def __init__(self, vocab: Vocabulary, namespace: str):
        super().__init__(vocab)

        self.namespace = namespace


    def forward(self, encoded_for_reordering: torch.Tensor,
                      source_mask: torch.Tensor,
                      target_toks: Optional[TextFieldTensors] = None,
                      target_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param encoded_for_reordering: shape (batch, seq_len, encoder dim)
        :param source_mask: shape (batch, seq_len)
        :param target_toks: shape (batch, seq_len)
        :param target_mask: shape (batch, seq_len)
        :return:
        """

        raise NotImplementedError()

@AbstractVariationalPermutationModel.register("var_bregman_bla")
class BregmanPermutationModelBla(AbstractVariationalPermutationModel):

    def __init__(self, vocab: Vocabulary, namespace: str, hidden_dim: int, encoder_dim: int,
                 clip: Optional[float] = 50,
                 train_max_iter: Optional[int] = None,
                 max_iter: int = 3000, tol: float = 0.001,
                 omega: Union[str, float] = 1.0,
                 posterior_max_iter: int = 150,
                 dropout: float = 0.1,
                 dev_solver: str = "bregman",
                 dev_beam_size: int = 30,
                 inference_temp: float = 10.0,
                 mlp: Optional[FeedForward] = None,
                 rel_dot: bool = False,
                 geometric_attention: bool = True,
                 postprocess_with_lap: bool = True,
                 max_len: int = 200):
        super().__init__(vocab, namespace)

        self.clip = clip
        self.encoder_dim = encoder_dim

        self.train_max_iter = train_max_iter if train_max_iter is not None else max_iter

        self.inference_temp = inference_temp

        assert dev_solver in ["bregman", "beam_search", "GLOP", "Clp", "SCIP"]

        self.dev_solver = dev_solver
        self.dev_beam_size = dev_beam_size
        # P(R|x, F)
        self.max_iter = max_iter
        self.tol = tol
        self.omega = omega

        self.posterior_max_iter = posterior_max_iter

        self.mlp = mlp

        self.postprocess_with_lap = postprocess_with_lap

        if self.mlp is not None:
            self.encoder_dim = self.mlp.get_output_dim()

        self.initial_scorer = FeedForward(self.encoder_dim, 2, [hidden_dim, 1],
                                          activations=[Activation.by_name("gelu")(), Activation.by_name("linear")()],
                                          dropout=dropout)

        self.end_scorer = FeedForward(self.encoder_dim, 2, [hidden_dim, 1],
                                          activations=[Activation.by_name("gelu")(), Activation.by_name("linear")()],
                                      dropout=dropout)


        self.rel_dot = rel_dot
        self.use_geom_att = geometric_attention
        if self.use_geom_att:
            self.geom_att_layer = GeometricAttention(self.encoder_dim, hidden_dim, max_len = max_len)
        elif rel_dot:
            self.rel_dot_layer = JumpDotProduct(hidden_dim, self.encoder_dim, dropout)
        else:
            self.concat_mlp = RelativeConcatMLP(hidden_dim, self.encoder_dim, self.encoder_dim, F.gelu)
            self.output_layer = torch.nn.Linear(hidden_dim, 1)


    def compute_prior_scores(self, encoded_for_reordering: torch.Tensor,
                source_mask: torch.Tensor):
        # Inference time
        neg_mask = ~source_mask

        initial_scores = self.initial_scorer(encoded_for_reordering) - 100_000 * neg_mask.unsqueeze(
            2)  # shape (batch_size, input_seq_len, 1)
        end_scores = self.end_scorer(encoded_for_reordering) - 100_000 * neg_mask.unsqueeze(
            2)  # shape (batch_size, input_seq_len, 1)


        if self.use_geom_att:
            transition_scores, transition_mask = self.geom_att_layer(encoded_for_reordering, source_mask)
        elif self.rel_dot:
            transition_scores, transition_mask = self.rel_dot_layer(encoded_for_reordering, source_mask)
        else:
            transition_repr, transition_mask = self.concat_mlp(encoded_for_reordering, encoded_for_reordering,
                                                               source_mask,
                                                               source_mask)  # shape (batch_size, seq len, seq len, hidden size)

            transition_scores = self.output_layer(transition_repr).squeeze(3)

        score_mask = transition_mask & ~torch.eye(transition_mask.shape[1], dtype=torch.bool,
                                                  device=transition_mask.device).unsqueeze(0)
        # can't transition from a state to itself.

        transition_scores = transition_scores - 100_000 * ~score_mask

        batch, n, _ = initial_scores.shape

        prep_log_z, prep_log_w = prepare_scores_with_mask(initial_scores.squeeze(2), transition_scores,
                                                          end_scores.squeeze(2), source_mask, transition_mask)

        if self.clip is not None:
            # if scores get too extreme, it slows down Bregman's method /
            # with insufficient iterations, the constraints can't be enforced
            prep_log_z = torch.clip(prep_log_z, -self.clip, self.clip)
            prep_log_w = torch.clip(prep_log_w, -self.clip, self.clip)

        return prep_log_z, prep_log_w


    def forward(self, encoded: torch.Tensor,
                source_mask: torch.Tensor,
                target_toks: Optional[TextFieldTensors] = None,
                target_mask: Optional[torch.Tensor] = None,
                ids: Optional[List[int]] = None,
                source_to_copy_mask: Optional[torch.Tensor] = None,
                permitted_alignments: Optional[torch.Tensor] = None,
                loss_mask: Optional[torch.Tensor] = None,
                test_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param encoded: shape (batch, seq_len, encoder dim)
        :param source_mask: shape (batch, seq_len)
        :param target_toks: shape (batch, seq_len)
        :param target_mask: shape (batch, seq_len)
        :return:
        """
        #TODO: source and target mask should contain the same thing, eliminate one of them.

        batch, n, _ = encoded.shape
        assert source_mask.shape == (batch, n)
        if target_toks is not None:
            assert target_mask.shape == (batch, n)

        lengths = source_mask.sum(dim=1).unsqueeze(1).unsqueeze(2)  # shape (batch,1, 1)

        alignment_mask = torch.einsum("bi, bj -> bij", target_mask,
                                      target_mask).bool()  # alignment_mask[b, i, j] = True iff the element is not in the padding.
        triple_alignment_mask = torch.einsum("bi, bj, bk -> bijk", target_mask, target_mask, target_mask).bool()

        if self.mlp is not None:
            encoded = self.mlp(encoded)

        prep_log_z, prep_log_w = self.compute_prior_scores(encoded, source_mask)

        #So the logits won't be too low for long ood sequences:
        prep_log_z = torch.log(lengths) * prep_log_z
        prep_log_w = torch.log(lengths).unsqueeze(3) * prep_log_w

        if not self.training and self.dev_solver != "bregman":

            if self.dev_solver == "beam_search":
                prior_log_x, prior_log_y = beam_search_perm(torch.log(lengths) * prep_log_z * self.inference_temp,
                                                            torch.log(lengths).unsqueeze(3) * prep_log_w * self.inference_temp,
                                                            beam_size=self.dev_beam_size)
            else:
                prior_log_x, prior_log_y = or_tools_solve(torch.log(lengths) * prep_log_z * self.inference_temp,
                                                          torch.log(lengths).unsqueeze(3) * prep_log_w * self.inference_temp,
                                                          solver=self.dev_solver)
        else:
            prior_log_x, prior_log_y = bregman(torch.log(lengths) * prep_log_z * (1.0 if self.training else self.inference_temp),
                                               torch.log(lengths).unsqueeze(3) * prep_log_w * (1.0 if self.training else self.inference_temp),
                                               iter=self.train_max_iter if self.training else self.max_iter, tol=self.tol,
                                               omega=self.omega,
                                               in_place=not self.training)  # this level of precision is required at least, for evaluation. Maybe training needs more?

        if not self.training and self.postprocess_with_lap:
            prior_log_x = apply_linear_sum_assignment(prior_log_x)

        if target_toks:
            if permitted_alignments is not None:
                posterior_log_x = prior_log_x - 100 * ~permitted_alignments

            posterior_log_x, posterior_log_y = bregman(posterior_log_x.detach().clone(), prior_log_y.detach().clone(),
                                                 self.posterior_max_iter, omega=1.0, tol=self.tol,
                                                 in_place=True)

            kl_term = kl_with_mask(posterior_log_x, prior_log_x, loss_mask.unsqueeze(-1).unsqueeze(-1) * alignment_mask)
            kl_term = kl_term + kl_with_mask(posterior_log_y, prior_log_y, loss_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * triple_alignment_mask)
            kl_term = kl_term / posterior_log_x.shape[0]
        else:
            kl_term = torch.zeros(1, device=prior_log_x.device)

        return (alignment_mask * prior_log_x.exp()).transpose(1, 2), kl_term

