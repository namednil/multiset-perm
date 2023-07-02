from allennlp.common import Lazy
from allennlp.nn.beam_search import FinalSequenceScorer

import torch

from fertility.decoding.decoding_grammar import DecodingGrammarFast


# @FinalSequenceScorer.register("grammar_decoding")
class GrammarFinalSequenceScorer(FinalSequenceScorer):


    def __init__(self, grammar_decoder: DecodingGrammarFast):
        self.grammar_decoder = grammar_decoder

    def score(
        self, predictions: torch.Tensor, log_probabilities: torch.Tensor, end_index: int
    ) -> torch.Tensor:
        (batch_size, beam_size, max_steps) = predictions.shape
        device = predictions.device
        modified_beam_logl = []
        for i in range(batch_size):
            beam_pred = predictions[i]
            beam_logl = log_probabilities[i]
            lengths = max_steps - (beam_pred == end_index).sum(dim=-1)
            accepted = self.grammar_decoder.recognize(beam_pred, lengths)
            modified_beam_logl.append(beam_logl - 1_000_000 * ~accepted)
        return torch.stack(modified_beam_logl, dim=0).to(device)
