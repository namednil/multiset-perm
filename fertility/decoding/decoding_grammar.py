from typing import List, Tuple, Dict, Union, Iterable, Set

import nltk
import numpy as np
import torch
from allennlp.common import Registrable
from nltk import CFG, Tree

from fertility.decoding.cnf_helper import decode, compute_derivable_lengths, NumbaGrammar


class DecodingGrammar(Registrable):

    def __init__(self, tok2id: Dict[str, int]):
        self.tok2id = tok2id
        self.grammar = None
        self.derivable_lengths = np.array([])

    def set_grammar(self, grammar: CFG):
        self.grammar = grammar

    def get_terminals(self) -> Set[str]:
        s = set()
        for prod in self.grammar.productions():
            for symbol in prod.rhs():
                if not isinstance(symbol, nltk.Nonterminal):
                    s.add(symbol)
        return s


    def get_derivable_lengths(self, max_length: int) -> torch.Tensor:
        if max_length > len(self.derivable_lengths):
            self.derivable_lengths = np.array(compute_derivable_lengths(self.grammar, max_length+10))

        return torch.from_numpy(self.derivable_lengths[:max_length])

    def decode(self, scores: np.array, length: int, use_ints) -> Tuple[List[str], float]:
        tree, score = decode(scores, self.grammar, length, self.tok2id, use_ints)

        return tree.leaves(), score

    def batch_decode(self, scores: torch.Tensor, lengths: torch.Tensor, use_ints: bool) -> Tuple[Union[List[List[str]], torch.Tensor], torch.Tensor]:
        np_scores = scores.detach().cpu().numpy()
        np_lengths = lengths.detach().cpu().numpy()
        ret_strings = []
        ret_probs = []
        for i in range(np_lengths.shape[0]):
            s, score = self.decode(np_scores[i], np_lengths[i], use_ints)
            ret_strings.append(s)
            ret_probs.append(score)

        log_probs = torch.from_numpy(np.array(ret_probs)).to(lengths.device)

        if use_ints:
            # We return a tensor so pad everything with 0s on the right
            max_len = scores.shape[1]
            tensor = np.array([s + (max_len - len(s)) * [0] for s in ret_strings])
            tensor = torch.from_numpy(tensor).to(lengths.device)
            return tensor, log_probs
        else:
            return ret_strings, log_probs




class DecodingGrammarFast(DecodingGrammar):

    def __init__(self, tok2id: Dict[str, int]):
        super().__init__(tok2id)
        self.tok2id = tok2id
        self.grammar = None
        self.derivable_lengths = np.array([])

    def set_grammar(self, grammar: CFG):
        self.compiled_grammar = NumbaGrammar(grammar, self.tok2id)
        self.grammar = grammar

    def get_derivable_lengths(self, max_length: int) -> torch.Tensor:
        if max_length > len(self.derivable_lengths):
            self.derivable_lengths = np.array(compute_derivable_lengths(self.grammar, max_length+10))

        return torch.from_numpy(self.derivable_lengths[:max_length])

    def decode(self, scores: np.array, length: int, use_ints) -> Tuple[List[str], float]:
        raise NotImplementedError()

    def batch_decode(self, scores: torch.Tensor, lengths: torch.Tensor, use_ints: bool) -> Tuple[Union[List[List[str]], torch.Tensor], torch.Tensor]:
        np_scores = scores.detach().cpu().numpy()
        np_lengths = lengths.detach().cpu().numpy()
        trees, log_probs = self.compiled_grammar.batch_viterbi(np_scores, np_lengths, use_ints)
        strings = [t.leaves() if isinstance(t, Tree) else [t] for t in trees]

        log_probs = torch.from_numpy(np.array(log_probs)).to(lengths.device)

        if use_ints:
            # We return a tensor so pad everything with 0s on the right
            max_len = scores.shape[1]
            tensor = np.array([s + (max_len - len(s)) * [0] for s in strings])
            tensor = torch.from_numpy(tensor).to(lengths.device)
            return tensor, log_probs
        else:
            return strings, log_probs

    def recognize(self, predictions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        device = predictions.device
        predictions = predictions.detach().cpu().numpy()
        lengths = lengths.detach().cpu().numpy()
        return torch.from_numpy(self.compiled_grammar.batch_recognize(predictions, lengths)).to(device)