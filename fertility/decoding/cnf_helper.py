import math
import multiprocessing
import sys
from collections import defaultdict
from typing import Callable, List, Optional, Dict, Tuple, Set

import nltk
import numba
import numpy as np
from nltk import CFG
from nltk.grammar import Nonterminal, Production

from nltk.parse.generate import generate


def default_formatter(s):
    return "_" + s + "_"


def remove_mixed_rules(g: CFG, non_terminal_formatter: Callable[[str], str] = default_formatter):
    """
    Returns a new equivalent CFG with mixed rules (e.g. A -> B "C" "D") removed.
    :param g:
    :param non_terminal_formatter: method to get a non-terminal symbol for a terminal symbol that appears in the RHS of a mixed rule.
    :return:
    """
    new_productions = []
    for rule in g.productions():
        if rule.is_lexical() and len(rule.rhs()) > 1:
            new_rhs = []
            for symbol in rule.rhs():
                if not isinstance(symbol, Nonterminal):
                    new_symbol = Nonterminal(non_terminal_formatter(symbol))
                    new_rhs.append(new_symbol)
                    new_productions.append(Production(new_symbol, [symbol]))
                else:
                    new_rhs.append(symbol)
            new_productions.append(Production(rule.lhs(), new_rhs))
        else:
            new_productions.append(rule)

    return CFG(g.start(), new_productions)


def to_cnf(grammar: CFG) -> CFG:
    return remove_mixed_rules(grammar).chomsky_normal_form()


def viterbi(scores: np.array, grammar: CFG, length: int, str2int: Dict[str, int], store_ints: bool = False) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    # backpointers[i][j][NT] is a dictionary mapping a non-terminal NT spanning i to j to a triple: the two non-terminals it was built from and the point where the split occurs
    # if j = i+1 then it doesn't contain a triple but the most likely word at that position in the string
    backpointers = [[dict() for j in range(length + 1)] for i in range(length)]
    viterbi_values = [[defaultdict(lambda: -np.inf) for j in range(length + 1)] for i in range(length)]
    # Initialize: Fill in first diagonal, the constituents of size 1

    pre_terminal_rules = [rule for rule in grammar.productions() if len(rule.rhs()) == 1]
    for i in range(length):
        # add all possible non-terminals that can expand to word to the cell [i][i+1]
        for rule in pre_terminal_rules:
            if rule.rhs()[0] in str2int:
                id = str2int[rule.rhs()[0]]
                score_of_rule = scores[i, id]
                if score_of_rule > viterbi_values[i][i + 1][rule.lhs()]:
                    viterbi_values[i][i + 1][rule.lhs()] = score_of_rule
                    if store_ints:
                        backpointers[i][i + 1][rule.lhs()] = id
                    else:
                        backpointers[i][i + 1][rule.lhs()] = rule.rhs()[0]

    for s in range(2, length + 1):  # for all constituent _sizes_ (increasing from 2)
        for i in range(length - s + 1):  # for all begin positions
            for k in range(1, s):  # for all positions where to split the span start at begin and of the given size s
                split_point = i + k
                for left_nt in backpointers[i][split_point].keys():  # go over all non-terminals in cell [i][i+k]
                    for matching_rule in grammar.productions(
                            rhs=left_nt):  # select all rules whose rhs starts with this non-terminal
                        right_nt = matching_rule.rhs()[1]  # get its right non-terminal
                        if right_nt in backpointers[i + k][
                            i + s]:  # check if the non-terminal following left_nt in this rule (right_nt) can be derived for cell i+k,i+s
                            parent_nt = matching_rule.lhs()
                            rule_score = viterbi_values[i][split_point][left_nt] + viterbi_values[split_point][i + s][
                                right_nt]
                            if rule_score > viterbi_values[i][i + s][parent_nt]:
                                # overwrite backpointer and viterbi value.
                                backpointers[i][i + s][parent_nt] = (left_nt, right_nt, k)
                                viterbi_values[i][i + s][parent_nt] = rule_score
    return backpointers, viterbi_values


class NumbaGrammar:
    def __init__(self, cfg, str2int: Dict[str, int]):
        self.pre_terminals = numba.typed.List()  # tuples of the form (nt id, score id according to str2int)
        self.rules = numba.typed.List()  # (parent, left, right)

        self.str2int = str2int
        self.int2str = {v:k for k,v in str2int.items()}

        non_terminals = set()
        for rule in cfg.productions():
            non_terminals.add(rule.lhs().symbol())
            for a in rule.rhs():
                if isinstance(a, Nonterminal):
                    non_terminals.add(a.symbol())

        self.id2nt = list(non_terminals)
        self.nt2id = {x: i for i, x in enumerate(self.id2nt)}

        self.num_nts = len(self.id2nt)

        self.start_id = self.nt2id[cfg.start().symbol()]

        for rule in cfg.productions():
            if len(rule.rhs()) == 1:
                rhs_symbol = rule.rhs()[0]
                if rhs_symbol in str2int:
                    self.pre_terminals.append((self.nt2id[rule.lhs().symbol()], str2int[rhs_symbol]))
            else:
                left, right = rule.rhs()
                left = self.nt2id[left.symbol()]
                right = self.nt2id[right.symbol()]
                self.rules.append((self.nt2id[rule.lhs().symbol()], left, right))


    def viterbi(self, scores: np.array, length: int, keep_ints: bool):
        backpointers, viterbi_values = fast_viterbi(scores, self.pre_terminals, self.rules, self.num_nts, length)

        return self._extract_tree(backpointers, viterbi_values, length, keep_ints)


    def _extract_tree(self, backpointers, viterbi_values, length, keep_ints):
        if viterbi_values[0][length][self.start_id] == -np.inf:  # no derivation
            return None, None

        return self.extract_backpointers_fast_viterbi(backpointers, 0, length, self.start_id, keep_ints),\
               viterbi_values[0][length][self.start_id]

    def batch_viterbi(self, scores: np.array, lengths: np.array, keep_ints: bool):
        r = batch_fast_viterbi(scores, self.pre_terminals, self.rules, self.num_nts, lengths)

        trees = []
        log_probs = []
        for (backpointers, viterbi_values), l in zip(r, lengths):
            t, log_prob = self._extract_tree(backpointers, viterbi_values, l, keep_ints)
            trees.append(t)
            log_probs.append(log_prob)

        return trees, log_probs


    def extract_backpointers_fast_viterbi(self, backpointers, i, j, nt, keep_ints):
        (left_backptr, right_backptr, k_backptr) = backpointers
        if j - i == 1:
            if keep_ints:
                return left_backptr[i][j][nt]
            else:
                return self.int2str[left_backptr[i][j][nt]]
        else:
            left = left_backptr[i][j][nt]
            right = right_backptr[i][j][nt]
            k = k_backptr[i][j][nt]

            children = [self.extract_backpointers_fast_viterbi(backpointers, i, i + k, left, keep_ints),
                        self.extract_backpointers_fast_viterbi(backpointers, i + k, j, right, keep_ints)]
            if keep_ints:
                return nltk.Tree(nt, children)
            else:
                return nltk.Tree(self.id2nt[nt], children=children)

    def batch_recognize(self, words: np.array, lengths: np.array):
        batch_size = words.shape[0]
        return np.stack([recognize(words[i], self.pre_terminals, self.rules, np.zeros((lengths[i], lengths[i]+1, self.num_nts), dtype=bool), lengths[i], self.start_id)
                         for i in range(batch_size)])

# @numba.jit(nopython=True, parallel=True)
@numba.jit(nopython=True)
def batch_fast_viterbi(scores: np.array,
                 preterminals: List[Tuple[int, str]],
                 rules: List[Tuple[int, int, int]],
                 num_nts: int, lengths: np.array):
    batch_size = scores.shape[0]
    # return [fast_viterbi(scores[i, :, :], preterminals, rules, num_nts, lengths[i]) for i in numba.prange(batch_size)]
    # Parallel doesn't seem to work :(
    return [fast_viterbi(scores[i, :, :], preterminals, rules, num_nts, lengths[i]) for i in range(batch_size)]

@numba.jit(nopython=True)
def fast_viterbi(scores: np.array,
                 preterminals: List[Tuple[int, str]],
                 rules: List[Tuple[int, int, int]],
                 num_nts: int, length: int) -> Tuple[Tuple[np.array, np.array, np.array], np.array]:

    left_backptr = np.zeros((length, length+1, num_nts), dtype=np.int64)
    right_backptr = np.zeros((length, length+1, num_nts), dtype=np.int64)
    k_backptr = np.zeros((length, length+1, num_nts), dtype=np.int64)

    viterbi_values = np.zeros((length, length+1, num_nts)) - np.inf
    # Initialize: Fill in first diagonal, the constituents of size 1

    for i in range(length):
        # add all possible non-terminals that can expand to word to the cell [i][i+1]
        for lhs, rhs in preterminals:
            score_of_rule = scores[i, rhs]
            vit_cell = viterbi_values[i][i + 1]
            if score_of_rule > vit_cell[lhs]:
                viterbi_values[i][i + 1][lhs] = score_of_rule
                left_backptr[i][i + 1][lhs] = rhs
                right_backptr[i][i + 1][lhs] = rhs

    for s in range(2, length + 1):  # for all constituent _sizes_ (increasing from 2)
        for i in range(length - s + 1):  # for all begin positions
            for k in range(1, s):  # for all positions where to split the span start at begin and of the given size s
                split_point = i + k
                for parent_nt, left_nt, right_nt in rules:
                    rule_score = viterbi_values[i][split_point][left_nt] + viterbi_values[split_point][i + s][
                        right_nt]
                    if rule_score > viterbi_values[i][i+s][parent_nt]:
                        viterbi_values[i][i+s][parent_nt] = rule_score
                        left_backptr[i][i+s][parent_nt] = left_nt
                        right_backptr[i][i+s][parent_nt] = right_nt
                        k_backptr[i][i+s][parent_nt] = k

    return (left_backptr, right_backptr, k_backptr), viterbi_values


@numba.jit(nopython=True)
def recognize(words: np.array,
                 preterminals: List[Tuple[int, str]],
                 rules: List[Tuple[int, int, int]],
                 chart: np.array,
                length: int,
              start_nt: int) -> bool:
    # Initialize: Fill in first diagonal, the constituents of size 1

    for i in range(length):
        # add all possible non-terminals that can expand to word to the cell [i][i+1]
        for lhs, rhs in preterminals:
            if words[i] == rhs:
                chart[i][i+1][lhs] = True

    for s in range(2, length + 1):  # for all constituent _sizes_ (increasing from 2)
        for i in range(length - s + 1):  # for all begin positions
            for k in range(1, s):  # for all positions where to split the span start at begin and of the given size s
                split_point = i + k
                for parent_nt, left_nt, right_nt in rules:
                    if chart[i, split_point, left_nt] and chart[split_point, i + s, right_nt]:
                        chart[i, i+s, parent_nt] = True

    return chart[0][length][start_nt]


def extract_backpointers(backpointers, i, j, nt):
    if j - i == 1:
        return backpointers[i][j][nt]
    left, right, k = backpointers[i][j][nt]
    return nltk.Tree(nt, [extract_backpointers(backpointers, i, i + k, left), extract_backpointers(backpointers, i + k, j, right)])

def decode(scores: np.array, grammar: CFG, length: int, str2int: Dict[str, int], store_ints: bool = False) -> Tuple[Optional[nltk.Tree], Optional[float]]:
    """
    Finds one of the most likely derivations (=most likely string if grammar is unambiguous) for a string of specified
    length with a variant of CYK. Returns None if there is no derivation for a string of the given length.
    The scores parameterize the terminal expansions but binary rules have no score/uniform score.

    Returns the most likely tree along with its score.
    :param scores: shape (max sequence length, vocab size) with log probabilities for the symbols in different positions
    :param grammar: a CFG in CNF
    :param length: length
    :param str2int
    :return:
    """
    backpointers, viterbi_values = viterbi(scores, grammar, length, str2int, store_ints)
    if grammar.start() not in backpointers[0][length]: #no derivation
        print("Warning: no derivation found", file=sys.stderr)
        return None, -np.inf

    return extract_backpointers(backpointers, 0, length, grammar.start()), viterbi_values[0][length][grammar.start()]


def compute_derivable_lengths(grammar: CFG, max_length) -> List[bool]:
    """
    Returns a list of booleans where the i-th value tells if the grammar in CNF permits a derivation with i leaves.
    Assumes S -> epsilon is not in the grammar.

    There might be a more efficient way to do this. This is O(max_length ^ 2).
    :param grammar:
    :param max_length:
    :return:
    """

    derivable_nts = [set() for j in range(max_length + 1)]
    # Initialize: Fill in first diagonal, the constituents of size 1

    for rule in grammar.productions():
        if len(rule.rhs()) == 1:
            derivable_nts[1].add(rule.lhs())

    for s in range(2, max_length + 1):  # for all constituent _sizes_ (increasing from 2)
        for k in range(1, s):  # for all positions where to split the span start at begin and of the given size s
            for left_nt in derivable_nts[k]:  # go over all non-terminals belonging to spans of length k
                for matching_rule in grammar.productions(
                        rhs=left_nt):  # select all rules whose rhs starts with this non-terminal
                    right_nt = matching_rule.rhs()[1]  # get its right non-terminal
                    if right_nt in derivable_nts[s-k]: #check if that non-terminal can produce a span of length s-k
                        derivable_nts[s].add(matching_rule.lhs())


    result = [False]  # ASSUMES THAT WE DON'T HAVE S -> EPSILON IN THE GRAMMAR.
    for i in range(1, max_length+1):
        a = grammar.start() in derivable_nts[i]
        result.append(a)

    return result



if __name__ == "__main__":
    import time
    grammar = CFG.fromstring("""
    S -> "answer" City
    City -> "city" "all"
    City -> "capital" "all"
    City -> "capital_1" State
    City -> "intersection" City City
    State -> "state" StateId
    StateId -> "stateid" StateName
    StateName -> "ohio"
    StateName -> "idaho"
    StateName -> "new" "york"
    """)

    removed = remove_mixed_rules(grammar)
    # print(removed)
    cnf = removed.chomsky_normal_form()
    print(cnf)
    # print(CFG.remove_unitary_rules(cnf))

    # print(list(generate(cnf, depth=10)))

    int2str = ["answer", "city", "capital", "capital_1", "intersection", "state", "stateid", "ohio", "idaho", "new", "york", "all"]
    str2int = {x: i for i, x in enumerate(int2str)}
    length = 50
    scores = np.zeros((length+1, len(int2str)))
    scores[-2, str2int["idaho"]] = 10.0

    t0 = time.time()
    t, score = decode(scores, cnf, length, str2int, False)
    t1 = time.time()

    #TODO: how to handle the COPY symbol?
    print("score", score)
    # print(t, t.leaves())
    print("Took", t1-t0)


    g = NumbaGrammar(cnf, str2int)

    # t2, score2 = g.viterbi(scores, length, False)
    # t0 = time.time()
    # t2, score2 = g.viterbi(scores, length, False)
    # t1 = time.time()
    # print("score2", score2)
    # # print(t2)
    # print("Took", t1-t0)
    # print(compute_derivable_lengths(cnf, 300))

    g.benchmark(scores, length, 1)
    t0 = time.time()
    g.benchmark(scores, length, 100)
    t1 = time.time()
    print("Time", t1-t0)