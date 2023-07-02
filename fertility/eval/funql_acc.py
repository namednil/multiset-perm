from typing import List, Dict

import nltk

from fertility.eval.lev import MyMetric
from fertility.tree_utils.read_funql import reconstruct_tree_with_partial_brackets, sort_tree_by_hash


@MyMetric.register("funql_acc")
class FunqlAcc(MyMetric):
    def __init__(self,  arities: Dict[str, int], sortable_nodes = ("intersection", "or")):
        self.sortable_nodes = set(sortable_nodes)
        self.reset()
        self.arities = arities

    def reset(self):
        self.correct = 0
        self.total = 0

    def get_metric(self, reset: bool) -> Dict[str, float]:
        if self.total == 0:
            return {"tree_acc": 0}
        a = self.correct / self.total
        if reset:
            self.reset()

        return {"tree_acc": a}

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        assert len(predictions) == len(gold)

        self.total += len(gold)

        for p, g in zip(predictions, gold):
            # For BART, which outputs strings instead of lists of tokens:
            if isinstance(p, str):
                p = p.split(" ")
                if p[0] == "": # if we start with a space (due to bart tokenization), remove it here
                    p = p[1:]
            if isinstance(g, str):
                g = g.split(" ")
                if g[0] == "": # if we start with a space (due to bart tokenization), remove it here
                    g = g[1:]

            try:
                g_t = reconstruct_tree_with_partial_brackets(g, self.arities)
            except Exception as ex:
                print("Warning, could not convert gold sequence into tree", g)
                continue
            try:
                p_t = reconstruct_tree_with_partial_brackets(p, self.arities)
            except Exception:
                continue

            self.correct += sort_tree_by_hash(nltk.ImmutableTree.convert(g_t), self.sortable_nodes) == \
                            sort_tree_by_hash(nltk.ImmutableTree.convert(p_t), self.sortable_nodes)


