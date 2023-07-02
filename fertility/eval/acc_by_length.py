import copy
from collections import defaultdict
from typing import Dict, List

from fertility.eval.lev import MyMetric


@MyMetric.register("acc_by_length")
class AccByLength(MyMetric):

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_instances = defaultdict(int)
        self.correct_instances = defaultdict(int)

    def get_metric(self, reset: bool) -> Dict[str, float]:
        d = {"seq_acc_"+str(k): self.correct_instances[k]/self.total_instances[k] if self.total_instances[k] > 0 else 0.0
                                                        for k in self.total_instances}

        if reset:
            self.reset()
        return d

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        assert len(predictions) == len(gold)

        for p, g in zip(predictions, gold):
            self.total_instances[len(g)] += 1
            self.correct_instances[len(g)] += (p == g)



class MetricByCat:

    def __init__(self, metrics: List[MyMetric]):
        self.metrics = copy.deepcopy(metrics)
        self.values_by_cat: Dict[str, List[MyMetric]] = dict()

    def get_metrics(self, reset: bool) -> Dict[str, float]:
        d = {}
        for cat in self.values_by_cat:
            for m in self.values_by_cat[cat]:
                for k, v in m.get_metric(reset).items():
                    d[cat+"_"+k] = v
        return d

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]], categories: List[str]) -> None:
        pgs_by_cat = defaultdict(list)
        for p, g, cat in zip(predictions, gold, categories):
            if cat not in self.values_by_cat:
                self.values_by_cat[cat] = copy.deepcopy(self.metrics)
            pgs_by_cat[cat].append((p, g))

        for cat in pgs_by_cat:
            ps, gs = tuple(zip(*pgs_by_cat[cat]))
            for m in self.values_by_cat[cat]:
                m.add_instances(ps, gs)

