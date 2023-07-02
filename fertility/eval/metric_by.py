import copy
from typing import List, Dict, Any

from allennlp.common import Registrable

from fertility.eval.lev import MyMetric


class MetricProperty(Registrable):

    def get_property(self, pred: List[str], gold: List[str]) -> str:
        raise NotImplementedError()

@MyMetric.register("metrics_by_property")
class MetricsByProperty(MyMetric):

    def __init__(self, metrics: List[MyMetric], property: MetricProperty):
        self.metrics = copy.deepcopy(metrics)
        self.property = property
        self.by_property = dict()

    def get_metric(self, reset: bool) -> Dict[str, float]:
        d = dict()
        for property in self.by_property:
            for metric in self.by_property[property]:
                for k, v in metric.get_metric(reset).items():
                    d[k+"_"+property] = v

        if reset:
            self.by_property = dict()

        return d

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        assert len(predictions) == len(gold)

        by_prop = dict()
        for p, g in zip(predictions, gold):
            value = self.property.get_property(p, g)
            if value not in by_prop:
                by_prop[value] = []
            by_prop[value].append((p,g))

        for property in by_prop:
            if property not in self.by_property:
                self.by_property[property] = copy.deepcopy(self.metrics)

            for metric in self.by_property[property]:
                metric.add_instances(*zip(*by_prop[property]))



@MetricProperty.register("cogs_recursion_depth")
class CogsRecursionDepth(MetricProperty):

    def get_property(self, pred: List[str], gold: List[str]) -> str:
        pp_depth = gold.count(".nmod")
        cp_depth = gold.count(".ccomp")

        return f"pp_{pp_depth}_cp_{cp_depth}"
