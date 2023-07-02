from typing import List, Dict

from fertility.eval.lev import MyMetric


@MyMetric.register("length_error")
class LengthError(MyMetric):

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_examples = 0
        self.total_deviation = 0

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        for p, g in zip(predictions, gold):
            if isinstance(p, str):
                p = p.split(" ")
            if isinstance(g, str):
                g = g.split(" ")

            self.num_examples += 1
            self.total_deviation += abs(len(p) - len(g))

    def get_metric(self, reset: bool) -> Dict[str, float]:
        if self.num_examples == 0:
            return {"mean_abs_length_error": 0}
        e = self.total_deviation / self.num_examples
        if reset:
            self.reset()
        return {"mean_abs_length_error": e}