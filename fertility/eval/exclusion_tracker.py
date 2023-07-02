from typing import List, Dict

from fertility.eval.lev import MyMetric


class InclusionMetric(MyMetric):
    """
    How many instances do we keep in the training stage of the multiset reordering? This is good to keep an eye on
    when things go wrong.
    """
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.total = 0
        self.true = 0

    def add_instances(self, results: List[bool]) -> None:
        self.total += len(results)
        self.true += sum(int(x) for x in results)

    def get_metric(self, reset: bool) -> Dict[str, float]:
        if self.total == 0:
            return {self.name: 0}
        r = self.true/self.total
        if reset:
            self.reset()
        return {self.name: r}
