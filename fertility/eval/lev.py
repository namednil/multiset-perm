from typing import List, Dict

import Levenshtein
#https://github.com/maxbachmann/Levenshtein
from allennlp.common import Registrable


class MyMetric(Registrable):
    def get_metric(self, reset: bool) -> Dict[str, float]:
        raise NotImplementedError()

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        raise NotImplementedError()

@MyMetric.register("levenshtein")
class LevenstheinMetric(MyMetric):

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_instances = 0
        self.total_distance = 0
        self.correct_length = 0
        self.correct = 0

    def get_metric(self, reset: bool) -> Dict[str, float]:
        if self.total_instances == 0:
            return {"avg_levenshtein": 0, "seq_acc": 0, "per": 0}
        r = self.total_distance / self.total_instances
        acc = self.correct / self.total_instances
        per = self.total_distance / self.correct_length
        if reset:
            self.reset()
        return {"avg_levenshtein": r, "seq_acc": acc, "per": per}

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        assert len(predictions) == len(gold)

        for p, g in zip(predictions, gold):
            self.total_instances += 1
            self.total_distance += Levenshtein.distance(p, g)
            self.correct += p == g
            self.correct_length += len(g)


@MyMetric.register("length_acc")
class LengthAcc(MyMetric):

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def get_metric(self, reset: bool) -> Dict[str, float]:
        if self.total == 0:
            return {"length_acc": 0}
        r = self.correct / self.total
        if reset:
            self.reset()
        return {"length_acc": r}

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        for p, g in zip(predictions, gold):
            self.total += 1
            self.correct += len(p) == len(g)