from typing import List, Dict

from fertility.dataset_readers.preprocessor import Preprocessor
from fertility.eval.lev import MyMetric


@MyMetric.register("cogs_acc")
class LenientCOGSEval(MyMetric):
    def __init__(self, preprocessor: Preprocessor):
        self.reset()
        self.preprocessor = preprocessor

    def reset(self):
        self.correct = 0
        self.total = 0

    def get_metric(self, reset: bool) -> Dict[str, float]:
        if self.total == 0:
            return {"cogs_acc": 0.0}
        acc = self.correct / self.total
        if reset:
            self.reset()
        return {"cogs_acc": acc}

    def process(self, input: List[str]):
        if isinstance(input, list):
            lf = " ".join(self.preprocessor.postprocess(input))  # original COGS representation
        else:
            input = input.strip()
            lf = " ".join(self.preprocessor.postprocess(input.split(" ")))
        splitted = lf.split(" ; ")
        preamble = set() if len(splitted) == 1 else set(splitted[:-1])
        conjuncts = set(splitted[-1].split(" AND "))

        return preamble, conjuncts

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        assert len(predictions) == len(gold)
        for p, g in zip(predictions, gold):
            self.total += 1
            c = self.process(p) == self.process(g)
            self.correct += int(c)
            #print("Pred",p, self.process(p))
            #print("Gold", g, self.process(g))
            #print(c)
            #print("---")

def test_metric():
    from fertility.dataset_readers.preprocessor import COGSSimplifiedAggressive
    p = COGSSimplifiedAggressive(False)

    gold = ["* cake 4 * hen 10 customer 1 lend .agent 2 1 lend .theme 2 4 lend .recipient 2 10 cake .nmod .on 4 7 tree 7".split(" ")]

    metric = LenientCOGSEval(p)
    metric.add_instances(["* cake 4 * hen 10 customer 1 lend .agent 2 1 lend .theme 2 4 lend .recipient 2 10 cake .nmod .on 4 7 tree 7".split(" ")],
                         gold)

    assert metric.get_metric(True)["cogs_acc"] == 1.0

    metric.add_instances(["* hen 10 * cake 4 customer 1 lend .agent 2 1 lend .theme 2 4 lend .recipient 2 10 cake .nmod .on 4 7 tree 7".split(" ")],
                         gold)

    assert metric.get_metric(True)["cogs_acc"] == 1.0

    metric.add_instances(["* hen 10 * cake 4 lend .agent 2 1 lend .theme 2 4 lend .recipient 2 10 cake .nmod .on 4 7 tree 7 customer 1".split(" ")],
                         gold)

    assert metric.get_metric(True)["cogs_acc"] == 1.0

    metric.add_instances(["* hen 10 * cake 4 lend .agent 2 1 lend .theme 2 4 cake .nmod .on 4 7 tree 7 lend .recipient 2 10 customer 1".split(" ")],
                         gold)

    assert metric.get_metric(True)["cogs_acc"] == 1.0

    metric.add_instances(["* hen 10 * cake 5 lend .agent 2 1 lend .theme 2 4 cake .nmod .on 4 7 tree 7 lend .recipient 2 10 customer 1".split(" ")],
                         gold)

    assert metric.get_metric(True)["cogs_acc"] == 0.0

    metric.add_instances(["* hen 10 * cake 4 lend .agent 2 1 lend .theme 2 4 cake .nmod .on 4 8 tree 7 lend .recipient 2 10 customer 1".split(" ")],
                         gold)

    assert metric.get_metric(True)["cogs_acc"] == 0.0

    metric.add_instances(["* cake 4 * hen 10 customer 1 lend .agent 2 1 lend .theme 2 4 lend .recipient 2 10".split(" ")],
                         gold)

    assert metric.get_metric(True)["cogs_acc"] == 0.0
