import re
from collections import defaultdict
from typing import Dict, List

from fertility.eval.lev import MyMetric

keywords = re.compile("(GET|FILTER|ORDERBY|COUNT|SEARCH|TOP)")


def get_repr(query_str):
    splitted = keywords.split(query_str)
    repr = set()
    i = 0
    # print(splitted)

    while i < len(splitted):
        tok = splitted[i]
        i += 1
        if not tok:
            continue
        if i < len(splitted):
            rest = splitted[i]
            i += 1
            repr.add((tok.strip().lower(), rest.strip().lower()))
        else:
            repr.add(tok.strip().lower())

    return repr

def okapi_correct(s1, s2):
    return get_repr(s1) == get_repr(s2)

@MyMetric.register("okapi_acc")
class OkapiAcc(MyMetric):

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.correct = 0
        self.total_by_clause = defaultdict(int)
        self.correct_by_clause = defaultdict(int)

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        assert len(predictions) == len(gold)
        self.total += len(gold)
        for p, g in zip(predictions, gold):
            if not isinstance(g, str):
                g = " ".join(g)
            if not isinstance(p, str):
                p = " ".join(p)
            order_invariant_g = get_repr(g)
            order_invariant_p = get_repr(p)

            correct = (order_invariant_p == order_invariant_g)

            self.total_by_clause[len(order_invariant_g)] += 1
            self.correct_by_clause[len(order_invariant_g)] += correct

            # if not correct:
            #     print("Predicted", " ".join(p))
            #     print("Gold", " ".join(g))
            #     print()

            self.correct += correct


    def get_metric(self, reset: bool) -> Dict[str, float]:
        if self.total == 0:
            return {"okapi_acc": 0.0}
        d = {"okapi_acc": self.correct / self.total}
        d.update({"okapi_acc_" + str(k): self.correct_by_clause[k] / v if v > 0.0 else 0.0
                  for k, v in self.total_by_clause.items()})
        d.update({"okapi_total_" + str(k): v for k, v in self.total_by_clause.items()})
        if reset:
            self.reset()
        return d


if __name__ == "__main__":
    # p = "GET drive.root.children COUNT FILTER file.name eq corean FILTER file.lastModifiedDateTime eq this Saturday ORDERBY file.name desc TOP 13"
    # q = "GET drive.root.children FILTER file.lastModifiedDateTime eq this Saturday FILTER file.name eq corean ORDERBY file.name desc TOP 13"

    p = "GET event FILTER event.hasAttachments eq True FILTER event.organizer eq melodee sodaro FILTER event.location eq conf room 1990"
    q = "GET event FILTER event.hasAttachments eq True FILTER event.location eq Conf Room 1990 FILTER event.organizer eq Melodee Sodaro"

    print(get_repr(p))
    print(get_repr(q))
    print(get_repr(p) == get_repr(q))