from typing import List, Dict

from allennlp.common import Lazy

from fertility.decoding.decoding_grammar import DecodingGrammar
from fertility.eval.lev import MyMetric

from nltk import parse

def recognize(parser, s, start_symbol):
    chart = parser.chart_parse(s)
    it = chart.parses(start_symbol)
    try:
        next(it)
        return True
    except StopIteration:
        return False

@MyMetric.register("wellformed")
class WellformednessMetric(MyMetric):

    def __init__(self, decoding_grammar: Lazy[DecodingGrammar], check_gold: bool = False):
        #We just snatch the CFG here
        self.grammar = decoding_grammar.construct(tok2id=dict()).grammar
        self.parser = parse.chart.BottomUpChartParser(self.grammar)
        self.reset()
        self.check_gold = check_gold

    def reset(self):
        self.total = 0
        self.well_formed = 0

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:

        for p, g in zip(predictions, gold):
            try:
                self.well_formed += recognize(self.parser, p, self.grammar.start())
            except ValueError:
                # word not covered?
                pass
                # print(p)
            self.total += 1
            if self.check_gold:
                r = recognize(self.parser, g, self.grammar.start())
                if not r:
                    raise ValueError("Could not parse gold sequence: "+ repr(g))

    def get_metric(self, reset: bool) -> Dict[str, float]:
        if self.total == 0:
            return {"well_formed": 0}
        w = self.well_formed / self.total
        if reset:
            self.reset()
        return {"well_formed": w}
