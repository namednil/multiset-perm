import copy
from typing import Dict, Any, List, Optional

from allennlp.common import Registrable
from allennlp.training import TrainerCallback

from fertility.eval.acc_by_length import MetricByCat
from fertility.eval.lev import MyMetric, LevenstheinMetric, LengthAcc

import logging
logger = logging.getLogger(__name__)


@TrainerCallback.register("test_mode")
class TestModeCallBack(TrainerCallback):
    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        trainer.model.is_in_test_mode = False

    def on_end(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        trainer.model.is_in_test_mode = True
        logger.info("Activating test mode")

class TestMode:
    """
    Test mode is used for evaluating with and without a grammar at test time.
    """
    def __init__(self, metrics: List[MyMetric], activate: bool):
        self.metrics = copy.deepcopy(metrics or [])
        self.metrics.append(LevenstheinMetric())
        self.metrics.append(LengthAcc())
        self.activate = activate
        self.metrics_by_cat = MetricByCat(metrics)

    def check_callback(self, model):
        if model.training and not hasattr(model, "is_in_test_mode"):
            raise ValueError("Please use the test_mode trainer callback!")

    def is_in_test_mode(self, model) -> bool:
        return self.activate and (not hasattr(model, "is_in_test_mode") or model.is_in_test_mode)

    def get_metrics(self, reset: bool) -> Dict[str, float]:
        d = dict()
        if self.activate:
            for m in self.metrics:
                for k, v in m.get_metric(reset).items():
                    d["test_mode_"+k] = v
            for k, v in self.metrics_by_cat.get_metrics(reset).items():
                    d["test_mode_"+k] = v
        return d

    def add_instances(self, pred: List[List[str]], gold: List[List[str]], categories: Optional[List[str]] = None) -> None:
        if not self.activate:
            return
        for m in self.metrics:
            m.add_instances(pred, gold)

        if categories and any(categories):
            self.metrics_by_cat.add_instances(pred, gold, categories)
