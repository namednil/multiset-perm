from typing import Optional

import torch
from allennlp.training.metrics import Metric


class AlignmentAccTracker(Metric):

    def __init__(self):
        self.reset()

    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions shape (batch_size, input_seq_len, output_seq_len)
        gold_labels : `torch.Tensor`, required.
            A tensor of predictions shape (batch_size, input_seq_len, output_seq_len)
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of predictions shape (batch_size, output_seq_len)
        """
        gold = torch.argmax(gold_labels, dim=1) #shape (batch, output)
        pred = torch.argmax(predictions, dim=1) #shape (batch, output)
        if mask is None:
            mask = torch.ones_like(gold)
        self.correct += ((gold == pred) * mask).sum().cpu().numpy()
        self.instances += mask.sum().cpu().numpy()


    def get_metric(self, reset: bool):
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        if self.instances == 0:
            return 0.0
        r = self.correct / self.instances
        if reset:
            self.reset()
        return r

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        self.instances = 0
        self.correct =  0




class SimpleAccTracker(Metric):

    def __init__(self):
        self.reset()

    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions shape (batch_size,)
        gold_labels : `torch.Tensor`, required.
            A tensor of predictions shape (batch_size,)
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of predictions shape (batch_size,)
        """
        if mask is None:
            mask = torch.ones_like(gold_labels)
        assert gold_labels.shape == predictions.shape
        self.correct += ((gold_labels == predictions) * mask).sum().cpu().numpy()
        self.instances += gold_labels.shape[0]


    def get_metric(self, reset: bool):
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        if self.instances == 0:
            return 0.0
        r = self.correct / self.instances
        if reset:
            self.reset()
        return r

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        self.instances = 0
        self.correct =  0