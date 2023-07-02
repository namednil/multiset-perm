from allennlp.common import Registrable


class RateScheduler(Registrable):

    def get_rate(self, epoch: int) -> float:
        raise NotImplementedError()

    def get_last(self) -> float:
        raise NotImplementedError()


@RateScheduler.register("on_off")
class OnOffRateScheduler(RateScheduler):

    def __init__(self, num_epochs: int, value: float):
        self.num_epochs = num_epochs
        self.value = value

    def get_rate(self, epoch: int) -> float:
        return 0.0 if epoch < self.num_epochs else self.value

    def get_last(self) -> float:
        return self.value

@RateScheduler.register("constant")
class ConstantScheduler(RateScheduler):

    def __init__(self, constant: float):
        self.constant = constant

    def get_rate(self, epoch: int) -> float:
        return self.constant

    def get_last(self) -> float:
        return self.constant


@RateScheduler.register("linear")
class LinearScheduler(RateScheduler):

    def __init__(self, begin_epoch: int, num_epochs: int, max_value: float):
        self.begin_epoch = begin_epoch
        self.total_epochs = num_epochs
        self.max_value = max_value

    def get_rate(self, epoch: int) -> float:
        if epoch < self.begin_epoch:
            return 0.0
        rel_epoch = epoch - self.begin_epoch
        d = self.max_value / (self.total_epochs - self.begin_epoch)
        return rel_epoch * d

    def get_last(self) -> float:
        return self.max_value



@RateScheduler.register("linear_start_end")
class LinearStartEndScheduler(RateScheduler):

    def __init__(self, start_value: float, end_value: float, num_epochs: int):
        """

        :type start_value: object
        """
        self.num_epochs = num_epochs
        self.end_value = end_value
        self.start_value = start_value

    def get_rate(self, epoch: int) -> float:
        slope = (self.end_value - self.start_value) / self.num_epochs
        return epoch * slope + self.start_value

    def get_last(self) -> float:
        return self.end_value