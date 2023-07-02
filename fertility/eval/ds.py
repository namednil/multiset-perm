import torch


class DSEval:
    """
    Measures how much we are away (in the extreme) from double-stochasticity on average per input.
    """

    def __init__(self):
        self.reset()


    def reset(self):
        self.total_instances = 0
        self.sum_min = 0
        self.sum_max = 0

    def get_metrics(self, reset: bool):
        d = dict()
        if self.total_instances == 0:
            d["avg_min_stoch"] = 0.0
            d["avg_max_stoch"] = 0.0
        else:
            d["avg_min_stoch"] = self.sum_min / self.total_instances
            d["avg_max_stoch"] = self.sum_max / self.total_instances


        if reset:
            self.reset()

        return d



    def add_matrix(self, m, mask):
        """

        :param m: shape (batch, n, n)
        :param mask: shape (batch_size, n) where mask[b,i] = True iff that element is present
        :return:
        """
        with torch.no_grad():
            batch_size = m.shape[0]
            s1 = m.sum(dim=1)
            s2 = m.sum(dim=2)

            mins1, _ = torch.min(s1 + ~mask, dim=1) #make sure that padding counts as 1s for this purpose.
            mins2, _ = torch.min(s2 + ~mask, dim=1) #shape (batch_size,)

            max1, _ = torch.max(s1, dim=1)
            max2, _ = torch.max(s2, dim=1)

            self.sum_min += torch.sum(torch.minimum(mins1, mins2)).detach().cpu().numpy()
            self.sum_max += torch.sum(torch.maximum(max1, max2)).detach().cpu().numpy()

            self.total_instances += batch_size