import random
from typing import Sequence, Iterable, List, Tuple, Dict

from allennlp.common.util import lazy_groups_of
from allennlp.data import BatchSampler, Instance


@BatchSampler.register("validation_bucket_length")
class BucketLengthBatchSampler(BatchSampler):
    """
    This batch sampler is meant for use at validation time mainly if we have a dataset where there are some
    really long sequences, of which only few fit into GPU memory, and some shorter sequences of which we can fit more into memory.
    """

    def __init__(
        self,
        batch_sizes: List[Tuple[int, int, int]],
        # [(a,b,c), (d,e,f) means if a <= length <= b, then batch c elements, if d <= length <= f, then batch f elements etc.
        length_key: str,
        shuffle: bool = True,
    ):
        self.length_key = length_key
        self.batch_sizes = batch_sizes
        self.length2batch_size = dict()
        self.smallest_bs = min(bs for (l, u, bs) in batch_sizes)

        self.shuffle = shuffle

    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        return sum(1 for _ in self.get_batch_indices(instances))

    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        # Partition instances by length
        batch_size2inst = dict()
        for i, ins in enumerate(instances):
            length = len(ins.fields[self.length_key])
            for (l, u, bs) in self.batch_sizes:
                if length in range(l, u+1):
                    if bs not in batch_size2inst:
                        batch_size2inst[bs] = []
                    batch_size2inst[bs].append(i)
                    break
            else:
                # didn't do a break, add instance to the smallest batch size, to be safe
                if self.smallest_bs not in batch_size2inst:
                    batch_size2inst[self.smallest_bs] = []
                batch_size2inst[self.smallest_bs].append(i)

        batches = []
        for (batch_size, instances_per_size) in batch_size2inst.items():
            # also sort by length within the group to reduce padding
            for batch in lazy_groups_of(sorted(instances_per_size, key=lambda i: len(instances[i].fields[self.length_key])), batch_size):
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

