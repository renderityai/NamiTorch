import operator

import numpy as np

from ..random import Generator, _validate_generator, choice, permutation
from ._utils import boolean, integer


class SequentialSampler:
    def __init__(self, data_source):
        len(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler:
    def __init__(self, data_source, replacement: bool = False, num_samples: int | None = None, *, generator: Generator | None = None):
        if type(replacement) is not bool:
            raise TypeError("replacement must be a Python bool.")
        _validate_generator(generator)
        if num_samples is not None:
            if isinstance(num_samples, (bool, np.bool_)):
                raise TypeError("num_samples must be an integer, not a boolean.")
            try:
                num_samples = operator.index(num_samples)
            except TypeError:
                raise TypeError("num_samples must be an integer or None.") from None
            if num_samples < 0:
                raise ValueError("num_samples must be nonnegative.")
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.generator = generator
        self._sizes()

    def _sizes(self) -> tuple[int, int]:
        size = len(self.data_source)
        count = size if self.num_samples is None else self.num_samples
        if count and size == 0:
            raise ValueError("Cannot sample from an empty data source.")
        if not self.replacement and count > size:
            raise ValueError("num_samples cannot exceed data source length without replacement.")
        return size, count

    def __len__(self) -> int:
        return self._sizes()[1]

    def __iter__(self):
        size, count = self._sizes()
        if count == 0:
            return
        indices = choice(size, (count,), generator=self.generator) if self.replacement else permutation(size, generator=self.generator)[:count]
        yield from indices.tolist()


class BatchSampler:
    def __init__(self, sampler, batch_size: int, drop_last: bool = False):
        if isinstance(sampler, (str, bytes, dict, set, frozenset)):
            raise TypeError("sampler must be an ordered iterable of indices.")
        try:
            iter(sampler)
        except TypeError:
            raise TypeError("sampler must be an ordered iterable of indices.") from None
        self.batch_size = integer(batch_size, "batch_size", 1)
        self.drop_last = boolean(drop_last, "drop_last")
        self.sampler = sampler

    def __iter__(self):
        batch = []
        for index in self.sampler:
            batch.append(integer(index, "sample index"))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        count = len(self.sampler)
        return count // self.batch_size if self.drop_last else (count + self.batch_size - 1) // self.batch_size


__all__ = ["SequentialSampler", "RandomSampler", "BatchSampler"]
