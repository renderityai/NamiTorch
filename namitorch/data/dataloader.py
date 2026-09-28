from ..random import Generator, _validate_generator
from ._utils import boolean, integer, normalize_index
from ._pin_memory import pin_batch
from .collate import default_collate, default_convert
from .dataset import IterableDataset, _map_dataset
from .sampler import BatchSampler, RandomSampler, SequentialSampler


_DEFAULT_BATCH_SIZE = object()


class DataLoader:
    def __init__(
        self, dataset, batch_size=_DEFAULT_BATCH_SIZE, shuffle: bool | None = None,
        sampler=None, batch_sampler=None, drop_last: bool = False, collate_fn=None,
        *, generator: Generator | None = None, pin_memory: bool = False,
    ):
        shuffle = False if shuffle is None else boolean(shuffle, "shuffle")
        drop_last = boolean(drop_last, "drop_last")
        self.pin_memory = boolean(pin_memory, "pin_memory")
        _validate_generator(generator, "cpu")
        if collate_fn is not None and not callable(collate_fn):
            raise TypeError("collate_fn must be callable or None.")
        if batch_sampler is not None and (batch_size is not _DEFAULT_BATCH_SIZE or shuffle or sampler is not None or drop_last):
            raise ValueError("batch_sampler cannot be combined with batch_size, shuffle=True, sampler or drop_last=True.")
        if sampler is not None and shuffle:
            raise ValueError("sampler cannot be combined with shuffle=True.")
        iterable = isinstance(dataset, IterableDataset)
        if iterable:
            if shuffle or sampler is not None or batch_sampler is not None:
                raise ValueError("IterableDataset cannot use shuffle or index samplers.")
        else:
            _map_dataset(dataset)
        size = 1 if batch_size is _DEFAULT_BATCH_SIZE else batch_size
        if batch_sampler is not None:
            size = None
        elif size is not None:
            size = integer(size, "batch_size", 1)
        elif drop_last:
            raise ValueError("drop_last requires automatic batching.")
        for name, value in (("sampler", sampler), ("batch_sampler", batch_sampler)):
            if value is not None:
                if isinstance(value, (str, bytes, dict, set, frozenset)):
                    raise TypeError(f"{name} must be an ordered iterable.")
                try:
                    iter(value)
                except TypeError:
                    raise TypeError(f"{name} must be iterable.") from None
        self.dataset = dataset
        self.batch_size = size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self._iterable = iterable
        self._custom_batch_sampler = batch_sampler is not None
        self.sampler = sampler
        if not iterable and sampler is None and batch_sampler is None:
            self.sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
        self.batch_sampler = batch_sampler
        if not iterable and batch_sampler is None and size is not None:
            self.batch_sampler = BatchSampler(self.sampler, size, drop_last)
        batching = size is not None or batch_sampler is not None
        self.collate_fn = collate_fn if collate_fn is not None else default_collate if batching else default_convert

    def _sample(self, index):
        return self.dataset[normalize_index(index, len(self.dataset))]

    def _collate(self, values):
        batch = self.collate_fn(values)
        return pin_batch(batch) if self.pin_memory else batch

    def __iter__(self):
        if self._iterable:
            if self.batch_size is None:
                for sample in self.dataset:
                    yield self._collate(sample)
            else:
                batch = []
                for sample in self.dataset:
                    batch.append(sample)
                    if len(batch) == self.batch_size:
                        yield self._collate(batch)
                        batch = []
                if batch and not self.drop_last:
                    yield self._collate(batch)
        elif self._custom_batch_sampler or self.batch_size is not None:
            batches = self.batch_sampler if self._custom_batch_sampler else BatchSampler(self.sampler, self.batch_size, self.drop_last)
            for indices in batches:
                yield self._collate([self._sample(index) for index in indices])
        else:
            for index in self.sampler:
                yield self._collate(self._sample(index))

    def __len__(self) -> int:
        if self._custom_batch_sampler:
            return len(self.batch_sampler)
        count = len(self.dataset) if self._iterable else len(self.sampler)
        if self.batch_size is None:
            return count
        return count // self.batch_size if self.drop_last else (count + self.batch_size - 1) // self.batch_size


__all__ = ["DataLoader"]
