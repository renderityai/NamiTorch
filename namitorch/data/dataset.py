from bisect import bisect_right
from itertools import accumulate

from ..tensor import Tensor
from ._utils import normalize_index


class Dataset:
    def __getitem__(self, index):
        raise RuntimeError("Dataset subclass must implement __getitem__.")

    def __len__(self):
        raise RuntimeError("Dataset subclass must implement __len__.")


class IterableDataset:
    def __iter__(self):
        raise RuntimeError("IterableDataset subclass must implement __iter__.")


def _map_dataset(dataset) -> None:
    if isinstance(dataset, IterableDataset):
        raise TypeError("This operation requires an indexable dataset, not IterableDataset.")
    if not callable(getattr(dataset, "__getitem__", None)) or not callable(getattr(dataset, "__len__", None)):
        raise TypeError("An indexable dataset must implement __getitem__ and __len__.")


class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor):
        if not tensors:
            raise ValueError("TensorDataset requires at least one Tensor.")
        if any(not isinstance(tensor, Tensor) for tensor in tensors):
            raise TypeError("TensorDataset accepts only NamiTorch Tensors.")
        if any(tensor.ndim == 0 for tensor in tensors):
            raise ValueError("TensorDataset tensors must have at least one dimension.")
        if any(tensor.shape[0] != tensors[0].shape[0] for tensor in tensors):
            raise ValueError("TensorDataset tensors must have the same first dimension.")
        self.tensors = tuple(tensors)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self) -> int:
        return self.tensors[0].shape[0]


class Subset(Dataset):
    def __init__(self, dataset, indices):
        _map_dataset(dataset)
        if isinstance(indices, Tensor):
            if not indices.dtype.is_integer or indices.ndim != 1:
                raise TypeError("Subset indices must be a one-dimensional integer Tensor.")
            indices = indices.tolist()
        if isinstance(indices, (str, bytes, dict, set, frozenset)):
            raise TypeError("Subset indices must be an ordered iterable of integers.")
        try:
            values = list(indices)
        except TypeError:
            raise TypeError("Subset indices must be an ordered iterable of integers.") from None
        size = len(dataset)
        self.indices = [normalize_index(index, size) for index in values]
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[self.indices[normalize_index(index, len(self))]]

    def __len__(self) -> int:
        return len(self.indices)


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        if isinstance(datasets, (str, bytes, dict, set, frozenset)):
            raise TypeError("datasets must be an ordered iterable of indexable datasets.")
        try:
            datasets = list(datasets)
        except TypeError:
            raise TypeError("datasets must be an ordered iterable of indexable datasets.") from None
        if not datasets:
            raise ValueError("ConcatDataset requires at least one dataset.")
        for dataset in datasets:
            _map_dataset(dataset)
        self.datasets = tuple(datasets)
        self.cumulative_sizes = tuple(accumulate(len(dataset) for dataset in datasets))

    def __getitem__(self, index):
        index = normalize_index(index, len(self))
        dataset_index = bisect_right(self.cumulative_sizes, index)
        offset = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        return self.datasets[dataset_index][index - offset]

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]


__all__ = ["Dataset", "IterableDataset", "TensorDataset", "Subset", "ConcatDataset"]
