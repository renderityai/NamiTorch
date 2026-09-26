from collections.abc import Mapping

import numpy as np

from ..ops import stack
from ..tensor import Tensor


def default_collate(batch):
    if not isinstance(batch, (list, tuple)):
        raise TypeError("default_collate expects a list or tuple of samples.")
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    first = batch[0]
    if isinstance(first, Tensor):
        if not all(isinstance(sample, Tensor) for sample in batch):
            raise TypeError("A Tensor batch cannot mix sample types.")
        return stack(batch, dim=0)
    if isinstance(first, np.ndarray):
        if not all(isinstance(sample, np.ndarray) for sample in batch):
            raise TypeError("An ndarray batch cannot mix sample types.")
        return stack([Tensor(sample) for sample in batch], dim=0)
    if isinstance(first, (str, bytes)):
        if not all(isinstance(sample, type(first)) for sample in batch):
            raise TypeError("String batches must contain matching string types.")
        return list(batch)
    if isinstance(first, (bool, int, float, np.generic)):
        if not all(isinstance(sample, (bool, int, float, np.generic)) for sample in batch):
            raise TypeError("Numeric batches must contain only numeric scalars.")
        return stack([Tensor(sample) for sample in batch], dim=0)
    if isinstance(first, Mapping):
        keys = set(first)
        if not all(isinstance(sample, Mapping) and set(sample) == keys for sample in batch):
            raise ValueError("Dictionary samples must have identical keys.")
        return {key: default_collate([sample[key] for sample in batch]) for key in first}
    if isinstance(first, (tuple, list)):
        if not all(type(sample) is type(first) and len(sample) == len(first) for sample in batch):
            raise ValueError("Sequence samples must have the same type and length.")
        fields = [default_collate(list(values)) for values in zip(*batch)]
        if isinstance(first, tuple):
            return type(first)(*fields) if hasattr(first, "_fields") else tuple(fields)
        return fields
    raise TypeError(f"Cannot collate samples of type {type(first).__name__}.")


def default_convert(sample):
    if isinstance(sample, np.ndarray) or isinstance(sample, np.generic) and not isinstance(sample, (str, bytes)):
        return Tensor(sample)
    if isinstance(sample, Mapping):
        return {key: default_convert(value) for key, value in sample.items()}
    if isinstance(sample, tuple):
        values = [default_convert(value) for value in sample]
        return type(sample)(*values) if hasattr(sample, "_fields") else tuple(values)
    if isinstance(sample, list):
        return [default_convert(value) for value in sample]
    return sample


__all__ = ["default_collate", "default_convert"]
