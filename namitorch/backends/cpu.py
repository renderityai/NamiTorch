import math
from contextlib import nullcontext

import numpy as np

from ..device import Device


class CPUBackend:
    __slots__ = ()
    device = Device("cpu")
    module = np
    namespace = np
    array_type = np.ndarray

    def context(self):
        return nullcontext()

    def array(self, value, dtype=None, copy=True):
        if copy:
            return np.array(value, dtype=dtype, copy=True, order="C", subok=False)
        return np.asarray(value, dtype=dtype, order="C")

    def to_numpy(self, array):
        return array.copy(order="C")

    def writable(self, array):
        return array.flags.writeable

    def readonly(self, array):
        array.flags.writeable = False
        return array

    def erfc(self, array):
        return np.fromiter((math.erfc(float(value)) for value in array.flat), dtype=array.dtype, count=array.size).reshape(array.shape)

    def random(self, generator, operation, shape, dtype, **options):
        if operation == "permutation":
            return generator.permutation(shape).astype(dtype, copy=False)
        if operation == "choice":
            return np.asarray(generator.choice(options["n"], size=shape, replace=options["replace"]), dtype=dtype)
        if operation == "integers":
            return generator.integers(size=shape, dtype=dtype, **options)
        return getattr(generator, operation)(shape) if dtype is None else getattr(generator, operation)(shape, dtype=dtype)

    def validate_index(self, array, index):
        return None
