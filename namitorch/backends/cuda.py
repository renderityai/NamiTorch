from threading import RLock

import numpy as np

from ._cuda_runtime import load_cupy, runtime_call
from ._namespace import ArrayNamespace


class CUDABackend:
    def __init__(self, device):
        module = load_cupy()
        count = runtime_call("getDeviceCount")
        if device.index >= count:
            raise RuntimeError(f"CUDA device {device} is unavailable; found {count} CUDA devices.")
        self.device = device
        self.module = module
        self.array_type = module.ndarray
        self.namespace = ArrayNamespace(self)
        self._erfc = module.ElementwiseKernel("T x", "T y", "y = erfc(x);", "namitorch_erfc")
        self._rng = None
        self._rng_lock = RLock()

    def context(self):
        return self.module.cuda.Device(self.device.index)

    def array(self, value, dtype=None, copy=True):
        with self.context():
            return self.module.array(value, dtype=dtype, copy=copy, order="C")

    def to_numpy(self, array):
        with self.context():
            return self.module.asnumpy(array, order="C")

    def writable(self, array):
        return True

    def readonly(self, array):
        return array

    def erfc(self, array):
        with self.context():
            return self._erfc(array)

    def add_at(self, array, index, values):
        with self.context():
            if array.dtype.kind in "biu" and array.dtype.itemsize < 4:
                working = array.astype(np.int32)
                self.module.add.at(working, index, values)
                self.module.copyto(array, working.astype(array.dtype))
            elif array.dtype == np.dtype("int64"):
                unsigned_values = self.module.asarray(values, dtype=np.int64).view(np.uint64)
                self.module.add.at(array.view(np.uint64), index, unsigned_values)
            else:
                self.module.add.at(array, index, values)

    def random(self, generator, operation, shape, dtype, **options):
        dtype = np.float64 if dtype is None else dtype
        seed = int(generator.integers(0, 2 ** 63, dtype=np.int64))
        with self._rng_lock, self.context():
            if self._rng is None:
                self._rng = self.module.random.RandomState(seed)
            else:
                self._rng.seed(seed)
            rng = self._rng
            if operation == "permutation":
                return rng.permutation(shape).astype(dtype, copy=False)
            if operation == "choice":
                return self.module.asarray(rng.choice(options["n"], size=shape, replace=options["replace"]), dtype=dtype)
            if operation == "integers":
                return rng.randint(options["low"], options["high"], size=shape, dtype=dtype)
            if operation == "random":
                return rng.random_sample(shape, dtype=dtype)
            return rng.standard_normal(shape, dtype=dtype)

    def validate_index(self, array, index):
        if sum(component is Ellipsis for component in index) > 1:
            raise IndexError("An index can have only one ellipsis.")
        consumed = sum(component.ndim if isinstance(component, self.array_type) and component.dtype.kind == "b" else 1 for component in index if component is not None and component is not Ellipsis and not isinstance(component, bool))
        if consumed > array.ndim:
            raise IndexError(f"Too many indices for Tensor with shape {array.shape}.")
        axis = 0
        boolean_arrays = 0
        for component in index:
            if component is None or isinstance(component, bool):
                continue
            if component is Ellipsis:
                axis += array.ndim - consumed
                continue
            if isinstance(component, self.array_type) and component.dtype.kind == "b":
                if component.shape != array.shape[axis:axis + component.ndim]:
                    raise IndexError(f"Boolean index shape {component.shape} does not match Tensor shape {array.shape} at axis {axis}.")
                axis += component.ndim
                boolean_arrays += 1
                if boolean_arrays > 1:
                    raise RuntimeError("CUDA indexing with multiple boolean arrays is not supported by the CuPy backend.")
                continue
            size = array.shape[axis]
            if isinstance(component, self.array_type):
                if bool(self.module.any((component < -size) | (component >= size))):
                    raise IndexError(f"Index is out of bounds for axis {axis} of Tensor with shape {array.shape}.")
            elif not isinstance(component, slice) and not -size <= component < size:
                raise IndexError(f"Index {component} is out of bounds for axis {axis} of Tensor with shape {array.shape}.")
            axis += 1
