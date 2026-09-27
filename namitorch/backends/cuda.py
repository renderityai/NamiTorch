import numpy as np

from ._cuda_random import CUDARandom
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

    def context(self):
        return self.module.cuda.Device(self.device.index)

    def array(self, value, dtype=None, copy=True, *, non_blocking=False):
        device_input = isinstance(value, self.array_type)
        cross_device = device_input and value.device.id != self.device.index
        if cross_device:
            with value.device:
                self.module.cuda.get_current_stream().synchronize()
            if not self.module.cuda.runtime.deviceCanAccessPeer(self.device.index, value.device.id):
                with value.device:
                    value = self.module.asnumpy(value, order="C", blocking=True)
                device_input = False
        with self.context():
            result = self.module.array(value, dtype=dtype, copy=copy, order="C")
            if not non_blocking or not device_input or cross_device:
                self.module.cuda.get_current_stream().synchronize()
            if result.device.id != self.device.index:
                raise RuntimeError(f"CUDA transfer produced storage on cuda:{result.device.id}, expected {self.device}.")
            return result

    def to_numpy(self, array):
        with self.context():
            return self.module.asnumpy(array, order="C", blocking=True)

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

    def create_rng(self, seed):
        return CUDARandom(self, seed)

    def seed_rng(self, generator, seed):
        generator.manual_seed(seed)

    def rng_state(self, generator):
        return generator.get_state()

    def restore_rng(self, generator, state):
        generator.set_state(state)

    def random(self, generator, operation, shape, dtype, **options):
        if not isinstance(generator, CUDARandom) or generator.device != self.device:
            raise RuntimeError(f"CUDA random operations on {self.device} require a generator on the same device.")
        return generator.draw(operation, shape, dtype, **options)

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
