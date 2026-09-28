from contextlib import contextmanager
from threading import RLock

import numpy as np

from .._cuda_execution import CUDAExecution, synchronize_array
from .._cuda_memory import CUDAMemory
from .._cuda_random import CUDARandom
from .._cuda_runtime import load_cupy, runtime_call
from .._namespace import ArrayNamespace
from .._pinned import is_pinned
from .kernels import CUDAKernels
from .fused import FusedCUDAKernels
from .softmax import SoftmaxCUDAKernels
from .normalization import NormalizationCUDAKernels


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
        self.memory = CUDAMemory(module, device)
        self.execution = CUDAExecution(module, device)
        self.kernels = CUDAKernels(self)
        self.fused = FusedCUDAKernels(self)
        self.softmax = SoftmaxCUDAKernels(self)
        self.normalization = NormalizationCUDAKernels(self)
        self._float16_supported = None
        self.memory.track_allocation = self.execution.allocated
        self.memory.collect_completed = self.collect_transfers
        self._pending_transfers = []
        self._transfer_lock = RLock()
        self._erfc = module.ElementwiseKernel("T x", "T y", "y = erfc(x);", "namitorch_erfc")

    def supports_float16(self):
        if self._float16_supported is None:
            properties = self.module.cuda.runtime.getDeviceProperties(self.device.index)
            capability = (int(properties["major"]), int(properties["minor"]))
            self._float16_supported = capability >= (5, 3) and self.module.dtype("float16") == np.dtype("float16") and callable(self.module.matmul)
        return self._float16_supported

    @contextmanager
    def context(self, *operands):
        with self.memory.context():
            self.collect_transfers()
            with self.execution.context(operands):
                yield
            self.collect_transfers()

    def collect_transfers(self):
        with self.module.cuda.Device(self.device.index), self._transfer_lock:
            self.execution.collect()
            self._pending_transfers = [entry for entry in self._pending_transfers if not entry[0].done]

    def result(self, value):
        return self.execution.result(value)

    def ready_for_host(self, array):
        self.execution.synchronize_array(array)
        return array

    def _retain_transfer(self, event, stream, source, destination):
        try:
            event.record(stream)
            with self._transfer_lock:
                self._pending_transfers.append((event, source, destination))
        except Exception:
            with self._transfer_lock:
                self._pending_transfers.append((stream, source, destination))
            stream.synchronize()
            raise

    def array(self, value, dtype=None, copy=True, *, non_blocking=False):
        if dtype is not None and np.dtype(dtype) == np.dtype("float16") and not self.supports_float16():
            raise RuntimeError(f"CUDA float16 is not supported on {self.device}.")
        device_input = isinstance(value, self.array_type)
        cross_device = device_input and value.device.id != self.device.index
        if cross_device:
            synchronize_array(value)
            with value.device:
                self.module.cuda.get_current_stream().synchronize()
            if not self.module.cuda.runtime.deviceCanAccessPeer(self.device.index, value.device.id):
                with value.device:
                    value = self.module.asnumpy(value, order="C", blocking=True)
                device_input = False
        with self.context(value):
            stream = self.module.cuda.get_current_stream()
            if np.isscalar(value):
                scalar = np.asarray(value, dtype=dtype).item()
                return self.module.full((), scalar, dtype=dtype)
            pinned_source = (
                not device_input and non_blocking and is_pinned(value)
                and value.flags.c_contiguous and value.dtype.isnative
                and (dtype is None or value.dtype == np.dtype(dtype))
            )
            if pinned_source:
                result = self.module.empty(value.shape, dtype=value.dtype)
                event = self.module.cuda.Event(disable_timing=True)
                try:
                    result.data.copy_from_host_async(value.ctypes.data, value.nbytes, stream)
                except Exception:
                    stream.synchronize()
                    raise
                self._retain_transfer(event, stream, value, result)
                return result
            result = self.module.array(value, dtype=dtype, copy=copy, order="C")
            if not device_input or cross_device:
                stream.synchronize()
            if result.device.id != self.device.index:
                raise RuntimeError(f"CUDA transfer produced storage on cuda:{result.device.id}, expected {self.device}.")
            return result

    def item(self, array):
        with self.context(array):
            self.execution.synchronize_array(array)
            return array.item()

    def to_numpy(self, array):
        with self.context(array):
            self.execution.synchronize_array(array)
            return self.module.asnumpy(array, order="C", blocking=True)

    def writable(self, array):
        return True

    def readonly(self, array):
        return array

    def erfc(self, array):
        with self.context(array):
            return self._erfc(array)

    def add_at(self, array, index, values):
        with self.context(array, index, values):
            if array.dtype == np.dtype("float16"):
                working = array.astype(np.float32)
                self.module.add.at(working, index, self.module.asarray(values, dtype=np.float32))
                self.module.copyto(array, working.astype(np.float16))
            elif array.dtype.kind in "biu" and array.dtype.itemsize < 4:
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
