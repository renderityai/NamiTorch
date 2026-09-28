from .._graph_capture import current_capture
from dataclasses import dataclass
from io import StringIO
from operator import index
from threading import RLock
from types import MappingProxyType

import numpy as np


_OPTIONS = ("--std=c++11",)
_TYPES = MappingProxyType({np.dtype("float32"): "float", np.dtype("float64"): "double"})
_OPERATIONS = MappingProxyType({
    "add": (2, "a[index] + b[index]"),
    "mul": (2, "a[index] * b[index]"),
    "relu": (1, "a[index] <= value_type(0) ? value_type(0) : a[index]"),
})


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool.")
    try:
        return index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer.") from None


def launch_dimensions(numel, block_size=256, max_grid_size=2147483647):
    count = _integer(numel, "numel")
    block = _integer(block_size, "block_size")
    limit = _integer(max_grid_size, "max_grid_size")
    if not 0 <= count <= 9223372036854775807:
        raise ValueError("numel must be between 0 and 2**63 - 1.")
    if not 1 <= block <= 1024:
        raise ValueError("block_size must be between 1 and 1024.")
    if not 1 <= limit <= 2147483647:
        raise ValueError("max_grid_size must be between 1 and 2**31 - 1.")
    if count == 0:
        return None
    return (min((count + block - 1) // block, limit),), (block,)


@dataclass(frozen=True)
class _KernelKey:
    source: str
    name: str
    dtype: str
    compute_capability: str
    options: tuple[str, ...]
    device_index: int
    compiler: str = "nvrtc"


def _kernel_key(operation, dtype, capability, device_index):
    if operation not in _OPERATIONS:
        raise ValueError(f"Unsupported internal CUDA kernel: {operation!r}.")
    if dtype not in _TYPES:
        raise TypeError("Custom CUDA kernels support only float32 and float64.")
    arity, expression = _OPERATIONS[operation]
    name = f"namitorch_{operation}_{dtype.name}"
    second = "const value_type* b, " if arity == 2 else ""
    source = (
        f"using value_type = {_TYPES[dtype]};\n"
        f'extern "C" __global__ void {name}(const value_type* a, {second}'
        "value_type* output, unsigned long long count) {\n"
        "    const unsigned long long stride = static_cast<unsigned long long>(blockDim.x) * gridDim.x;\n"
        "    for (unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n"
        "         index < count; index += stride) {\n"
        f"        output[index] = {expression};\n"
        "    }\n"
        "}\n"
    )
    return _KernelKey(source, name, dtype.str, str(capability), _OPTIONS, device_index)


class CUDAKernelCompilationError(RuntimeError):
    def __init__(self, key, device, compiler_log):
        self.function_name = key.name
        self.dtype = key.dtype
        self.device = device
        self.compute_capability = key.compute_capability
        self.options = key.options
        self.compiler_log = compiler_log
        super().__init__(
            f"CUDA kernel {key.name!r} compilation failed on {device} "
            f"(dtype={key.dtype}, compute capability={key.compute_capability}, options={key.options}):\n{compiler_log}"
        )


class CUDAKernels:
    def __init__(self, backend):
        self._backend = backend
        self._cache = {}
        self._lock = RLock()
        self._capability = None
        self._block_size = None
        self._max_grid_size = None

    def _configure_device(self):
        if self._capability is None:
            device = self._backend.module.cuda.Device(self._backend.device.index)
            attributes = device.attributes
            self._block_size = min(256, attributes["MaxThreadsPerBlock"], attributes["MaxBlockDimX"])
            self._max_grid_size = min(2147483647, attributes["MaxGridDimX"])
            self._capability = device.compute_capability

    def _get_kernel(self, operation, dtype):
        with self._lock:
            self._configure_device()
            key = _kernel_key(operation, dtype, self._capability, self._backend.device.index)
            return self._compile_key(key)

    def _compile_key(self, key):
        with self._lock:
            kernel = self._cache.get(key)
            if kernel is None:
                if current_capture() is not None and current_capture().phase == "capture":
                    raise RuntimeError("CUDA graph kernel was not compiled during warmup.")
                log = StringIO()
                try:
                    kernel = self._backend.module.RawKernel(key.source, key.name, options=key.options, backend=key.compiler)
                    kernel.compile(log_stream=log)
                except MemoryError:
                    raise
                except Exception as error:
                    compiler_log = "\n".join(part for part in (log.getvalue(), str(error)) if part)
                    raise CUDAKernelCompilationError(key, self._backend.device, compiler_log) from error
                self._cache[key] = kernel
            return kernel

    def _validate(self, arrays):
        for array in arrays:
            if not isinstance(array, self._backend.array_type):
                raise TypeError("Custom CUDA kernels require CuPy storage arrays.")
            if array.device.id != self._backend.device.index:
                raise RuntimeError(f"Custom CUDA kernel expected {self._backend.device}, received cuda:{array.device.id}; transfer explicitly.")
            if array.dtype not in _TYPES:
                raise TypeError("Custom CUDA kernels support only float32 and float64.")
            if not array.flags.c_contiguous:
                raise ValueError("Custom CUDA kernels require C-contiguous arrays; make an explicit contiguous copy.")
            if array.data.ptr % array.dtype.alignment:
                raise ValueError("Custom CUDA kernels require dtype-aligned storage.")
        first = arrays[0]
        if any(array.dtype != first.dtype for array in arrays[1:]):
            raise TypeError("Custom CUDA kernel operands must have the same dtype; convert explicitly.")
        if any(array.shape != first.shape for array in arrays[1:]):
            raise ValueError(f"Custom CUDA kernel operands must have identical shapes, received {tuple(array.shape for array in arrays)}.")

    def _elementwise(self, operation, arrays):
        self._validate(arrays)
        first = arrays[0]
        count = int(first.size)
        backend = self._backend
        with backend.context(arrays):
            output = backend.module.empty(first.shape, dtype=first.dtype)
            if count:
                kernel = self._get_kernel(operation, first.dtype)
                grid, block = launch_dimensions(count, self._block_size, self._max_grid_size)
                stream = backend.module.cuda.get_current_stream()
                backend.execution.hold(kernel)
                kernel(grid, block, (*arrays, output, np.uint64(count)), stream=stream)
            return backend.result(output)

    def add(self, left, right):
        return self._elementwise("add", (left, right))

    def mul(self, left, right):
        return self._elementwise("mul", (left, right))

    def relu(self, array):
        return self._elementwise("relu", (array,))
