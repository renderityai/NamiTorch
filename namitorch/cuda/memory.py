from numbers import Integral

from ..backends import get_backend
from ..backends._cuda_memory import NamiTorchCUDAOutOfMemoryError
from ..backends._cuda_runtime import runtime_call
from ..device import Device


def _backend(device):
    if device is None:
        device = Device("cuda", int(runtime_call("getDevice")))
    elif isinstance(device, Integral) and not isinstance(device, bool):
        device = Device("cuda", device)
    else:
        device = Device(device)
    if device.type != "cuda":
        raise ValueError("CUDA memory operations require a CUDA device.")
    return get_backend(device)


def memory_allocated(device=None):
    return _backend(device).memory.stats()[0]


def memory_reserved(device=None):
    return _backend(device).memory.stats()[1]


def max_memory_allocated(device=None):
    return _backend(device).memory.stats()[2]


def max_memory_reserved(device=None):
    return _backend(device).memory.stats()[3]


def empty_cache(device=None):
    backend = _backend(device)
    backend.collect_transfers()
    backend.memory.empty_cache()


def reset_peak_memory_stats(device=None):
    _backend(device).memory.reset_peaks()


def mem_get_info(device=None):
    return _backend(device).memory.mem_get_info()


def synchronize(device=None):
    backend = _backend(device)
    with backend.module.cuda.Device(backend.device.index):
        backend.module.cuda.runtime.deviceSynchronize()
    backend.collect_transfers()


__all__ = [
    "NamiTorchCUDAOutOfMemoryError", "memory_allocated", "memory_reserved",
    "max_memory_allocated", "max_memory_reserved", "empty_cache",
    "reset_peak_memory_stats", "mem_get_info", "synchronize",
]
