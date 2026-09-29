from ..backends._cuda_peer import can_access_peer, enable_peer_access, disable_peer_access
from ..backends._cuda_runtime import runtime_call
from ..device import Device
from ..backends._settings import enable_fused_kernels, fused_kernels_enabled


def is_available():
    try:
        return runtime_call("getDeviceCount") > 0
    except RuntimeError:
        return False


def device_count():
    try:
        return int(runtime_call("getDeviceCount"))
    except RuntimeError:
        return 0


def current_device():
    return int(runtime_call("getDevice"))


def _index(index):
    if index is None:
        return current_device()
    target = Device("cuda", index)
    count = runtime_call("getDeviceCount")
    if target.index >= count:
        raise RuntimeError(f"CUDA device {target} is unavailable; found {count} CUDA devices.")
    return target.index


def set_device(index):
    if index is None:
        raise TypeError("set_device requires an explicit CUDA device index.")
    runtime_call("setDevice", _index(index))


def get_device_properties(index=None):
    properties = runtime_call("getDeviceProperties", _index(index))
    return {
        key.decode("utf-8") if isinstance(key, bytes) else key:
        value.rstrip(b"\x00").decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        for key, value in properties.items()
    }


def get_device_name(index=None):
    return get_device_properties(index)["name"]


def manual_seed(seed):
    from ..random import _get_default_generator, _seed
    value = _seed(seed)
    _get_default_generator(Device("cuda", current_device())).manual_seed(value)


def manual_seed_all(seed):
    from ..random import _manual_seed_cuda_all
    _manual_seed_cuda_all(seed)


from .streams import Event, Stream, current_stream, default_stream, elapsed_time, stream
from . import memory, graphs
from .graphs import CUDAGraph
from .memory import NamiTorchCUDAOutOfMemoryError, empty_cache, max_memory_allocated, max_memory_reserved, mem_get_info, memory_allocated, memory_reserved, reset_peak_memory_stats, synchronize


__all__ = ["can_access_peer", "enable_peer_access", "disable_peer_access", "graphs", "CUDAGraph", "enable_fused_kernels", "fused_kernels_enabled", "Stream", "Event", "current_stream", "default_stream", "stream", "elapsed_time", "is_available", "device_count", "current_device", "set_device", "get_device_name", "get_device_properties", "manual_seed", "manual_seed_all", "memory", "memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved", "empty_cache", "reset_peak_memory_stats", "mem_get_info", "synchronize", "NamiTorchCUDAOutOfMemoryError"]
