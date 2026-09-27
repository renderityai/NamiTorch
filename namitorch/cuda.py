from .backends._cuda_runtime import runtime_call
from .device import Device


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


__all__ = ["is_available", "device_count", "current_device", "set_device", "get_device_name", "get_device_properties"]
