import sys
from functools import wraps
from threading import RLock

import numpy as np

from ..device import Device
from .cpu import CPUBackend
from ._settings import fused_kernels_enabled


_backends = {Device("cpu"): CPUBackend()}
_lock = RLock()


def is_array(value):
    if isinstance(value, np.ndarray):
        return True
    module = sys.modules.get("cupy")
    return module is not None and isinstance(value, getattr(module, "ndarray", ()))


def array_device(value):
    if isinstance(value, np.ndarray):
        return Device("cpu")
    module = sys.modules.get("cupy")
    if module is not None and isinstance(value, getattr(module, "ndarray", ())):
        return Device("cuda", int(value.device.id))
    raise TypeError(f"Tensor storage must be a registered backend array, got {type(value).__name__}.")


def get_backend(value=None):
    if value is None or isinstance(value, (np.generic, bool, int, float)):
        target = Device("cpu")
    elif isinstance(value, (str, Device)):
        target = Device(value)
    elif is_array(value):
        target = array_device(value)
    else:
        backend = getattr(value, "_backend", None)
        if backend is None or not is_array(getattr(value, "_data", None)) or array_device(value._data) != backend.device:
            raise TypeError("Expected a Tensor, backend array or Device.")
        target = backend.device
    with _lock:
        if target not in _backends:
            from .cuda import CUDABackend
            _backends[target] = CUDABackend(target)
        return _backends[target]


def get_array_module(value=None):
    return get_backend(value).module


def try_fused(operation, *arrays):
    if not fused_kernels_enabled():
        return None
    backend = get_backend(arrays[0])
    fused = getattr(backend, "fused", None)
    return None if fused is None else fused.run(operation, arrays)


def try_fused_softmax(operation, *arrays, **parameters):
    if not fused_kernels_enabled():
        return None
    backend = get_backend(arrays[0])
    kernels = getattr(backend, "softmax", None)
    return None if kernels is None else kernels.run(operation, arrays, **parameters)


def try_fused_normalization(operation, *arrays, **parameters):
    if not fused_kernels_enabled():
        return None
    backend = get_backend(arrays[0])
    kernels = getattr(backend, "normalization", None)
    return None if kernels is None else kernels.run(operation, arrays, **parameters)


def namespace(value=None):
    return get_backend(value).namespace


def ensure_same_device(*values, _active=None):
    active = set() if _active is None else _active
    devices = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            if id(value) in active:
                raise ValueError("Operand sequences cannot contain cycles.")
            active.add(id(value))
            nested = ensure_same_device(*value, _active=active)
            active.remove(id(value))
            if nested is not None:
                devices.append(nested)
        elif is_array(value) or hasattr(value, "_backend"):
            devices.append(get_backend(value).device)
    if devices and any(value != devices[0] for value in devices[1:]):
        raise RuntimeError(f"Operands are on different devices: {', '.join(map(str, devices))}; use an explicit .to() transfer.")
    return devices[0] if devices else None


def same_device(function):
    @wraps(function)
    def checked(*args, **kwargs):
        target = ensure_same_device(*args, *kwargs.values())
        backend = get_backend(target)
        with backend.context(args, kwargs):
            return backend.result(function(*args, **kwargs))
    return checked


def transfer(array, target, dtype=None, copy=True, *, non_blocking=False):
    if type(copy) is not bool or type(non_blocking) is not bool:
        raise TypeError("Transfer copy and non_blocking arguments must be Python bools.")
    source = get_backend(array)
    destination = get_backend(target)
    if source.device.type != destination.device.type and destination.device.type == "cpu":
        array = source.to_numpy(array)
    return destination.array(array, dtype=dtype, copy=copy, non_blocking=non_blocking)


def writable(array):
    return get_backend(array).writable(array)


def readonly(array):
    return get_backend(array).readonly(array)


__all__ = ["get_backend", "get_array_module"]
