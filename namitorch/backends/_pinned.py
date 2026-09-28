import math
import sys

import numpy as np

from ._cuda_runtime import load_cupy, runtime_call


def is_pinned(array):
    if not isinstance(array, np.ndarray):
        return False
    module = sys.modules.get("cupy")
    if module is None:
        return False
    owner = array
    while owner is not None:
        if isinstance(owner, module.cuda.PinnedMemoryPointer):
            return True
        owner = owner.obj if isinstance(owner, memoryview) else getattr(owner, "base", None)
    return False


def pinned_empty(shape, dtype):
    module = load_cupy()
    if runtime_call("getDeviceCount") < 1:
        raise RuntimeError("Pinned CPU memory requires an available CUDA device and runtime.")
    dtype = np.dtype(dtype)
    count = math.prod(shape)
    try:
        owner = module.cuda.alloc_pinned_memory(max(1, count * dtype.itemsize))
    except Exception as error:
        raise RuntimeError(f"Cannot allocate {count * dtype.itemsize} bytes of pinned CPU memory: {error}") from error
    return np.frombuffer(owner, dtype=dtype, count=count).reshape(shape)
