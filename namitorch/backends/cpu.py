import math
from copy import deepcopy
from contextlib import nullcontext

import numpy as np

from ._graph_capture import reject_during_capture
from ..device import Device


class CPUBackend:
    __slots__ = ()
    device = Device("cpu")
    module = np
    namespace = np
    array_type = np.ndarray

    def context(self, *operands):
        reject_during_capture("CPU Tensor operations")
        return nullcontext()

    def result(self, value):
        return value

    def ready_for_host(self, array):
        return array

    def array(self, value, dtype=None, copy=True, *, non_blocking=False):
        if copy:
            return np.array(value, dtype=dtype, copy=True, order="C", subok=False)
        return np.asarray(value, dtype=dtype, order="C")

    def item(self, array):
        return array.item()

    def to_numpy(self, array):
        return array.copy(order="C")

    def writable(self, array):
        return array.flags.writeable

    def readonly(self, array):
        array.flags.writeable = False
        return array

    def erfc(self, array):
        return np.fromiter((math.erfc(float(value)) for value in array.flat), dtype=array.dtype, count=array.size).reshape(array.shape)

    def create_rng(self, seed):
        return np.random.Generator(np.random.PCG64(seed))

    def seed_rng(self, generator, seed):
        generator.bit_generator.state = np.random.PCG64(seed).state

    def rng_state(self, generator):
        state = deepcopy(generator.bit_generator.state)
        state["version"] = 1
        return state

    def restore_rng(self, generator, state):
        if not isinstance(state, dict):
            raise TypeError("Generator state must be a dictionary.")
        expected_keys = {"version", "bit_generator", "state", "has_uint32", "uinteger"}
        if set(state) != expected_keys or state.get("bit_generator") != "PCG64":
            raise ValueError("Expected a versioned PCG64 generator state.")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("Unsupported generator state version.")
        inner = state["state"]
        if not isinstance(inner, dict) or set(inner) != {"state", "inc"}:
            raise ValueError("Invalid PCG64 state and increment fields.")
        for name, value, maximum in (
            ("state", inner["state"], 2 ** 128), ("inc", inner["inc"], 2 ** 128),
            ("has_uint32", state["has_uint32"], 2), ("uinteger", state["uinteger"], 2 ** 32),
        ):
            if type(value) is not int or not 0 <= value < maximum:
                raise ValueError(f"Invalid generator state field {name!r}.")
        if inner["inc"] % 2 != 1:
            raise ValueError("The PCG64 increment must be odd.")
        restored = deepcopy(state)
        del restored["version"]
        generator.bit_generator.state = restored


    def random(self, generator, operation, shape, dtype, **options):
        if dtype is not None and np.dtype(dtype) == np.dtype("float16") and operation in ("random", "standard_normal"):
            result = getattr(generator, operation)(shape, dtype=np.float32).astype(np.float16)
            if operation == "random":
                np.minimum(result, np.nextafter(np.float16(1), np.float16(0)), out=result)
            return result
        if operation == "permutation":
            return generator.permutation(shape).astype(dtype, copy=False)
        if operation == "choice":
            return np.asarray(generator.choice(options["n"], size=shape, replace=options["replace"]), dtype=dtype)
        if operation == "integers":
            return generator.integers(size=shape, dtype=dtype, **options)
        return getattr(generator, operation)(shape) if dtype is None else getattr(generator, operation)(shape, dtype=dtype)

    def validate_index(self, array, index):
        return None
