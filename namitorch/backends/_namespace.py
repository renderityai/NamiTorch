from functools import wraps

import numpy as np


class ArrayNamespace:
    def __init__(self, backend):
        self.backend = backend

    def _validate(self, value):
        if isinstance(value, (tuple, list)):
            for item in value:
                self._validate(item)
        elif isinstance(value, np.ndarray):
            raise RuntimeError("CUDA primitives cannot consume CPU arrays; transfer explicitly.")
        elif isinstance(value, self.backend.array_type) and value.device.id != self.backend.device.index:
            raise RuntimeError(f"CUDA primitive expected {self.backend.device}, received cuda:{value.device.id}; transfer explicitly.")

    def __getattr__(self, name):
        if name == "errstate":
            return np.errstate
        try:
            attribute = getattr(self.backend.module, name)
        except AttributeError:
            raise RuntimeError(f"CUDA primitive {name!r} is not available in the installed CuPy; no CPU fallback is used.") from None
        if not callable(attribute) or isinstance(attribute, type):
            return attribute

        @wraps(attribute)
        def call(*args, **kwargs):
            self._validate(args)
            self._validate(tuple(kwargs.values()))
            with self.backend.context():
                if name == "array":
                    kwargs.pop("subok", None)
                if name != "copyto" and "where" in kwargs:
                    mask = kwargs.pop("where")
                    output = kwargs.pop("out", None)
                    computed = attribute(*args, **kwargs)
                    if output is None:
                        output = self.backend.module.empty_like(computed)
                    self.backend.module.copyto(output, computed, where=mask)
                    return output
                if name in ("max", "amax", "min", "amin") and "initial" in kwargs:
                    initial = kwargs.pop("initial")
                    value = args[0]
                    axis = kwargs.get("axis")
                    axes = tuple(range(value.ndim)) if axis is None else (axis,) if isinstance(axis, int) else axis
                    if any(value.shape[dim] == 0 for dim in axes):
                        axes = {dim % value.ndim for dim in axes}
                        shape = tuple(1 if dim in axes else size for dim, size in enumerate(value.shape)) if kwargs.get("keepdims", False) else tuple(size for dim, size in enumerate(value.shape) if dim not in axes)
                        return self.backend.module.full(shape, initial, dtype=value.dtype)
                    result = attribute(*args, **kwargs)
                    combine = self.backend.module.maximum if name in ("max", "amax") else self.backend.module.minimum
                    return combine(result, initial)
                return attribute(*args, **kwargs)

        if hasattr(attribute, "at"):
            def at(*args, **kwargs):
                self._validate(args)
                self._validate(tuple(kwargs.values()))
                with self.backend.context():
                    if name == "add":
                        return self.backend.add_at(*args, **kwargs)
                    return attribute.at(*args, **kwargs)
            call.at = at
        return call
