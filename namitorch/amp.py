from .backends._graph_capture import reject_during_capture
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from types import MappingProxyType
import math
import operator
from numbers import Real

import numpy as np

from .backends import get_backend
from .backends._cuda_runtime import runtime_call
from .device import Device
from .dtype import float16, float32, normalize_dtype


OPERATION_POLICIES = MappingProxyType({
    "matmul": "float16",
    "linear": "float16",
    **dict.fromkeys(("sum", "mean", "prod", "var", "std", "logsumexp", "softmax", "log_softmax", "layer_norm", "rms_norm", "batch_norm", "cross_entropy", "nll_loss", "mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss", "binary_cross_entropy", "binary_cross_entropy_with_logits", "exp", "expm1", "log", "log1p", "log2", "log10", "sqrt", "rsqrt", "gelu", "sigmoid", "silu", "softplus", "softsign"), "float32"),
})
_autocast_state = ContextVar("namitorch_autocast", default=(False, float16))


@contextmanager
def autocast(device_type="cuda", dtype=float16, enabled=True):
    if device_type not in ("cpu", "cuda"):
        raise ValueError("autocast device_type must be 'cuda' or 'cpu'.")
    if type(enabled) is not bool:
        raise TypeError("autocast enabled must be a Python bool.")
    selected = normalize_dtype(dtype)
    if selected is not float16:
        raise ValueError("autocast currently supports only float16 compute dtype.")
    if device_type == "cpu":
        if enabled:
            raise RuntimeError("CPU autocast is not supported; use explicit Tensor.to(dtype) conversions.")
        yield
        return
    if enabled:
        backend = get_backend(Device("cuda", int(runtime_call("getDevice"))))
        if not backend.supports_float16():
            raise RuntimeError(f"CUDA float16 autocast is unavailable on {backend.device}; compatible CuPy and compute capability >= 5.3 are required.")
    token = _autocast_state.set((enabled, selected))
    try:
        yield
    finally:
        _autocast_state.reset(token)


def autocast_dtype(operation, *operands):
    enabled, dtype = _autocast_state.get()
    values = tuple(value for value in operands if value is not None)
    if not enabled or OPERATION_POLICIES.get(operation) != "float16" or not values:
        return None
    if any(value.device.type != "cuda" or value.dtype not in (float16, float32) for value in values):
        return None
    backend = get_backend(values[0])
    if not backend.supports_float16():
        raise RuntimeError(f"CUDA float16 autocast is unavailable on {backend.device}.")
    return dtype


def float32_function(function):
    if OPERATION_POLICIES.get(function.__name__) != "float32":
        raise ValueError(f"No float32 policy for {function.__name__}.")

    def promote(value):
        return value.to(float32) if getattr(value, "dtype", None) is float16 else value

    @wraps(function)
    def checked(*args, **kwargs):
        return function(*(promote(value) for value in args), **{key: promote(value) for key, value in kwargs.items()})

    return checked


class GradScaler:
    _MIN_SCALE = float(np.finfo(np.float32).tiny)
    _MAX_SCALE = float(np.finfo(np.float32).max)

    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        if type(enabled) is not bool:
            raise TypeError("GradScaler enabled must be a Python bool.")
        self._scale = self._scale_value(init_scale)
        self._growth_factor = self._real(growth_factor, "growth_factor")
        self._backoff_factor = self._real(backoff_factor, "backoff_factor")
        if self._growth_factor <= 1:
            raise ValueError("growth_factor must be greater than 1.")
        if not 0 < self._backoff_factor < 1:
            raise ValueError("backoff_factor must be between 0 and 1.")
        if isinstance(growth_interval, (bool, np.bool_)):
            raise TypeError("growth_interval must be a positive integer.")
        self._growth_interval = operator.index(growth_interval)
        if self._growth_interval <= 0:
            raise ValueError("growth_interval must be positive.")
        self._enabled = enabled
        self._growth_tracker = 0
        self._optimizer_states = {}
        self._scale_used = False

    @staticmethod
    def _real(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite real number.")
        try:
            result = float(value)
        except OverflowError:
            raise ValueError(f"{name} must be finite.") from None
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite.")
        return result

    @classmethod
    def _scale_value(cls, value):
        scale = cls._real(value, "scale")
        if not cls._MIN_SCALE <= scale <= cls._MAX_SCALE:
            raise ValueError(f"scale must be between {cls._MIN_SCALE} and {cls._MAX_SCALE}.")
        return float(np.float32(scale))

    def is_enabled(self):
        return self._enabled

    def get_scale(self):
        return self._scale if self._enabled else 1.0

    def scale(self, loss):
        reject_during_capture('GradScaler')
        dtype = getattr(loss, "dtype", None)
        if not hasattr(dtype, "can_require_grad") or not dtype.can_require_grad or not hasattr(loss, "_data"):
            raise TypeError("GradScaler.scale requires a floating NamiTorch Tensor.")
        get_backend(loss)
        if not self._enabled:
            return loss
        if self._optimizer_states:
            raise RuntimeError("Call update() before scaling another loss after unscale_ or step.")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = loss * self._scale
        self._scale_used = True
        return result

    @staticmethod
    def _gradients(optimizer):
        if not callable(getattr(optimizer, "_parameters_with_grad", None)) or not callable(getattr(optimizer, "step", None)):
            raise TypeError("GradScaler requires a NamiTorch optimizer.")
        gradients = tuple((parameter, gradient) for _, parameter, gradient in optimizer._parameters_with_grad())
        for parameter, gradient in gradients:
            if not parameter.dtype.can_require_grad or gradient.requires_grad or gradient.grad_fn is not None:
                raise RuntimeError("GradScaler requires detached floating gradients.")
        return gradients

    @staticmethod
    def _finite(gradients):
        devices = {}
        for _, gradient in gradients:
            devices.setdefault(gradient.device, []).append(gradient)
        finite = True
        for device, values in devices.items():
            backend = get_backend(device)
            xp = backend.namespace
            with backend.context(values):
                valid = xp.asarray(True)
                for gradient in values:
                    valid = xp.logical_and(valid, xp.all(xp.isfinite(gradient._data)))
                valid = backend.result(xp.asarray(valid))
            finite = bool(backend.item(valid)) and finite
        return finite

    def unscale_(self, optimizer):
        reject_during_capture("GradScaler")
        if not self._enabled:
            return True
        if not self._scale_used:
            raise RuntimeError("Call scale(loss) and backward before unscale_.")
        identity = id(optimizer)
        if identity in self._optimizer_states:
            raise RuntimeError("unscale_ may be called only once per optimizer between update() calls.")
        gradients = self._gradients(optimizer)
        if not gradients:
            raise RuntimeError("No gradients were found for this optimizer.")
        previous = {id(parameter) for state in self._optimizer_states.values() for parameter, _ in state["gradients"]}
        if any(id(parameter) in previous for parameter, _ in gradients):
            raise RuntimeError("Optimizers cannot share gradients in the same GradScaler iteration.")
        state = {"optimizer": optimizer, "stage": "unscaling", "gradients": gradients, "finite": False}
        self._optimizer_states[identity] = state
        for _, gradient in gradients:
            backend = get_backend(gradient)
            xp = backend.namespace
            with backend.context(gradient), np.errstate(over="ignore", under="ignore", invalid="ignore"):
                data = gradient._data.astype(np.float32, copy=False) if gradient.dtype is float16 else gradient._data
                gradient.copy_(xp.divide(data, self._scale))
        state["finite"] = self._finite(gradients)
        state["stage"] = "unscaled"
        return state["finite"]

    def step(self, optimizer):
        reject_during_capture("GradScaler")
        if not self._enabled:
            optimizer.step()
            return True
        identity = id(optimizer)
        if identity not in self._optimizer_states:
            self.unscale_(optimizer)
        state = self._optimizer_states[identity]
        if state["stage"] != "unscaled":
            raise RuntimeError("step() may be called only once per optimizer before update().")
        current = self._gradients(optimizer)
        if len(current) != len(state["gradients"]) or any(first[0] is not second[0] or first[1] is not second[1] for first, second in zip(current, state["gradients"])):
            raise RuntimeError("Optimizer gradients changed after unscale_; clipping must modify the existing gradients in-place.")
        state["finite"] = self._finite(current) and state["finite"]
        if state["finite"]:
            state["stage"] = "stepping"
            optimizer.step()
            state["stage"] = "stepped"
            return True
        state["stage"] = "stepped"
        return False

    def update(self):
        reject_during_capture("GradScaler")
        if not self._enabled:
            return
        if not self._optimizer_states or any(state["stage"] != "stepped" for state in self._optimizer_states.values()):
            raise RuntimeError("Call step() for every unscaled optimizer before update().")
        if any(not state["finite"] for state in self._optimizer_states.values()):
            self._scale = self._scale_value(max(self._MIN_SCALE, self._scale * self._backoff_factor))
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker == self._growth_interval:
                self._scale = self._scale_value(min(self._MAX_SCALE, self._scale * self._growth_factor))
                self._growth_tracker = 0
        self._optimizer_states.clear()
        self._scale_used = False

    def _ensure_idle(self):
        if self._optimizer_states or self._scale_used:
            raise RuntimeError("GradScaler state can be saved or loaded only between iterations, after update().")

    def state_dict(self):
        self._ensure_idle()
        return dict(version=1, scale=self._scale, growth_factor=self._growth_factor, backoff_factor=self._backoff_factor, growth_interval=self._growth_interval, enabled=self._enabled, growth_tracker=self._growth_tracker)

    def _prepare_load_state_dict(self, state):
        self._ensure_idle()
        if type(state) is not dict or set(state) != {"version", "scale", "growth_factor", "backoff_factor", "growth_interval", "enabled", "growth_tracker"}:
            raise ValueError("Invalid GradScaler state fields.")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("Unsupported GradScaler state version.")
        prepared = GradScaler(state["scale"], state["growth_factor"], state["backoff_factor"], state["growth_interval"], state["enabled"])
        tracker = state["growth_tracker"]
        if type(tracker) is not int or not 0 <= tracker < prepared._growth_interval:
            raise ValueError("Invalid GradScaler growth_tracker.")
        prepared._growth_tracker = tracker
        return prepared

    def load_state_dict(self, state):
        prepared = self._prepare_load_state_dict(state)
        self.__dict__.update(prepared.__dict__)


__all__ = ["autocast", "GradScaler"]
