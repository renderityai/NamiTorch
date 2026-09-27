import math
from numbers import Real
from typing import Any as Array

import numpy as np

from ..backends import ensure_same_device, is_array, namespace, writable

from ..autograd import enable_grad, no_grad
from ..dtype import from_numpy_dtype
from ..tensor import Tensor
from .optimizer import Optimizer


def _real_option(options: dict, name: str, positive: bool = False, unit_interval: bool = False) -> float:
    if name not in options:
        raise ValueError(f"Missing optimizer option {name!r}.")
    value = options[name]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"Optimizer {name} must be a real scalar.")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"Optimizer {name} must be finite.") from None
    if not math.isfinite(value) or value < 0 or (positive and value == 0) or (unit_interval and value >= 1):
        constraint = "in [0, 1)" if unit_interval else "positive" if positive else "nonnegative"
        raise ValueError(f"Optimizer {name} must be finite and {constraint}.")
    options[name] = value
    return value


def _bool_option(options: dict, name: str) -> bool:
    if name not in options or type(options[name]) is not bool:
        raise TypeError(f"Optimizer {name} must be a Python bool.")
    return options[name]


def _scalar(parameter: Tensor, value: float, name: str = "coefficient"):
    dtype = parameter.dtype.numpy_dtype
    if not math.isfinite(value) or abs(value) > float(np.finfo(dtype).max):
        raise ValueError(f"Optimizer {name} must be representable in {parameter.dtype.name}.")
    with np.errstate(under="ignore"):
        result = dtype.type(value)
    if name == "eps" and result == 0:
        raise ValueError(f"Optimizer eps must remain positive in {parameter.dtype.name}.")
    return result


def _buffer(state: dict, name: str, parameter: Tensor) -> Tensor:
    xp = namespace(parameter)
    if name not in state:
        state[name] = Tensor._from_array(xp.zeros_like(parameter._data), False)
    return state[name]


def _gradient(parameter: Tensor, gradient: Tensor, options: dict, decoupled: bool = False) -> Array:
    xp = namespace(parameter)
    result = gradient._data.copy()
    if options["maximize"]:
        xp.negative(result, out=result)
    if options["weight_decay"] != 0 and not decoupled:
        result += _scalar(parameter, options["weight_decay"]) * parameter._data
    return result


def _apply_update(parameter: Tensor, update: Array, options: dict, decoupled: bool = False) -> None:
    if options["lr"] == 0:
        return
    values = parameter._data
    if decoupled and options["weight_decay"] != 0:
        values = values * _scalar(parameter, 1 - options["lr"] * options["weight_decay"], "decay factor")
    parameter.copy_(values - _scalar(parameter, options["lr"]) * update)


class _FirstOrderOptimizer(Optimizer):
    _state_buffers = ()
    _required_buffers = ()
    _has_step = False
    _nonnegative_buffers = ()
    _decoupled_weight_decay = False

    def _prepare_options(self, options: dict, path: str) -> dict:
        prepared = super()._prepare_options(options, path)
        _real_option(prepared, "lr")
        _real_option(prepared, "weight_decay")
        _bool_option(prepared, "maximize")
        return prepared

    def _check_state(self, values: dict, parameter: Tensor, path: str, runtime: bool = False) -> None:
        if not isinstance(values, dict):
            raise TypeError(f"{path} must be a dictionary.")
        if not values:
            return
        allowed = set(self._state_buffers) | ({"step"} if self._has_step else set())
        required = set(self._required_buffers) | ({"step"} if self._has_step else set())
        if not set(values) <= allowed or not required <= values.keys():
            raise ValueError(f"{path} has invalid state fields for {type(self).__name__}.")
        if self._has_step and (type(values["step"]) is not int or values["step"] < 0):
            raise ValueError(f"{path} step must be a nonnegative Python integer.")
        for name in self._state_buffers:
            if name not in values:
                continue
            value = values[name]
            if not (isinstance(value, Tensor) or is_array(value)):
                raise TypeError(f"{path}[{name!r}] must be a floating Tensor or NumPy array.")
            dtype = value.dtype if isinstance(value, Tensor) else from_numpy_dtype(value.dtype)
            if not dtype.is_floating_point:
                raise TypeError(f"{path}[{name!r}] must have a floating dtype.")
            if value.shape != parameter.shape:
                raise ValueError(f"{path}[{name!r}] shape {value.shape} must equal parameter shape {parameter.shape}.")
            array = value._data if isinstance(value, Tensor) else value
            xp = namespace(array)
            if name in self._nonnegative_buffers and xp.any(xp.less(array, 0)):
                raise ValueError(f"{path}[{name!r}] must be nonnegative.")
            if runtime:
                ensure_same_device(parameter, value)
                if not isinstance(value, Tensor) or dtype is not parameter.dtype:
                    raise TypeError(f"{path}[{name!r}] must be a Tensor with parameter dtype {parameter.dtype.name}.")
                if value.requires_grad or value.grad_fn is not None or not value._writable or not writable(array):
                    raise RuntimeError(f"{path}[{name!r}] must be writable and detached from autograd.")

    def _prepare_state(self, values: dict, parameter: Tensor, path: str) -> dict:
        copied = super()._prepare_state(values, parameter, path)
        self._check_state(copied, parameter, path)
        if copied and not parameter.dtype.is_floating_point:
            raise TypeError("Nonfloating parameters cannot have optimizer accumulation buffers.")
        for name in self._state_buffers:
            if name in copied:
                copied[name] = Tensor(copied[name], dtype=parameter.dtype, device=parameter.device)
        return copied

    def step(self, closure=None):
        if closure is not None and not callable(closure):
            raise TypeError("Optimizer closure must be callable or None.")
        loss = None
        if closure is not None:
            with enable_grad():
                loss = closure()
        entries = list(self._parameters_with_grad())
        options = {
            id(group): self._prepare_options({key: value for key, value in group.items() if key != "params"}, "param_group")
            for group in self.param_groups
        }
        for group, parameter, _ in entries:
            if not parameter.dtype.is_floating_point:
                raise TypeError("Optimizer updates require floating parameters.")
            if not parameter._writable or not writable(parameter._data):
                raise RuntimeError("Optimizer cannot update read-only parameter storage.")
            prepared = options[id(group)]
            for name, value in prepared.items():
                if type(value) is float:
                    _scalar(parameter, value, name)
            if self._decoupled_weight_decay:
                _scalar(parameter, 1 - prepared["lr"] * prepared["weight_decay"], "decay factor")
            self._check_state(self.state.get(id(parameter), {}), parameter, "parameter state", runtime=True)
        update = getattr(self, "_update_parameter", None)
        if not callable(update):
            raise RuntimeError("Optimizer subclass must implement parameter updates.")
        with no_grad():
            for group, parameter, gradient in entries:
                update(parameter, gradient, options[id(group)])
        return loss
