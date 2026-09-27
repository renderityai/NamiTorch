import math
from numbers import Real

import numpy as np

from ..backends import ensure_same_device, namespace, same_device, writable

from ..autograd import no_grad
from ..tensor import Tensor


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real scalar.")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite.") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _norm_type(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("norm_type must be a positive real scalar or positive infinity.")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError("norm_type is outside the supported range.") from None
    if math.isnan(value) or value <= 0:
        raise ValueError("norm_type must be positive or positive infinity.")
    return value


def _gradients(parameters) -> list[Tensor]:
    if isinstance(parameters, Tensor):
        parameters = (parameters,)
    if isinstance(parameters, (dict, str, bytes, set, frozenset)):
        raise TypeError("parameters must be a Tensor or an ordered iterable of Tensors.")
    try:
        parameters = iter(parameters)
    except TypeError:
        raise TypeError("parameters must be a Tensor or an ordered iterable of Tensors.") from None
    gradients = []
    seen = set()
    for parameter in parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only NamiTorch Tensors.")
        gradient = parameter.grad
        if gradient is None or id(gradient) in seen:
            continue
        if not isinstance(gradient, Tensor) or not gradient.dtype.is_floating_point:
            raise TypeError("Gradients must be floating NamiTorch Tensors.")
        if gradient.shape != parameter.shape:
            raise ValueError(f"Gradient shape {gradient.shape} does not match parameter shape {parameter.shape}.")
        ensure_same_device(parameter, gradient)
        seen.add(id(gradient))
        gradients.append(gradient)
    return gradients


@same_device
def _total_norm(gradients: list[Tensor], norm_type: float):
    xp = namespace(gradients[0] if gradients else None)
    scale = xp.asarray(0.0, dtype=np.float64)
    for gradient in gradients:
        scale = xp.maximum(scale, xp.max(xp.abs(gradient._data), initial=0))
    if not bool(xp.isfinite(scale)) or bool(scale == 0) or math.isinf(norm_type):
        return xp.asarray(scale, dtype=np.float64)
    total = xp.asarray(0.0, dtype=np.float64)
    with xp.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        for gradient in gradients:
            scaled = xp.abs(xp.asarray(gradient._data, dtype=np.float64)) / scale
            total = total + xp.sum(xp.power(scaled, norm_type), dtype=np.float64)
        result = scale * xp.power(total, 1 / norm_type)
        if not bool(xp.isfinite(result)):
            result = xp.exp(xp.log(scale) + xp.log(total) / norm_type)
    return xp.asarray(result, dtype=np.float64)


def _norm_tensor(norm) -> Tensor:
    return Tensor._from_array(norm, False)


def get_grad_norm(parameters, norm_type: float = 2) -> Tensor:
    order = _norm_type(norm_type)
    return _norm_tensor(_total_norm(_gradients(parameters), order))


def _require_writable(gradients: list[Tensor]) -> None:
    if any(not gradient._writable or not writable(gradient._data) for gradient in gradients):
        raise RuntimeError("Cannot clip read-only gradient storage.")


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2, error_if_nonfinite: bool = False) -> Tensor:
    maximum = _nonnegative(max_norm, "max_norm")
    order = _norm_type(norm_type)
    if type(error_if_nonfinite) is not bool:
        raise TypeError("error_if_nonfinite must be a Python bool.")
    gradients = _gradients(parameters)
    norm = _total_norm(gradients, order)
    xp = namespace(norm)
    if error_if_nonfinite and not bool(xp.isfinite(norm)):
        raise RuntimeError(f"Cannot clip gradients with nonfinite total norm {norm}.")
    coefficient = xp.divide(maximum, xp.add(norm, 1e-6))
    if coefficient < 1:
        _require_writable(gradients)
        with no_grad(), np.errstate(invalid="ignore", under="ignore"):
            for gradient in gradients:
                gradient.copy_(xp.multiply(xp.asarray(gradient._data, dtype=np.float64), coefficient))
    return _norm_tensor(norm)


def clip_grad_value_(parameters, clip_value: float) -> None:
    bound = _nonnegative(clip_value, "clip_value")
    gradients = _gradients(parameters)
    _require_writable(gradients)
    with no_grad():
        for gradient in gradients:
            gradient.copy_(namespace(gradient).clip(namespace(gradient).asarray(gradient._data, dtype=np.float64), -bound, bound))


__all__ = ["get_grad_norm", "clip_grad_norm_", "clip_grad_value_"]
