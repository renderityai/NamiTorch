import math
from numbers import Real

import numpy as np

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
        seen.add(id(gradient))
        gradients.append(gradient)
    return gradients


def _total_norm(gradients: list[Tensor], norm_type: float) -> float:
    scale = 0.0
    for gradient in gradients:
        data = gradient._data
        if np.any(np.isnan(data)):
            return math.nan
        scale = max(scale, float(np.max(np.abs(data), initial=0)))
    if scale == 0 or math.isinf(scale) or math.isinf(norm_type):
        return scale
    totals = []
    with np.errstate(under="ignore"):
        for gradient in gradients:
            scaled = np.abs(gradient._data.astype(np.float64, copy=False)) / scale
            totals.append(float(np.sum(np.power(scaled, norm_type), dtype=np.float64)))
    total = math.fsum(totals)
    try:
        result = scale * math.pow(total, 1 / norm_type)
        if math.isfinite(result):
            return result
    except OverflowError:
        result = math.inf
    logarithm = math.log(scale) + math.log(total) / norm_type
    try:
        return math.exp(logarithm)
    except OverflowError:
        return math.inf


def _norm_tensor(norm: float) -> Tensor:
    return Tensor._from_array(np.asarray(norm, dtype=np.float64), False)


def get_grad_norm(parameters, norm_type: float = 2) -> Tensor:
    order = _norm_type(norm_type)
    return _norm_tensor(_total_norm(_gradients(parameters), order))


def _require_writable(gradients: list[Tensor]) -> None:
    if any(not gradient._data.flags.writeable for gradient in gradients):
        raise RuntimeError("Cannot clip read-only gradient storage.")


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2, error_if_nonfinite: bool = False) -> Tensor:
    maximum = _nonnegative(max_norm, "max_norm")
    order = _norm_type(norm_type)
    if type(error_if_nonfinite) is not bool:
        raise TypeError("error_if_nonfinite must be a Python bool.")
    gradients = _gradients(parameters)
    norm = _total_norm(gradients, order)
    if error_if_nonfinite and not math.isfinite(norm):
        raise RuntimeError(f"Cannot clip gradients with nonfinite total norm {norm}.")
    coefficient = maximum / (norm + 1e-6)
    if coefficient < 1:
        _require_writable(gradients)
        with no_grad(), np.errstate(invalid="ignore", under="ignore"):
            for gradient in gradients:
                gradient.copy_(gradient._data.astype(np.float64, copy=False) * coefficient)
    return _norm_tensor(norm)


def clip_grad_value_(parameters, clip_value: float) -> None:
    bound = _nonnegative(clip_value, "clip_value")
    gradients = _gradients(parameters)
    _require_writable(gradients)
    with no_grad():
        for gradient in gradients:
            gradient.copy_(np.clip(gradient._data.astype(np.float64, copy=False), -bound, bound))


__all__ = ["get_grad_norm", "clip_grad_norm_", "clip_grad_value_"]
