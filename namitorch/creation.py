import math

import numpy as np

from .dtype import DType, get_default_dtype, normalize_dtype, result_type
from .tensor import Tensor, _validate_requires_grad
from .utils import normalize_shape


def _factory_dtype(dtype: object, requires_grad: bool, default: DType | None = None) -> DType:
    resolved = normalize_dtype(dtype) if dtype is not None else default or get_default_dtype()
    _validate_requires_grad(resolved, requires_grad)
    return resolved


def _scalar(value: object) -> object:
    if not isinstance(value, (bool, int, float, np.generic)):
        raise TypeError("Expected a Python or NumPy numeric scalar.")
    result_type(value)
    return value.item() if isinstance(value, np.generic) else value


def _finite_real(value: object) -> int | float:
    scalar = _scalar(value)
    if isinstance(scalar, bool):
        raise TypeError("Expected a real numeric scalar, not a boolean.")
    if not math.isfinite(scalar):
        raise ValueError("Range arguments must be finite.")
    return scalar


def zeros(*shape: object, dtype: object = None, requires_grad: bool = False) -> Tensor:
    dimensions = normalize_shape(*shape)
    target = _factory_dtype(dtype, requires_grad)
    return Tensor._from_array(np.zeros(dimensions, dtype=target.numpy_dtype), requires_grad)


def ones(*shape: object, dtype: object = None, requires_grad: bool = False) -> Tensor:
    dimensions = normalize_shape(*shape)
    target = _factory_dtype(dtype, requires_grad)
    return Tensor._from_array(np.ones(dimensions, dtype=target.numpy_dtype), requires_grad)


def empty(*shape: object, dtype: object = None, requires_grad: bool = False) -> Tensor:
    dimensions = normalize_shape(*shape)
    target = _factory_dtype(dtype, requires_grad)
    return Tensor._from_array(np.empty(dimensions, dtype=target.numpy_dtype), requires_grad)


def full(shape: object, fill_value: object, *, dtype: object = None, requires_grad: bool = False) -> Tensor:
    dimensions = normalize_shape(shape)
    scalar = _scalar(fill_value)
    target = _factory_dtype(dtype, requires_grad, result_type(fill_value))
    array = np.full(dimensions, scalar, dtype=target.numpy_dtype)
    return Tensor._from_array(array, requires_grad)


def _like_dtype(input: Tensor, dtype: object, requires_grad: bool) -> DType:
    if not isinstance(input, Tensor):
        raise TypeError("A like factory requires a NamiTorch Tensor as input.")
    return _factory_dtype(dtype, requires_grad, input.dtype)


def zeros_like(input: Tensor, *, dtype: object = None, requires_grad: bool = False) -> Tensor:
    target = _like_dtype(input, dtype, requires_grad)
    return zeros(input.shape, dtype=target, requires_grad=requires_grad)


def ones_like(input: Tensor, *, dtype: object = None, requires_grad: bool = False) -> Tensor:
    target = _like_dtype(input, dtype, requires_grad)
    return ones(input.shape, dtype=target, requires_grad=requires_grad)


def empty_like(input: Tensor, *, dtype: object = None, requires_grad: bool = False) -> Tensor:
    target = _like_dtype(input, dtype, requires_grad)
    return empty(input.shape, dtype=target, requires_grad=requires_grad)


def full_like(
    input: Tensor, fill_value: object, *, dtype: object = None, requires_grad: bool = False
) -> Tensor:
    target = _like_dtype(input, dtype, requires_grad)
    return full(input.shape, fill_value, dtype=target, requires_grad=requires_grad)


def arange(
    start: object, stop: object = None, step: object = None,
    *, dtype: object = None, requires_grad: bool = False,
) -> Tensor:
    arguments = [start]
    if stop is not None:
        arguments.append(stop)
    if step is not None:
        arguments.append(step)
    first = 0 if stop is None else _finite_real(start)
    last = _finite_real(start if stop is None else stop)
    stride = 1 if step is None else _finite_real(step)
    if stride == 0:
        raise ValueError("arange step must not be zero.")
    target = _factory_dtype(dtype, requires_grad, result_type(*arguments))
    if target.is_boolean:
        raise TypeError("arange requires a floating or integer dtype.")
    if target.is_integer and any(isinstance(value, float) for value in (first, last, stride)):
        array = np.arange(first, last, stride, dtype=np.float64).astype(target.numpy_dtype)
    else:
        array = np.arange(first, last, stride, dtype=target.numpy_dtype)
    return Tensor._from_array(array, requires_grad)


def linspace(
    start: object, end: object, steps: object,
    *, dtype: object = None, requires_grad: bool = False,
) -> Tensor:
    first, last = _finite_real(start), _finite_real(end)
    count = normalize_shape((steps,))[0]
    target = _factory_dtype(dtype, requires_grad)
    array = np.linspace(first, last, num=count, dtype=target.numpy_dtype)
    return Tensor._from_array(array, requires_grad)


def logspace(
    start: object, end: object, steps: object, base: object = 10.0,
    *, dtype: object = None, requires_grad: bool = False,
) -> Tensor:
    first, last, radix = _finite_real(start), _finite_real(end), _finite_real(base)
    if radix <= 0:
        raise ValueError("logspace base must be positive.")
    count = normalize_shape((steps,))[0]
    target = _factory_dtype(dtype, requires_grad)
    exponents = np.linspace(first, last, num=count, dtype=np.float64)
    array = np.power(radix, exponents).astype(target.numpy_dtype)
    return Tensor._from_array(array, requires_grad)


def eye(n: object, m: object = None, *, dtype: object = None, requires_grad: bool = False) -> Tensor:
    rows, columns = normalize_shape(n, n if m is None else m)
    target = _factory_dtype(dtype, requires_grad)
    array = np.eye(rows, columns, dtype=target.numpy_dtype)
    return Tensor._from_array(array, requires_grad)


def identity(n: object, *, dtype: object = None, requires_grad: bool = False) -> Tensor:
    return eye(n, dtype=dtype, requires_grad=requires_grad)


__all__ = [
    "zeros", "ones", "empty", "full", "zeros_like", "ones_like", "empty_like",
    "full_like", "arange", "linspace", "logspace", "eye", "identity",
]
