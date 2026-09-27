import math
from types import MappingProxyType

import numpy as np

from .backends import get_backend, namespace, readonly, same_device

from .dtype import DType
from .utils import broadcast_shapes, matmul_shape


BINARY_UFUNCS = MappingProxyType({
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "true_divide": np.true_divide,
    "floor_divide": np.floor_divide,
    "remainder": np.remainder,
    "power": np.power,
    "matmul": np.matmul,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "equal": np.equal,
    "not_equal": np.not_equal,
    "less": np.less,
    "less_equal": np.less_equal,
    "greater": np.greater,
    "greater_equal": np.greater_equal,
})

UNARY_UFUNCS = MappingProxyType({
    "positive": np.positive,
    "negative": np.negative,
    "absolute": np.absolute,
    "exp": np.exp,
    "expm1": np.expm1,
    "log": np.log,
    "log1p": np.log1p,
    "log2": np.log2,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "square": np.square,
    "sign": np.sign,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
})


@same_device
def binary_forward(
    operation: str,
    left: np.ndarray,
    right: np.ndarray,
    compute_dtype: DType,
    output_dtype: DType,
) -> np.ndarray:
    xp = namespace(left)
    shape = matmul_shape(left.shape, right.shape) if operation == "matmul" else broadcast_shapes(left.shape, right.shape)
    if compute_dtype.is_boolean and operation in ("subtract", "floor_divide", "remainder", "power"):
        raise TypeError(f"{operation} is not defined for boolean operands.")
    first = left.astype(compute_dtype.numpy_dtype, copy=False)
    second = right.astype(compute_dtype.numpy_dtype, copy=False)
    output = xp.empty(shape, dtype=output_dtype.numpy_dtype)
    try:
        getattr(xp, BINARY_UFUNCS[operation].__name__)(first, second, out=output)
    except ValueError as error:
        raise ValueError(f"{operation} failed for operand shapes {left.shape} and {right.shape}: {error}") from None
    return output


@same_device
def axis_index_coordinates(shape: tuple[int, ...], axis: int, index: np.ndarray) -> tuple[np.ndarray, ...]:
    xp = namespace(index)
    if index.ndim != len(shape):
        raise ValueError(f"Input and index must have the same rank, got shapes {shape} and {index.shape}.")
    if any(size > shape[dimension] for dimension, size in enumerate(index.shape) if dimension != axis):
        raise ValueError(f"Index shape {index.shape} exceeds input shape {shape} outside dim {axis}.")
    if xp.any(index < 0) or xp.any(index >= shape[axis]):
        raise IndexError(f"Indices must be in [0, {shape[axis]}) for dim {axis} of input shape {shape}.")
    coordinates = []
    for dimension, size in enumerate(index.shape):
        if dimension == axis:
            coordinates.append(index)
        else:
            coordinate_shape = tuple(size if position == dimension else 1 for position in range(index.ndim))
            coordinate = xp.arange(size, dtype=np.intp).reshape(coordinate_shape)
            readonly(coordinate)
            coordinates.append(coordinate)
    return tuple(coordinates)


@same_device
def scatter_add_forward(
    value: np.ndarray, coordinates: tuple[np.ndarray, ...], source: np.ndarray,
    index_shape: tuple[int, ...], dtype: DType,
) -> np.ndarray:
    xp = namespace(value)
    try:
        broadcasted = xp.broadcast_to(source.astype(dtype.numpy_dtype, copy=False), index_shape)
    except ValueError:
        raise ValueError(f"scatter_add cannot broadcast src shape {source.shape} to index shape {index_shape}.") from None
    output = xp.array(value, dtype=dtype.numpy_dtype, copy=True, order="C")
    xp.add.at(output, coordinates, broadcasted)
    return output


@same_device
def unary_forward(operation: str, value: np.ndarray, dtype: DType) -> np.ndarray:
    xp = namespace(value)
    if dtype.is_boolean and operation in ("positive", "negative"):
        raise TypeError(f"{operation} is not defined for boolean operands.")
    output = xp.empty(value.shape, dtype=dtype.numpy_dtype)
    if dtype.is_boolean and operation in ("square", "sign"):
        xp.copyto(output, value)
        return output
    getattr(xp, UNARY_UFUNCS[operation].__name__)(value, out=output)
    return output


@same_device
def _sigmoid(value: np.ndarray) -> np.ndarray:
    xp = namespace(value)
    output = xp.empty_like(value)
    positive = value >= 0
    with np.errstate(under="ignore"):
        output[positive] = 1 / (1 + xp.exp(-value[positive]))
        exponent = xp.exp(value[~positive])
        output[~positive] = exponent / (1 + exponent)
    return output


@same_device
def _gelu(value: np.ndarray, approximation: str) -> np.ndarray:
    xp = namespace(value)
    output = xp.empty_like(value)
    if approximation == "exact":
        divisor = math.sqrt(2)
        weights = 0.5 * get_backend(value).erfc(-value / divisor)
        negative_infinity = xp.isneginf(value)
        output[negative_infinity] = 0
        xp.multiply(value, weights, out=output, where=~negative_infinity)
    else:
        xp.maximum(value, 0, out=output)
        output[value < 0] = -0.0
        central = xp.abs(value) <= 20
        selected = value[central]
        inner = math.sqrt(2 / math.pi) * (selected + 0.044715 * selected ** 3)
        output[central] = (0.5 * selected) * (1 + xp.tanh(inner))
    return output


@same_device
def elementwise_forward(
    operation: str, value: np.ndarray, dtype: DType,
    parameter: float | None = None, approximation: str | None = None,
) -> np.ndarray:
    xp = namespace(value)
    data = value.astype(dtype.numpy_dtype, copy=False)
    if operation in UNARY_UFUNCS:
        return unary_forward(operation, data, dtype)
    if operation == "sigmoid":
        return _sigmoid(data)
    if operation == "gelu":
        return _gelu(data, approximation)
    output = xp.empty(data.shape, dtype=dtype.numpy_dtype)
    if operation == "rsqrt":
        xp.sqrt(data, out=output)
        xp.reciprocal(output, out=output)
    elif operation == "relu":
        xp.maximum(data, dtype.numpy_dtype.type(0), out=output)
    elif operation in ("leaky_relu", "elu"):
        xp.copyto(output, data)
        negative = data < 0
        scale = dtype.numpy_dtype.type(parameter)
        if scale == 0:
            output[negative] = 0
        elif operation == "leaky_relu":
            output[negative] = scale * data[negative]
        else:
            with np.errstate(under="ignore"):
                output[negative] = scale * xp.expm1(data[negative])
    elif operation == "silu":
        negative_infinity = xp.isneginf(data)
        output[negative_infinity] = 0
        xp.multiply(data, _sigmoid(data), out=output, where=~negative_infinity)
    elif operation == "softplus":
        with np.errstate(under="ignore"):
            xp.add(xp.maximum(data, 0), xp.log1p(xp.exp(-xp.abs(data))), out=output)
    elif operation == "softsign":
        infinite = xp.isinf(data)
        output[infinite] = xp.sign(data[infinite])
        xp.divide(data, 1 + xp.abs(data), out=output, where=~infinite)
    else:
        raise ValueError(f"Unknown elementwise operation: {operation}.")
    return output


@same_device
def clamp_forward(
    value: np.ndarray, minimum: np.ndarray | None, maximum: np.ndarray | None, dtype: DType
) -> np.ndarray:
    xp = namespace(value)
    output = xp.array(value, dtype=dtype.numpy_dtype, copy=True, order="C")
    if minimum is not None:
        xp.maximum(output, minimum.astype(dtype.numpy_dtype, copy=False), out=output)
    if maximum is not None:
        xp.minimum(output, maximum.astype(dtype.numpy_dtype, copy=False), out=output)
    return output


@same_device
def _gelu_derivative(value: np.ndarray, approximation: str) -> np.ndarray:
    xp = namespace(value)
    if approximation == "exact":
        output = xp.asarray(0.5 * get_backend(value).erfc(-value / math.sqrt(2)))
        central = xp.abs(value) <= 40
        selected = value[central]
        with np.errstate(under="ignore"):
            output[central] += selected * xp.exp(-0.5 * selected * selected) / math.sqrt(2 * math.pi)
        return output
    output = xp.where(xp.isnan(value), np.nan, xp.where(value > 0, 1, 0)).astype(value.dtype)
    central = xp.abs(value) <= 20
    selected = value[central]
    scale = math.sqrt(2 / math.pi)
    activation = xp.tanh(scale * (selected + 0.044715 * selected ** 3))
    inner_derivative = scale * (1 + 3 * 0.044715 * selected * selected)
    output[central] = 0.5 * (1 + activation) + 0.5 * selected * (1 - activation * activation) * inner_derivative
    return output


@same_device
def elementwise_derivative(operation: str, value: np.ndarray, parameters: dict) -> np.ndarray:
    xp = namespace(value)
    if operation == "exp":
        derivative = value
    elif operation == "expm1":
        derivative = xp.exp(value)
    elif operation == "log":
        derivative = 1 / value
    elif operation == "log1p":
        derivative = 1 / (1 + value)
    elif operation == "log2":
        derivative = (1 / value) / math.log(2)
    elif operation == "log10":
        derivative = (1 / value) / math.log(10)
    elif operation == "sqrt":
        derivative = 0.5 / value
    elif operation == "rsqrt":
        derivative = -0.5 * value * value * value
    elif operation == "square":
        derivative = 2 * value
    elif operation == "absolute":
        derivative = xp.sign(value)
    elif operation == "sin":
        derivative = xp.cos(value)
    elif operation == "cos":
        derivative = -xp.sin(value)
    elif operation == "tan":
        cosine = xp.cos(value)
        derivative = 1 / (cosine * cosine)
    elif operation == "tanh":
        derivative = 1 - value * value
    elif operation == "sigmoid":
        derivative = value * (1 - value)
    elif operation == "relu":
        derivative = xp.where(xp.isnan(value), np.nan, value > 0)
    elif operation == "leaky_relu":
        derivative = xp.where(xp.isnan(value), np.nan, xp.where(value > 0, 1, parameters["negative_slope"]))
    elif operation == "elu":
        derivative = xp.ones_like(value)
        negative = value <= 0
        scale = value.dtype.type(parameters["alpha"])
        with np.errstate(under="ignore"):
            derivative[negative] = scale * xp.exp(value[negative])
        derivative[xp.isnan(value)] = np.nan
    elif operation == "gelu":
        derivative = _gelu_derivative(value, parameters["approximation"])
    elif operation == "silu":
        sigmoid = _sigmoid(value)
        correction = xp.zeros_like(value)
        with np.errstate(under="ignore"):
            xp.multiply(value, sigmoid * (1 - sigmoid), out=correction, where=xp.isfinite(value))
        derivative = sigmoid + correction
    elif operation == "softplus":
        derivative = _sigmoid(value)
    elif operation == "softsign":
        inverse = 1 / (1 + xp.abs(value))
        with np.errstate(under="ignore"):
            derivative = inverse * inverse
    else:
        raise ValueError(f"Unknown elementwise derivative: {operation}.")
    return xp.asarray(derivative, dtype=value.dtype)


@same_device
def where_forward(condition: np.ndarray, left: np.ndarray, right: np.ndarray, dtype: DType) -> np.ndarray:
    xp = namespace(condition)
    try:
        broadcast_shapes(condition.shape, broadcast_shapes(left.shape, right.shape))
    except ValueError:
        raise ValueError(
            f"where cannot broadcast operand shapes {condition.shape}, {left.shape} and {right.shape}."
        ) from None
    return xp.where(condition, left.astype(dtype.numpy_dtype, copy=False), right.astype(dtype.numpy_dtype, copy=False))


@same_device
def normalized_exponential_forward(
    operation: str, value: np.ndarray, axes: tuple[int, ...], keepdim: bool = True,
) -> np.ndarray:
    xp = namespace(value)
    if operation == "logsumexp" and not axes:
        return value.copy()
    maximum = xp.max(value, axis=axes, keepdims=True, initial=-np.inf)
    shifted = xp.full_like(value, np.nan)
    with np.errstate(over="ignore"):
        xp.subtract(value, maximum, out=shifted, where=xp.isfinite(maximum))
    xp.copyto(shifted, -np.inf, where=xp.isneginf(maximum))
    with np.errstate(under="ignore"):
        exponentials = xp.exp(shifted)
    denominator = xp.sum(exponentials, axis=axes, keepdims=True, dtype=value.dtype)
    if operation == "softmax":
        output = xp.zeros_like(value)
        with np.errstate(under="ignore"):
            xp.divide(exponentials, denominator, out=output, where=denominator != 0)
        return output
    log_denominator = xp.zeros_like(denominator)
    xp.log(denominator, out=log_denominator, where=denominator != 0)
    if operation == "log_softmax":
        with np.errstate(over="ignore"):
            return xp.asarray(shifted - log_denominator)
    if operation == "logsumexp":
        with np.errstate(over="ignore"):
            output = xp.asarray(maximum + log_denominator)
        xp.copyto(output, maximum, where=xp.isinf(maximum))
        return output if keepdim else xp.squeeze(output, axis=axes)
    raise ValueError(f"Unknown normalized exponential operation: {operation}.")


@same_device
def reduction_forward(
    operation: str,
    value: np.ndarray,
    axes: tuple[int, ...],
    keepdim: bool,
    dtype: DType,
    count: int,
    correction: float | None,
) -> np.ndarray:
    xp = namespace(value)
    if operation in ("mean", "min", "max", "amin", "amax", "argmin", "argmax", "var", "std") and count == 0:
        raise ValueError(f"{operation} cannot reduce an empty group for shape {value.shape} and axes {axes}.")
    if operation in ("var", "std") and count - correction <= 0:
        raise ValueError(f"{operation} requires N - correction > 0, got N={count} and correction={correction}.")
    output_shape = tuple(1 if axis in axes else size for axis, size in enumerate(value.shape)) if keepdim else tuple(
        size for axis, size in enumerate(value.shape) if axis not in axes
    )
    if operation == "logsumexp":
        result = normalized_exponential_forward(operation, value.astype(dtype.numpy_dtype, copy=False), axes, keepdim)
    elif operation in ("argmin", "argmax"):
        remaining = tuple(axis for axis in range(value.ndim) if axis not in axes)
        group_shape = tuple(value.shape[axis] for axis in remaining) + (count,)
        grouped = value.transpose(remaining + axes).reshape(group_shape)
        function = xp.argmin if operation == "argmin" else xp.argmax
        result = function(grouped, axis=-1).reshape(output_shape)
    elif operation in ("min", "amin", "max", "amax", "all", "any"):
        function = {"min": xp.min, "amin": xp.min, "max": xp.max, "amax": xp.max, "all": xp.all, "any": xp.any}[operation]
        result = function(value, axis=axes, keepdims=keepdim)
    elif operation in ("sum", "prod"):
        function = xp.sum if operation == "sum" else xp.prod
        result = function(value, axis=axes, dtype=dtype.numpy_dtype, keepdims=keepdim)
    elif operation == "mean":
        result = xp.asarray(xp.sum(value, axis=axes, dtype=dtype.numpy_dtype, keepdims=keepdim))
        xp.divide(result, xp.asarray(count, dtype=dtype.numpy_dtype), out=result)
    elif operation in ("var", "std"):
        data = value.astype(dtype.numpy_dtype, copy=False)
        center = xp.asarray(xp.sum(data, axis=axes, dtype=dtype.numpy_dtype, keepdims=True))
        xp.divide(center, xp.asarray(count, dtype=dtype.numpy_dtype), out=center)
        deviations = xp.asarray(xp.subtract(data, center))
        xp.square(deviations, out=deviations)
        result = xp.asarray(xp.sum(deviations, axis=axes, dtype=dtype.numpy_dtype, keepdims=keepdim))
        xp.divide(result, xp.asarray(count - correction, dtype=dtype.numpy_dtype), out=result)
        if operation == "std":
            xp.sqrt(result, out=result)
    else:
        raise ValueError(f"Unknown reduction operation: {operation}.")
    return xp.array(result, dtype=dtype.numpy_dtype, copy=True, order="C")
