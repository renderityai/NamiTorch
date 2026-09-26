import math
from types import MappingProxyType

import numpy as np

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


def binary_forward(
    operation: str,
    left: np.ndarray,
    right: np.ndarray,
    compute_dtype: DType,
    output_dtype: DType,
) -> np.ndarray:
    shape = matmul_shape(left.shape, right.shape) if operation == "matmul" else broadcast_shapes(left.shape, right.shape)
    if compute_dtype.is_boolean and operation in ("subtract", "floor_divide", "remainder", "power"):
        raise TypeError(f"{operation} is not defined for boolean operands.")
    first = left.astype(compute_dtype.numpy_dtype, copy=False)
    second = right.astype(compute_dtype.numpy_dtype, copy=False)
    output = np.empty(shape, dtype=output_dtype.numpy_dtype)
    try:
        BINARY_UFUNCS[operation](first, second, out=output)
    except ValueError as error:
        raise ValueError(f"{operation} failed for operand shapes {left.shape} and {right.shape}: {error}") from None
    return output


def axis_index_coordinates(shape: tuple[int, ...], axis: int, index: np.ndarray) -> tuple[np.ndarray, ...]:
    if index.ndim != len(shape):
        raise ValueError(f"Input and index must have the same rank, got shapes {shape} and {index.shape}.")
    if any(size > shape[dimension] for dimension, size in enumerate(index.shape) if dimension != axis):
        raise ValueError(f"Index shape {index.shape} exceeds input shape {shape} outside dim {axis}.")
    if np.any(index < 0) or np.any(index >= shape[axis]):
        raise IndexError(f"Indices must be in [0, {shape[axis]}) for dim {axis} of input shape {shape}.")
    coordinates = []
    for dimension, size in enumerate(index.shape):
        if dimension == axis:
            coordinates.append(index)
        else:
            coordinate_shape = tuple(size if position == dimension else 1 for position in range(index.ndim))
            coordinate = np.arange(size, dtype=np.intp).reshape(coordinate_shape)
            coordinate.flags.writeable = False
            coordinates.append(coordinate)
    return tuple(coordinates)


def scatter_add_forward(
    value: np.ndarray, coordinates: tuple[np.ndarray, ...], source: np.ndarray,
    index_shape: tuple[int, ...], dtype: DType,
) -> np.ndarray:
    try:
        broadcasted = np.broadcast_to(source.astype(dtype.numpy_dtype, copy=False), index_shape)
    except ValueError:
        raise ValueError(f"scatter_add cannot broadcast src shape {source.shape} to index shape {index_shape}.") from None
    output = np.array(value, dtype=dtype.numpy_dtype, copy=True, order="C")
    np.add.at(output, coordinates, broadcasted)
    return output


def unary_forward(operation: str, value: np.ndarray, dtype: DType) -> np.ndarray:
    if dtype.is_boolean and operation in ("positive", "negative"):
        raise TypeError(f"{operation} is not defined for boolean operands.")
    output = np.empty(value.shape, dtype=dtype.numpy_dtype)
    if dtype.is_boolean and operation in ("square", "sign"):
        np.copyto(output, value)
        return output
    UNARY_UFUNCS[operation](value, out=output)
    return output


def _sigmoid(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value)
    positive = value >= 0
    with np.errstate(under="ignore"):
        output[positive] = 1 / (1 + np.exp(-value[positive]))
        exponent = np.exp(value[~positive])
        output[~positive] = exponent / (1 + exponent)
    return output


def _gelu(value: np.ndarray, approximation: str) -> np.ndarray:
    output = np.empty_like(value)
    if approximation == "exact":
        divisor = math.sqrt(2)
        weights = np.fromiter(
            (0.5 * math.erfc(-float(item) / divisor) for item in value.flat),
            dtype=value.dtype, count=value.size,
        ).reshape(value.shape)
        negative_infinity = np.isneginf(value)
        output[negative_infinity] = 0
        np.multiply(value, weights, out=output, where=~negative_infinity)
    else:
        np.maximum(value, 0, out=output)
        output[value < 0] = -0.0
        central = np.abs(value) <= 20
        selected = value[central]
        inner = math.sqrt(2 / math.pi) * (selected + 0.044715 * selected ** 3)
        output[central] = (0.5 * selected) * (1 + np.tanh(inner))
    return output


def elementwise_forward(
    operation: str, value: np.ndarray, dtype: DType,
    parameter: float | None = None, approximation: str | None = None,
) -> np.ndarray:
    data = value.astype(dtype.numpy_dtype, copy=False)
    if operation in UNARY_UFUNCS:
        return unary_forward(operation, data, dtype)
    if operation == "sigmoid":
        return _sigmoid(data)
    if operation == "gelu":
        return _gelu(data, approximation)
    output = np.empty(data.shape, dtype=dtype.numpy_dtype)
    if operation == "rsqrt":
        np.sqrt(data, out=output)
        np.reciprocal(output, out=output)
    elif operation == "relu":
        np.maximum(data, dtype.numpy_dtype.type(0), out=output)
    elif operation in ("leaky_relu", "elu"):
        np.copyto(output, data)
        negative = data < 0
        scale = dtype.numpy_dtype.type(parameter)
        if scale == 0:
            output[negative] = 0
        elif operation == "leaky_relu":
            output[negative] = scale * data[negative]
        else:
            with np.errstate(under="ignore"):
                output[negative] = scale * np.expm1(data[negative])
    elif operation == "silu":
        negative_infinity = np.isneginf(data)
        output[negative_infinity] = 0
        np.multiply(data, _sigmoid(data), out=output, where=~negative_infinity)
    elif operation == "softplus":
        with np.errstate(under="ignore"):
            np.add(np.maximum(data, 0), np.log1p(np.exp(-np.abs(data))), out=output)
    elif operation == "softsign":
        infinite = np.isinf(data)
        output[infinite] = np.sign(data[infinite])
        np.divide(data, 1 + np.abs(data), out=output, where=~infinite)
    else:
        raise ValueError(f"Unknown elementwise operation: {operation}.")
    return output


def clamp_forward(
    value: np.ndarray, minimum: np.ndarray | None, maximum: np.ndarray | None, dtype: DType
) -> np.ndarray:
    output = np.array(value, dtype=dtype.numpy_dtype, copy=True, order="C")
    if minimum is not None:
        np.maximum(output, minimum.astype(dtype.numpy_dtype, copy=False), out=output)
    if maximum is not None:
        np.minimum(output, maximum.astype(dtype.numpy_dtype, copy=False), out=output)
    return output


def _gelu_derivative(value: np.ndarray, approximation: str) -> np.ndarray:
    if approximation == "exact":
        output = np.fromiter(
            (0.5 * math.erfc(-float(item) / math.sqrt(2)) for item in value.flat),
            dtype=value.dtype, count=value.size,
        ).reshape(value.shape)
        central = np.abs(value) <= 40
        selected = value[central]
        with np.errstate(under="ignore"):
            output[central] += selected * np.exp(-0.5 * selected * selected) / math.sqrt(2 * math.pi)
        return output
    output = np.where(np.isnan(value), np.nan, np.where(value > 0, 1, 0)).astype(value.dtype)
    central = np.abs(value) <= 20
    selected = value[central]
    scale = math.sqrt(2 / math.pi)
    activation = np.tanh(scale * (selected + 0.044715 * selected ** 3))
    inner_derivative = scale * (1 + 3 * 0.044715 * selected * selected)
    output[central] = 0.5 * (1 + activation) + 0.5 * selected * (1 - activation * activation) * inner_derivative
    return output


def elementwise_derivative(operation: str, value: np.ndarray, parameters: dict) -> np.ndarray:
    if operation == "exp":
        derivative = value
    elif operation == "expm1":
        derivative = np.exp(value)
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
        derivative = np.sign(value)
    elif operation == "sin":
        derivative = np.cos(value)
    elif operation == "cos":
        derivative = -np.sin(value)
    elif operation == "tan":
        cosine = np.cos(value)
        derivative = 1 / (cosine * cosine)
    elif operation == "tanh":
        derivative = 1 - value * value
    elif operation == "sigmoid":
        derivative = value * (1 - value)
    elif operation == "relu":
        derivative = np.where(np.isnan(value), np.nan, value > 0)
    elif operation == "leaky_relu":
        derivative = np.where(np.isnan(value), np.nan, np.where(value > 0, 1, parameters["negative_slope"]))
    elif operation == "elu":
        derivative = np.ones_like(value)
        negative = value <= 0
        scale = value.dtype.type(parameters["alpha"])
        with np.errstate(under="ignore"):
            derivative[negative] = scale * np.exp(value[negative])
        derivative[np.isnan(value)] = np.nan
    elif operation == "gelu":
        derivative = _gelu_derivative(value, parameters["approximation"])
    elif operation == "silu":
        sigmoid = _sigmoid(value)
        correction = np.zeros_like(value)
        with np.errstate(under="ignore"):
            np.multiply(value, sigmoid * (1 - sigmoid), out=correction, where=np.isfinite(value))
        derivative = sigmoid + correction
    elif operation == "softplus":
        derivative = _sigmoid(value)
    elif operation == "softsign":
        inverse = 1 / (1 + np.abs(value))
        with np.errstate(under="ignore"):
            derivative = inverse * inverse
    else:
        raise ValueError(f"Unknown elementwise derivative: {operation}.")
    return np.asarray(derivative, dtype=value.dtype)


def where_forward(condition: np.ndarray, left: np.ndarray, right: np.ndarray, dtype: DType) -> np.ndarray:
    try:
        broadcast_shapes(condition.shape, broadcast_shapes(left.shape, right.shape))
    except ValueError:
        raise ValueError(
            f"where cannot broadcast operand shapes {condition.shape}, {left.shape} and {right.shape}."
        ) from None
    return np.where(condition, left.astype(dtype.numpy_dtype, copy=False), right.astype(dtype.numpy_dtype, copy=False))


def normalized_exponential_forward(
    operation: str, value: np.ndarray, axes: tuple[int, ...], keepdim: bool = True,
) -> np.ndarray:
    if operation == "logsumexp" and not axes:
        return value.copy()
    maximum = np.max(value, axis=axes, keepdims=True, initial=-np.inf)
    shifted = np.full_like(value, np.nan)
    with np.errstate(over="ignore"):
        np.subtract(value, maximum, out=shifted, where=np.isfinite(maximum))
    np.copyto(shifted, -np.inf, where=np.isneginf(maximum))
    with np.errstate(under="ignore"):
        exponentials = np.exp(shifted)
    denominator = np.sum(exponentials, axis=axes, keepdims=True, dtype=value.dtype)
    if operation == "softmax":
        output = np.zeros_like(value)
        with np.errstate(under="ignore"):
            np.divide(exponentials, denominator, out=output, where=denominator != 0)
        return output
    log_denominator = np.zeros_like(denominator)
    np.log(denominator, out=log_denominator, where=denominator != 0)
    if operation == "log_softmax":
        with np.errstate(over="ignore"):
            return np.asarray(shifted - log_denominator)
    if operation == "logsumexp":
        with np.errstate(over="ignore"):
            output = np.asarray(maximum + log_denominator)
        np.copyto(output, maximum, where=np.isinf(maximum))
        return output if keepdim else np.squeeze(output, axis=axes)
    raise ValueError(f"Unknown normalized exponential operation: {operation}.")


def reduction_forward(
    operation: str,
    value: np.ndarray,
    axes: tuple[int, ...],
    keepdim: bool,
    dtype: DType,
    count: int,
    correction: float | None,
) -> np.ndarray:
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
        function = np.argmin if operation == "argmin" else np.argmax
        result = function(grouped, axis=-1).reshape(output_shape)
    elif operation in ("min", "amin", "max", "amax", "all", "any"):
        function = {"min": np.min, "amin": np.min, "max": np.max, "amax": np.max, "all": np.all, "any": np.any}[operation]
        result = function(value, axis=axes, keepdims=keepdim)
    elif operation in ("sum", "prod"):
        function = np.sum if operation == "sum" else np.prod
        result = function(value, axis=axes, dtype=dtype.numpy_dtype, keepdims=keepdim)
    elif operation == "mean":
        result = np.asarray(np.sum(value, axis=axes, dtype=dtype.numpy_dtype, keepdims=keepdim))
        np.divide(result, np.asarray(count, dtype=dtype.numpy_dtype), out=result)
    elif operation in ("var", "std"):
        data = value.astype(dtype.numpy_dtype, copy=False)
        center = np.asarray(np.sum(data, axis=axes, dtype=dtype.numpy_dtype, keepdims=True))
        np.divide(center, np.asarray(count, dtype=dtype.numpy_dtype), out=center)
        deviations = np.asarray(np.subtract(data, center))
        np.square(deviations, out=deviations)
        result = np.asarray(np.sum(deviations, axis=axes, dtype=dtype.numpy_dtype, keepdims=keepdim))
        np.divide(result, np.asarray(count - correction, dtype=dtype.numpy_dtype), out=result)
        if operation == "std":
            np.sqrt(result, out=result)
    else:
        raise ValueError(f"Unknown reduction operation: {operation}.")
    return np.array(result, dtype=dtype.numpy_dtype, copy=True, order="C")
