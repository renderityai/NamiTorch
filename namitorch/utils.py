import math
import operator
from itertools import zip_longest

import numpy as np


def normalize_shape(*shape: object, allow_inferred: bool = False) -> tuple[int, ...]:
    dimensions = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    normalized = []
    for dimension in dimensions:
        if isinstance(dimension, (bool, np.bool_)):
            raise TypeError("Shape dimensions must be integers, not booleans.")
        try:
            value = operator.index(dimension)
        except TypeError:
            raise TypeError(f"Shape dimensions must be integers, got {dimension!r}.") from None
        if value < 0 and not (allow_inferred and value == -1):
            raise ValueError(f"Shape dimensions must be nonnegative, got {value}.")
        normalized.append(value)
    return tuple(normalized)


def normalize_reshape_shape(numel: int, *shape: object) -> tuple[int, ...]:
    dimensions = normalize_shape(*shape, allow_inferred=True)
    inferred_count = dimensions.count(-1)
    if inferred_count > 1:
        raise ValueError("Only one reshape dimension may be -1.")
    known_size = math.prod(dimension for dimension in dimensions if dimension != -1)
    if inferred_count:
        if known_size == 0:
            raise ValueError("Cannot infer a reshape dimension when another dimension is zero.")
        if numel % known_size:
            raise ValueError(f"Cannot reshape {numel} elements into shape {dimensions}.")
        inferred = numel // known_size
        return tuple(inferred if dimension == -1 else dimension for dimension in dimensions)
    if known_size != numel:
        raise ValueError(f"Cannot reshape {numel} elements into shape {dimensions}.")
    return dimensions


def normalize_axis(axis: object, ndim: int) -> int:
    if isinstance(axis, (bool, np.bool_)):
        raise TypeError("An axis must be an integer, not a boolean.")
    try:
        value = operator.index(axis)
    except TypeError:
        raise TypeError(f"An axis must be an integer, got {axis!r}.") from None
    if not -ndim <= value < ndim:
        raise IndexError(f"Axis {value} is out of bounds for {ndim} dimensions.")
    return value % ndim


def normalize_axes(axes: object, ndim: int) -> tuple[int, ...]:
    sequence = axes if isinstance(axes, (list, tuple)) else (axes,)
    normalized = tuple(normalize_axis(axis, ndim) for axis in sequence)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Axes must not contain duplicates.")
    return normalized


def normalize_reduction_axes(dim: object, ndim: int) -> tuple[int, ...]:
    if dim is None:
        return tuple(range(ndim))
    if isinstance(dim, list):
        raise TypeError("Reduction dim must be None, an integer or a tuple of integers.")
    return tuple(sorted(normalize_axes(dim, ndim)))


def broadcast_shapes(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    left, right = normalize_shape(left), normalize_shape(right)
    result = []
    for first, second in zip_longest(reversed(left), reversed(right), fillvalue=1):
        if first == second or second == 1:
            result.append(first)
        elif first == 1:
            result.append(second)
        else:
            raise ValueError(f"Cannot broadcast operand shapes {left} and {right}.")
    return tuple(reversed(result))


def matmul_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    left, right = normalize_shape(left), normalize_shape(right)
    if not left or not right:
        raise ValueError(f"matmul requires operands with at least one dimension, got shapes {left} and {right}.")
    right_inner = right[-2] if len(right) > 1 else right[-1]
    if left[-1] != right_inner:
        raise ValueError(f"matmul inner dimensions do not match for operand shapes {left} and {right}.")
    try:
        batch = broadcast_shapes(left[:-2], right[:-2])
    except ValueError:
        raise ValueError(f"matmul cannot broadcast batch dimensions for operand shapes {left} and {right}.") from None
    rows = (left[-2],) if len(left) > 1 else ()
    columns = (right[-1],) if len(right) > 1 else ()
    return batch + rows + columns


__all__ = [
    "normalize_shape", "normalize_reshape_shape", "normalize_axis", "normalize_axes",
    "normalize_reduction_axes", "broadcast_shapes", "matmul_shape",
]
