from __future__ import annotations

import builtins
from enum import Enum
from types import MappingProxyType

import numpy as np


class UnsupportedDTypeError(TypeError):
    def __init__(self, value: object):
        supported = ", ".join(dtype.value for dtype in DType)
        super().__init__(f"Unsupported NamiTorch dtype: {value!r}. Supported: {supported}.")


class DType(Enum):
    float32 = "float32"
    float64 = "float64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"
    bool = "bool"

    @classmethod
    def _missing_(cls, value: object) -> DType:
        raise UnsupportedDTypeError(value)

    def __repr__(self) -> str:
        return f"namitorch.{self.value}"

    def __str__(self) -> str:
        return repr(self)

    @property
    def bits(self) -> int:
        return self.itemsize * 8

    @property
    def itemsize(self) -> int:
        return self.numpy_dtype.itemsize

    @property
    def numpy_dtype(self) -> np.dtype:
        return _DTYPE_TO_NUMPY[self]

    @property
    def is_floating_point(self) -> builtins.bool:
        return self in (DType.float32, DType.float64)

    @property
    def is_integer(self) -> builtins.bool:
        return self in (DType.int8, DType.int16, DType.int32, DType.int64, DType.uint8)

    @property
    def is_signed(self) -> builtins.bool:
        return self.is_floating_point or (self.is_integer and not self.is_unsigned)

    @property
    def is_unsigned(self) -> builtins.bool:
        return self is DType.uint8

    @property
    def is_boolean(self) -> builtins.bool:
        return self is DType.bool

    @property
    def can_require_grad(self) -> builtins.bool:
        return self.is_floating_point


float32 = DType.float32
float64 = DType.float64
int8 = DType.int8
int16 = DType.int16
int32 = DType.int32
int64 = DType.int64
uint8 = DType.uint8
bool = DType.bool

_NAME_TO_DTYPE = MappingProxyType({dtype.value: dtype for dtype in DType})
_DTYPE_TO_NUMPY = MappingProxyType({dtype: np.dtype(dtype.value) for dtype in DType})
_NUMPY_TO_DTYPE = MappingProxyType({value: key for key, value in _DTYPE_TO_NUMPY.items()})
_NUMPY_SCALAR_TO_DTYPE = MappingProxyType(
    {
        scalar_type: _NUMPY_TO_DTYPE[np.dtype(scalar_type)]
        for scalar_type in {numpy_dtype.type for numpy_dtype in _NUMPY_TO_DTYPE}
        | {np.longlong}
    }
)

_PROMOTION_ORDER = (bool, uint8, int8, int16, int32, int64, float32, float64)
_PROMOTION_INDEX = MappingProxyType(
    {dtype: index for index, dtype in enumerate(_PROMOTION_ORDER)}
)
_PROMOTION_TABLE = (
    (bool, uint8, int8, int16, int32, int64, float32, float64),
    (uint8, uint8, int16, int16, int32, int64, float32, float64),
    (int8, int16, int8, int16, int32, int64, float32, float64),
    (int16, int16, int16, int16, int32, int64, float32, float64),
    (int32, int32, int32, int32, int32, int64, float32, float64),
    (int64, int64, int64, int64, int64, int64, float32, float64),
    (float32, float32, float32, float32, float32, float32, float32, float64),
    (float64, float64, float64, float64, float64, float64, float64, float64),
)


def get_default_dtype() -> DType:
    return float32


def from_numpy_dtype(dtype: np.dtype) -> DType:
    if not isinstance(dtype, np.dtype):
        raise UnsupportedDTypeError(dtype)
    if dtype.fields is not None or dtype.subdtype is not None or dtype.metadata is not None:
        raise UnsupportedDTypeError(dtype)
    normalized = _NUMPY_TO_DTYPE.get(dtype.newbyteorder("="))
    if normalized is None:
        raise UnsupportedDTypeError(dtype)
    return normalized


def normalize_dtype(dtype: object) -> DType:
    if isinstance(dtype, DType):
        return dtype
    if dtype is builtins.float:
        return get_default_dtype()
    if dtype is builtins.int:
        return int64
    if dtype is builtins.bool:
        return bool
    if isinstance(dtype, np.dtype):
        return from_numpy_dtype(dtype)
    if type(dtype) is str:
        normalized = _NAME_TO_DTYPE.get(dtype)
        if normalized is not None:
            return normalized
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        normalized = _NUMPY_SCALAR_TO_DTYPE.get(dtype)
        if normalized is not None:
            return normalized
    raise UnsupportedDTypeError(dtype)


def to_numpy_dtype(dtype: object) -> np.dtype:
    return _DTYPE_TO_NUMPY[normalize_dtype(dtype)]


def is_floating_point(dtype: object) -> builtins.bool:
    return normalize_dtype(dtype).is_floating_point


def is_integer(dtype: object) -> builtins.bool:
    return normalize_dtype(dtype).is_integer


def is_signed(dtype: object) -> builtins.bool:
    return normalize_dtype(dtype).is_signed


def is_unsigned(dtype: object) -> builtins.bool:
    return normalize_dtype(dtype).is_unsigned


def is_boolean(dtype: object) -> builtins.bool:
    return normalize_dtype(dtype).is_boolean


def can_require_grad(dtype: object) -> builtins.bool:
    return normalize_dtype(dtype).can_require_grad


def _infer_dtype(value: object) -> DType:
    if isinstance(value, (np.generic, np.ndarray)):
        return from_numpy_dtype(value.dtype)
    if isinstance(value, builtins.bool):
        return bool
    if isinstance(value, builtins.int):
        if not -(1 << 63) <= value < (1 << 63):
            raise OverflowError("Python integer is outside the supported int64 range.")
        return int64
    if isinstance(value, builtins.float):
        return get_default_dtype()
    return normalize_dtype(value)


def _apply_operation(dtype: DType, operation: str) -> DType:
    if operation == "arithmetic":
        return dtype
    if operation == "true_divide":
        return dtype if dtype.is_floating_point else get_default_dtype()
    if operation == "comparison":
        return bool
    raise ValueError(
        f"Unsupported promotion operation: {operation!r}. "
        "Expected 'arithmetic', 'true_divide' or 'comparison'."
    )


def promote_types(left: object, right: object, *, operation: str = "arithmetic") -> DType:
    left_dtype = normalize_dtype(left)
    right_dtype = normalize_dtype(right)
    promoted = _PROMOTION_TABLE[_PROMOTION_INDEX[left_dtype]][_PROMOTION_INDEX[right_dtype]]
    return _apply_operation(promoted, operation)


def result_type(*values_or_dtypes: object, operation: str = "arithmetic") -> DType:
    if not values_or_dtypes:
        raise TypeError("result_type requires at least one value or dtype.")
    promoted = _infer_dtype(values_or_dtypes[0])
    for value in values_or_dtypes[1:]:
        promoted = promote_types(promoted, _infer_dtype(value))
    return _apply_operation(promoted, operation)


__all__ = [
    "DType",
    "UnsupportedDTypeError",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
    "get_default_dtype",
    "normalize_dtype",
    "from_numpy_dtype",
    "to_numpy_dtype",
    "is_floating_point",
    "is_integer",
    "is_signed",
    "is_unsigned",
    "is_boolean",
    "can_require_grad",
    "promote_types",
    "result_type",
]
