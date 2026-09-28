from __future__ import annotations

import builtins
import math
import operator
import weakref
from dataclasses import dataclass
from numbers import Real

from typing import Any as Array

import numpy as np

from .backends import array_device, ensure_same_device, get_backend, is_array, namespace, readonly, same_device, transfer, writable
from .backends._pinned import is_pinned, pinned_empty
from .device import Device
from .amp import OPERATION_POLICIES, autocast_dtype

from ._backend import (
    axis_index_coordinates,
    binary_forward,
    clamp_forward,
    elementwise_forward,
    normalized_exponential_forward,
    reduction_forward,
    scatter_add_forward,
    unary_forward,
    where_forward,
)
from .autograd import (
    ARITHMETIC_BACKWARD_RULES,
    ELEMENTWISE_BACKWARD_OPERATIONS,
    REDUCTION_BACKWARD_RULES,
    SELECTION_BACKWARD_RULES,
    BackwardNode,
    Context,
    accumulate_gradient,
    arithmetic_node,
    elementwise_node,
    identity_backward,
    indexing_node,
    is_grad_enabled,
    normalized_exponential_node,
    reduction_node,
    run_backward,
    scatter_add_node,
    selection_node,
    transform_node,
    transfer_backward,
    validate_accumulation,
)
from .dtype import (
    DType,
    UnsupportedDTypeError,
    bool as bool_dtype,
    float16,
    float32,
    float64,
    from_numpy_dtype,
    get_default_dtype,
    int32,
    int64,
    normalize_dtype,
    promote_types,
    result_type,
)
from .utils import broadcast_shapes, normalize_axes, normalize_axis, normalize_reduction_axes, normalize_reshape_shape, normalize_shape


@dataclass(slots=True)
class _VersionCounter:
    value: int = 0

    def increment(self) -> None:
        self.value += 1


@dataclass(frozen=True, slots=True)
class _TransformContext:
    operation: str
    original_shape: tuple[int, ...]
    permutation: tuple[int, ...] | None = None
    axes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _IndexContext:
    original_shape: tuple[int, ...]
    index: tuple[object, ...]
    is_advanced: bool
    source_version: int


@dataclass(frozen=True, slots=True)
class _OperationContext:
    operation: str
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[DType, ...]
    input_versions: tuple[int, ...]
    compute_dtype: DType
    parameters: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class _ReductionContext:
    operation: str
    original_shape: tuple[int, ...]
    axes: tuple[int, ...]
    keepdim: bool
    count: int
    correction: float | None
    source_version: int


_storage_versions: dict[int, tuple[dict[int, weakref.ReferenceType[Array]], _VersionCounter]] = {}


def _storage_owner(array: Array) -> object:
    owner = array
    while True:
        base = owner.obj if isinstance(owner, memoryview) else getattr(owner, "base", None)
        if base is None:
            return owner
        owner = base


def _shares_storage(left: Array, right: Array) -> bool:
    return array_device(left) == array_device(right) and (_storage_owner(left) is _storage_owner(right) or namespace(left).shares_memory(left, right))


def _storage_version(array: Array, preferred: _VersionCounter | None = None) -> _VersionCounter:
    owner = _storage_owner(array)
    identity = id(owner)
    existing = _storage_versions.get(identity)
    if existing is None:
        existing = ({}, preferred if preferred is not None else _VersionCounter())
        _storage_versions[identity] = existing
    references, counter = existing
    array_identity = id(array)
    if array_identity in references:
        return counter

    def discard(reference: weakref.ReferenceType[Array]) -> None:
        entry = _storage_versions.get(identity)
        if entry is not None and entry[0].get(array_identity) is reference:
            del entry[0][array_identity]
            if not entry[0]:
                del _storage_versions[identity]

    references[array_identity] = weakref.ref(array, discard)
    return counter


def _slice_bound(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        if value.ndim != 0 or not value.dtype.is_integer:
            raise TypeError("Slice bounds must be scalar integers.")
        value = value.item()
    if isinstance(value, (builtins.bool, np.bool_)):
        return builtins.int(value)
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError("Slice bounds and step must be integers or None.") from None


def _index_integer(value: object) -> int:
    try:
        integer = operator.index(value)
    except TypeError:
        raise TypeError("Tensor indices must be integers, slices, ellipsis, None or integer/boolean arrays.") from None
    limits = np.iinfo(np.intp)
    if integer < limits.min or integer > limits.max:
        raise IndexError(f"Integer index {integer} is outside the supported indexing range.")
    return integer


def _index_sequence(sequence: list | tuple, active: set[int]) -> list:
    identity = id(sequence)
    if identity in active:
        raise ValueError("Index sequences cannot contain cycles.")
    active.add(identity)
    result = []
    try:
        for value in sequence:
            if isinstance(value, (list, tuple)):
                result.append(_index_sequence(value, active))
            elif isinstance(value, (builtins.bool, np.bool_)):
                result.append(builtins.bool(value))
            else:
                result.append(_index_integer(value))
    finally:
        active.remove(identity)
    return result


def _index_array(array: Array) -> Array:
    if array.dtype.kind not in "iub":
        raise TypeError(f"Index arrays must have integer or boolean dtype, got {array.dtype}.")
    if array.dtype.kind in "iu" and array.size:
        limits = np.iinfo(np.intp)
        if builtins.int(array.min()) < limits.min or builtins.int(array.max()) > limits.max:
            raise IndexError("An integer array index is outside the supported indexing range.")
    return readonly(namespace(array).array(array, copy=True, order="C", subok=False))


def _normalize_index_component(index: object, device: Device) -> object:
    if index is None or index is Ellipsis:
        return index
    if isinstance(index, slice):
        start, stop, step = (_slice_bound(value) for value in (index.start, index.stop, index.step))
        if step == 0:
            raise ValueError("Slice step must not be zero.")
        return slice(start, stop, step)
    if isinstance(index, Tensor):
        if index.device != device:
            raise RuntimeError(f"Index is on {index.device}, but Tensor is on {device}; use .to() explicitly.")
        if not (index.dtype.is_integer or index.dtype.is_boolean):
            raise TypeError(f"Tensor indices must have integer or boolean dtype, got {index.dtype.name}.")
        return _index_array(index._data)
    if is_array(index):
        if array_device(index) != device:
            raise RuntimeError(f"Index array is on {array_device(index)}, but Tensor is on {device}.")
        return _index_array(index)
    if isinstance(index, (list, tuple)):
        sequence = _index_sequence(index, set())
        try:
            array = np.asarray(sequence)
        except ValueError:
            raise IndexError("Index sequences must form a rectangular integer or boolean array.") from None
        if array.size == 0:
            array = array.astype(np.intp)
        return _index_array(get_backend(device).array(array))
    if isinstance(index, (builtins.bool, np.bool_)):
        return builtins.bool(index)
    return _index_integer(index)


def _normalize_index(index: object, device: Device) -> tuple[object, ...]:
    components = index if isinstance(index, tuple) else (index,)
    return tuple(_normalize_index_component(component, device) for component in components)


def _scalar_index_view(array: Array, index: tuple[object, ...]) -> Array:
    integers = (component for component in index if component is not Ellipsis)
    slices = tuple(slice(integer % length, integer % length + 1) for integer, length in zip(integers, array.shape))
    return array[slices].reshape(()) if slices else array.view()


def _infer_data_dtype(data: object, active_sequences: set[int]) -> DType | None:
    if isinstance(data, Tensor):
        return data.dtype
    if is_array(data) or isinstance(data, np.generic):
        return from_numpy_dtype(data.dtype)
    if isinstance(data, (builtins.bool, builtins.int, builtins.float)):
        return result_type(data)
    if isinstance(data, (list, tuple)):
        identity = id(data)
        if identity in active_sequences:
            raise ValueError("Tensor data cannot contain cyclic sequences.")
        active_sequences.add(identity)
        inferred = None
        try:
            for value in data:
                candidate = _infer_data_dtype(value, active_sequences)
                if candidate is not None:
                    inferred = candidate if inferred is None else promote_types(inferred, candidate)
        finally:
            active_sequences.remove(identity)
        return inferred
    raise UnsupportedDTypeError(type(data))


def _resolve_dtype(data: object, dtype: object) -> DType:
    requested = None if dtype is None else normalize_dtype(dtype)
    inferred = _infer_data_dtype(data, set())
    return requested if requested is not None else inferred or get_default_dtype()


def _validate_requires_grad(dtype: DType, requires_grad: builtins.bool) -> None:
    if type(requires_grad) is not builtins.bool:
        raise TypeError("requires_grad must be a Python bool.")
    if requires_grad and not dtype.can_require_grad:
        raise ValueError(f"requires_grad=True requires a floating dtype, got {dtype.name}.")


def _integer_argument(value, name, minimum=None):
    if isinstance(value, (builtins.bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean.")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer.") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _make_array(data: object, dtype: DType, *, copy: builtins.bool, device=None):
    backend = get_backend(device)
    source = data._data if isinstance(data, Tensor) else data
    if is_array(source):
        can_share = (
            array_device(source) == backend.device
            and source.dtype == dtype.numpy_dtype
            and source.dtype.isnative
            and source.flags.c_contiguous
            and getattr(source.flags, "aligned", True)
            and writable(source)
            and all(stride >= 0 for stride in source.strides)
        )
        if not copy and can_share:
            return backend.array(source, dtype=dtype.numpy_dtype, copy=False).view()
        return transfer(source, backend.device, dtype.numpy_dtype, copy=True)
    return backend.array(source, dtype=dtype.numpy_dtype, copy=True)


class Tensor:
    __array_priority__ = 1000
    __array_ufunc__ = None

    __slots__ = (
        "_backend", "_writable",
        "_data", "_dtype", "_requires_grad", "_grad", "_grad_fn", "_parents",
        "_is_leaf", "_retain_grad", "_version_counter", "_transform_context", "_index_context",
        "_operation_context", "_reduction_context",
    )

    def __init__(
        self, data: object, dtype: object = None, requires_grad: builtins.bool = False, *, device=None
    ):
        resolved_dtype = _resolve_dtype(data, dtype)
        _validate_requires_grad(resolved_dtype, requires_grad)
        self._initialize(_make_array(data, resolved_dtype, copy=True, device=device), requires_grad)

    def _initialize(
        self,
        array: Array,
        requires_grad: builtins.bool,
        version_counter: _VersionCounter | None = None,
    ) -> None:
        if not is_array(array):
            raise TypeError("Tensor storage must be a registered backend array.")
        self._backend = get_backend(array)
        self._writable = writable(array)
        self._dtype = from_numpy_dtype(array.dtype)
        if self._dtype is float16 and self._backend.device.type == "cuda" and not self._backend.supports_float16():
            raise RuntimeError(f"CUDA float16 is not supported on {self._backend.device}.")
        _validate_requires_grad(self._dtype, requires_grad)
        self._data = array
        self._requires_grad = requires_grad
        self._grad: Tensor | None = None
        self._grad_fn = None
        self._parents: tuple[Tensor, ...] = ()
        self._is_leaf = True
        self._retain_grad = False
        self._version_counter = _storage_version(array, version_counter)
        self._transform_context: _TransformContext | None = None
        self._index_context: _IndexContext | None = None
        self._operation_context: _OperationContext | None = None
        self._reduction_context: _ReductionContext | None = None

    @classmethod
    def _from_array(
        cls,
        array: Array,
        requires_grad: builtins.bool,
        version_counter: _VersionCounter | None = None,
    ) -> Tensor:
        instance = cls.__new__(cls)
        instance._initialize(array, requires_grad, version_counter)
        return instance

    @property
    def device(self) -> Device:
        return self._backend.device

    @property
    def shape(self) -> tuple[builtins.int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> builtins.int:
        return self._data.ndim

    @property
    def size(self) -> builtins.int:
        return self._data.size

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def requires_grad(self) -> builtins.bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: builtins.bool) -> None:
        _validate_requires_grad(self.dtype, value)
        if not self.is_leaf:
            raise ValueError("requires_grad can only be changed on leaf tensors.")
        self._requires_grad = value

    def requires_grad_(self, mode: builtins.bool = True) -> Tensor:
        self.requires_grad = mode
        return self

    @property
    def grad(self) -> Tensor | None:
        return self._grad

    @property
    def grad_fn(self) -> BackwardNode | None:
        return self._grad_fn

    def retain_grad(self) -> None:
        if not self.requires_grad:
            raise RuntimeError("retain_grad() requires a Tensor with requires_grad=True.")
        self._retain_grad = True

    def backward(self, gradient: object = None, retain_graph: builtins.bool = False) -> None:
        if type(retain_graph) is not builtins.bool:
            raise TypeError("retain_graph must be a Python bool.")
        if not self.requires_grad:
            raise RuntimeError("backward() requires a Tensor with requires_grad=True.")
        if gradient is None:
            if self.numel() != 1:
                raise RuntimeError("An explicit gradient is required unless the output has exactly one element.")
            seed = namespace(self).ones(self.shape, dtype=self.dtype.numpy_dtype)
        else:
            ensure_same_device(self, gradient)
            supplied = Tensor(gradient, dtype=self.dtype, device=self.device)
            if supplied.shape != self.shape:
                raise RuntimeError(f"Gradient shape {supplied.shape} does not match output shape {self.shape}.")
            seed = supplied._data
        run_backward(self, seed, retain_graph)

    @same_device
    def _accumulate_grad(self, gradient: Array) -> None:
        validate_accumulation(self, gradient)
        accumulated = accumulate_gradient(self, gradient, None if self._grad is None else self._grad._data)
        if self._grad is None:
            self._grad = Tensor._from_array(accumulated, False)
        else:
            self._grad.copy_(accumulated)

    @property
    def is_leaf(self) -> builtins.bool:
        return self._is_leaf

    @property
    def nbytes(self) -> builtins.int:
        return self._data.nbytes

    @property
    def _version(self) -> builtins.int:
        return self._version_counter.value

    def numel(self) -> builtins.int:
        return self.size

    def element_size(self) -> builtins.int:
        return self._data.itemsize

    def item(self) -> builtins.bool | builtins.int | builtins.float:
        if self.size != 1:
            raise ValueError(f"item() requires exactly one element, got {self.size}.")
        return self._backend.item(self._data)

    def tolist(self) -> list | builtins.bool | builtins.int | builtins.float:
        return self.numpy().tolist()

    def numpy(self) -> np.ndarray:
        return self._backend.to_numpy(self._data)

    def is_pinned(self) -> builtins.bool:
        return is_pinned(self._data)

    def pin_memory(self) -> Tensor:
        if self.device.type != "cpu":
            raise RuntimeError("pin_memory() requires a CPU Tensor; use .cpu() explicitly first.")
        if self.is_pinned():
            return self if is_grad_enabled() or not self.requires_grad else self.detach()
        array = pinned_empty(self.shape, self.dtype.numpy_dtype)
        np.copyto(array, self._data)
        result = Tensor._from_array(array, self.requires_grad and is_grad_enabled())
        if result.requires_grad:
            result._is_leaf = False
            result._grad_fn = BackwardNode("pin_memory", (self,), identity_backward, Context())
        return result

    def clone(self) -> Tensor:
        result = Tensor(self, requires_grad=self.requires_grad and is_grad_enabled(), device=self.device)
        if result.requires_grad:
            result._is_leaf = False
            result._grad_fn = BackwardNode("clone", (self,), identity_backward, Context())
        return result

    def copy(self) -> Tensor:
        return self.clone()

    def detach(self) -> Tensor:
        result = Tensor._from_array(self._data.view(), False, self._version_counter)
        result._writable = self._writable
        return result

    def to(self, *args, device=None, dtype=None, copy=False, non_blocking=False) -> Tensor:
        if type(copy) is not builtins.bool:
            raise TypeError("copy must be a Python bool.")
        if type(non_blocking) is not builtins.bool:
            raise TypeError("non_blocking must be a Python bool.")
        if len(args) > 2:
            raise TypeError("to() accepts at most a device and a dtype as positional arguments.")
        if args:
            first = args[0]
            dtype_argument = isinstance(first, DType) or isinstance(first, np.dtype) or isinstance(first, type) or (isinstance(first, str) and first in {value.value for value in DType})
            if dtype_argument:
                if len(args) != 1 or dtype is not None:
                    raise TypeError("to() dtype must be specified only once.")
                dtype = first
            else:
                if device is not None:
                    raise TypeError("to() device must be specified only once.")
                device = first
                if len(args) == 2:
                    if dtype is not None:
                        raise TypeError("to() dtype must be specified only once.")
                    dtype = args[1]
        target_device = self.device if device is None else Device(device)
        target_dtype = self.dtype if dtype is None else normalize_dtype(dtype)
        if target_device == self.device and target_dtype is self.dtype and not copy:
            return self if is_grad_enabled() or not self.requires_grad else self.detach()
        array = transfer(self._data, target_device, target_dtype.numpy_dtype, copy=True, non_blocking=non_blocking)
        tracked = self.requires_grad and target_dtype.can_require_grad and is_grad_enabled()
        result = Tensor._from_array(array, tracked)
        if tracked:
            result._is_leaf = False
            result._grad_fn = BackwardNode("to", (self,), transfer_backward, Context(device=self.device, dtype=self.dtype.numpy_dtype), device_transfer=True)
        return result

    def cpu(self) -> Tensor:
        return self.to("cpu")

    def cuda(self, device=None, non_blocking=False, *, index=None) -> Tensor:
        if index is not None:
            if device is not None:
                raise TypeError("cuda() device index must be specified only once.")
            device = index
        target = Device(device) if isinstance(device, (Device, str)) else Device("cuda", device)
        if target.type != "cuda":
            raise ValueError("cuda() requires a CUDA device.")
        return self.to(device=target, non_blocking=non_blocking)

    def astype(self, dtype: object) -> Tensor:
        target = normalize_dtype(dtype)
        result = Tensor(self, dtype=target, requires_grad=self.requires_grad and target.can_require_grad and is_grad_enabled(), device=self.device)
        if result.requires_grad:
            result._is_leaf = False
            result._grad_fn = BackwardNode("astype", (self,), identity_backward, Context())
        return result

    def float(self) -> Tensor:
        return self.astype(float32)

    def double(self) -> Tensor:
        return self.astype(float64)

    def long(self) -> Tensor:
        return self.astype(int64)

    def int(self) -> Tensor:
        return self.astype(int32)

    def bool(self) -> Tensor:
        return self.astype(bool_dtype)

    def _operand(self, value: object) -> Tensor:
        if isinstance(value, Tensor):
            ensure_same_device(self, value)
            return value
        if isinstance(value, (builtins.bool, builtins.int, builtins.float, np.generic)):
            return Tensor(value, device=self.device)
        raise TypeError(
            f"Operators require Tensor, Python scalar or NumPy scalar operands, got {type(value).__name__}. "
            "Convert sequences and ndarrays explicitly with tensor() or as_tensor()."
        )

    def _operation_result(
        self,
        array: Array,
        operation: str,
        operands: tuple[Tensor, ...],
        compute_dtype: DType,
        differentiable: builtins.bool,
        parameters: tuple[tuple[str, object], ...] = (),
    ) -> Tensor:
        requires_grad = (
            differentiable
            and is_grad_enabled()
            and from_numpy_dtype(array.dtype).can_require_grad
            and any(operand.requires_grad for operand in operands)
        )
        result = Tensor._from_array(array, requires_grad)
        result._is_leaf = not requires_grad
        result._operation_context = _OperationContext(
            operation,
            tuple(operand.shape for operand in operands),
            tuple(operand.dtype for operand in operands),
            tuple(operand._version for operand in operands),
            compute_dtype,
            parameters,
        )
        if requires_grad and operation in ARITHMETIC_BACKWARD_RULES:
            result._grad_fn = arithmetic_node(operation, operands, compute_dtype)
        elif requires_grad and operation in ELEMENTWISE_BACKWARD_OPERATIONS:
            result._grad_fn = elementwise_node(operation, self, result, parameters)
        elif requires_grad and operation in SELECTION_BACKWARD_RULES:
            result._grad_fn = selection_node(operation, operands, compute_dtype, parameters)
        return result

    @same_device
    def _binary_operation(self, other: object, operation: str, *, reverse: builtins.bool = False) -> Tensor:
        xp = namespace(self)
        operand = self._operand(other)
        left, right = (operand, self) if reverse else (self, operand)
        cast_dtype = autocast_dtype(operation, left, right)
        if cast_dtype is not None:
            left, right = left.to(cast_dtype), right.to(cast_dtype)
        comparison = operation in ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal")
        promotion = "true_divide" if operation == "true_divide" else "arithmetic"
        compute_dtype = promote_types(left.dtype, right.dtype, operation=promotion)
        output_dtype = promote_types(left.dtype, right.dtype, operation="comparison") if comparison else compute_dtype
        if operation == "power" and right.requires_grad and is_grad_enabled() and not xp.all(left._data > 0):
            raise ValueError("Power requires a strictly positive base when the exponent requires_grad=True.")
        array = binary_forward(operation, left._data, right._data, compute_dtype, output_dtype)
        differentiable = not comparison and operation not in ("floor_divide", "remainder")
        return self._operation_result(array, operation, (left, right), compute_dtype, differentiable)

    @same_device
    def _unary_operation(self, operation: str) -> Tensor:
        array = unary_forward(operation, self._data, self.dtype)
        return self._operation_result(array, operation, (self,), self.dtype, True)

    @same_device
    def _elementwise(
        self, operation: str, parameter: object = None, approximation: object = None
    ) -> Tensor:
        if self.dtype is float16 and OPERATION_POLICIES.get(operation) == "float32":
            return self.to(float32)._elementwise(operation, parameter, approximation)
        dtype = self.dtype
        if not dtype.is_floating_point and operation not in ("square", "absolute", "sign", "relu"):
            dtype = get_default_dtype()
        parameters = ()
        if operation in ("leaky_relu", "elu"):
            name = "negative_slope" if operation == "leaky_relu" else "alpha"
            if isinstance(parameter, (builtins.bool, np.bool_)) or not isinstance(parameter, Real):
                raise TypeError(f"{name} must be a finite real scalar.")
            try:
                parameter = builtins.float(parameter)
            except OverflowError:
                raise ValueError(f"{name} must be finite and representable in {dtype.name}.") from None
            if not math.isfinite(parameter) or builtins.abs(parameter) > builtins.float(np.finfo(dtype.numpy_dtype).max):
                raise ValueError(f"{name} must be finite and representable in {dtype.name}.")
            parameters = ((name, parameter),)
        if operation == "gelu":
            if not isinstance(approximation, str):
                raise TypeError("GELU approximation must be 'exact' or 'tanh'.")
            if approximation not in ("exact", "tanh"):
                raise ValueError("GELU approximation must be 'exact' or 'tanh'.")
            parameters = (("approximation", approximation),)
        array = elementwise_forward(operation, self._data, dtype, parameter, approximation)
        return self._operation_result(array, operation, (self,), dtype, operation != "sign", parameters)

    def exp(self) -> Tensor:
        return self._elementwise("exp")

    def expm1(self) -> Tensor:
        return self._elementwise("expm1")

    def log(self) -> Tensor:
        return self._elementwise("log")

    def log1p(self) -> Tensor:
        return self._elementwise("log1p")

    def log2(self) -> Tensor:
        return self._elementwise("log2")

    def log10(self) -> Tensor:
        return self._elementwise("log10")

    def sqrt(self) -> Tensor:
        return self._elementwise("sqrt")

    def rsqrt(self) -> Tensor:
        return self._elementwise("rsqrt")

    def square(self) -> Tensor:
        return self._elementwise("square")

    def abs(self) -> Tensor:
        return self.__abs__()

    def sign(self) -> Tensor:
        return self._elementwise("sign")

    def sin(self) -> Tensor:
        return self._elementwise("sin")

    def cos(self) -> Tensor:
        return self._elementwise("cos")

    def tan(self) -> Tensor:
        return self._elementwise("tan")

    def tanh(self) -> Tensor:
        return self._elementwise("tanh")

    def sigmoid(self) -> Tensor:
        return self._elementwise("sigmoid")

    def relu(self) -> Tensor:
        return self._elementwise("relu")

    def leaky_relu(self, negative_slope: object = 0.01) -> Tensor:
        return self._elementwise("leaky_relu", parameter=negative_slope)

    def elu(self, alpha: object = 1.0) -> Tensor:
        return self._elementwise("elu", parameter=alpha)

    def gelu(self, approximation: str = "exact") -> Tensor:
        return self._elementwise("gelu", approximation=approximation)

    def silu(self) -> Tensor:
        return self._elementwise("silu")

    def softplus(self) -> Tensor:
        return self._elementwise("softplus")

    def softsign(self) -> Tensor:
        return self._elementwise("softsign")

    def maximum(self, other: object) -> Tensor:
        return self._binary_operation(other, "maximum")

    def minimum(self, other: object) -> Tensor:
        return self._binary_operation(other, "minimum")

    @same_device
    def clamp(self, min: object = None, max: object = None) -> Tensor:
        if min is None and max is None:
            raise ValueError("clamp requires at least one of min or max.")
        lower = None if min is None else self._operand(min)
        upper = None if max is None else self._operand(max)
        bounds = tuple(bound for bound in (lower, upper) if bound is not None)
        if any(bound.ndim != 0 for bound in bounds):
            raise TypeError("clamp bounds must be scalars or zero-dimensional Tensors.")
        if lower is not None and upper is not None:
            lower_value = min.item() if isinstance(min, Tensor) else min
            upper_value = max.item() if isinstance(max, Tensor) else max
            if lower_value > upper_value:
                raise ValueError("clamp requires min <= max.")
        dtype = self.dtype
        for bound in bounds:
            dtype = promote_types(dtype, bound.dtype)
        array = clamp_forward(self._data, None if lower is None else lower._data, None if upper is None else upper._data, dtype)
        parameters = (("has_min", lower is not None), ("has_max", upper is not None))
        return self._operation_result(array, "clamp", (self,) + bounds, dtype, True, parameters)

    @same_device
    def where(self, condition: object, other: object) -> Tensor:
        mask = self._operand(condition)
        if not mask.dtype.is_boolean:
            raise TypeError("where condition must have boolean dtype.")
        right = self._operand(other)
        dtype = promote_types(self.dtype, right.dtype)
        array = where_forward(mask._data, self._data, right._data, dtype)
        return self._operation_result(array, "where", (mask, self, right), dtype, True)

    @same_device
    def masked_fill(self, mask: object, value: object) -> Tensor:
        if is_array(mask):
            ensure_same_device(self, mask)
        condition = Tensor(mask, device=self.device) if is_array(mask) else self._operand(mask)
        if not condition.dtype.is_boolean:
            raise TypeError("masked_fill mask must have boolean dtype.")
        if broadcast_shapes(condition.shape, self.shape) != self.shape:
            raise ValueError(f"masked_fill mask shape {condition.shape} must broadcast to input shape {self.shape} without expanding it.")
        fill = self._operand(value)
        if fill.ndim != 0:
            raise ValueError("masked_fill value must be a scalar or a zero-dimensional Tensor.")
        return fill.where(condition, self)

    def __add__(self, other: object) -> Tensor:
        return self._binary_operation(other, "add")

    def __radd__(self, other: object) -> Tensor:
        return self._binary_operation(other, "add", reverse=True)

    def __sub__(self, other: object) -> Tensor:
        return self._binary_operation(other, "subtract")

    def __rsub__(self, other: object) -> Tensor:
        return self._binary_operation(other, "subtract", reverse=True)

    def __mul__(self, other: object) -> Tensor:
        return self._binary_operation(other, "multiply")

    def __rmul__(self, other: object) -> Tensor:
        return self._binary_operation(other, "multiply", reverse=True)

    def __truediv__(self, other: object) -> Tensor:
        return self._binary_operation(other, "true_divide")

    def __rtruediv__(self, other: object) -> Tensor:
        return self._binary_operation(other, "true_divide", reverse=True)

    def __floordiv__(self, other: object) -> Tensor:
        return self._binary_operation(other, "floor_divide")

    def __rfloordiv__(self, other: object) -> Tensor:
        return self._binary_operation(other, "floor_divide", reverse=True)

    def __mod__(self, other: object) -> Tensor:
        return self._binary_operation(other, "remainder")

    def __rmod__(self, other: object) -> Tensor:
        return self._binary_operation(other, "remainder", reverse=True)

    def __pow__(self, other: object) -> Tensor:
        return self._binary_operation(other, "power")

    def __rpow__(self, other: object) -> Tensor:
        return self._binary_operation(other, "power", reverse=True)

    def matmul(self, other: object) -> Tensor:
        return self._binary_operation(other, "matmul")

    def bmm(self, other: object) -> Tensor:
        operand = self._operand(other)
        if self.ndim != 3 or operand.ndim != 3:
            raise ValueError(f"bmm requires rank 3 operands, got shapes {self.shape} and {operand.shape}.")
        if self.shape[0] != operand.shape[0]:
            raise ValueError(f"bmm requires equal batch sizes, got shapes {self.shape} and {operand.shape}.")
        return self.matmul(operand)

    def dot(self, other: object) -> Tensor:
        operand = self._operand(other)
        if self.ndim != 1 or operand.ndim != 1:
            raise ValueError(f"dot requires rank 1 operands, got shapes {self.shape} and {operand.shape}.")
        return self.matmul(operand)

    def __matmul__(self, other: object) -> Tensor:
        return self.matmul(other)

    def __rmatmul__(self, other: object) -> Tensor:
        return self._binary_operation(other, "matmul", reverse=True)

    def __pos__(self) -> Tensor:
        return self._unary_operation("positive")

    def __neg__(self) -> Tensor:
        return self._unary_operation("negative")

    def __abs__(self) -> Tensor:
        return self._unary_operation("absolute")

    def __eq__(self, other: object) -> Tensor:
        return self._binary_operation(other, "equal")

    def __ne__(self, other: object) -> Tensor:
        return self._binary_operation(other, "not_equal")

    def __lt__(self, other: object) -> Tensor:
        return self._binary_operation(other, "less")

    def __le__(self, other: object) -> Tensor:
        return self._binary_operation(other, "less_equal")

    def __gt__(self, other: object) -> Tensor:
        return self._binary_operation(other, "greater")

    def __ge__(self, other: object) -> Tensor:
        return self._binary_operation(other, "greater_equal")

    def __bool__(self) -> builtins.bool:
        if self.numel() != 1:
            raise ValueError(
                f"The truth value of a Tensor with {self.numel()} elements is ambiguous; expected exactly one element."
            )
        return builtins.bool(self.item())

    @same_device
    def _reduce(
        self, operation: str, dim: object, keepdim: builtins.bool, correction: object = None
    ) -> Tensor:
        if self.dtype is float16 and OPERATION_POLICIES.get(operation) == "float32":
            return self.to(float32)._reduce(operation, dim, keepdim, correction)
        axes = normalize_reduction_axes(dim, self.ndim)
        if type(keepdim) is not builtins.bool:
            raise TypeError("keepdim must be a Python bool.")
        if operation in ("var", "std"):
            if isinstance(correction, (builtins.bool, np.bool_)) or not isinstance(correction, Real):
                raise TypeError("correction must be a finite nonnegative real number.")
            try:
                correction = builtins.float(correction)
            except OverflowError:
                raise ValueError("correction must be a finite nonnegative real number.") from None
            if not math.isfinite(correction) or correction < 0:
                raise ValueError("correction must be a finite nonnegative real number.")
        count = math.prod(self.shape[axis] for axis in axes)
        if operation in ("argmin", "argmax"):
            dtype = int64
        elif operation in ("all", "any"):
            dtype = bool_dtype
        elif operation in ("sum", "prod") and not self.dtype.is_floating_point:
            dtype = int64
        elif operation in ("mean", "var", "std", "logsumexp") and not self.dtype.is_floating_point:
            dtype = get_default_dtype()
        else:
            dtype = self.dtype
        array = reduction_forward(operation, self._data, axes, keepdim, dtype, count, correction)
        differentiable = operation not in ("argmin", "argmax", "all", "any")
        result = Tensor._from_array(array, self.requires_grad and dtype.can_require_grad and differentiable and is_grad_enabled())
        result._is_leaf = not result.requires_grad
        result._reduction_context = _ReductionContext(
            operation, self.shape, axes, keepdim, count, correction, self._version
        )
        if result.requires_grad and operation in REDUCTION_BACKWARD_RULES:
            result._grad_fn = reduction_node(operation, self, axes, keepdim, count, correction)
        return result

    def logsumexp(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("logsumexp", dim, keepdim)

    @same_device
    def _normalized_exponential(self, operation: str, dim: object) -> Tensor:
        if self.dtype is float16:
            return self.to(float32)._normalized_exponential(operation, dim)
        axis = normalize_axis(dim, self.ndim)
        dtype = self.dtype if self.dtype.is_floating_point else get_default_dtype()
        array = normalized_exponential_forward(operation, self._data.astype(dtype.numpy_dtype, copy=False), (axis,))
        result = Tensor._from_array(array, self.requires_grad and is_grad_enabled())
        result._is_leaf = not result.requires_grad
        if result.requires_grad:
            result._grad_fn = normalized_exponential_node(operation, self, result, axis)
        return result

    def softmax(self, dim: object) -> Tensor:
        return self._normalized_exponential("softmax", dim)

    def log_softmax(self, dim: object) -> Tensor:
        return self._normalized_exponential("log_softmax", dim)

    def sum(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("sum", dim, keepdim)

    def mean(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("mean", dim, keepdim)

    def prod(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("prod", dim, keepdim)

    def min(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("min", dim, keepdim)

    def max(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("max", dim, keepdim)

    def amin(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("amin", dim, keepdim)

    def amax(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("amax", dim, keepdim)

    def argmin(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("argmin", dim, keepdim)

    def argmax(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("argmax", dim, keepdim)

    def var(self, dim: object = None, keepdim: builtins.bool = False, *, correction: object = 1) -> Tensor:
        return self._reduce("var", dim, keepdim, correction)

    def std(self, dim: object = None, keepdim: builtins.bool = False, *, correction: object = 1) -> Tensor:
        return self._reduce("std", dim, keepdim, correction)

    def all(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("all", dim, keepdim)

    def any(self, dim: object = None, keepdim: builtins.bool = False) -> Tensor:
        return self._reduce("any", dim, keepdim)

    @same_device
    def _transform_result(
        self, array: Array, operation: str,
        permutation: tuple[builtins.int, ...] | None = None,
        *, axes: tuple[builtins.int, ...] = (),
        **metadata,
    ) -> Tensor:
        version = self._version_counter if _shares_storage(self._data, array) else None
        result = Tensor._from_array(array, self.requires_grad and is_grad_enabled(), version)
        result._is_leaf = not result.requires_grad
        result._writable = (self._writable if version is not None else writable(array)) and operation != "expand"
        result._transform_context = _TransformContext(operation, self.shape, permutation, axes)
        if result.requires_grad:
            result._grad_fn = transform_node(operation, self, permutation, axes, **metadata)
        return result

    @same_device
    def expand(self, *shape: object) -> Tensor:
        xp = namespace(self)
        dimensions = normalize_shape(*shape, allow_inferred=True)
        if len(dimensions) < self.ndim:
            raise ValueError(f"Cannot expand shape {self.shape} to a lower-rank shape {dimensions}.")
        leading = len(dimensions) - self.ndim
        original = (1,) * leading + self.shape
        resolved = []
        for axis, (source, target) in enumerate(zip(original, dimensions)):
            if target == -1:
                if axis < leading:
                    raise ValueError("expand -1 may only preserve an existing dimension.")
                target = source
            if source != target and source != 1:
                raise ValueError(f"Cannot expand non-singleton dimension from shape {self.shape} to {dimensions}.")
            resolved.append(target)
        return self._transform_result(xp.broadcast_to(self._data, tuple(resolved)), "expand")

    def broadcast_to(self, *shape: object) -> Tensor:
        return self.expand(*shape)

    @same_device
    def repeat(self, *repeats: object) -> Tensor:
        xp = namespace(self)
        counts = normalize_shape(*repeats)
        if len(counts) < self.ndim:
            raise ValueError(f"repeat requires at least {self.ndim} repetition counts.")
        return self._transform_result(xp.tile(self._data, counts), "repeat", repeats=counts)

    @same_device
    def repeat_interleave(self, repeats: object, dim: object = None) -> Tensor:
        xp = namespace(self)
        count = _integer_argument(repeats, "repeats", 0)
        if dim is None:
            return self.reshape(-1).repeat_interleave(count, 0)
        axis = normalize_axis(dim, self.ndim)
        return self._transform_result(xp.repeat(self._data, count, axis=axis), "repeat_interleave", repeats=count, axis=axis)

    def split(self, split_size_or_sections: object, dim: object = 0) -> tuple[Tensor, ...]:
        axis = normalize_axis(dim, self.ndim)
        size = self.shape[axis]
        if isinstance(split_size_or_sections, (list, tuple)):
            sections = tuple(_integer_argument(value, "split section", 0) for value in split_size_or_sections)
            if sum(sections) != size:
                raise ValueError(f"split sections must sum to dimension length {size}, got {sections}.")
        else:
            width = _integer_argument(split_size_or_sections, "split_size", 1)
            sections = tuple(min(width, size - start) for start in range(0, size, width)) if size else (0,)
        outputs = []
        start = 0
        for length in sections:
            index = (slice(None),) * axis + (slice(start, start + length),)
            outputs.append(self[index])
            start += length
        return tuple(outputs)

    def chunk(self, chunks: object, dim: object = 0) -> tuple[Tensor, ...]:
        count = _integer_argument(chunks, "chunks", 1)
        axis = normalize_axis(dim, self.ndim)
        base, remainder = divmod(self.shape[axis], count)
        return self.split(tuple(base + (index < remainder) for index in range(count)), axis)

    @same_device
    def tril(self, diagonal: object = 0) -> Tensor:
        if self.ndim < 2:
            raise ValueError("tril requires a Tensor with at least two dimensions.")
        offset = _integer_argument(diagonal, "diagonal")
        return self._transform_result(namespace(self).tril(self._data, k=offset), "tril", diagonal=offset)

    @same_device
    def triu(self, diagonal: object = 0) -> Tensor:
        if self.ndim < 2:
            raise ValueError("triu requires a Tensor with at least two dimensions.")
        offset = _integer_argument(diagonal, "diagonal")
        return self._transform_result(namespace(self).triu(self._data, k=offset), "triu", diagonal=offset)

    @same_device
    def reshape(self, *shape: object) -> Tensor:
        dimensions = normalize_reshape_shape(self.numel(), *shape)
        return self._transform_result(self._data.reshape(dimensions, order="C"), "reshape")

    @same_device
    def view(self, *shape: object) -> Tensor:
        xp = namespace(self)
        dimensions = normalize_reshape_shape(self.numel(), *shape)
        try:
            array = xp.reshape(self._data, dimensions, order="C", copy=False)
        except TypeError:
            array = self._data.reshape(dimensions, order="C")
        except ValueError:
            raise ValueError("view requires shared storage; use reshape or contiguous first.") from None
        if not _shares_storage(self._data, array):
            raise ValueError("view requires shared storage; use reshape or contiguous first.")
        return self._transform_result(array, "view")

    @same_device
    def flatten(self, start_dim: object = 0, end_dim: object = -1) -> Tensor:
        start = normalize_axis(start_dim, max(self.ndim, 1))
        end = normalize_axis(end_dim, max(self.ndim, 1))
        if start > end:
            raise ValueError("flatten requires start_dim <= end_dim.")
        dimensions = self.shape[:start] + (math.prod(self.shape[start:end + 1]),) + self.shape[end + 1:]
        return self._transform_result(self._data.reshape(dimensions, order="C"), "flatten")

    @same_device
    def ravel(self) -> Tensor:
        return self._transform_result(self._data.ravel(order="C"), "ravel")

    @same_device
    def squeeze(self, dim: object = None) -> Tensor:
        xp = namespace(self)
        if dim is None:
            axes = tuple(axis for axis, size in enumerate(self.shape) if size == 1)
            array = xp.squeeze(self._data)
        else:
            axis = normalize_axis(dim, max(self.ndim, 1))
            axes = (axis,) if self.ndim and self.shape[axis] == 1 else ()
            array = xp.squeeze(self._data, axis=axes) if axes else self._data.view()
        return self._transform_result(array, "squeeze", axes=axes)

    @same_device
    def unsqueeze(self, dim: object) -> Tensor:
        xp = namespace(self)
        axis = normalize_axis(dim, self.ndim + 1)
        return self._transform_result(xp.expand_dims(self._data, axis=axis), "unsqueeze", axes=(axis,))

    @same_device
    def transpose(self, dim0: object, dim1: object) -> Tensor:
        first, second = normalize_axis(dim0, self.ndim), normalize_axis(dim1, self.ndim)
        permutation = list(range(self.ndim))
        permutation[first], permutation[second] = permutation[second], permutation[first]
        axes = tuple(permutation)
        return self._transform_result(self._data.transpose(axes), "transpose", axes, axes=(first, second))

    @same_device
    def swapaxes(self, dim0: object, dim1: object) -> Tensor:
        return self.transpose(dim0, dim1)

    @same_device
    def moveaxis(self, source: object, destination: object) -> Tensor:
        sources = normalize_axes(source, self.ndim)
        destinations = normalize_axes(destination, self.ndim)
        if len(sources) != len(destinations):
            raise ValueError("moveaxis source and destination must contain the same number of axes.")
        permutation = [axis for axis in range(self.ndim) if axis not in sources]
        for destination_axis, source_axis in sorted(zip(destinations, sources)):
            permutation.insert(destination_axis, source_axis)
        axes = tuple(permutation)
        return self._transform_result(self._data.transpose(axes), "moveaxis", axes)

    @same_device
    def permute(self, *dims: object) -> Tensor:
        sequence = dims[0] if len(dims) == 1 and isinstance(dims[0], (list, tuple)) else dims
        axes = normalize_axes(sequence, self.ndim)
        if len(axes) != self.ndim:
            raise ValueError("permute requires every axis exactly once.")
        return self._transform_result(self._data.transpose(axes), "permute", axes)

    @property
    def T(self) -> Tensor:
        return self.permute(tuple(reversed(range(self.ndim))))

    def is_contiguous(self) -> builtins.bool:
        return self._data.flags.c_contiguous

    @same_device
    def contiguous(self) -> Tensor:
        if self.is_contiguous():
            if self.requires_grad and not is_grad_enabled():
                return self.detach()
            return self
        return self._transform_result(self._data.copy(order="C"), "contiguous")

    def _indexed_data(self, index: tuple[object, ...]) -> Array | np.generic:
        self._backend.validate_index(self._data, index)
        try:
            return self._data[index]
        except IndexError as error:
            raise IndexError(f"Invalid index for Tensor with shape {self.shape}: {error}") from None

    def _axis_index(self, dim: object, index: Tensor) -> tuple[Array, ...]:
        ensure_same_device(self, index)
        axis = normalize_axis(dim, self.ndim)
        if not isinstance(index, Tensor) or not index.dtype.is_integer:
            raise TypeError("index must be a NamiTorch Tensor with integer dtype.")
        return axis_index_coordinates(self.shape, axis, _index_array(index._data))

    @same_device
    def gather(self, dim: object, index: Tensor) -> Tensor:
        xp = namespace(self)
        coordinates = self._axis_index(dim, index)
        array = xp.asarray(self._data[coordinates])
        result = Tensor._from_array(array, self.requires_grad and is_grad_enabled())
        result._is_leaf = not result.requires_grad
        if result.requires_grad:
            result._grad_fn = indexing_node("gather", self, coordinates, True)
        return result

    @same_device
    def scatter_add(self, dim: object, index: Tensor, src: Tensor) -> Tensor:
        ensure_same_device(self, index, src)
        coordinates = self._axis_index(dim, index)
        if not isinstance(src, Tensor):
            raise TypeError("src must be a NamiTorch Tensor.")
        dtype = promote_types(self.dtype, src.dtype)
        array = scatter_add_forward(self._data, coordinates, src._data, index.shape, dtype)
        result = Tensor._from_array(array, is_grad_enabled() and (self.requires_grad or src.requires_grad))
        result._is_leaf = not result.requires_grad
        if result.requires_grad:
            result._grad_fn = scatter_add_node(self, src, coordinates)
        return result

    @same_device
    def __getitem__(self, index: object) -> Tensor:
        xp = namespace(self)
        normalized = _normalize_index(index, self.device)
        advanced = any(is_array(component) or isinstance(component, builtins.bool) for component in normalized)
        selected = self._indexed_data(normalized)
        if is_array(selected):
            array = selected
        elif advanced:
            array = xp.array(selected, dtype=self.dtype.numpy_dtype)
        else:
            array = _scalar_index_view(self._data, normalized)
        version = self._version_counter if _shares_storage(self._data, array) else None
        result = Tensor._from_array(array, self.requires_grad and is_grad_enabled(), version)
        result._writable = self._writable if version is not None else writable(array)
        result._is_leaf = not result.requires_grad
        result._index_context = _IndexContext(self.shape, normalized, advanced, self._version)
        if result.requires_grad:
            result._grad_fn = indexing_node("getitem", self, normalized, advanced)
        return result

    def _ensure_inplace_allowed(self) -> None:
        if self.requires_grad and is_grad_enabled():
            raise RuntimeError("In-place assignment is not allowed on a Tensor with requires_grad=True.")
        if not self._writable or not writable(self._data):
            raise RuntimeError("Cannot assign to read-only Tensor storage.")

    def zero_(self) -> Tensor:
        return self.fill_(0)

    def fill_(self, value: object) -> Tensor:
        self._ensure_inplace_allowed()
        scalar = value if isinstance(value, Tensor) else Tensor(value, dtype=self.dtype, device=self.device)
        if scalar.ndim != 0:
            raise ValueError("fill_ requires a scalar value or a zero-dimensional Tensor.")
        return self.copy_(scalar)

    def copy_(self, src: object) -> Tensor:
        self._ensure_inplace_allowed()
        if isinstance(src, Tensor) and src.requires_grad and is_grad_enabled():
            raise RuntimeError("copy_ from a Tensor with requires_grad=True must run inside no_grad().")
        self[...] = src
        return self

    @same_device
    def __setitem__(self, index: object, value: object) -> None:
        xp = namespace(self)
        self._ensure_inplace_allowed()
        ensure_same_device(self, value)
        normalized = _normalize_index(index, self.device)
        selected = self._indexed_data(normalized)
        values = Tensor(value, dtype=self.dtype, device=self.device)._data
        staged = xp.empty(selected.shape, dtype=self.dtype.numpy_dtype)
        staged[...] = values
        self._data[normalized] = staged
        self._version_counter.increment()

    def __len__(self) -> builtins.int:
        if self.ndim == 0:
            raise TypeError("len() is not defined for a scalar Tensor.")
        return self.shape[0]

    @same_device
    def __repr__(self) -> str:
        self._backend.ready_for_host(self._data)
        summarized = self.size > 1000
        display = self._data.reshape(-1) if summarized else self._data
        if summarized:
            display = namespace(self).concatenate((display[:3], display[-3:]))
        display = self._backend.to_numpy(display)
        values = np.array2string(
            display, separator=", ", threshold=5 if summarized else 1000, edgeitems=3 if not summarized else 2, max_line_width=80,
            precision=8, suppress_small=False, formatter={}, floatmode="maxprec_equal",
            sign="-", legacy=False,
        )
        shape = f", shape={self.shape}" if summarized or self.size == 0 else ""
        location = "" if self.device.type == "cpu" else f", device='{self.device}'"
        return f"tensor({values}{shape}, dtype={self.dtype.name}{location}, requires_grad={self.requires_grad})"


def tensor(data: object, dtype: object = None, requires_grad: builtins.bool = False, *, device=None) -> Tensor:
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, device=device)


def as_tensor(data: object, dtype: object = None, *, device=None) -> Tensor:
    target = _resolve_dtype(data, dtype)
    destination = get_backend(data).device if device is None and (is_array(data) or isinstance(data, Tensor)) else Device("cpu" if device is None else device)
    if isinstance(data, Tensor):
        if data.device != destination:
            return data.to(destination, target)
        if data.dtype is target:
            if data.requires_grad and not is_grad_enabled():
                return data.detach()
            return data
        return data.astype(target)
    return Tensor._from_array(_make_array(data, target, copy=False, device=destination), False)


__all__ = ["Tensor", "tensor", "as_tensor"]
