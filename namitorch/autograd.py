from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any as Array, Callable

from .backends import array_device, ensure_same_device, get_backend, is_array, namespace, transfer

from ._backend import elementwise_derivative, normalized_exponential_forward
from ._convolution import col2im, im2col
from ._pooling import avg_pool2d_backward, max_pool2d_backward


_grad_enabled = ContextVar("namitorch_grad_enabled", default=True)


def is_grad_enabled() -> bool:
    return _grad_enabled.get()


@contextmanager
def _grad_mode(mode: bool):
    token = _grad_enabled.set(mode)
    try:
        yield
    finally:
        _grad_enabled.reset(token)


def set_grad_enabled(mode: bool):
    if type(mode) is not bool:
        raise TypeError("mode must be a Python bool.")
    return _grad_mode(mode)


def no_grad():
    return set_grad_enabled(False)


def enable_grad():
    return set_grad_enabled(True)


class SavedArray:
    __slots__ = ("_array", "_counter", "_version", "_device")

    def __init__(self, array: Array, version_counter):
        if not is_array(array):
            raise TypeError("Saved backward values must be registered backend arrays.")
        self._array = array
        self._counter = version_counter
        self._version = version_counter.value
        self._device = array_device(array)

    @property
    def device(self):
        return self._device

    def validate(self) -> None:
        if array_device(self._array) != self._device:
            raise RuntimeError(f"A value saved for backward changed device; expected {self._device}.")
        if self._counter.value != self._version:
            raise RuntimeError(
                "A value saved for backward was modified in-place: "
                f"expected storage version {self._version}, got {self._counter.value}."
            )

    def unpack(self) -> Array:
        self.validate()
        return self._array


class Context:
    __slots__ = ("metadata", "_saved", "_released")

    def __init__(self, **metadata):
        self.metadata = metadata
        self._saved: tuple[SavedArray, ...] = ()
        self._released = False

    def _ensure_live(self) -> None:
        if self._released:
            raise RuntimeError("The backward graph has been freed; use retain_graph=True on the earlier backward call.")

    def save_array(self, array: Array, version_counter) -> None:
        self._ensure_live()
        self._saved += (SavedArray(array, version_counter),)

    @property
    def saved_arrays(self) -> tuple[Array, ...]:
        self._ensure_live()
        return tuple(value.unpack() for value in self._saved)

    def validate(self) -> None:
        self._ensure_live()
        for value in self._saved:
            value.validate()

    def release(self) -> None:
        self.metadata.clear()
        self._saved = ()
        self._released = True


class BackwardNode:
    __slots__ = ("name", "context", "_parents", "_backward", "_released", "_device_transfer")

    def __init__(
        self,
        name: str,
        parents: tuple,
        backward: Callable[[Context, Array], tuple[Array | None, ...]],
        context: Context,
        *, device_transfer: bool = False,
    ):
        if not parents or any(not parent.requires_grad for parent in parents):
            raise ValueError("Backward nodes require parents with requires_grad=True.")
        self.name = name
        self.context = context
        self._parents = tuple(parents)
        self._backward = backward
        self._released = False
        if type(device_transfer) is not bool:
            raise TypeError("device_transfer must be a Python bool.")
        self._device_transfer = device_transfer
        if not device_transfer:
            ensure_same_device(*parents)

    @property
    def parents(self) -> tuple:
        return self._parents

    @property
    def released(self) -> bool:
        return self._released

    def validate(self) -> None:
        if self._released:
            raise RuntimeError("The backward graph has been freed; use retain_graph=True on the earlier backward call.")
        self.context.validate()

    def apply(self, gradient: Array) -> tuple[Array | None, ...]:
        self.validate()
        if not is_array(gradient) or gradient.dtype.kind != "f":
            raise RuntimeError(f"Backward rule for {self.name} requires a floating backend gradient array.")
        if not self._device_transfer:
            ensure_same_device(gradient, *self.parents, *self.context.saved_arrays)
            _validate_metadata_device(self.context.metadata, array_device(gradient))
        with get_backend(gradient).context():
            contributions = self._backward(self.context, gradient)
        if not isinstance(contributions, tuple) or len(contributions) != len(self.parents):
            raise RuntimeError(f"Backward rule for {self.name} must return one contribution per parent.")
        return contributions

    def release(self) -> None:
        self.context.release()
        self._parents = ()
        self._backward = None
        self._released = True


def _validate_metadata_device(value, device):
    if is_array(value):
        if array_device(value) != device:
            raise RuntimeError(f"Saved backward metadata is on {array_device(value)}, expected {device}.")
    elif isinstance(value, dict):
        for item in value.values():
            _validate_metadata_device(item, device)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _validate_metadata_device(item, device)


def validate_gradient(parent, gradient, operation="gradient", *, exact_dtype=False) -> None:
    if not is_array(gradient):
        raise RuntimeError(f"{operation} must be a registered backend gradient array.")
    if gradient.shape != parent.shape:
        raise RuntimeError(f"{operation} has invalid gradient shape {gradient.shape}; expected {parent.shape}.")
    if not parent.dtype.can_require_grad or gradient.dtype.kind != "f" or gradient.dtype.itemsize not in (4, 8):
        raise RuntimeError(f"{operation} requires supported floating gradients and a floating parent, got {gradient.dtype} and {parent.dtype}.")
    ensure_same_device(parent, gradient)
    if exact_dtype and gradient.dtype != parent.dtype.numpy_dtype:
        raise RuntimeError(f"{operation} has gradient dtype {gradient.dtype}; expected {parent.dtype}.")


def validate_accumulation(parent, gradient) -> None:
    validate_gradient(parent, gradient, "Gradient accumulation")
    existing = parent.grad
    if existing is not None:
        array = getattr(existing, "_data", None)
        validate_gradient(parent, array, "Stored gradient", exact_dtype=True)
        if existing.requires_grad or existing.grad_fn is not None:
            raise RuntimeError("Stored gradients must be detached from autograd.")


def accumulate_gradient(parent, gradient, existing=None, operation="Gradient accumulation"):
    validate_gradient(parent, gradient, operation)
    xp = namespace(parent)
    converted = xp.asarray(gradient, dtype=parent.dtype.numpy_dtype)
    if existing is None:
        return xp.array(converted, copy=True, order="C")
    validate_gradient(parent, existing, "Stored gradient", exact_dtype=True)
    return xp.asarray(xp.add(existing, converted), dtype=parent.dtype.numpy_dtype)


def sum_to_shape(gradient: Array, shape: tuple[int, ...]) -> Array:
    xp = namespace(gradient)
    gradient = xp.asarray(gradient)
    leading = gradient.ndim - len(shape)
    if leading < 0 or any(
        expected != actual and expected != 1
        for expected, actual in zip(shape, gradient.shape[leading:])
    ):
        raise RuntimeError(f"Cannot reduce gradient shape {gradient.shape} to parent shape {shape}.")
    axes = tuple(range(leading)) + tuple(
        leading + axis for axis, size in enumerate(shape)
        if size == 1 and gradient.shape[leading + axis] != 1
    )
    if axes:
        gradient = gradient.sum(axis=axes, keepdims=True, dtype=gradient.dtype)
    return gradient.reshape(shape)


def addition_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    return tuple(sum_to_shape(gradient, shape) for shape in context.metadata["parent_shapes"])


def identity_backward(context: Context, gradient: Array) -> tuple[Array]:
    return (gradient,)


def transfer_backward(context: Context, gradient) -> tuple:
    return (transfer(gradient, context.metadata["device"], context.metadata["dtype"]),)


def subtraction_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    return tuple(
        sum_to_shape(gradient if index == 0 else -gradient, shape)
        for index, shape in zip(context.metadata["parent_indices"], context.metadata["parent_shapes"])
    )


def negation_backward(context: Context, gradient: Array) -> tuple[Array]:
    return (sum_to_shape(-gradient, context.metadata["parent_shapes"][0]),)


def _saved_operands(context: Context) -> dict[int, Array]:
    return {
        index: array.astype(context.metadata["compute_dtype"], copy=False)
        for index, array in zip(context.metadata["saved_indices"], context.saved_arrays)
    }


def conv2d_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    xp = namespace(gradient)
    metadata = context.metadata
    operands = _saved_operands(context)
    batch, channels, _, _ = metadata["input_shape"]
    out_channels, _, kh, kw = metadata["weight_shape"]
    groups = metadata["groups"]
    oh, ow = metadata["output_spatial"]
    inner = channels // groups * kh * kw
    upstream = gradient.astype(metadata["compute_dtype"], copy=False).reshape(batch, groups, out_channels // groups, oh * ow)
    contributions = []
    for index in metadata["parent_indices"]:
        if index == 0:
            kernels = operands[1].reshape(groups, out_channels // groups, inner)
            columns = xp.matmul(kernels.swapaxes(-2, -1), upstream).reshape(batch, channels, kh, kw, oh, ow)
            value = col2im(columns, metadata["input_shape"], metadata["stride"], metadata["padding"], metadata["dilation"])
        elif index == 1:
            columns = im2col(operands[0], (kh, kw), metadata["stride"], metadata["padding"], metadata["dilation"], (oh, ow))
            columns = columns.reshape(batch, groups, inner, oh * ow)
            value = xp.matmul(upstream, columns.swapaxes(-2, -1)).sum(axis=0, dtype=upstream.dtype).reshape(metadata["weight_shape"])
        else:
            value = upstream.sum(axis=(0, 3), dtype=upstream.dtype).reshape(out_channels)
        contributions.append(value)
    return tuple(contributions)


def conv2d_node(input, weight, bias, stride, padding, dilation, groups, output_spatial, compute_dtype) -> BackwardNode:
    operands = (input, weight, bias)
    parent_indices = tuple(index for index, value in enumerate(operands) if value is not None and value.requires_grad)
    saved_indices = tuple(index for index, needed in ((0, weight.requires_grad), (1, input.requires_grad)) if needed)
    context = Context(
        input_shape=input.shape, weight_shape=weight.shape, stride=stride, padding=padding,
        dilation=dilation, groups=groups, output_spatial=output_spatial, compute_dtype=compute_dtype,
        parent_indices=parent_indices, saved_indices=saved_indices,
    )
    for index in saved_indices:
        context.save_array(operands[index]._data, operands[index]._version_counter)
    return BackwardNode("conv2d", tuple(operands[index] for index in parent_indices), conv2d_backward, context)


def pooling_backward(context: Context, gradient: Array) -> tuple[Array]:
    metadata = context.metadata
    if metadata["operation"] == "max_pool2d":
        result = max_pool2d_backward(gradient, metadata["input_shape"], metadata["indices"])
    else:
        result = avg_pool2d_backward(gradient, metadata["input_shape"], metadata["kernel_size"], metadata["stride"], metadata["padding"], metadata["count_include_pad"])
    return (result,)


def pooling_node(operation, input, **metadata) -> BackwardNode:
    context = Context(operation=operation, input_shape=input.shape, **metadata)
    return BackwardNode(operation, (input,), pooling_backward, context)


def multiplication_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    operands = _saved_operands(context)
    return tuple(
        sum_to_shape(gradient * operands[1 - index], shape)
        for index, shape in zip(context.metadata["parent_indices"], context.metadata["parent_shapes"])
    )


def division_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    operands = _saved_operands(context)
    denominator = operands[1]
    contributions = []
    for index, shape in zip(context.metadata["parent_indices"], context.metadata["parent_shapes"]):
        if index == 0:
            local = gradient / denominator
        else:
            local = -(gradient * operands[0]) / (denominator * denominator)
        contributions.append(sum_to_shape(local, shape))
    return tuple(contributions)


def power_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    xp = namespace(gradient)
    operands = _saved_operands(context)
    exponent = operands[1]
    contributions = []
    for index, shape in zip(context.metadata["parent_indices"], context.metadata["parent_shapes"]):
        if index == 0:
            if 0 not in operands:
                derivative = exponent
            else:
                derivative = xp.zeros_like(gradient)
                xp.power(operands[0], exponent - 1, out=derivative, where=exponent != 0)
                derivative *= exponent
            local = gradient * derivative
        else:
            base = operands[0]
            local = gradient * xp.power(base, exponent) * xp.log(base)
        contributions.append(sum_to_shape(local, shape))
    return tuple(contributions)


def matmul_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    xp = namespace(gradient)
    operands = _saved_operands(context)
    left_vector, right_vector = context.metadata["vector_operands"]
    if right_vector:
        gradient = xp.expand_dims(gradient, axis=-1)
    if left_vector:
        gradient = xp.expand_dims(gradient, axis=-2)
    contributions = []
    for index, shape in zip(context.metadata["parent_indices"], context.metadata["parent_shapes"]):
        if index == 0:
            right = operands[1]
            if right_vector:
                right = xp.expand_dims(right, axis=-1)
            local = xp.matmul(gradient, right.swapaxes(-1, -2))
            if left_vector:
                local = xp.squeeze(local, axis=-2)
        else:
            left = operands[0]
            if left_vector:
                left = xp.expand_dims(left, axis=-2)
            local = xp.matmul(left.swapaxes(-1, -2), gradient)
            if right_vector:
                local = xp.squeeze(local, axis=-1)
        contributions.append(sum_to_shape(local, shape))
    return tuple(contributions)


ARITHMETIC_BACKWARD_RULES = MappingProxyType({
    "add": addition_backward,
    "positive": identity_backward,
    "subtract": subtraction_backward,
    "multiply": multiplication_backward,
    "true_divide": division_backward,
    "negative": negation_backward,
    "power": power_backward,
    "matmul": matmul_backward,
})


def arithmetic_node(operation: str, operands: tuple, compute_dtype) -> BackwardNode:
    xp = namespace(operands[0])
    indices = tuple(index for index, operand in enumerate(operands) if operand.requires_grad)
    parents = tuple(operands[index] for index in indices)
    context = Context(parent_shapes=tuple(parent.shape for parent in parents))
    if operation not in ("add", "negative"):
        context.metadata["parent_indices"] = indices
    if operation == "matmul":
        context.metadata["vector_operands"] = tuple(operand.ndim == 1 for operand in operands)
    if operation in ("multiply", "matmul"):
        saved_indices = tuple(sorted({1 - index for index in indices}))
    elif operation == "true_divide":
        saved_indices = (0, 1) if 1 in indices else (1,)
    elif operation == "power":
        exponent = operands[1]._data
        constant_derivative = indices == (0,) and xp.all((exponent == 0) | (exponent == 1))
        saved_indices = (1,) if constant_derivative else (0, 1)
    else:
        saved_indices = ()
    if saved_indices:
        context.metadata["compute_dtype"] = compute_dtype.numpy_dtype
        context.metadata["saved_indices"] = saved_indices
        for index in saved_indices:
            operand = operands[index]
            context.save_array(operand._data, operand._version_counter)
    return BackwardNode(operation, parents, ARITHMETIC_BACKWARD_RULES[operation], context)


def reshape_backward(context: Context, gradient: Array) -> tuple[Array]:
    return (gradient.reshape(context.metadata["original_shape"], order="C"),)


def squeeze_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    return (xp.expand_dims(gradient, axis=context.metadata["removed_axes"]),)


def unsqueeze_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    return (xp.squeeze(gradient, axis=context.metadata["added_axis"]),)


def transpose_backward(context: Context, gradient: Array) -> tuple[Array]:
    first, second = context.metadata["axes"]
    return (gradient.swapaxes(first, second),)


def permutation_backward(context: Context, gradient: Array) -> tuple[Array]:
    permutation = context.metadata["permutation"]
    inverse = [0] * len(permutation)
    for output_axis, input_axis in enumerate(permutation):
        inverse[input_axis] = output_axis
    return (gradient.transpose(tuple(inverse)),)


def expand_backward(context: Context, gradient: Array) -> tuple[Array]:
    return (sum_to_shape(gradient, context.metadata["original_shape"]),)


def repeat_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    shape = context.metadata["original_shape"]
    repeats = context.metadata["repeats"]
    padded_shape = (1,) * (len(repeats) - len(shape)) + shape
    alternating = tuple(size for pair in zip(repeats, padded_shape) for size in pair)
    axes = tuple(range(0, len(alternating), 2))
    contribution = gradient.reshape(alternating).sum(axis=axes, dtype=gradient.dtype)
    return (xp.asarray(contribution).reshape(shape),)


def repeat_interleave_backward(context: Context, gradient: Array) -> tuple[Array]:
    shape = context.metadata["original_shape"]
    axis = context.metadata["axis"]
    expanded = shape[:axis + 1] + (context.metadata["repeats"],) + shape[axis + 1:]
    return (gradient.reshape(expanded).sum(axis=axis + 1, dtype=gradient.dtype),)


def triangular_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    function = xp.triu if context.metadata["upper"] else xp.tril
    return (function(gradient, k=context.metadata["diagonal"]),)


TRANSFORM_BACKWARD_RULES = MappingProxyType({
    "reshape": reshape_backward,
    "view": reshape_backward,
    "flatten": reshape_backward,
    "ravel": reshape_backward,
    "squeeze": squeeze_backward,
    "unsqueeze": unsqueeze_backward,
    "transpose": transpose_backward,
    "permute": permutation_backward,
    "moveaxis": permutation_backward,
    "contiguous": identity_backward,
    "expand": expand_backward,
    "repeat": repeat_backward,
    "repeat_interleave": repeat_interleave_backward,
    "tril": triangular_backward,
    "triu": triangular_backward,
})


def transform_node(operation: str, parent, permutation: tuple[int, ...] | None, axes: tuple[int, ...], **metadata) -> BackwardNode:
    if operation in ("reshape", "view", "flatten", "ravel"):
        context = Context(original_shape=parent.shape)
    elif operation in ("expand", "repeat", "repeat_interleave"):
        context = Context(original_shape=parent.shape, **metadata)
    elif operation in ("tril", "triu"):
        context = Context(upper=operation == "triu", **metadata)
    elif operation == "squeeze":
        context = Context(removed_axes=axes)
    elif operation == "unsqueeze":
        context = Context(added_axis=axes[0])
    elif operation == "transpose":
        context = Context(axes=axes)
    elif operation in ("permute", "moveaxis"):
        context = Context(permutation=permutation)
    else:
        context = Context()
    return BackwardNode(operation, (parent,), TRANSFORM_BACKWARD_RULES[operation], context)


def _broadcast_reduction_gradient(context: Context, gradient: Array) -> Array:
    xp = namespace(gradient)
    if not context.metadata["keepdim"]:
        gradient = xp.expand_dims(gradient, axis=context.metadata["axes"])
    return xp.broadcast_to(gradient, context.metadata["original_shape"])


def sum_backward(context: Context, gradient: Array) -> tuple[Array]:
    return (_broadcast_reduction_gradient(context, gradient),)


def logsumexp_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    if not context.metadata["axes"]:
        return (gradient,)
    value, = context.saved_arrays
    weights = normalized_exponential_forward("softmax", value, context.metadata["axes"])
    expanded = _broadcast_reduction_gradient(context, gradient)
    output = xp.zeros_like(value)
    with xp.errstate(under="ignore"):
        xp.multiply(expanded, weights, out=output, where=weights != 0)
    return (output,)


def normalized_exponential_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    output, = context.saved_arrays
    axis = context.metadata["axis"]
    with xp.errstate(under="ignore"):
        if context.metadata["operation"] == "softmax":
            dot = xp.sum(gradient * output, axis=axis, keepdims=True, dtype=gradient.dtype)
            contribution = output * (gradient - dot)
        else:
            total = xp.sum(gradient, axis=axis, keepdims=True, dtype=gradient.dtype)
            contribution = gradient - xp.exp(output) * total
            xp.copyto(contribution, 0, where=xp.all(xp.isneginf(output), axis=axis, keepdims=True))
    return (xp.asarray(contribution),)


def normalized_exponential_node(operation: str, parent, output, axis: int) -> BackwardNode:
    context = Context(operation=operation, axis=axis)
    context.save_array(output._data, output._version_counter)
    return BackwardNode(operation, (parent,), normalized_exponential_backward, context)


def mean_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    count = xp.asarray(context.metadata["count"], dtype=gradient.dtype)
    return (xp.asarray(_broadcast_reduction_gradient(context, gradient) / count),)


def product_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    data, = context.saved_arrays
    axes = context.metadata["axes"]
    zeros = data == 0
    zero_count = xp.sum(zeros, axis=axes, keepdims=True)
    nonzero_product = xp.prod(xp.where(zeros, 1, data), axis=axes, keepdims=True, dtype=data.dtype)
    derivative = xp.zeros_like(data)
    xp.divide(nonzero_product, data, out=derivative, where=zero_count == 0)
    xp.copyto(derivative, xp.broadcast_to(nonzero_product, data.shape), where=(zero_count == 1) & zeros)
    return (xp.asarray(_broadcast_reduction_gradient(context, gradient) * derivative),)


def extremum_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    data, = context.saved_arrays
    axes = context.metadata["axes"]
    function = xp.min if context.metadata["operation"] in ("min", "amin") else xp.max
    extremum = function(data, axis=axes, keepdims=True)
    mask = data == extremum
    count = xp.sum(mask, axis=axes, keepdims=True)
    derivative = xp.full_like(data, float("nan"))
    xp.divide(mask, count, out=derivative, where=count != 0)
    return (xp.asarray(_broadcast_reduction_gradient(context, gradient) * derivative),)


def moment_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    data, = context.saved_arrays
    axes = context.metadata["axes"]
    count = context.metadata["count"]
    correction = context.metadata["correction"]
    if count - correction <= 0:
        raise ValueError(f"{context.metadata['operation']} requires N - correction > 0, got N={count} and correction={correction}.")
    denominator = xp.asarray(count - correction, dtype=data.dtype)
    center = xp.sum(data, axis=axes, keepdims=True, dtype=data.dtype) / xp.asarray(count, dtype=data.dtype)
    centered = xp.asarray(data - center)
    if context.metadata["operation"] == "var":
        derivative = 2 * centered / denominator
    else:
        variance = xp.sum(centered * centered, axis=axes, keepdims=True, dtype=data.dtype) / denominator
        deviation = xp.sqrt(variance)
        derivative = xp.zeros_like(data)
        xp.divide(centered, denominator * deviation, out=derivative, where=deviation != 0)
    return (xp.asarray(_broadcast_reduction_gradient(context, gradient) * derivative),)


REDUCTION_BACKWARD_RULES = MappingProxyType({
    "logsumexp": logsumexp_backward,
    "sum": sum_backward,
    "mean": mean_backward,
    "prod": product_backward,
    "min": extremum_backward,
    "max": extremum_backward,
    "amin": extremum_backward,
    "amax": extremum_backward,
    "var": moment_backward,
    "std": moment_backward,
})


def reduction_node(operation: str, parent, axes: tuple[int, ...], keepdim: bool, count: int, correction) -> BackwardNode:
    context = Context(original_shape=parent.shape, axes=axes, keepdim=keepdim)
    if operation in ("mean", "var", "std"):
        context.metadata["count"] = count
    if operation in ("min", "max", "amin", "amax", "var", "std"):
        context.metadata["operation"] = operation
    if operation in ("var", "std"):
        context.metadata["correction"] = correction
    if operation not in ("sum", "mean") and not (operation == "logsumexp" and not axes):
        context.save_array(parent._data, parent._version_counter)
    return BackwardNode(operation, (parent,), REDUCTION_BACKWARD_RULES[operation], context)


ELEMENTWISE_BACKWARD_OPERATIONS = frozenset((
    "exp", "expm1", "log", "log1p", "log2", "log10", "sqrt", "rsqrt", "square",
    "absolute", "sin", "cos", "tan", "tanh", "sigmoid", "relu", "leaky_relu",
    "elu", "gelu", "silu", "softplus", "softsign",
))

_OUTPUT_DERIVATIVES = frozenset(("exp", "sqrt", "rsqrt", "tanh", "sigmoid"))


def elementwise_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    value, = context.saved_arrays
    derivative = elementwise_derivative(context.metadata["operation"], value, context.metadata)
    return (xp.asarray(gradient * derivative),)


def elementwise_node(operation: str, parent, output, parameters: tuple) -> BackwardNode:
    context = Context(operation=operation, **dict(parameters))
    saved = output if operation in _OUTPUT_DERIVATIVES else parent
    context.save_array(saved._data, saved._version_counter)
    return BackwardNode(operation, (parent,), elementwise_backward, context)


def binary_extremum_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    xp = namespace(gradient)
    operands = _saved_operands(context)
    left, right = operands[0], operands[1]
    comparison = xp.greater if context.metadata["operation"] == "maximum" else xp.less
    equal = left == right
    invalid = xp.isnan(left) | xp.isnan(right)
    contributions = []
    for index, shape in zip(context.metadata["parent_indices"], context.metadata["parent_shapes"]):
        first, second = (left, right) if index == 0 else (right, left)
        local = xp.where(comparison(first, second), gradient, xp.where(equal, gradient * 0.5, 0))
        contributions.append(sum_to_shape(xp.where(invalid, float("nan"), local), shape))
    return tuple(contributions)


def where_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    xp = namespace(gradient)
    condition, = context.saved_arrays
    return tuple(
        sum_to_shape(xp.where(condition, gradient, 0) if index == 1 else xp.where(condition, 0, gradient), shape)
        for index, shape in zip(context.metadata["parent_indices"], context.metadata["parent_shapes"])
    )


def clamp_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    xp = namespace(gradient)
    operands = _saved_operands(context)
    value = operands[0]
    lower_index = 1 if context.metadata["has_min"] else None
    upper_index = 1 + int(context.metadata["has_min"]) if context.metadata["has_max"] else None
    interior = xp.ones(value.shape, dtype=bool)
    invalid = xp.isnan(value)
    if lower_index is not None:
        interior &= value > operands[lower_index]
        invalid |= xp.isnan(operands[lower_index])
    if upper_index is not None:
        interior &= value < operands[upper_index]
        invalid |= xp.isnan(operands[upper_index])
    contributions = []
    for index, shape in zip(context.metadata["parent_indices"], context.metadata["parent_shapes"]):
        if index == 0:
            mask = interior
        elif index == lower_index:
            mask = value <= operands[index]
        else:
            mask = value >= operands[index]
        local = xp.where(mask, gradient, 0)
        if index != 0 and lower_index is not None and upper_index is not None:
            shared_boundary = (value == operands[lower_index]) & (value == operands[upper_index])
            local = xp.where(shared_boundary, gradient * 0.5, local)
        local = xp.where(invalid, float("nan"), local)
        contributions.append(sum_to_shape(local, shape))
    return tuple(contributions)


SELECTION_BACKWARD_RULES = MappingProxyType({
    "maximum": binary_extremum_backward,
    "minimum": binary_extremum_backward,
    "clamp": clamp_backward,
    "where": where_backward,
})


def selection_node(operation: str, operands: tuple, compute_dtype, parameters: tuple) -> BackwardNode:
    indices = tuple(index for index, operand in enumerate(operands) if operand.requires_grad)
    parents = tuple(operands[index] for index in indices)
    context = Context(
        operation=operation, parent_indices=indices, parent_shapes=tuple(parent.shape for parent in parents),
        **dict(parameters),
    )
    saved_indices = (0,) if operation == "where" else tuple(range(len(operands)))
    if operation != "where":
        context.metadata["compute_dtype"] = compute_dtype.numpy_dtype
        context.metadata["saved_indices"] = saved_indices
    for index in saved_indices:
        context.save_array(operands[index]._data, operands[index]._version_counter)
    return BackwardNode(operation, parents, SELECTION_BACKWARD_RULES[operation], context)


def stack_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    xp = namespace(gradient)
    return tuple(xp.asarray(xp.take(gradient, index, axis=context.metadata["axis"])) for index in context.metadata["indices"])


def stack_node(operands: tuple, axis: int) -> BackwardNode:
    indices = tuple(index for index, operand in enumerate(operands) if operand.requires_grad)
    context = Context(axis=axis, indices=indices)
    return BackwardNode("stack", tuple(operands[index] for index in indices), stack_backward, context)


def cat_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    axis = context.metadata["axis"]
    contributions = []
    for start, stop in context.metadata["segments"]:
        index = (slice(None),) * axis + (slice(start, stop),)
        contributions.append(gradient[index])
    return tuple(contributions)


def cat_node(operands: tuple, axis: int) -> BackwardNode:
    parents, segments = [], []
    offset = 0
    for operand in operands:
        end = offset + operand.shape[axis]
        if operand.requires_grad:
            parents.append(operand)
            segments.append((offset, end))
        offset = end
    return BackwardNode("cat", tuple(parents), cat_backward, Context(axis=axis, segments=tuple(segments)))


def indexing_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    contribution = xp.zeros(context.metadata["original_shape"], dtype=gradient.dtype)
    index = context.metadata["index"]
    if context.metadata["is_advanced"]:
        xp.add.at(contribution, index, gradient)
    else:
        contribution[index] += gradient
    return (contribution,)


def indexing_node(operation: str, parent, index: tuple, is_advanced: bool) -> BackwardNode:
    context = Context(original_shape=parent.shape, index=index, is_advanced=is_advanced)
    return BackwardNode(operation, (parent,), indexing_backward, context)


def embedding_backward(context: Context, gradient: Array) -> tuple[Array]:
    xp = namespace(gradient)
    shape = context.metadata["weight_shape"]
    indices = context.metadata["indices"].reshape(-1)
    values = gradient.reshape(-1, shape[1])
    padding = context.metadata["padding_idx"]
    if padding is not None:
        selected = indices != padding
        indices, values = indices[selected], values[selected]
    contribution = xp.zeros(shape, dtype=gradient.dtype)
    xp.add.at(contribution, indices, values)
    return (contribution,)


def embedding_node(weight, indices: Array, padding_idx: int | None) -> BackwardNode:
    context = Context(weight_shape=weight.shape, indices=indices, padding_idx=padding_idx)
    return BackwardNode("embedding", (weight,), embedding_backward, context)


def scatter_add_backward(context: Context, gradient: Array) -> tuple[Array, ...]:
    xp = namespace(gradient)
    return tuple(
        gradient if position == 0 else sum_to_shape(xp.asarray(gradient[context.metadata["index"]]), shape)
        for position, shape in zip(context.metadata["parent_indices"], context.metadata["parent_shapes"])
    )


def scatter_add_node(input, source, coordinates: tuple[Array, ...]) -> BackwardNode:
    operands = (input, source)
    positions = tuple(position for position, operand in enumerate(operands) if operand.requires_grad)
    parents = tuple(operands[position] for position in positions)
    context = Context(parent_indices=positions, parent_shapes=tuple(parent.shape for parent in parents))
    if source.requires_grad:
        context.metadata["index"] = coordinates
    return BackwardNode("scatter_add", parents, scatter_add_backward, context)


def _topological_order(output) -> list:
    order = []
    states: dict[int, int] = {}
    pending = [(output, False)]
    while pending:
        value, exiting = pending.pop()
        identity = id(value)
        if exiting:
            states[identity] = 2
            order.append(value)
            continue
        state = states.get(identity, 0)
        if state == 2:
            continue
        if state == 1:
            raise RuntimeError("The backward graph contains a cycle.")
        node = value.grad_fn
        if node is not None:
            node.validate()
        elif not value.is_leaf:
            raise RuntimeError("This operation has no registered backward rule.")
        states[identity] = 1
        pending.append((value, True))
        if node is not None:
            pending.extend((parent, False) for parent in reversed(node.parents))
    return order


def run_backward(output, gradient: Array, retain_graph: bool) -> None:
    with no_grad():
        validate_gradient(output, gradient, "Backward seed")
        order = _topological_order(output)
        gradients = {id(output): gradient}
        accumulated = []
        for value in reversed(order):
            incoming = gradients.pop(id(value), None)
            if incoming is None:
                continue
            if value.requires_grad and (value.is_leaf or value._retain_grad):
                accumulated.append((value, incoming))
            node = value.grad_fn
            if node is None:
                continue
            for parent, contribution in zip(node.parents, node.apply(incoming)):
                if contribution is None:
                    continue
                identity = id(parent)
                gradients[identity] = accumulate_gradient(parent, contribution, gradients.get(identity), f"Backward rule for {node.name}")
        for value, incoming in accumulated:
            validate_accumulation(value, incoming)
        for value, incoming in accumulated:
            value._accumulate_grad(incoming)
        if not retain_graph:
            for value in reversed(order):
                if value.grad_fn is not None:
                    value.grad_fn.release()
