import math
import operator
from numbers import Real

import numpy as np

from ..backends import get_backend, namespace, readonly, same_device, writable, try_fused, try_fused_softmax

from .._convolution import conv2d_forward
from .._pooling import avg_pool2d_forward, max_pool2d_forward
from ..autograd import conv2d_node, embedding_node, fused_combine_node, is_grad_enabled, no_grad, normalized_exponential_node, pooling_node
from ..dtype import result_type
from ..random import Generator, _get_rng, _validate_generator
from ..tensor import Tensor
from ..utils import broadcast_shapes


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean.")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer.") from None


def _positive_dimension(value: object, name: str) -> int:
    dimension = _integer(value, name)
    if dimension <= 0:
        raise ValueError(f"{name} must be positive, got {dimension}.")
    return dimension


def _padding_index(padding_idx: object, num_embeddings: int) -> int | None:
    if padding_idx is None:
        return None
    index = _integer(padding_idx, "padding_idx")
    if not -num_embeddings <= index < num_embeddings:
        raise IndexError(f"padding_idx must be in [{-num_embeddings}, {num_embeddings}), got {index}.")
    return index % num_embeddings


@same_device
def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    if not isinstance(input, Tensor) or not isinstance(weight, Tensor):
        raise TypeError("linear input and weight must be NamiTorch Tensors.")
    if weight.ndim != 2:
        raise ValueError(f"linear weight must have rank 2, got shape {weight.shape}.")
    if input.ndim < 1 or input.shape[-1] != weight.shape[1]:
        raise ValueError(f"linear input must have rank >= 1 and last dimension {weight.shape[1]}, got shape {input.shape}.")
    if bias is not None:
        if not isinstance(bias, Tensor):
            raise TypeError("linear bias must be a NamiTorch Tensor or None.")
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"linear bias must have shape {(weight.shape[0],)}, got {bias.shape}.")
    matrix = input.reshape(math.prod(input.shape[:-1]), input.shape[-1]) if input.ndim > 2 else input
    output = matrix @ weight.transpose(-2, -1)
    if bias is not None:
        output = output + bias
    return output.reshape(*input.shape[:-1], weight.shape[0]) if input.ndim > 2 else output


def _spatial_tuple(value, dimensions, name, minimum):
    values = tuple(value) if isinstance(value, (tuple, list)) else (value,) * dimensions
    if len(values) != dimensions:
        raise ValueError(f"{name} must contain exactly {dimensions} integers.")
    normalized = tuple(_integer(item, name) for item in values)
    if any(item < minimum for item in normalized):
        raise ValueError(f"{name} values must be at least {minimum}, got {normalized}.")
    return normalized


def _convolution_arguments(input, weight, bias, stride, padding, dilation, groups, dimensions):
    for name, value in (("input", input), ("weight", weight)):
        if not isinstance(value, Tensor) or not value.dtype.is_floating_point:
            raise TypeError(f"Convolution {name} must be a floating NamiTorch Tensor.")
        if value.ndim != dimensions + 2:
            raise ValueError(f"Convolution {name} must have rank {dimensions + 2}, got shape {value.shape}.")
    if any(size <= 0 for size in input.shape[1:]) or any(size <= 0 for size in weight.shape):
        raise ValueError("Convolution channel, spatial and kernel dimensions must be positive.")
    stride = _spatial_tuple(stride, dimensions, "stride", 1)
    padding = _spatial_tuple(padding, dimensions, "padding", 0)
    dilation = _spatial_tuple(dilation, dimensions, "dilation", 1)
    groups = _positive_dimension(groups, "groups")
    if input.shape[1] % groups or weight.shape[0] % groups:
        raise ValueError("Convolution input and output channels must be divisible by groups.")
    if weight.shape[1] != input.shape[1] // groups:
        raise ValueError(f"Convolution input shape {input.shape} and weight shape {weight.shape} disagree for groups={groups}.")
    if bias is not None:
        if not isinstance(bias, Tensor) or not bias.dtype.is_floating_point:
            raise TypeError("Convolution bias must be a floating NamiTorch Tensor or None.")
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"Convolution bias must have shape {(weight.shape[0],)}, got {bias.shape}.")
    output_spatial = tuple(
        (size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
        for size, kernel, step, pad, dil in zip(input.shape[2:], weight.shape[2:], stride, padding, dilation)
    )
    if any(size <= 0 for size in output_spatial):
        raise ValueError(f"Convolution output spatial dimensions must be positive, got {output_spatial} for input {input.shape} and weight {weight.shape}.")
    return stride, padding, dilation, groups, output_spatial


@same_device
def conv2d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    stride, padding, dilation, groups, output_spatial = _convolution_arguments(input, weight, bias, stride, padding, dilation, groups, 2)
    operands = tuple(value for value in (input, weight, bias) if value is not None)
    dtype = result_type(*(value.dtype for value in operands))
    array = conv2d_forward(
        input._data.astype(dtype.numpy_dtype, copy=False), weight._data.astype(dtype.numpy_dtype, copy=False),
        None if bias is None else bias._data.astype(dtype.numpy_dtype, copy=False),
        stride, padding, dilation, groups, output_spatial,
    )
    output = Tensor._from_array(array, is_grad_enabled() and any(value.requires_grad for value in operands))
    output._is_leaf = not output.requires_grad
    if output.requires_grad:
        output._grad_fn = conv2d_node(input, weight, bias, stride, padding, dilation, groups, output_spatial, dtype.numpy_dtype)
    return output


@same_device
def conv1d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    stride, padding, dilation, groups, _ = _convolution_arguments(input, weight, bias, stride, padding, dilation, groups, 1)
    return conv2d(
        input.unsqueeze(-2), weight.unsqueeze(-2), bias,
        (1, stride[0]), (0, padding[0]), (1, dilation[0]), groups,
    ).squeeze(-2)


def _pool_parameters(kernel_size, stride, padding, dilation, dimensions):
    kernel = _spatial_tuple(kernel_size, dimensions, "kernel_size", 1)
    stride = kernel if stride is None else _spatial_tuple(stride, dimensions, "stride", 1)
    padding = _spatial_tuple(padding, dimensions, "padding", 0)
    dilation = _spatial_tuple(dilation, dimensions, "dilation", 1)
    if any(pad > size // 2 for pad, size in zip(padding, kernel)):
        raise ValueError("Pooling padding must not exceed half the kernel size.")
    return kernel, stride, padding, dilation


def _pool_arguments(input, kernel_size, stride, padding, dilation, dimensions):
    if not isinstance(input, Tensor) or not input.dtype.is_floating_point:
        raise TypeError("Pooling input must be a floating NamiTorch Tensor.")
    if input.ndim != dimensions + 2 or any(size <= 0 for size in input.shape[1:]):
        raise ValueError(f"Pooling input must have rank {dimensions + 2} and positive channel and spatial dimensions, got {input.shape}.")
    kernel, stride, padding, dilation = _pool_parameters(kernel_size, stride, padding, dilation, dimensions)
    spatial = tuple((size + 2 * pad - dil * (k - 1) - 1) // step + 1 for size, k, step, pad, dil in zip(input.shape[2:], kernel, stride, padding, dilation))
    if any(size <= 0 for size in spatial):
        raise ValueError(f"Pooling output spatial dimensions must be positive, got {spatial}.")
    return kernel, stride, padding, dilation, spatial


@same_device
def max_pool2d(input: Tensor, kernel_size, stride=None, padding=0, dilation=1) -> Tensor:
    kernel, stride, padding, dilation, spatial = _pool_arguments(input, kernel_size, stride, padding, dilation, 2)
    array, indices = max_pool2d_forward(input._data, kernel, stride, padding, dilation, spatial)
    output = Tensor._from_array(array, is_grad_enabled() and input.requires_grad)
    output._is_leaf = not output.requires_grad
    if output.requires_grad:
        output._grad_fn = pooling_node("max_pool2d", input, indices=indices)
    return output


@same_device
def avg_pool2d(input: Tensor, kernel_size, stride=None, padding=0, count_include_pad=True) -> Tensor:
    if type(count_include_pad) is not bool:
        raise TypeError("count_include_pad must be a Python bool.")
    kernel, stride, padding, _, spatial = _pool_arguments(input, kernel_size, stride, padding, 1, 2)
    array = avg_pool2d_forward(input._data, kernel, stride, padding, spatial, count_include_pad)
    output = Tensor._from_array(array, is_grad_enabled() and input.requires_grad)
    output._is_leaf = not output.requires_grad
    if output.requires_grad:
        output._grad_fn = pooling_node("avg_pool2d", input, kernel_size=kernel, stride=stride, padding=padding, count_include_pad=count_include_pad)
    return output


@same_device
def max_pool1d(input: Tensor, kernel_size, stride=None, padding=0, dilation=1) -> Tensor:
    kernel, stride, padding, dilation, _ = _pool_arguments(input, kernel_size, stride, padding, dilation, 1)
    return max_pool2d(input.unsqueeze(-2), (1, kernel[0]), (1, stride[0]), (0, padding[0]), (1, dilation[0])).squeeze(-2)


@same_device
def avg_pool1d(input: Tensor, kernel_size, stride=None, padding=0, count_include_pad=True) -> Tensor:
    kernel, stride, padding, _, _ = _pool_arguments(input, kernel_size, stride, padding, 1, 1)
    return avg_pool2d(input.unsqueeze(-2), (1, kernel[0]), (1, stride[0]), (0, padding[0]), count_include_pad).squeeze(-2)


@same_device
def embedding(input: Tensor, weight: Tensor, padding_idx: object = None) -> Tensor:
    xp = namespace(input)
    if not isinstance(input, Tensor) or not input.dtype.is_integer:
        raise TypeError("embedding input must be an integer NamiTorch Tensor.")
    if not isinstance(weight, Tensor):
        raise TypeError("embedding weight must be a NamiTorch Tensor.")
    if weight.ndim != 2 or any(size == 0 for size in weight.shape):
        raise ValueError(f"embedding weight must have rank 2 with positive dimensions, got shape {weight.shape}.")
    padding = _padding_index(padding_idx, weight.shape[0])
    indices = input._data.copy()
    if xp.any(indices < 0) or xp.any(indices >= weight.shape[0]):
        raise IndexError(f"embedding indices must be in [0, {weight.shape[0]}).")
    readonly(indices)
    output = Tensor._from_array(xp.asarray(weight._data[indices]), weight.requires_grad and is_grad_enabled())
    output._is_leaf = not output.requires_grad
    if output.requires_grad:
        output._grad_fn = embedding_node(weight, indices, padding)
    return output


def _require_input(input: Tensor) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("Functional operations require a NamiTorch Tensor input.")
    return input


def _combined_output(array, operation, first, second, **metadata):
    requires_grad = is_grad_enabled() and (first.requires_grad or second.requires_grad)
    output = Tensor._from_array(array, requires_grad)
    output._is_leaf = not requires_grad
    if requires_grad:
        output._grad_fn = fused_combine_node(operation, first, second, **metadata)
    return output


@same_device
def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    _require_input(gate)
    _require_input(up)
    if not gate.dtype.is_floating_point or not up.dtype.is_floating_point:
        raise TypeError("swiglu requires floating Tensors.")
    array = try_fused("swiglu", gate._data, up._data)
    if array is None:
        return gate.silu() * up
    return _combined_output(array, "swiglu", gate, up)


@same_device
def bias_activation(input: Tensor, bias: Tensor, activation: str = "gelu", *, approximation: str = "tanh") -> Tensor:
    _require_input(input)
    _require_input(bias)
    if not input.dtype.is_floating_point or not bias.dtype.is_floating_point:
        raise TypeError("bias_activation requires floating Tensors.")
    if input.ndim < 1 or bias.shape != (input.shape[-1],):
        raise ValueError("bias_activation requires a one-dimensional bias matching the last input dimension.")
    if activation not in ("relu", "sigmoid", "silu", "gelu"):
        raise ValueError("activation must be relu, sigmoid, silu or gelu.")
    if approximation not in ("tanh", "exact"):
        raise ValueError("approximation must be tanh or exact.")
    operation = "bias_" + ("gelu_tanh" if activation == "gelu" else activation)
    array = None if activation == "gelu" and approximation == "exact" else try_fused(operation, input._data, bias._data)
    if array is None:
        value = input + bias
        return value.gelu(approximation) if activation == "gelu" else getattr(value, activation)()
    return _combined_output(array, operation, input, bias, activation=activation, approximation=approximation)


def _validate_inplace(inplace: bool) -> None:
    if type(inplace) is not bool:
        raise TypeError("inplace must be a Python bool.")
    if inplace:
        raise RuntimeError("inplace=True is not supported; use inplace=False to preserve autograd.")


def _real_parameter(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar.")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite.") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _dropout_probability(p: object) -> float:
    probability = _real_parameter(p, "p")
    if not 0 <= probability < 1:
        raise ValueError("Dropout p must satisfy 0 <= p < 1.")
    return probability


def _gelu_approximation(approximation: str) -> str:
    if not isinstance(approximation, str):
        raise TypeError("GELU approximation must be 'exact' or 'tanh'.")
    if approximation not in ("exact", "tanh"):
        raise ValueError("GELU approximation must be 'exact' or 'tanh'.")
    return approximation


@same_device
def relu(input: Tensor, inplace: bool = False) -> Tensor:
    _validate_inplace(inplace)
    return _require_input(input).relu()


@same_device
def leaky_relu(input: Tensor, negative_slope: object = 0.01, inplace: bool = False) -> Tensor:
    _validate_inplace(inplace)
    return _require_input(input).leaky_relu(negative_slope)


@same_device
def elu(input: Tensor, alpha: object = 1.0, inplace: bool = False) -> Tensor:
    _validate_inplace(inplace)
    return _require_input(input).elu(alpha)


@same_device
def gelu(input: Tensor, approximation: str = "exact") -> Tensor:
    return _require_input(input).gelu(approximation)


@same_device
def silu(input: Tensor, inplace: bool = False) -> Tensor:
    _validate_inplace(inplace)
    return _require_input(input).silu()


@same_device
def sigmoid(input: Tensor) -> Tensor:
    return _require_input(input).sigmoid()


@same_device
def tanh(input: Tensor) -> Tensor:
    return _require_input(input).tanh()


@same_device
def softplus(input: Tensor) -> Tensor:
    return _require_input(input).softplus()


@same_device
def softmax(input: Tensor, dim: object) -> Tensor:
    return _require_input(input).softmax(dim)


@same_device
def log_softmax(input: Tensor, dim: object) -> Tensor:
    return _require_input(input).log_softmax(dim)


def _attention_inputs(query, key, value):
    xp = namespace(query)
    for name, operand in (("query", query), ("key", key), ("value", value)):
        if not isinstance(operand, Tensor) or not operand.dtype.is_floating_point:
            raise TypeError(f"Attention {name} must be a floating NamiTorch Tensor.")
        if operand.ndim != 4:
            raise ValueError(f"Attention {name} must have rank 4, got shape {operand.shape}.")
        if not xp.all(xp.isfinite(operand._data)):
            raise ValueError(f"Attention {name} must contain finite values.")
    batch, heads, query_length, head_dim = query.shape
    if key.shape[:2] != (batch, heads) or value.shape[:2] != (batch, heads):
        raise ValueError(f"Attention query, key and value must have the same B/H dimensions, got {query.shape}, {key.shape}, {value.shape}.")
    if key.shape[2] != value.shape[2]:
        raise ValueError("Attention key and value must have the same sequence length Tk.")
    if key.shape[3] != head_dim:
        raise ValueError("Attention query and key must have the same head dimension D.")
    if heads <= 0 or head_dim <= 0 or key.shape[2] <= 0 or value.shape[3] <= 0:
        raise ValueError("Attention requires positive H, D, Tk and Dv dimensions.")
    return batch, heads, query_length, key.shape[2], head_dim


def _attention_mask(attn_mask, score_shape):
    xp = namespace(attn_mask)
    if attn_mask is None:
        return None
    if not isinstance(attn_mask, Tensor) or not (attn_mask.dtype.is_boolean or attn_mask.dtype.is_floating_point):
        raise TypeError("attn_mask must be a boolean or floating NamiTorch Tensor, or None.")
    if broadcast_shapes(attn_mask.shape, score_shape) != score_shape:
        raise ValueError(f"attn_mask shape {attn_mask.shape} must broadcast to attention scores {score_shape} without expanding them.")
    if attn_mask.dtype.is_floating_point and xp.any(xp.isnan(attn_mask._data) | xp.isposinf(attn_mask._data)):
        raise ValueError("Additive attn_mask may contain finite values or -inf, but not NaN or +inf.")
    return attn_mask


@same_device
def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None = None,
    dropout_p: float = 0.0, is_causal: bool = False, *, scale: float | None = None,
    query_position_offset: int = 0, training: bool = True, need_weights: bool = False,
    generator: Generator | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    xp = namespace(query)
    batch, heads, query_length, key_length, head_dim = _attention_inputs(query, key, value)
    for name, flag in (("is_causal", is_causal), ("training", training), ("need_weights", need_weights)):
        if type(flag) is not bool:
            raise TypeError(f"{name} must be a Python bool.")
    offset = _integer(query_position_offset, "query_position_offset")
    if offset < 0:
        raise ValueError("query_position_offset must be nonnegative.")
    probability = _dropout_probability(dropout_p)
    _validate_generator(generator)
    mask = _attention_mask(attn_mask, (batch, heads, query_length, key_length))
    scale_value = 1 / math.sqrt(head_dim) if scale is None else _real_parameter(scale, "scale")
    dtype = result_type(query.dtype, key.dtype)
    if abs(scale_value) > float(np.finfo(dtype.numpy_dtype).max):
        raise ValueError(f"Attention scale must be representable in {dtype.name}.")
    with np.errstate(under="ignore"):
        factor = Tensor(scale_value, dtype=dtype, device=query.device)
    if scale_value != 0 and factor.item() == 0:
        raise ValueError(f"Nonzero attention scale must remain nonzero in {dtype.name}.")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scores = (query @ key.transpose(-2, -1)) * factor
    if not xp.all(xp.isfinite(scores._data)):
        raise ValueError("Attention scores must be finite before masking; input magnitudes or scale exceed the compute dtype range.")
    boolean_mask = mask if mask is not None and mask.dtype.is_boolean else None
    if mask is not None and not mask.dtype.is_boolean:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scores = scores + mask
    operation = "causal_softmax" if is_causal else "masked_softmax" if boolean_mask is not None else "softmax"
    arrays = (scores._data, None if boolean_mask is None else boolean_mask._data) if is_causal or boolean_mask is not None else (scores._data,)
    fused = try_fused_softmax(operation, *arrays, query_position_offset=offset, require_nonempty=True)
    if fused is not None:
        weights = Tensor._from_array(fused, scores.requires_grad and is_grad_enabled())
        weights._is_leaf = not weights.requires_grad
        if weights.requires_grad:
            weights._grad_fn = normalized_exponential_node(
                "softmax", scores, weights, scores.ndim - 1, mask=boolean_mask,
                is_causal=is_causal, query_position_offset=offset,
            )
        weights = dropout(weights, p=probability, training=training, generator=generator)
        output = weights @ value
        return (output, weights) if need_weights else output
    allowed = None
    if is_causal:
        key_positions = xp.arange(key_length) - min(offset, key_length)
        allowed = key_positions[None, :] <= xp.arange(query_length)[:, None]
    if boolean_mask is not None:
        allowed = boolean_mask._data if allowed is None else xp.logical_and(allowed, boolean_mask._data)
    if allowed is not None:
        blocked = Tensor._from_array(xp.asarray(xp.logical_not(allowed)), False)
        scores = scores.masked_fill(blocked, -float("inf"))
    if xp.any(xp.isnan(scores._data) | xp.isposinf(scores._data)):
        raise ValueError("Masked attention scores must be finite or -inf.")
    empty_rows = ~xp.any(xp.isfinite(scores._data), axis=-1)
    if xp.any(empty_rows):
        row = tuple(int(index) for index in xp.argwhere(empty_rows)[0])
        raise ValueError(f"Attention row {row} has no allowed keys.")
    weights = dropout(scores.softmax(-1), p=probability, training=training, generator=generator)
    output = weights @ value
    return (output, weights) if need_weights else output


def _normalized_shape(normalized_shape: object) -> tuple[int, ...]:
    dimensions = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
    if not dimensions:
        raise ValueError("normalized_shape must contain at least one dimension.")
    return tuple(_positive_dimension(value, "normalized_shape dimension") for value in dimensions)


def _normalization_eps(eps: object) -> float:
    epsilon = _real_parameter(eps, "eps")
    if epsilon <= 0:
        raise ValueError("eps must be positive.")
    return epsilon


def _normalization_arguments(input, normalized_shape, weight, bias, eps):
    _require_input(input)
    if not input.dtype.is_floating_point:
        raise TypeError("Normalization requires a floating Tensor input.")
    shape = _normalized_shape(normalized_shape)
    if input.ndim < len(shape) or input.shape[-len(shape):] != shape:
        raise ValueError(f"Input shape {input.shape} must end with normalized_shape {shape}.")
    for name, parameter in (("weight", weight), ("bias", bias)):
        if parameter is not None:
            if not isinstance(parameter, Tensor) or not parameter.dtype.is_floating_point:
                raise TypeError(f"Normalization {name} must be a floating Tensor or None.")
            if parameter.shape != shape:
                raise ValueError(f"Normalization {name} must have shape {shape}, got {parameter.shape}.")
    epsilon = _normalization_eps(eps)
    if epsilon > float(np.finfo(input.dtype.numpy_dtype).max):
        raise ValueError(f"eps must be finite and positive in {input.dtype.name}.")
    with np.errstate(under="ignore"):
        constant = Tensor(epsilon, dtype=input.dtype, device=input.device)
    if constant.item() == 0:
        raise ValueError(f"eps must remain positive in {input.dtype.name}.")
    axes = tuple(range(input.ndim - len(shape), input.ndim))
    return axes, constant


@same_device
def layer_norm(
    input: Tensor, normalized_shape: object, weight: Tensor | None = None,
    bias: Tensor | None = None, eps: float = 1e-5,
) -> Tensor:
    axes, epsilon = _normalization_arguments(input, normalized_shape, weight, bias, eps)
    mean = input.mean(dim=axes, keepdim=True)
    variance = input.var(dim=axes, keepdim=True, correction=0)
    output = (input - mean) * (variance + epsilon).rsqrt()
    if weight is not None:
        output = output * weight
    return output if bias is None else output + bias


@same_device
def rms_norm(
    input: Tensor, normalized_shape: object, weight: Tensor | None = None, eps: float = 1e-5,
) -> Tensor:
    axes, epsilon = _normalization_arguments(input, normalized_shape, weight, None, eps)
    mean_square = input.square().mean(dim=axes, keepdim=True)
    output = input * (mean_square + epsilon).rsqrt()
    return output if weight is None else output * weight


def _batch_norm_momentum(momentum):
    value = _real_parameter(momentum, "momentum")
    if not 0 <= value <= 1:
        raise ValueError("BatchNorm momentum must be in [0, 1].")
    return value


@same_device
def batch_norm(input: Tensor, running_mean=None, running_var=None, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5) -> Tensor:
    xp = namespace(input)
    _require_input(input)
    if not input.dtype.is_floating_point:
        raise TypeError("BatchNorm input must have a floating dtype.")
    if input.ndim not in (2, 3, 4) or input.shape[1] <= 0:
        raise ValueError(f"BatchNorm input must have shape (N, C), (N, C, L) or (N, C, H, W) with C > 0, got {input.shape}.")
    if type(training) is not bool:
        raise TypeError("training must be a Python bool.")
    momentum = _batch_norm_momentum(momentum)
    epsilon = _loss_constant(_normalization_eps(eps), "eps", input)
    channels = input.shape[1]
    for name, value in (("weight", weight), ("bias", bias), ("running_mean", running_mean), ("running_var", running_var)):
        if value is not None:
            if not isinstance(value, Tensor) or not value.dtype.is_floating_point:
                raise TypeError(f"BatchNorm {name} must be a floating Tensor or None.")
            if value.shape != (channels,):
                raise ValueError(f"BatchNorm {name} must have shape {(channels,)}, got {value.shape}.")
    if (running_mean is None) != (running_var is None):
        raise ValueError("running_mean and running_var must both be supplied or both be None.")
    if not training and running_mean is None:
        raise ValueError("BatchNorm evaluation requires running statistics.")
    if running_mean is not None:
        if running_mean.requires_grad or running_var.requires_grad:
            raise ValueError("BatchNorm running statistics must not require gradients.")
        if xp.shares_memory(running_mean._data, running_var._data):
            raise ValueError("BatchNorm running_mean and running_var must have independent storage.")
        if training and (not writable(running_mean._data) or not writable(running_var._data)):
            raise RuntimeError("Cannot update read-only BatchNorm running statistics.")
    axes = (0, *range(2, input.ndim))
    shape = (1, channels, *((1,) * (input.ndim - 2)))
    if training:
        count = math.prod(input.shape[axis] for axis in axes)
        if count == 0:
            raise ValueError("BatchNorm batch statistics require at least one value per channel.")
        mean = input.mean(dim=axes, keepdim=True)
        variance = input.var(dim=axes, keepdim=True, correction=0)
    else:
        mean = running_mean.reshape(shape)
        variance = running_var.reshape(shape)
    output = (input - mean) * (variance + epsilon).rsqrt()
    if weight is not None:
        output = output * weight.reshape(shape)
    if bias is not None:
        output = output + bias.reshape(shape)
    if training and running_mean is not None:
        mean_values = mean._data.reshape(channels).astype(running_mean.dtype.numpy_dtype)
        variance_values = variance._data.reshape(channels).astype(running_var.dtype.numpy_dtype)
        if count > 1:
            variance_values *= count / (count - 1)
        next_mean = (1 - momentum) * running_mean._data + momentum * mean_values
        next_variance = (1 - momentum) * running_var._data + momentum * variance_values
        with no_grad():
            running_mean.copy_(next_mean)
            running_var.copy_(next_variance)
    return output


@same_device
def dropout(input: Tensor, p: object = 0.5, training: bool = True, inplace: bool = False, *, generator: Generator | None = None) -> Tensor:
    xp = namespace(input)
    _require_input(input)
    _validate_inplace(inplace)
    probability = _dropout_probability(p)
    _validate_generator(generator)
    if type(training) is not bool:
        raise TypeError("training must be a Python bool.")
    if not training or probability == 0:
        return input
    if not input.dtype.is_floating_point:
        raise TypeError("Training dropout requires a floating Tensor input.")
    rng = _get_rng(generator, device=input.device)
    mask = Tensor._from_array(xp.asarray(get_backend(input).random(rng, "random", input.shape, None) >= probability, dtype=input.dtype.numpy_dtype), False)
    denominator = Tensor(1 - probability, dtype=input.dtype, device=input.device)
    return input * mask / denominator


def _loss_reduction(reduction: str) -> str:
    if not isinstance(reduction, str):
        raise TypeError("Loss reduction must be 'none', 'sum' or 'mean'.")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("Loss reduction must be 'none', 'sum' or 'mean'.")
    return reduction


def _reduce_loss(loss: Tensor, reduction: str, mean_denominator: Tensor | None = None) -> Tensor:
    reduction = _loss_reduction(reduction)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if mean_denominator is not None:
        return loss.sum() / mean_denominator
    if loss.numel() == 0:
        return loss.sum() * Tensor(float("nan"), dtype=loss.dtype, device=loss.device)
    return loss.mean()


def _loss_broadcast(value: Tensor, input: Tensor, name: str) -> None:
    if broadcast_shapes(input.shape, value.shape) != input.shape:
        raise ValueError(f"Loss {name} shape {value.shape} must broadcast to input shape {input.shape} without expanding it.")


def _loss_inputs(input: Tensor, target: Tensor, reduction: str) -> None:
    _loss_reduction(reduction)
    _require_input(input)
    if not input.dtype.is_floating_point:
        raise TypeError("Loss input must be a floating NamiTorch Tensor.")
    if not isinstance(target, Tensor):
        raise TypeError("Loss target must be a NamiTorch Tensor.")
    _loss_broadcast(target, input, "target")


def _loss_scale(value: object, name: str, allow_zero: bool = False) -> float:
    scale = _real_parameter(value, name)
    if scale < 0 or (scale == 0 and not allow_zero):
        constraint = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {constraint}.")
    return scale


def _loss_constant(value: float, name: str, input: Tensor) -> Tensor:
    if value > float(np.finfo(input.dtype.numpy_dtype).max):
        raise ValueError(f"{name} must be representable in {input.dtype.name}.")
    with np.errstate(under="ignore"):
        constant = Tensor(value, dtype=input.dtype, device=input.device)
    if value > 0 and constant.item() == 0:
        raise ValueError(f"{name} must remain positive in {input.dtype.name}.")
    return constant


@same_device
def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    _loss_inputs(input, target, reduction)
    return _reduce_loss((input - target).square(), reduction)


@same_device
def l1_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    _loss_inputs(input, target, reduction)
    return _reduce_loss((input - target).abs(), reduction)


@same_device
def smooth_l1_loss(input: Tensor, target: Tensor, reduction: str = "mean", beta: float = 1.0) -> Tensor:
    _loss_inputs(input, target, reduction)
    beta = _loss_scale(beta, "beta", allow_zero=True)
    distance = (input - target).abs()
    if beta == 0:
        return _reduce_loss(distance, reduction)
    threshold = _loss_constant(beta, "beta", distance)
    quadratic = distance.clamp(max=threshold)
    loss = 0.5 * (quadratic / threshold) * quadratic + (distance - quadratic)
    return _reduce_loss(loss, reduction)


@same_device
def huber_loss(input: Tensor, target: Tensor, reduction: str = "mean", delta: float = 1.0) -> Tensor:
    _loss_inputs(input, target, reduction)
    delta = _loss_scale(delta, "delta")
    distance = (input - target).abs()
    threshold = _loss_constant(delta, "delta", distance)
    quadratic = distance.clamp(max=threshold)
    loss = threshold * (0.5 * (quadratic / threshold) * quadratic + (distance - quadratic))
    return _reduce_loss(loss, reduction)


def _loss_weight(weight: Tensor | None, name: str) -> None:
    xp = namespace(weight)
    if weight is None:
        return
    if not isinstance(weight, Tensor):
        raise TypeError(f"Loss {name} must be a NamiTorch Tensor or None.")
    if not xp.all(xp.isfinite(weight._data)) or xp.any(weight._data < 0):
        raise ValueError(f"Loss {name} must contain finite, nonnegative values.")


def _binary_loss_inputs(input: Tensor, target: Tensor, weight: Tensor | None, reduction: str, pos_weight: Tensor | None = None) -> None:
    xp = namespace(input)
    _loss_inputs(input, target, reduction)
    if not xp.all(xp.isfinite(target._data)) or xp.any(target._data < 0) or xp.any(target._data > 1):
        raise ValueError("Binary loss target values must be finite and in [0, 1].")
    for name, value in (("weight", weight), ("pos_weight", pos_weight)):
        _loss_weight(value, name)
        if value is not None:
            _loss_broadcast(value, input, name)


@same_device
def binary_cross_entropy(input: Tensor, target: Tensor, weight: Tensor | None = None, reduction: str = "mean") -> Tensor:
    xp = namespace(input)
    _binary_loss_inputs(input, target, weight, reduction)
    if not xp.all(xp.isfinite(input._data)) or xp.any(input._data < 0) or xp.any(input._data > 1):
        raise ValueError("binary_cross_entropy input probabilities must be finite and in [0, 1].")
    epsilon = float(np.finfo(input.dtype.numpy_dtype).eps)
    probability = input.clamp(min=Tensor(epsilon, dtype=input.dtype, device=input.device), max=Tensor(1 - epsilon, dtype=input.dtype, device=input.device))
    one = Tensor(1, dtype=input.dtype, device=input.device)
    loss = -(target * probability.log() + (one - target) * (-probability).log1p())
    if weight is not None:
        loss = loss * weight
    return _reduce_loss(loss, reduction)


@same_device
def binary_cross_entropy_with_logits(
    input: Tensor, target: Tensor, weight: Tensor | None = None,
    reduction: str = "mean", pos_weight: Tensor | None = None,
) -> Tensor:
    xp = namespace(input)
    _binary_loss_inputs(input, target, weight, reduction, pos_weight)
    if not xp.all(xp.isfinite(input._data)):
        raise ValueError("binary_cross_entropy_with_logits input must contain finite logits.")
    one = Tensor(1, dtype=input.dtype, device=input.device)
    positive = target * (-input).softplus()
    if pos_weight is not None:
        positive = positive * pos_weight
    loss = (one - target) * input.softplus() + positive
    if weight is not None:
        loss = loss * weight
    return _reduce_loss(loss, reduction)


def _class_weight(weight: Tensor | None) -> None:
    _loss_weight(weight, "weight")
    if weight is not None:
        if not weight.dtype.is_floating_point:
            raise TypeError("Class weight must be a floating Tensor.")
        if weight.ndim != 1 or weight.shape[0] == 0:
            raise ValueError(f"Class weight must have shape [C] with C > 0, got {weight.shape}.")


def _label_smoothing(value: object) -> float:
    epsilon = _real_parameter(value, "label_smoothing")
    if not 0 <= epsilon < 1:
        raise ValueError("label_smoothing must be in [0, 1).")
    return epsilon


def _class_loss_arguments(input: Tensor, target: Tensor, weight: Tensor | None, ignore_index: int, reduction: str):
    xp = namespace(input)
    _loss_reduction(reduction)
    _require_input(input)
    if not input.dtype.is_floating_point:
        raise TypeError("Classification loss input must be a floating Tensor.")
    if input.ndim < 1 or input.shape[-1] == 0:
        raise ValueError(f"Classification loss input must have shape [..., C] with C > 0, got {input.shape}.")
    if not isinstance(target, Tensor) or not target.dtype.is_integer:
        raise TypeError("Classification loss target must be an integer Tensor.")
    if target.shape != input.shape[:-1]:
        raise ValueError(f"Target shape {target.shape} must equal input shape without the last class axis {input.shape[:-1]}.")
    ignored = _integer(ignore_index, "ignore_index")
    _class_weight(weight)
    classes = input.shape[-1]
    if weight is not None and weight.shape != (classes,):
        raise ValueError(f"Class weight must have shape {(classes,)}, got {weight.shape}.")
    target_values = target._data.reshape(-1)
    valid = target_values != ignored
    selected = target_values[valid]
    if xp.any(selected < 0) or xp.any(selected >= classes):
        raise IndexError(f"Every non-ignored target must be in [0, {classes}).")
    indices = Tensor._from_array(xp.asarray(selected, dtype=np.int64), False)
    positions = Tensor._from_array(xp.flatnonzero(valid).astype(np.int64, copy=False), False)
    return input.reshape(target.numel(), classes), indices, positions


def _classification_loss(
    input: Tensor, target: Tensor, weight: Tensor | None, ignore_index: int,
    reduction: str, label_smoothing: float, from_logits: bool,
) -> Tensor:
    xp = namespace(input)
    flat_input, indices, positions = _class_loss_arguments(input, target, weight, ignore_index, reduction)
    example_weights = None if weight is None else weight[indices]
    denominator = Tensor(indices.numel(), dtype=input.dtype, device=input.device) if weight is None else example_weights.sum()
    if reduction == "mean" and denominator.item() == 0:
        zero = flat_input[:0].sum()
        return zero if weight is None else zero + weight[:0].sum()
    selected = flat_input[positions]
    log_probabilities = selected.log_softmax(dim=-1) if from_logits else selected
    loss = -log_probabilities.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    if example_weights is not None:
        loss = loss * example_weights
    if label_smoothing != 0:
        weighted_log_probabilities = log_probabilities if weight is None else log_probabilities * weight
        smooth_loss = -weighted_log_probabilities.mean(dim=-1)
        epsilon = Tensor(label_smoothing, dtype=loss.dtype, device=loss.device)
        confidence = Tensor(1 - label_smoothing, dtype=loss.dtype, device=loss.device)
        loss = confidence * loss + epsilon * smooth_loss
    if reduction == "none":
        output = Tensor._from_array(xp.zeros(target.numel(), dtype=loss.dtype.numpy_dtype), False)
        return output.scatter_add(0, positions, loss).reshape(target.shape)
    return _reduce_loss(loss, reduction, denominator)


@same_device
def nll_loss(
    input: Tensor, target: Tensor, weight: Tensor | None = None,
    ignore_index: int = -100, reduction: str = "mean",
) -> Tensor:
    return _classification_loss(input, target, weight, ignore_index, reduction, 0.0, False)


@same_device
def cross_entropy(
    input: Tensor, target: Tensor, weight: Tensor | None = None,
    ignore_index: int = -100, reduction: str = "mean", label_smoothing: float = 0.0,
) -> Tensor:
    epsilon = _label_smoothing(label_smoothing)
    return _classification_loss(input, target, weight, ignore_index, reduction, epsilon, True)


__all__ = [
    "swiglu", "bias_activation",
    "linear", "conv1d", "conv2d", "max_pool1d", "max_pool2d", "avg_pool1d", "avg_pool2d",
    "embedding", "relu", "leaky_relu", "elu", "gelu", "silu", "sigmoid",
    "tanh", "softplus", "softmax", "log_softmax", "dropout",
    "scaled_dot_product_attention",
    "layer_norm", "rms_norm", "batch_norm",
    "mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss",
    "binary_cross_entropy", "binary_cross_entropy_with_logits",
    "nll_loss", "cross_entropy",
]
