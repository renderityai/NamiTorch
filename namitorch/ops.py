import builtins

import numpy as np

from .backends import ensure_same_device, get_backend, namespace

from .autograd import cat_node, is_grad_enabled, stack_node
from .dtype import result_type
from .tensor import Tensor
from .utils import normalize_axis


def _require_tensor(input: Tensor) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("This operation requires a NamiTorch Tensor.")
    return input


def stack(tensors, dim: object = 0) -> Tensor:
    if isinstance(tensors, (Tensor, str, bytes, dict, set, frozenset)):
        raise TypeError("stack requires an ordered iterable of Tensors.")
    try:
        operands = tuple(tensors)
    except TypeError:
        raise TypeError("stack requires an ordered iterable of Tensors.") from None
    if not operands:
        raise ValueError("stack requires at least one Tensor.")
    for operand in operands:
        _require_tensor(operand)
    ensure_same_device(*operands)
    xp = namespace(operands[0])
    shape = operands[0].shape
    if builtins.any(operand.shape != shape for operand in operands):
        raise ValueError(f"stack requires identical shapes, got {[operand.shape for operand in operands]}.")
    axis = normalize_axis(dim, len(shape) + 1)
    dtype = result_type(*(operand.dtype for operand in operands))
    with get_backend(operands[0]).context():
        array = xp.stack([operand._data.astype(dtype.numpy_dtype, copy=False) for operand in operands], axis=axis)
    requires_grad = is_grad_enabled() and builtins.any(operand.requires_grad for operand in operands)
    output = Tensor._from_array(array, requires_grad)
    output._is_leaf = not requires_grad
    if requires_grad:
        output._grad_fn = stack_node(operands, axis)
    return output


def cat(tensors, dim: object = 0) -> Tensor:
    if isinstance(tensors, (Tensor, str, bytes, dict, set, frozenset)):
        raise TypeError("cat requires an ordered iterable of Tensors.")
    try:
        operands = tuple(tensors)
    except TypeError:
        raise TypeError("cat requires an ordered iterable of Tensors.") from None
    if not operands:
        raise ValueError("cat requires at least one Tensor.")
    for operand in operands:
        _require_tensor(operand)
    ensure_same_device(*operands)
    xp = namespace(operands[0])
    shape = operands[0].shape
    if not shape:
        raise ValueError("cat requires non-scalar Tensors; use stack for scalars.")
    axis = normalize_axis(dim, len(shape))
    if builtins.any(operand.ndim != len(shape) or operand.shape[:axis] + operand.shape[axis + 1:] != shape[:axis] + shape[axis + 1:] for operand in operands):
        raise ValueError(f"cat requires matching ranks and dimensions except dim={axis}, got {[operand.shape for operand in operands]}.")
    dtype = result_type(*(operand.dtype for operand in operands))
    with get_backend(operands[0]).context():
        array = xp.concatenate([operand._data.astype(dtype.numpy_dtype, copy=False) for operand in operands], axis=axis)
    requires_grad = is_grad_enabled() and builtins.any(operand.requires_grad for operand in operands)
    output = Tensor._from_array(array, requires_grad)
    output._is_leaf = not requires_grad
    if requires_grad:
        output._grad_fn = cat_node(operands, axis)
    return output


def expand(input: Tensor, *shape: object) -> Tensor:
    return _require_tensor(input).expand(*shape)


def broadcast_to(input: Tensor, *shape: object) -> Tensor:
    return _require_tensor(input).broadcast_to(*shape)


def repeat(input: Tensor, *repeats: object) -> Tensor:
    return _require_tensor(input).repeat(*repeats)


def repeat_interleave(input: Tensor, repeats: object, dim: object = None) -> Tensor:
    return _require_tensor(input).repeat_interleave(repeats, dim)


def split(input: Tensor, split_size_or_sections: object, dim: object = 0) -> tuple[Tensor, ...]:
    return _require_tensor(input).split(split_size_or_sections, dim)


def chunk(input: Tensor, chunks: object, dim: object = 0) -> tuple[Tensor, ...]:
    return _require_tensor(input).chunk(chunks, dim)


def tril(input: Tensor, diagonal: object = 0) -> Tensor:
    return _require_tensor(input).tril(diagonal)


def triu(input: Tensor, diagonal: object = 0) -> Tensor:
    return _require_tensor(input).triu(diagonal)


def masked_fill(input: Tensor, mask: object, value: object) -> Tensor:
    return _require_tensor(input).masked_fill(mask, value)


def reshape(input: Tensor, *shape: object) -> Tensor:
    return _require_tensor(input).reshape(*shape)


def view(input: Tensor, *shape: object) -> Tensor:
    return _require_tensor(input).view(*shape)


def flatten(input: Tensor, start_dim: object = 0, end_dim: object = -1) -> Tensor:
    return _require_tensor(input).flatten(start_dim, end_dim)


def ravel(input: Tensor) -> Tensor:
    return _require_tensor(input).ravel()


def squeeze(input: Tensor, dim: object = None) -> Tensor:
    return _require_tensor(input).squeeze(dim)


def unsqueeze(input: Tensor, dim: object) -> Tensor:
    return _require_tensor(input).unsqueeze(dim)


def transpose(input: Tensor, dim0: object, dim1: object) -> Tensor:
    return _require_tensor(input).transpose(dim0, dim1)


def swapaxes(input: Tensor, dim0: object, dim1: object) -> Tensor:
    return _require_tensor(input).swapaxes(dim0, dim1)


def moveaxis(input: Tensor, source: object, destination: object) -> Tensor:
    return _require_tensor(input).moveaxis(source, destination)


def permute(input: Tensor, *dims: object) -> Tensor:
    return _require_tensor(input).permute(*dims)


def is_contiguous(input: Tensor) -> bool:
    return _require_tensor(input).is_contiguous()


def contiguous(input: Tensor) -> Tensor:
    return _require_tensor(input).contiguous()


def matmul(input: Tensor, other: Tensor) -> Tensor:
    return _require_tensor(input).matmul(other)


def bmm(input: Tensor, other: Tensor) -> Tensor:
    return _require_tensor(input).bmm(other)


def dot(input: Tensor, other: Tensor) -> Tensor:
    return _require_tensor(input).dot(other)


def gather(input: Tensor, dim: object, index: Tensor) -> Tensor:
    return _require_tensor(input).gather(dim, index)


def scatter_add(input: Tensor, dim: object, index: Tensor, src: Tensor) -> Tensor:
    return _require_tensor(input).scatter_add(dim, index, src)


def sum(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).sum(dim, keepdim)


def mean(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).mean(dim, keepdim)


def prod(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).prod(dim, keepdim)


def min(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).min(dim, keepdim)


def max(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).max(dim, keepdim)


def amin(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).amin(dim, keepdim)


def amax(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).amax(dim, keepdim)


def argmin(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).argmin(dim, keepdim)


def argmax(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).argmax(dim, keepdim)


def var(input: Tensor, dim: object = None, keepdim: bool = False, *, correction: object = 1) -> Tensor:
    return _require_tensor(input).var(dim, keepdim, correction=correction)


def std(input: Tensor, dim: object = None, keepdim: bool = False, *, correction: object = 1) -> Tensor:
    return _require_tensor(input).std(dim, keepdim, correction=correction)


def all(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).all(dim, keepdim)


def any(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).any(dim, keepdim)


def logsumexp(input: Tensor, dim: object = None, keepdim: bool = False) -> Tensor:
    return _require_tensor(input).logsumexp(dim, keepdim)


def softmax(input: Tensor, dim: object) -> Tensor:
    return _require_tensor(input).softmax(dim)


def log_softmax(input: Tensor, dim: object) -> Tensor:
    return _require_tensor(input).log_softmax(dim)


def exp(input: Tensor) -> Tensor:
    return _require_tensor(input).exp()


def expm1(input: Tensor) -> Tensor:
    return _require_tensor(input).expm1()


def log(input: Tensor) -> Tensor:
    return _require_tensor(input).log()


def log1p(input: Tensor) -> Tensor:
    return _require_tensor(input).log1p()


def log2(input: Tensor) -> Tensor:
    return _require_tensor(input).log2()


def log10(input: Tensor) -> Tensor:
    return _require_tensor(input).log10()


def sqrt(input: Tensor) -> Tensor:
    return _require_tensor(input).sqrt()


def rsqrt(input: Tensor) -> Tensor:
    return _require_tensor(input).rsqrt()


def square(input: Tensor) -> Tensor:
    return _require_tensor(input).square()


def abs(input: Tensor) -> Tensor:
    return _require_tensor(input).abs()


def sign(input: Tensor) -> Tensor:
    return _require_tensor(input).sign()


def sin(input: Tensor) -> Tensor:
    return _require_tensor(input).sin()


def cos(input: Tensor) -> Tensor:
    return _require_tensor(input).cos()


def tan(input: Tensor) -> Tensor:
    return _require_tensor(input).tan()


def tanh(input: Tensor) -> Tensor:
    return _require_tensor(input).tanh()


def sigmoid(input: Tensor) -> Tensor:
    return _require_tensor(input).sigmoid()


def relu(input: Tensor) -> Tensor:
    return _require_tensor(input).relu()


def leaky_relu(input: Tensor, negative_slope: object = 0.01) -> Tensor:
    return _require_tensor(input).leaky_relu(negative_slope)


def elu(input: Tensor, alpha: object = 1.0) -> Tensor:
    return _require_tensor(input).elu(alpha)


def gelu(input: Tensor, approximation: str = "exact") -> Tensor:
    return _require_tensor(input).gelu(approximation)


def silu(input: Tensor) -> Tensor:
    return _require_tensor(input).silu()


def softplus(input: Tensor) -> Tensor:
    return _require_tensor(input).softplus()


def softsign(input: Tensor) -> Tensor:
    return _require_tensor(input).softsign()


def maximum(input: object, other: object) -> Tensor:
    return _scalar_operand(input, other).maximum(other)


def minimum(input: object, other: object) -> Tensor:
    return _scalar_operand(input, other).minimum(other)


def clamp(input: Tensor, min: object = None, max: object = None) -> Tensor:
    return _require_tensor(input).clamp(min=min, max=max)


def where(condition: object, input: object, other: object) -> Tensor:
    return _scalar_operand(input, other, condition).where(condition, other)


def _scalar_operand(value, *peers):
    ensure_same_device(value, *peers)
    if isinstance(value, Tensor):
        return value
    if not isinstance(value, (builtins.bool, builtins.int, builtins.float, np.generic)):
        raise TypeError("Operands must be Tensors or numeric scalars.")
    target = next((peer.device for peer in peers if isinstance(peer, Tensor)), None)
    return Tensor(value, device=target)


__all__ = [
    "stack", "cat", "expand", "broadcast_to", "repeat", "repeat_interleave",
    "split", "chunk", "tril", "triu", "masked_fill",
    "reshape", "view", "flatten", "ravel", "squeeze", "unsqueeze", "transpose",
    "swapaxes", "moveaxis", "permute", "is_contiguous", "contiguous",
    "matmul", "bmm", "dot",
    "gather", "scatter_add",
    "sum", "mean", "prod", "min", "max", "amin", "amax", "argmin", "argmax",
    "var", "std", "all", "any",
    "logsumexp", "softmax", "log_softmax",
    "exp", "expm1", "log", "log1p", "log2", "log10", "sqrt", "rsqrt", "square",
    "abs", "sign", "sin", "cos", "tan", "tanh", "sigmoid", "relu", "leaky_relu",
    "elu", "gelu", "silu", "softplus", "softsign", "maximum", "minimum", "clamp", "where",
]
