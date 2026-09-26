from ..creation import ones, zeros
from ..dtype import get_default_dtype, normalize_dtype
from ..tensor import Tensor
from . import functional as F
from . import init
from .module import Module
from .parameter import Parameter


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
        super().__init__()
        self.normalized_shape = F._normalized_shape(normalized_shape)
        self.eps = F._normalization_eps(eps)
        if type(elementwise_affine) is not bool or type(bias) is not bool:
            raise TypeError("elementwise_affine and bias must be Python bools.")
        self.elementwise_affine = elementwise_affine
        self.register_parameter("weight", Parameter(ones(self.normalized_shape)) if elementwise_affine else None)
        self.register_parameter("bias", Parameter(zeros(self.normalized_shape)) if elementwise_affine and bias else None)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)

    def __repr__(self) -> str:
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, bias={self.bias is not None})"


class RMSNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, *, dtype=None):
        super().__init__()
        self.normalized_shape = F._normalized_shape(normalized_shape)
        self.eps = F._normalization_eps(eps)
        if type(elementwise_affine) is not bool:
            raise TypeError("elementwise_affine must be a Python bool.")
        dtype = get_default_dtype() if dtype is None else normalize_dtype(dtype)
        if not dtype.is_floating_point:
            raise TypeError("RMSNorm parameters require a floating dtype.")
        self.elementwise_affine = elementwise_affine
        self.register_parameter("weight", Parameter(ones(self.normalized_shape, dtype=dtype)) if elementwise_affine else None)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            init.ones_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        return F.rms_norm(input, self.normalized_shape, weight=self.weight, eps=self.eps)

    def __repr__(self) -> str:
        return f"RMSNorm(normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine})"


__all__ = ["LayerNorm", "RMSNorm"]
