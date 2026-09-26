import math

from ..creation import empty
from ..dtype import get_default_dtype, normalize_dtype
from ..random import Generator
from ..tensor import Tensor
from . import functional as F
from . import init
from .module import Module
from .parameter import Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, dtype=None, generator: Generator | None = None):
        super().__init__()
        self.in_features = F._positive_dimension(in_features, "in_features")
        self.out_features = F._positive_dimension(out_features, "out_features")
        if type(bias) is not bool:
            raise TypeError("bias must be a Python bool.")
        dtype = get_default_dtype() if dtype is None else normalize_dtype(dtype)
        if not dtype.is_floating_point:
            raise TypeError("Linear parameters require a floating dtype.")
        self.weight = Parameter(empty(self.out_features, self.in_features, dtype=dtype))
        self.register_parameter("bias", Parameter(empty(self.out_features, dtype=dtype)) if bias else None)
        self.reset_parameters(generator=generator)

    def reset_parameters(self, *, generator: Generator | None = None) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5), generator=generator)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound, generator=generator)

    def forward(self, input: Tensor) -> Tensor:
        if not isinstance(input, Tensor):
            raise TypeError("Linear input must be a NamiTorch Tensor.")
        if input.ndim < 1 or input.shape[-1] != self.in_features:
            raise ValueError(f"Linear input must have rank >= 1 and last dimension {self.in_features}, got shape {input.shape}.")
        return F.linear(input, self.weight, self.bias)

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


__all__ = ["Linear"]
