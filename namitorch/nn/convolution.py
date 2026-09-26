import math

from ..creation import empty
from ..dtype import get_default_dtype, normalize_dtype
from ..random import Generator
from ..tensor import Tensor
from . import functional as F
from . import init
from .module import Module
from .parameter import Parameter


class _ConvNd(Module):
    def __init__(self, dimensions, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype, generator):
        super().__init__()
        self.in_channels = F._positive_dimension(in_channels, "in_channels")
        self.out_channels = F._positive_dimension(out_channels, "out_channels")
        self.groups = F._positive_dimension(groups, "groups")
        if self.in_channels % self.groups or self.out_channels % self.groups:
            raise ValueError("in_channels and out_channels must be divisible by groups.")
        self.kernel_size = F._spatial_tuple(kernel_size, dimensions, "kernel_size", 1)
        self.stride = F._spatial_tuple(stride, dimensions, "stride", 1)
        self.padding = F._spatial_tuple(padding, dimensions, "padding", 0)
        self.dilation = F._spatial_tuple(dilation, dimensions, "dilation", 1)
        if type(bias) is not bool:
            raise TypeError("bias must be a Python bool.")
        dtype = get_default_dtype() if dtype is None else normalize_dtype(dtype)
        if not dtype.is_floating_point:
            raise TypeError("Convolution parameters require a floating dtype.")
        self.weight = Parameter(empty(self.out_channels, self.in_channels // self.groups, *self.kernel_size, dtype=dtype))
        self.register_parameter("bias", Parameter(empty(self.out_channels, dtype=dtype)) if bias else None)
        self.reset_parameters(generator=generator)

    def reset_parameters(self, *, generator: Generator | None = None) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5), generator=generator)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound, generator=generator)

    def _validate_input(self, input):
        if not isinstance(input, Tensor):
            raise TypeError("Convolution input must be a NamiTorch Tensor.")
        rank = len(self.kernel_size) + 2
        if input.ndim != rank or input.shape[1] != self.in_channels:
            raise ValueError(f"{type(self).__name__} input must have rank {rank} and {self.in_channels} channels, got shape {input.shape}.")

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None})"
        )


class Conv2d(_ConvNd):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, *, dtype=None, generator: Generator | None = None):
        super().__init__(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype, generator)

    def forward(self, input: Tensor) -> Tensor:
        self._validate_input(input)
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv1d(_ConvNd):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, *, dtype=None, generator: Generator | None = None):
        super().__init__(1, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, dtype, generator)

    def forward(self, input: Tensor) -> Tensor:
        self._validate_input(input)
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


__all__ = ["Conv1d", "Conv2d"]
