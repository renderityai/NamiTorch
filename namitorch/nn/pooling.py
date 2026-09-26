from ..tensor import Tensor
from . import functional as F
from .module import Module


class _MaxPoolNd(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.kernel_size, self.stride, self.padding, self.dilation = F._pool_parameters(kernel_size, stride, padding, dilation, self._dimensions)

    def forward(self, input: Tensor) -> Tensor:
        function = F.max_pool1d if self._dimensions == 1 else F.max_pool2d
        return function(input, self.kernel_size, self.stride, self.padding, self.dilation)

    def __repr__(self):
        return f"{type(self).__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})"


class MaxPool1d(_MaxPoolNd):
    _dimensions = 1


class MaxPool2d(_MaxPoolNd):
    _dimensions = 2


class _AvgPoolNd(Module):
    def __init__(self, kernel_size, stride=None, padding=0, count_include_pad=True):
        super().__init__()
        if type(count_include_pad) is not bool:
            raise TypeError("count_include_pad must be a Python bool.")
        self.kernel_size, self.stride, self.padding, _ = F._pool_parameters(kernel_size, stride, padding, 1, self._dimensions)
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        function = F.avg_pool1d if self._dimensions == 1 else F.avg_pool2d
        return function(input, self.kernel_size, self.stride, self.padding, self.count_include_pad)

    def __repr__(self):
        return f"{type(self).__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, count_include_pad={self.count_include_pad})"


class AvgPool1d(_AvgPoolNd):
    _dimensions = 1


class AvgPool2d(_AvgPoolNd):
    _dimensions = 2


__all__ = ["MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d"]
