from ..backends import writable
from ..autograd import no_grad
from ..creation import ones, zeros
from ..dtype import get_default_dtype, int64, normalize_dtype
from ..tensor import Tensor
from . import functional as F
from . import init
from .module import Module
from .parameter import Parameter


class _BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, *, dtype=None):
        super().__init__()
        self.num_features = F._positive_dimension(num_features, "num_features")
        self.eps = F._normalization_eps(eps)
        self.momentum = F._batch_norm_momentum(momentum)
        if type(affine) is not bool or type(track_running_stats) is not bool:
            raise TypeError("affine and track_running_stats must be Python bools.")
        self.affine = affine
        self.track_running_stats = track_running_stats
        dtype = get_default_dtype() if dtype is None else normalize_dtype(dtype)
        if not dtype.is_floating_point:
            raise TypeError("BatchNorm requires a floating dtype.")
        self.register_parameter("weight", Parameter(ones(num_features, dtype=dtype)) if affine else None)
        self.register_parameter("bias", Parameter(zeros(num_features, dtype=dtype)) if affine else None)
        self.register_buffer("running_mean", zeros(num_features, dtype=dtype) if track_running_stats else None)
        self.register_buffer("running_var", ones(num_features, dtype=dtype) if track_running_stats else None)
        self.register_buffer("num_batches_tracked", zeros((), dtype=int64) if track_running_stats else None)

    def reset_running_stats(self):
        if self.track_running_stats:
            init.zeros_(self.running_mean)
            init.ones_(self.running_var)
            init.zeros_(self.num_batches_tracked)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        if not isinstance(input, Tensor):
            raise TypeError("BatchNorm input must be a NamiTorch Tensor.")
        if input.ndim not in self._input_ranks or input.shape[1] != self.num_features:
            raise ValueError(f"{type(self).__name__} expects rank in {self._input_ranks} and {self.num_features} channels, got {input.shape}.")
        next_count = None
        if self.training and self.track_running_stats:
            counter = self.num_batches_tracked
            if not isinstance(counter, Tensor) or counter.shape != () or counter.dtype is not int64:
                raise ValueError("num_batches_tracked must be a scalar int64 Tensor.")
            if not writable(counter._data):
                raise RuntimeError("Cannot update read-only num_batches_tracked.")
            count = counter.item()
            if not 0 <= count < 2 ** 63 - 1:
                raise OverflowError("num_batches_tracked is outside its incrementable int64 range.")
            next_count = count + 1
        output = F.batch_norm(
            input, self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps,
        )
        if next_count is not None:
            with no_grad():
                self.num_batches_tracked.fill_(next_count)
        return output

    def __repr__(self):
        return f"{type(self).__name__}(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats})"


class BatchNorm1d(_BatchNorm):
    _input_ranks = (2, 3)


class BatchNorm2d(_BatchNorm):
    _input_ranks = (4,)


__all__ = ["BatchNorm1d", "BatchNorm2d"]
