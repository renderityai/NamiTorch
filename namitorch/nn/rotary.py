import numpy as np

from ..backends import namespace

from ..ops import cat, stack
from ..tensor import Tensor
from . import functional as F
from .module import Module


class RotaryEmbedding(Module):
    def __init__(self, head_dim: int, rotary_dim: int | None = None, *, theta: float = 10000.0, max_seq_len: int = 2048, initial_cache_length: int = 0):
        super().__init__()
        self.head_dim = F._positive_dimension(head_dim, "head_dim")
        self.rotary_dim = self.head_dim if rotary_dim is None else F._positive_dimension(rotary_dim, "rotary_dim")
        if self.rotary_dim > self.head_dim or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be even and no greater than head_dim.")
        self.theta = F._real_parameter(theta, "theta")
        if self.theta <= 0:
            raise ValueError("theta must be positive.")
        self.max_seq_len = F._positive_dimension(max_seq_len, "max_seq_len")
        initial_cache_length = F._integer(initial_cache_length, "initial_cache_length")
        if not 0 <= initial_cache_length <= self.max_seq_len:
            raise ValueError("initial_cache_length must be between zero and max_seq_len.")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            inv_freq = 1 / np.power(self.theta, np.arange(0, self.rotary_dim, 2, dtype=np.float64) / self.rotary_dim)
        if not np.all(np.isfinite(inv_freq)):
            raise ValueError("theta produces nonfinite rotary frequencies.")
        self.register_buffer("inv_freq", Tensor(inv_freq), persistent=False)
        self.register_buffer("cos_cached", Tensor(np.empty((0, self.rotary_dim // 2), dtype=np.float64)), persistent=False)
        self.register_buffer("sin_cached", Tensor(np.empty((0, self.rotary_dim // 2), dtype=np.float64)), persistent=False)
        self._ensure_cache(initial_cache_length)

    def _ensure_cache(self, required_length: int) -> None:
        xp = namespace(self.inv_freq)
        if required_length > self.max_seq_len:
            raise ValueError(f"RoPE positions require length {required_length}, exceeding max_seq_len={self.max_seq_len}.")
        current_length = self.cos_cached.shape[0]
        if required_length <= current_length:
            return
        length = min(self.max_seq_len, max(required_length, 2 * current_length))
        with np.errstate(over="ignore", invalid="ignore"):
            angles = xp.arange(length, dtype=np.float64)[:, None] * self.inv_freq._data[None, :]
        if not xp.all(xp.isfinite(angles)):
            raise ValueError("Requested positions produce nonfinite rotary angles.")
        self.cos_cached = Tensor._from_array(xp.cos(angles), False)
        self.sin_cached = Tensor._from_array(xp.sin(angles), False)

    def forward(self, input: Tensor, offset: int = 0) -> Tensor:
        if not isinstance(input, Tensor):
            raise TypeError("RotaryEmbedding input must be a NamiTorch Tensor.")
        if input.ndim != 4 or input.shape[-1] != self.head_dim:
            raise ValueError(f"RotaryEmbedding expects [B, H, T, {self.head_dim}], got {input.shape}.")
        if not input.dtype.is_floating_point:
            raise TypeError("RotaryEmbedding requires floating input.")
        offset = F._integer(offset, "offset")
        if offset < 0:
            raise ValueError("offset must be nonnegative.")
        length = input.shape[2]
        self._ensure_cache(offset + length)
        cos = self.cos_cached[offset:offset + length].astype(input.dtype).reshape(1, 1, length, self.rotary_dim // 2)
        sin = self.sin_cached[offset:offset + length].astype(input.dtype).reshape(1, 1, length, self.rotary_dim // 2)
        even = input[..., :self.rotary_dim:2]
        odd = input[..., 1:self.rotary_dim:2]
        rotated = stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        rotated = rotated.reshape(*input.shape[:-1], self.rotary_dim)
        if self.rotary_dim == self.head_dim:
            return rotated
        return cat((rotated, input[..., self.rotary_dim:]), dim=-1)

    def __repr__(self) -> str:
        return f"RotaryEmbedding(head_dim={self.head_dim}, rotary_dim={self.rotary_dim}, theta={self.theta}, max_seq_len={self.max_seq_len})"


__all__ = ["RotaryEmbedding"]
