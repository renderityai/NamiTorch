from ..autograd import is_grad_enabled
from ..ops import cat
from ..random import Generator, _validate_generator
from ..tensor import Tensor
from . import functional as F
from .cache import LayerKVCache
from .dropout import Dropout
from .linear import Linear
from .module import Module
from .rotary import RotaryEmbedding


class CausalSelfAttention(Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None, *, bias: bool = True, attn_dropout: float = 0.0, resid_dropout: float = 0.0, rotary_dim: int | None = None, rope_theta: float = 10000.0, max_seq_len: int = 2048, dtype=None, generator: Generator | None = None):
        super().__init__()
        self.d_model = F._positive_dimension(d_model, "d_model")
        self.n_heads = F._positive_dimension(n_heads, "n_heads")
        self.n_kv_heads = self.n_heads if n_kv_heads is None else F._positive_dimension(n_kv_heads, "n_kv_heads")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if type(bias) is not bool:
            raise TypeError("bias must be a Python bool.")
        self.head_dim = self.d_model // self.n_heads
        self.attn_dropout = F._dropout_probability(attn_dropout)
        resid_dropout = F._dropout_probability(resid_dropout)
        _validate_generator(generator)
        self.generator = generator
        self.rope = RotaryEmbedding(self.head_dim, rotary_dim, theta=rope_theta, max_seq_len=max_seq_len)
        self.q_proj = Linear(self.d_model, self.n_heads * self.head_dim, bias, dtype=dtype, generator=generator)
        self.k_proj = Linear(self.d_model, self.n_kv_heads * self.head_dim, bias, dtype=dtype, generator=generator)
        self.v_proj = Linear(self.d_model, self.n_kv_heads * self.head_dim, bias, dtype=dtype, generator=generator)
        self.o_proj = Linear(self.d_model, self.d_model, bias, dtype=dtype, generator=generator)
        self.o_proj.is_residual_projection = True
        self.resid_dropout = Dropout(resid_dropout, generator=generator)

    def _cache_length(self, cache, batch: int) -> int:
        if cache is None:
            return 0
        if isinstance(cache, LayerKVCache):
            cache.validate(batch, self.n_kv_heads, self.head_dim)
            return cache.length
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise TypeError("cache must be LayerKVCache, a (key, value) tuple of Tensors or None.")
        for name, tensor in zip(("key", "value"), cache):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Cached {name} must be a NamiTorch Tensor.")
            if tensor.ndim != 4 or tensor.shape[:2] != (batch, self.n_kv_heads) or tensor.shape[-1] != self.head_dim:
                raise ValueError(f"Cached {name} must have shape [{batch}, {self.n_kv_heads}, T, {self.head_dim}], got {tensor.shape}.")
            if not tensor.dtype.is_floating_point:
                raise TypeError(f"Cached {name} must have a floating dtype.")
        if cache[0].shape != cache[1].shape:
            raise ValueError("Cached key and value must have the same shape.")
        return cache[0].shape[2]

    def _repeat_kv(self, input: Tensor) -> Tensor:
        if self.n_kv_heads == self.n_heads:
            return input
        return input.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

    def forward(self, input: Tensor, *, cache: LayerKVCache | tuple[Tensor, Tensor] | None = None, use_cache: bool = False, attn_mask: Tensor | None = None) -> Tensor | tuple[Tensor, LayerKVCache | tuple[Tensor, Tensor]]:
        if not isinstance(input, Tensor):
            raise TypeError("CausalSelfAttention input must be a NamiTorch Tensor.")
        if input.ndim != 3 or input.shape[-1] != self.d_model:
            raise ValueError(f"CausalSelfAttention expects [B, T, {self.d_model}], got {input.shape}.")
        if not input.dtype.is_floating_point:
            raise TypeError("CausalSelfAttention requires floating input.")
        if type(use_cache) is not bool:
            raise TypeError("use_cache must be a Python bool.")
        batch, length, _ = input.shape
        if length == 0:
            raise ValueError("CausalSelfAttention requires at least one input token.")
        offset = self._cache_length(cache, batch)
        mutable_cache = isinstance(cache, LayerKVCache)
        if mutable_cache and use_cache:
            if is_grad_enabled():
                raise RuntimeError("Writing LayerKVCache is inference-only; use no_grad().")
            cache.validate(batch, self.n_kv_heads, self.head_dim, append_length=length)
        if offset + length > self.rope.max_seq_len:
            raise ValueError(f"Sequence including cache exceeds max_seq_len={self.rope.max_seq_len}.")
        query = self.q_proj(input).reshape(batch, length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.k_proj(input).reshape(batch, length, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.v_proj(input).reshape(batch, length, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        if mutable_cache:
            cache.validate(batch, self.n_kv_heads, self.head_dim, key.dtype)
            cache.validate(batch, self.n_kv_heads, self.head_dim, value.dtype)
        elif cache is not None and (cache[0].dtype != key.dtype or cache[1].dtype != value.dtype):
            raise TypeError("Cache dtype must match the corresponding projected key/value dtype.")
        query = self.rope(query, offset=offset)
        key = self.rope(key, offset=offset)
        if mutable_cache and use_cache:
            cache.append(key, value)
            key, value = cache.get()
        elif cache is not None:
            previous_key, previous_value = cache.get() if mutable_cache else cache
            key = cat((previous_key, key), dim=2)
            value = cat((previous_value, value), dim=2)
        try:
            output = F.scaled_dot_product_attention(
                query, self._repeat_kv(key), self._repeat_kv(value), attn_mask,
                dropout_p=self.attn_dropout, is_causal=True, query_position_offset=offset,
                training=self.training, generator=self.generator,
            )
            output = output.transpose(1, 2).reshape(batch, length, self.d_model)
            output = self.resid_dropout(self.o_proj(output))
        except Exception:
            if mutable_cache and use_cache:
                cache._length = offset
            raise
        if use_cache:
            return output, cache if mutable_cache else (key, value)
        return output

    def __repr__(self) -> str:
        return (
            f"CausalSelfAttention(d_model={self.d_model}, n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, "
            f"bias={self.q_proj.bias is not None}, attn_dropout={self.attn_dropout}, "
            f"resid_dropout={self.resid_dropout.p}, rotary_dim={self.rope.rotary_dim}, "
            f"rope_theta={self.rope.theta}, max_seq_len={self.rope.max_seq_len})"
        )


__all__ = ["CausalSelfAttention"]
