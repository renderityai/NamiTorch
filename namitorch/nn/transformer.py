from ..random import Generator, _validate_generator
from ..tensor import Tensor
from . import functional as F
from .attention import CausalSelfAttention
from .cache import LayerKVCache
from .dropout import Dropout
from .linear import Linear
from .module import Module
from .normalization import RMSNorm


class SwiGLU(Module):
    def __init__(self, d_model: int, hidden_dim: int, *, bias: bool = True, dropout: float = 0.0, dtype=None, generator: Generator | None = None):
        super().__init__()
        self.d_model = F._positive_dimension(d_model, "d_model")
        self.hidden_dim = F._positive_dimension(hidden_dim, "hidden_dim")
        if type(bias) is not bool:
            raise TypeError("bias must be a Python bool.")
        dropout = F._dropout_probability(dropout)
        _validate_generator(generator)
        self.gate_proj = Linear(self.d_model, self.hidden_dim, bias, dtype=dtype, generator=generator)
        self.up_proj = Linear(self.d_model, self.hidden_dim, bias, dtype=dtype, generator=generator)
        self.down_proj = Linear(self.hidden_dim, self.d_model, bias, dtype=dtype, generator=generator)
        self.down_proj.is_residual_projection = True
        self.dropout = Dropout(dropout, generator=generator)

    def forward(self, input: Tensor) -> Tensor:
        if not isinstance(input, Tensor):
            raise TypeError("SwiGLU input must be a NamiTorch Tensor.")
        if input.ndim < 1 or input.shape[-1] != self.d_model:
            raise ValueError(f"SwiGLU expects last dimension {self.d_model}, got {input.shape}.")
        if not input.dtype.is_floating_point:
            raise TypeError("SwiGLU requires floating input.")
        hidden = F.swiglu(self.gate_proj(input), self.up_proj(input))
        return self.dropout(self.down_proj(hidden))

    def __repr__(self) -> str:
        return f"SwiGLU(d_model={self.d_model}, hidden_dim={self.hidden_dim}, bias={self.gate_proj.bias is not None}, dropout={self.dropout.p})"


class TransformerBlock(Module):
    def __init__(self, d_model: int, n_heads: int, hidden_dim: int, n_kv_heads: int | None = None, *, bias: bool = True, eps: float = 1e-5, attn_dropout: float = 0.0, resid_dropout: float = 0.0, mlp_dropout: float = 0.0, rotary_dim: int | None = None, rope_theta: float = 10000.0, max_seq_len: int = 2048, dtype=None, generator: Generator | None = None):
        super().__init__()
        self.d_model = F._positive_dimension(d_model, "d_model")
        self.hidden_dim = F._positive_dimension(hidden_dim, "hidden_dim")
        mlp_dropout = F._dropout_probability(mlp_dropout)
        self.attn_norm = RMSNorm(self.d_model, eps=eps, dtype=dtype)
        self.attention = CausalSelfAttention(
            self.d_model, n_heads, n_kv_heads, bias=bias, attn_dropout=attn_dropout,
            resid_dropout=resid_dropout, rotary_dim=rotary_dim, rope_theta=rope_theta,
            max_seq_len=max_seq_len, dtype=dtype, generator=generator,
        )
        self.ffn_norm = RMSNorm(self.d_model, eps=eps, dtype=dtype)
        self.mlp = SwiGLU(self.d_model, self.hidden_dim, bias=bias, dropout=mlp_dropout, dtype=dtype, generator=generator)

    def forward(self, input: Tensor, *, cache: LayerKVCache | tuple[Tensor, Tensor] | None = None, use_cache: bool = False, attn_mask: Tensor | None = None) -> Tensor | tuple[Tensor, LayerKVCache | tuple[Tensor, Tensor]]:
        if not isinstance(input, Tensor):
            raise TypeError("TransformerBlock input must be a NamiTorch Tensor.")
        if input.ndim != 3 or input.shape[-1] != self.d_model:
            raise ValueError(f"TransformerBlock expects [B, T, {self.d_model}], got {input.shape}.")
        if not input.dtype.is_floating_point:
            raise TypeError("TransformerBlock requires floating input.")
        offset = cache.length if isinstance(cache, LayerKVCache) else None
        try:
            result = self.attention(self.attn_norm(input), cache=cache, use_cache=use_cache, attn_mask=attn_mask)
            if use_cache:
                attn_output, new_cache = result
            else:
                attn_output = result
            residual = input + attn_output
            output = residual + self.mlp(self.ffn_norm(residual))
        except Exception:
            if isinstance(cache, LayerKVCache) and use_cache:
                cache._length = offset
            raise
        if use_cache:
            return output, new_cache
        return output

    def __repr__(self) -> str:
        return (
            f"TransformerBlock(d_model={self.d_model}, n_heads={self.attention.n_heads}, "
            f"hidden_dim={self.hidden_dim}, n_kv_heads={self.attention.n_kv_heads}, "
            f"bias={self.attention.q_proj.bias is not None}, eps={self.attn_norm.eps}, "
            f"attn_dropout={self.attention.attn_dropout}, resid_dropout={self.attention.resid_dropout.p}, "
            f"mlp_dropout={self.mlp.dropout.p}, rotary_dim={self.attention.rope.rotary_dim}, "
            f"max_seq_len={self.attention.rope.max_seq_len})"
        )


__all__ = ["SwiGLU", "TransformerBlock"]
