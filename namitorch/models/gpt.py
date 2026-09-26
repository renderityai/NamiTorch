import math
from dataclasses import dataclass

import numpy as np

from ..autograd import no_grad
from ..nn import Dropout, Embedding, LayerKVCache, Linear, Module, ModuleList, RMSNorm, TransformerBlock, init
from ..nn import functional as F
from ..random import Generator, _validate_generator
from ..tensor import Tensor
from .generation import _sampling_options, sample_next_token


@dataclass(frozen=True, slots=True)
class GPTConfig:
    vocab_size: int = 32000
    context_length: int = 1024
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int | None = None
    d_model: int = 512
    d_ff: int = 1536
    emb_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    mlp_dropout: float = 0.0
    rope_theta: float = 10000.0
    rotary_dim: int | None = None
    eps: float = 1e-5
    initializer_range: float = 0.02
    bias: bool = False
    tie_weights: bool = True

    def __post_init__(self):
        for name in ("vocab_size", "context_length", "n_layers", "n_heads", "d_model", "d_ff"):
            object.__setattr__(self, name, F._positive_dimension(getattr(self, name), name))
        kv_heads = self.n_heads if self.n_kv_heads is None else F._positive_dimension(self.n_kv_heads, "n_kv_heads")
        object.__setattr__(self, "n_kv_heads", kv_heads)
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        head_dim = self.d_model // self.n_heads
        rotary_dim = head_dim if self.rotary_dim is None else F._positive_dimension(self.rotary_dim, "rotary_dim")
        if rotary_dim > head_dim or rotary_dim % 2:
            raise ValueError("rotary_dim must be even and no greater than head_dim.")
        object.__setattr__(self, "rotary_dim", rotary_dim)
        for name in ("emb_dropout", "attn_dropout", "resid_dropout", "mlp_dropout"):
            object.__setattr__(self, name, F._dropout_probability(getattr(self, name)))
        for name in ("rope_theta", "eps", "initializer_range"):
            value = F._real_parameter(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        for name in ("bias", "tie_weights"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a Python bool.")


@dataclass(frozen=True, slots=True, eq=False)
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    cache: list[LayerKVCache | tuple[Tensor, Tensor]] | None = None


class GPT(Module):
    def __init__(self, config: GPTConfig, *, dtype=None, generator: Generator | None = None):
        super().__init__()
        if not isinstance(config, GPTConfig):
            raise TypeError("GPT config must be a GPTConfig.")
        _validate_generator(generator)
        self.config = config
        self.token_embedding = Embedding(config.vocab_size, config.d_model, dtype=dtype, generator=generator)
        self.embedding_dropout = Dropout(config.emb_dropout, generator=generator)
        self.blocks = ModuleList(
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.n_kv_heads,
                bias=config.bias, eps=config.eps, attn_dropout=config.attn_dropout,
                resid_dropout=config.resid_dropout, mlp_dropout=config.mlp_dropout,
                rotary_dim=config.rotary_dim, rope_theta=config.rope_theta,
                max_seq_len=config.context_length, dtype=dtype, generator=generator,
            )
            for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.eps, dtype=dtype)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False, dtype=dtype, generator=generator)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        self.reset_parameters(generator=generator)

    def reset_parameters(self, *, generator: Generator | None = None) -> None:
        _validate_generator(generator)
        initialized = set()

        def initialize(module):
            if isinstance(module, (Embedding, Linear)):
                if id(module.weight) not in initialized:
                    std = self.config.initializer_range
                    if getattr(module, "is_residual_projection", False):
                        std /= math.sqrt(2 * self.config.n_layers)
                    init.normal_(module.weight, mean=0, std=std, generator=generator)
                    initialized.add(id(module.weight))
                if isinstance(module, Linear) and module.bias is not None and id(module.bias) not in initialized:
                    init.zeros_(module.bias)
                    initialized.add(id(module.bias))
            elif isinstance(module, RMSNorm) and module.weight is not None and id(module.weight) not in initialized:
                init.ones_(module.weight)
                initialized.add(id(module.weight))

        self.apply(initialize)

    def num_parameters(self, trainable_only: bool = False) -> int:
        if type(trainable_only) is not bool:
            raise TypeError("trainable_only must be a Python bool.")
        return sum(parameter.numel() for parameter in self.parameters() if not trainable_only or parameter.requires_grad)

    def create_cache(self, batch_size: int, *, preallocate: bool = True) -> list[LayerKVCache]:
        if type(preallocate) is not bool:
            raise TypeError("preallocate must be a Python bool.")
        return [
            LayerKVCache(
                batch_size, self.config.n_kv_heads, self.config.d_model // self.config.n_heads,
                max_seq_len=self.config.context_length if preallocate else None,
                dtype=self.token_embedding.weight.dtype,
            )
            for _ in self.blocks
        ]

    def _validate_tokens(self, tokens, name, shape=None, allow_ignore=False):
        if not isinstance(tokens, Tensor) or not tokens.dtype.is_integer:
            raise TypeError(f"{name} must be an integer NamiTorch Tensor.")
        if tokens.ndim != 2 or (shape is not None and tokens.shape != shape):
            expected = "[B, T]" if shape is None else str(shape)
            raise ValueError(f"{name} must have shape {expected}, got {tokens.shape}.")
        values = tokens._data
        invalid = (values < 0) | (values >= self.config.vocab_size)
        if allow_ignore:
            invalid &= values != -1
        if np.any(invalid):
            raise ValueError(f"{name} contains token IDs outside [0, {self.config.vocab_size}).")

    def _prepare_cache(self, cache, batch, length, use_cache):
        if cache is None:
            return [None] * self.config.n_layers, 0
        if not isinstance(cache, (list, tuple)) or len(cache) != self.config.n_layers:
            raise ValueError(f"cache must contain exactly {self.config.n_layers} layer caches.")
        layers = list(cache)
        lengths, seen = [], set()
        for block, entry in zip(self.blocks, layers):
            if entry is None:
                raise TypeError("Each layer cache must be LayerKVCache or a (key, value) tuple.")
            lengths.append(block.attention._cache_length(entry, batch))
            if isinstance(entry, LayerKVCache):
                if id(entry) in seen:
                    raise ValueError("Each layer must have a distinct LayerKVCache.")
                seen.add(id(entry))
                entry.validate(batch, self.config.n_kv_heads, self.config.d_model // self.config.n_heads, self.token_embedding.weight.dtype, length if use_cache else 0)
            elif any(tensor.dtype != self.token_embedding.weight.dtype for tensor in entry):
                raise TypeError("Cache dtype must match the model dtype.")
        if len(set(lengths)) != 1:
            raise ValueError("All layer caches must have the same length.")
        return layers, lengths[0]

    def forward(self, input_ids: Tensor, targets: Tensor | None = None, *, cache: list[LayerKVCache | tuple[Tensor, Tensor]] | None = None, use_cache: bool = False, attn_mask: Tensor | None = None) -> CausalLMOutput:
        if type(use_cache) is not bool:
            raise TypeError("use_cache must be a Python bool.")
        self._validate_tokens(input_ids, "input_ids")
        batch, length = input_ids.shape
        if length == 0:
            raise ValueError("GPT requires at least one input token.")
        if targets is not None:
            self._validate_tokens(targets, "targets", input_ids.shape, allow_ignore=True)
        layer_caches, offset = self._prepare_cache(cache, batch, length, use_cache)
        if offset + length > self.config.context_length:
            raise ValueError(f"Sequence including cache exceeds context_length={self.config.context_length}.")
        new_cache = [] if use_cache else None
        try:
            hidden = self.embedding_dropout(self.token_embedding(input_ids))
            for block, entry in zip(self.blocks, layer_caches):
                result = block(hidden, cache=entry, use_cache=use_cache, attn_mask=attn_mask)
                if use_cache:
                    hidden, updated = result
                    new_cache.append(updated)
                else:
                    hidden = result
            logits = self.lm_head(self.final_norm(hidden))
            loss = None if targets is None else F.cross_entropy(
                logits.reshape(batch * length, self.config.vocab_size), targets.reshape(batch * length), ignore_index=-1,
            )
        except Exception:
            if use_cache:
                for entry in layer_caches:
                    if isinstance(entry, LayerKVCache):
                        entry._length = offset
            raise
        return CausalLMOutput(logits=logits, loss=loss, cache=new_cache)

    def generate(
        self, input_ids: Tensor, max_new_tokens: int = 20, *, temperature: float = 1.0,
        top_k: int | None = None, top_p: float = 1.0, repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
        eos_token_id: int | None = None, pad_token_id: int | None = None,
        generator: Generator | None = None,
    ) -> Tensor:
        self._validate_tokens(input_ids, "input_ids")
        batch, prompt_length = input_ids.shape
        if batch == 0 or prompt_length == 0:
            raise ValueError("generate requires a nonempty batch and prompt.")
        max_new_tokens = F._integer(max_new_tokens, "max_new_tokens")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")
        if prompt_length + max_new_tokens > self.config.context_length:
            raise ValueError("Prompt length plus max_new_tokens exceeds context_length.")
        options = _sampling_options(temperature, top_k, top_p, repetition_penalty, frequency_penalty, presence_penalty)
        _validate_generator(generator)
        for name, token in (("eos_token_id", eos_token_id), ("pad_token_id", pad_token_id)):
            if token is not None:
                value = F._integer(token, name)
                if not 0 <= value < self.config.vocab_size:
                    raise ValueError(f"{name} must be within the vocabulary.")
                if name == "eos_token_id":
                    eos_token_id = value
                else:
                    pad_token_id = value
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            with no_grad():
                if max_new_tokens == 0:
                    return input_ids.clone()
                history = np.empty((batch, prompt_length + max_new_tokens), dtype=np.int64)
                history[:, :prompt_length] = input_ids._data
                length = prompt_length
                finished = np.zeros(batch, dtype=np.bool_)
                cache = self.create_cache(batch)
                result = self(input_ids, cache=cache, use_cache=True)
                for step in range(max_new_tokens):
                    active = np.flatnonzero(~finished)
                    sampled = sample_next_token(
                        result.logits[active, -1, :], Tensor._from_array(history[active, :length], False),
                        **options, generator=generator,
                    )
                    fill = pad_token_id if pad_token_id is not None else eos_token_id
                    next_tokens = np.full((batch, 1), 0 if fill is None else fill, dtype=np.int64)
                    next_tokens[active] = sampled._data
                    history[:, length:length + 1] = next_tokens
                    length += 1
                    if eos_token_id is not None:
                        finished[active] |= sampled._data[:, 0] == eos_token_id
                    if np.all(finished) or step + 1 == max_new_tokens:
                        break
                    result = self(Tensor._from_array(next_tokens, False), cache=result.cache, use_cache=True)
                return Tensor(history[:, :length])
        finally:
            for module, training in modes:
                module.training = training

    def __repr__(self):
        return f"GPT(config={self.config!r})"


__all__ = ["GPTConfig", "GPT", "CausalLMOutput"]
