import os
import sys
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 259
    context_length: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    d_model: int = 768
    d_ff: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    bias: bool = False


@dataclass
class TrainConfig:
    train_file: str = "data/train.txt"
    val_file: Optional[str] = None
    output_dir: str = "checkpoints"
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_steps: int = 10000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_steps: int = 50
    save_interval: int = 500
    log_interval: int = 10
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    resume: Optional[str] = None


class ByteTokenizer:
    PAD = 256
    BOS = 257
    EOS = 258

    def __init__(self):
        self.vocab_size = 259

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[int]:
        tokens = list(text.encode("utf-8", errors="replace"))
        if add_bos:
            tokens.insert(0, self.BOS)
        if add_eos:
            tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        values = []
        for token in tokens:
            if 0 <= token <= 255:
                values.append(token)
        return bytes(values).decode("utf-8", errors="replace")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (x_norm * self.weight.float()).to(dtype)


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (indices / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0
) -> torch.Tensor:
    seq_len = x.size(-2)
    cos = cos[offset:offset + seq_len]
    sin = sin[offset:offset + seq_len]
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.groups = config.n_heads // config.n_kv_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(
            config.d_model,
            config.n_heads * self.head_dim,
            bias=config.bias
        )

        self.k_proj = nn.Linear(
            config.d_model,
            config.n_kv_heads * self.head_dim,
            bias=config.bias
        )

        self.v_proj = nn.Linear(
            config.d_model,
            config.n_kv_heads * self.head_dim,
            bias=config.bias
        )

        self.o_proj = nn.Linear(
            config.n_heads * self.head_dim,
            config.d_model,
            bias=config.bias
        )

        self.resid_dropout = nn.Dropout(config.dropout)

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.groups == 1:
            return x
        return x.repeat_interleave(self.groups, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ):
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            bsz,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            bsz,
            seq_len,
            self.n_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            bsz,
            seq_len,
            self.n_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        cache_len = 0

        if cache is not None:
            cache_len = cache[0].size(-2)

        q = apply_rope(
            q,
            rope_cos,
            rope_sin,
            offset=cache_len
        )

        k = apply_rope(
            k,
            rope_cos,
            rope_sin,
            offset=cache_len
        )

        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat((cached_k, k), dim=-2)
            v = torch.cat((cached_v, v), dim=-2)

        new_cache = (k, v) if use_cache else None

        k_attn = self.repeat_kv(k)
        v_attn = self.repeat_kv(v)

        if hasattr(F, "scaled_dot_product_attention"):
            if cache is None:
                attn = F.scaled_dot_product_attention(
                    q,
                    k_attn,
                    v_attn,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True
                )
            else:
                total_len = k_attn.size(-2)
                q_len = q.size(-2)

                query_positions = torch.arange(
                    cache_len,
                    cache_len + q_len,
                    device=x.device
                )

                key_positions = torch.arange(
                    total_len,
                    device=x.device
                )

                mask = (
                    key_positions.unsqueeze(0)
                    <= query_positions.unsqueeze(1)
                )

                mask = mask.unsqueeze(0).unsqueeze(0)

                attn = F.scaled_dot_product_attention(
                    q,
                    k_attn,
                    v_attn,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False
                )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)

            scores = torch.matmul(
                q,
                k_attn.transpose(-2, -1)
            ) * scale

            q_len = q.size(-2)
            k_len = k_attn.size(-2)

            query_positions = torch.arange(
                cache_len,
                cache_len + q_len,
                device=x.device
            )

            key_positions = torch.arange(
                k_len,
                device=x.device
            )

            mask = (
                key_positions.unsqueeze(0)
                > query_positions.unsqueeze(1)
            )

            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0),
                float("-inf")
            )

            probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
            probs = F.dropout(
                probs,
                p=self.dropout,
                training=self.training
            )

            attn = torch.matmul(probs, v_attn)

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(
            bsz,
            seq_len,
            self.n_heads * self.head_dim
        )

        out = self.o_proj(attn)
        out = self.resid_dropout(out)

        return out, new_cache


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=config.bias
        )

        self.up_proj = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=config.bias
        )

        self.down_proj = nn.Linear(
            config.d_ff,
            config.d_model,
            bias=config.bias
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.attn_norm = RMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        self.ffn_norm = RMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        self.attn = CausalSelfAttention(config)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cache=None,
        use_cache=False
    ):
        attn_out, new_cache = self.attn(
            self.attn_norm(x),
            rope_cos,
            rope_sin,
            cache=cache,
            use_cache=use_cache
        )

        x = x + attn_out
        x = x + self.mlp(self.ffn_norm(x))

        return x, new_cache


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight"):
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layers)
                )

            if name.endswith("down_proj.weight"):
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layers)
                )

        self.register_buffer(
            "rope_cos",
            torch.empty(0),
            persistent=False
        )

        self.register_buffer(
            "rope_sin",
            torch.empty(0),
            persistent=False
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def build_rope(self, device: torch.device):
        head_dim = self.config.d_model // self.config.n_heads

        cos, sin = precompute_rope_frequencies(
            head_dim,
            self.config.context_length,
            self.config.rope_theta,
            device
        )

        self.rope_cos = cos
        self.rope_sin = sin

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        caches=None,
        use_cache: bool = False
    ):
        device = input_ids.device

        if (
            self.rope_cos.numel() == 0
            or self.rope_cos.device != device
        ):
            self.build_rope(device)

        batch_size, seq_len = input_ids.shape

        cache_len = 0

        if caches is not None and len(caches) > 0:
            if caches[0] is not None:
                cache_len = caches[0][0].size(-2)

        if cache_len + seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {cache_len + seq_len} exceeds context length "
                f"{self.config.context_length}"
            )

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        new_caches = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            cache = None

            if caches is not None:
                cache = caches[i]

            x, new_cache = block(
                x,
                self.rope_cos,
                self.rope_sin,
                cache=cache,
                use_cache=use_cache
            )

            if use_cache:
                new_caches.append(new_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1
            )

        if use_cache:
            return logits, loss, new_caches

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        self.eval()

        device = input_ids.device

        if input_ids.size(1) > self.config.context_length:
            input_ids = input_ids[:, -self.config.context_length:]

        generated = input_ids.clone()
        caches = None

        if generated.size(1) > 1:
            prefill = generated[:, :-1]

            _, _, caches = self(
                prefill,
                caches=None,
                use_cache=True
            )

            current = generated[:, -1:]
        else:
            current = generated

        for _ in range(max_new_tokens):
            if generated.size(1) >= self.config.context_length:
                context = generated[:, -self.config.context_length:]
                logits, _ = self(context)
                caches = None
            else:
                logits, _, caches = self(
                    current,
                    caches=caches,
                    use_cache=True
                )

            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for batch_index in range(generated.size(0)):
                    unique_tokens = torch.unique(generated[batch_index])

                    token_logits = logits[
                        batch_index,
                        unique_tokens
                    ]

                    token_logits = torch.where(
                        token_logits < 0,
                        token_logits * repetition_penalty,
                        token_logits / repetition_penalty
                    )

                    logits[
                        batch_index,
                        unique_tokens
                    ] = token_logits

            if temperature <= 0:
                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )
            else:
                logits = logits / temperature

                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))

                    threshold = torch.topk(
                        logits,
                        k
                    ).values[:, -1:]

                    logits = torch.where(
                        logits < threshold,
                        torch.full_like(
                            logits,
                            float("-inf")
                        ),
                        logits
                    )

                if top_p is not None and 0 < top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(
                        logits,
                        descending=True,
                        dim=-1
                    )

                    sorted_probs = F.softmax(
                        sorted_logits,
                        dim=-1
                    )

                    cumulative_probs = torch.cumsum(
                        sorted_probs,
                        dim=-1
                    )

                    remove_mask = cumulative_probs > top_p

                    remove_mask[..., 1:] = (
                        remove_mask[..., :-1].clone()
                    )

                    remove_mask[..., 0] = False

                    sorted_logits = sorted_logits.masked_fill(
                        remove_mask,
                        float("-inf")
                    )

                    filtered_logits = torch.full_like(
                        logits,
                        float("-inf")
                    )

                    filtered_logits.scatter_(
                        -1,
                        sorted_indices,
                        sorted_logits
                    )

                    logits = filtered_logits

                probs = F.softmax(logits, dim=-1)

                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )

            generated = torch.cat(
                (generated, next_token),
                dim=1
            )

            current = next_token

            if (
                eos_token_id is not None
                and torch.all(
                    next_token == eos_token_id
                )
            ):
                break

        return generated

    def configure_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
        device_type: str
    ):
        params = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad
        }

        decay_params = [
            p
            for _, p in params.items()
            if p.dim() >= 2
        ]

        no_decay_params = [
            p
            for _, p in params.items()
            if p.dim() < 2
        ]

        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": weight_decay
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0
            }
        ]

        fused_available = (
            "fused"
            in torch.optim.AdamW.__init__.__code__.co_varnames
        )

        use_fused = (
            fused_available
            and device_type == "cuda"
        )

        extra = {}

        if fused_available:
            extra["fused"] = use_fused

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra
        )

        return optimizer

    def num_parameters(self, non_embedding=False):
        total = sum(
            p.numel()
            for p in self.parameters()
        )

        if non_embedding:
            total -= self.token_embedding.weight.numel()

        return total


class TokenDataset:
    def __init__(
        self,
        path: str,
        tokenizer: ByteTokenizer,
        add_eos_between_documents: bool = True
    ):
        self.path = path
        self.tokenizer = tokenizer

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as f:
            text = f.read()

        tokens = tokenizer.encode(text)

        if add_eos_between_documents:
            tokens.append(tokenizer.EOS)

        if len(tokens) < 2:
            raise ValueError(
                f"Dataset {path} is too small"
            )

        dtype = (
            torch.int16
            if tokenizer.vocab_size < 32768
            else torch.int32
        )

        self.tokens = torch.tensor(
            tokens,
            dtype=dtype
        )

    def __len__(self):
        return self.tokens.numel()

    def get_batch(
        self,
        batch_size: int,
        context_length: int,
        device: torch.device
    ):
        max_start = len(self.tokens) - context_length - 1

        if max_start <= 0:
            raise ValueError(
                "Dataset is smaller than context_length"
            )

        starts = torch.randint(
            0,
            max_start,
            (batch_size,)
        )

        x = torch.stack([
            self.tokens[
                i:i + context_length
            ].long()
            for i in starts
        ])

        y = torch.stack([
            self.tokens[
                i + 1:i + 1 + context_length
            ].long()
            for i in starts
        ])

        if device.type == "cuda":
            x = x.pin_memory().to(
                device,
                non_blocking=True
            )

            y = y.pin_memory().to(
                device,
                non_blocking=True
            )
        else:
            x = x.to(device)
            y = y.to(device)

        return x, y


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


def get_dtype(
    requested: str,
    device: torch.device
):
    if requested == "float32":
        return torch.float32

    if requested == "float16":
        return torch.float16

    if requested == "bfloat16":
        return torch.bfloat16

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if device.type == "mps":
        return torch.float16

    return torch.float32


def get_autocast_context(
    device: torch.device,
    dtype: torch.dtype
):
    if device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=dtype
        )

    if device.type == "mps":
        return torch.autocast(
            device_type="mps",
            dtype=dtype
        )

    return torch.autocast(
        device_type="cpu",
        enabled=False
    )


def get_lr(
    step: int,
    config: TrainConfig
) -> float:
    if step < config.warmup_steps:
        return (
            config.learning_rate
            * (step + 1)
            / max(1, config.warmup_steps)
        )

    if step >= config.max_steps:
        return config.min_learning_rate

    decay_ratio = (
        step - config.warmup_steps
    ) / max(
        1,
        config.max_steps - config.warmup_steps
    )

    coeff = 0.5 * (
        1.0 + math.cos(
            math.pi * decay_ratio
        )
    )

    return (
        config.min_learning_rate
        + coeff
        * (
            config.learning_rate
            - config.min_learning_rate
        )
    )


def save_checkpoint(
    path: str,
    model: GPT,
    optimizer,
    step: int,
    model_config: GPTConfig,
    train_config: TrainConfig,
    best_val_loss: Optional[float] = None
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "best_val_loss": best_val_loss
    }

    torch.save(state, path)


def load_checkpoint(
    path: str,
    device: torch.device
):
    return torch.load(
        path,
        map_location=device,
        weights_only=False
    )


@torch.no_grad()
def estimate_loss(
    model: GPT,
    dataset: TokenDataset,
    train_config: TrainConfig,
    device: torch.device,
    dtype: torch.dtype
):
    model.eval()

    losses = []

    for _ in range(train_config.eval_steps):
        x, y = dataset.get_batch(
            train_config.batch_size,
            model.config.context_length,
            device
        )

        with get_autocast_context(
            device,
            dtype
        ):
            _, loss = model(x, y)

        losses.append(loss.detach().float())

    model.train()

    return torch.stack(losses).mean().item()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_number(value: int):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"{value / 1_000:.2f}K"

    return str(value)


def train(
    model_config: GPTConfig,
    train_config: TrainConfig
):
    seed_everything(train_config.seed)

    device = get_device(train_config.device)
    dtype = get_dtype(
        train_config.dtype,
        device
    )

    tokenizer = ByteTokenizer()

    train_dataset = TokenDataset(
        train_config.train_file,
        tokenizer
    )

    val_dataset = None

    if (
        train_config.val_file
        and os.path.exists(
            train_config.val_file
        )
    ):
        val_dataset = TokenDataset(
            train_config.val_file,
            tokenizer
        )

    start_step = 0
    best_val_loss = None

    if train_config.resume:
        checkpoint = load_checkpoint(
            train_config.resume,
            device
        )

        model_config = GPTConfig(
            **checkpoint["model_config"]
        )

        model = GPT(model_config).to(device)
        model.load_state_dict(checkpoint["model"])

        start_step = checkpoint.get(
            "step",
            0
        )

        best_val_loss = checkpoint.get(
            "best_val_loss"
        )
    else:
        model = GPT(model_config).to(device)

    raw_model = model

    optimizer = raw_model.configure_optimizer(
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(
            train_config.beta1,
            train_config.beta2
        ),
        device_type=device.type
    )

    if train_config.resume:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

    use_scaler = (
        device.type == "cuda"
        and dtype == torch.float16
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler
    )

    if train_config.compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model)

    os.makedirs(
        train_config.output_dir,
        exist_ok=True
    )

    params = raw_model.num_parameters()

    print(
        json.dumps(
            {
                "device": str(device),
                "dtype": str(dtype),
                "parameters": params,
                "parameters_human": format_number(params),
                "train_tokens": len(train_dataset),
                "context_length": model_config.context_length,
                "batch_size": train_config.batch_size,
                "gradient_accumulation_steps":
                    train_config.gradient_accumulation_steps,
                "effective_tokens_per_step":
                    train_config.batch_size
                    * model_config.context_length
                    * train_config.gradient_accumulation_steps
            },
            indent=2
        )
    )

    model.train()

    running_loss = 0.0
    running_steps = 0
    last_time = time.time()

    optimizer.zero_grad(
        set_to_none=True
    )

    for step in range(
        start_step,
        train_config.max_steps
    ):
        lr = get_lr(
            step,
            train_config
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        step_loss = 0.0

        for micro_step in range(
            train_config.gradient_accumulation_steps
        ):
            x, y = train_dataset.get_batch(
                train_config.batch_size,
                model_config.context_length,
                device
            )

            with get_autocast_context(
                device,
                dtype
            ):
                _, loss = model(x, y)

                loss = (
                    loss
                    / train_config.gradient_accumulation_steps
                )

            step_loss += loss.detach().float().item()

            scaler.scale(loss).backward()

        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(),
                train_config.grad_clip
            )

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss += step_loss
        running_steps += 1

        completed_step = step + 1

        if (
            completed_step
            % train_config.log_interval
            == 0
        ):
            now = time.time()
            elapsed = now - last_time

            avg_loss = (
                running_loss
                / max(1, running_steps)
            )

            tokens_processed = (
                train_config.log_interval
                * train_config.batch_size
                * model_config.context_length
                * train_config.gradient_accumulation_steps
            )

            tokens_per_second = (
                tokens_processed
                / max(elapsed, 1e-6)
            )

            print(
                json.dumps(
                    {
                        "step": completed_step,
                        "loss": round(avg_loss, 6),
                        "lr": lr,
                        "tokens_per_second":
                            round(tokens_per_second, 2)
                    }
                )
            )

            running_loss = 0.0
            running_steps = 0
            last_time = now

        if (
            val_dataset is not None
            and completed_step
            % train_config.eval_interval
            == 0
        ):
            val_loss = estimate_loss(
                raw_model,
                val_dataset,
                train_config,
                device,
                dtype
            )

            print(
                json.dumps(
                    {
                        "step": completed_step,
                        "val_loss": round(
                            val_loss,
                            6
                        ),
                        "val_perplexity": round(
                            math.exp(
                                min(val_loss, 20)
                            ),
                            4
                        )
                    }
                )
            )

            if (
                best_val_loss is None
                or val_loss < best_val_loss
            ):
                best_val_loss = val_loss

                save_checkpoint(
                    os.path.join(
                        train_config.output_dir,
                        "best.pt"
                    ),
                    raw_model,
                    optimizer,
                    completed_step,
                    model_config,
                    train_config,
                    best_val_loss
                )

        if (
            completed_step
            % train_config.save_interval
            == 0
        ):
            save_checkpoint(
                os.path.join(
                    train_config.output_dir,
                    f"step_{completed_step}.pt"
                ),
                raw_model,
                optimizer,
                completed_step,
                model_config,
                train_config,
                best_val_loss
            )

            save_checkpoint(
                os.path.join(
                    train_config.output_dir,
                    "latest.pt"
                ),
                raw_model,
                optimizer,
                completed_step,
                model_config,
                train_config,
                best_val_loss
            )

    save_checkpoint(
        os.path.join(
            train_config.output_dir,
            "final.pt"
        ),
        raw_model,
        optimizer,
        train_config.max_steps,
        model_config,
        train_config,
        best_val_loss
    )


def load_model_for_inference(
    checkpoint_path: str,
    device_string: str = "auto"
):
    device = get_device(device_string)

    checkpoint = load_checkpoint(
        checkpoint_path,
        device
    )

    config = GPTConfig(
        **checkpoint["model_config"]
    )

    model = GPT(config).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    return model, config, device


def generate_text(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device_string: str
):
    tokenizer = ByteTokenizer()

    model, config, device = load_model_for_inference(
        checkpoint_path,
        device_string
    )

    tokens = tokenizer.encode(
        prompt,
        add_bos=True
    )

    if len(tokens) >= config.context_length:
        tokens = tokens[
            -(config.context_length - 1):
        ]

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.EOS
    )

    generated_tokens = output[0].tolist()

    text = tokenizer.decode(
        generated_tokens
    )

    print(text)


def chat(
    checkpoint_path: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device_string: str,
    system_prompt: str
):
    tokenizer = ByteTokenizer()

    model, config, device = load_model_for_inference(
        checkpoint_path,
        device_string
    )

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (
            KeyboardInterrupt,
            EOFError
        ):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in {
            "/exit",
            "/quit",
            "exit",
            "quit"
        }:
            break

        if user_input.lower() == "/clear":
            history = []
            continue

        history.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        parts = []

        if system_prompt:
            parts.append(
                "<|system|>\n"
                + system_prompt
                + "\n"
            )

        for item in history:
            if item["role"] == "user":
                parts.append(
                    "<|user|>\n"
                    + item["content"]
                    + "\n"
                )
            else:
                parts.append(
                    "<|assistant|>\n"
                    + item["content"]
                    + "\n"
                )

        parts.append("<|assistant|>\n")

        prompt = "".join(parts)

        prompt_tokens = tokenizer.encode(
            prompt,
            add_bos=True
        )

        max_prompt_length = (
            config.context_length
            - max_new_tokens
            - 1
        )

        if max_prompt_length < 1:
            max_prompt_length = 1

        if len(prompt_tokens) > max_prompt_length:
            prompt_tokens = prompt_tokens[
                -max_prompt_length:
            ]

        input_ids = torch.tensor(
            [prompt_tokens],
            dtype=torch.long,
            device=device
        )

        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.EOS
        )

        completion = output[
            0,
            len(prompt_tokens):
        ].tolist()

        response = tokenizer.decode(
            completion
        )

        stop_markers = [
            "<|user|>",
            "<|system|>",
            "<|assistant|>"
        ]

        for marker in stop_markers:
            if marker in response:
                response = response.split(
                    marker,
                    1
                )[0]

        response = response.strip()

        history.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        print(
            "GPT:",
            response
        )


def build_instruction_dataset(
    input_path: str,
    output_path: str
):
    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        else:
            data = [data]

    output_parts = []

    for item in data:
        system = str(
            item.get(
                "system",
                ""
            )
        ).strip()

        instruction = str(
            item.get(
                "instruction",
                item.get(
                    "prompt",
                    item.get(
                        "user",
                        ""
                    )
                )
            )
        ).strip()

        response = str(
            item.get(
                "response",
                item.get(
                    "output",
                    item.get(
                        "assistant",
                        ""
                    )
                )
            )
        ).strip()

        if not instruction or not response:
            continue

        sample = []

        if system:
            sample.append(
                "<|system|>\n"
                + system
                + "\n"
            )

        sample.append(
            "<|user|>\n"
            + instruction
            + "\n"
        )

        sample.append(
            "<|assistant|>\n"
            + response
            + "\n"
        )

        output_parts.append(
            "".join(sample)
        )

    os.makedirs(
        os.path.dirname(
            os.path.abspath(output_path)
        ),
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "\n".join(output_parts)
        )


def split_dataset(
    input_path: str,
    train_path: str,
    val_path: str,
    val_ratio: float,
    seed: int
):
    with open(
        input_path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:
        text = f.read()

    random.seed(seed)

    length = len(text)

    split_index = int(
        length * (1.0 - val_ratio)
    )

    train_text = text[:split_index]
    val_text = text[split_index:]

    os.makedirs(
        os.path.dirname(
            os.path.abspath(train_path)
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(
            os.path.abspath(val_path)
        ),
        exist_ok=True
    )

    with open(
        train_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(train_text)

    with open(
        val_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(val_text)


def parameter_count_for_config(
    config: GPTConfig
):
    model = GPT(config)

    return model.num_parameters()


def preset_config(name: str) -> GPTConfig:
    presets = {
        "tiny": GPTConfig(
            context_length=512,
            n_layers=4,
            n_heads=4,
            n_kv_heads=4,
            d_model=256,
            d_ff=768
        ),
        "small": GPTConfig(
            context_length=1024,
            n_layers=8,
            n_heads=8,
            n_kv_heads=4,
            d_model=512,
            d_ff=1536
        ),
        "medium": GPTConfig(
            context_length=1024,
            n_layers=12,
            n_heads=12,
            n_kv_heads=4,
            d_model=768,
            d_ff=2048
        ),
        "large": GPTConfig(
            context_length=2048,
            n_layers=24,
            n_heads=16,
            n_kv_heads=8,
            d_model=1024,
            d_ff=2816
        ),
        "xl": GPTConfig(
            context_length=2048,
            n_layers=32,
            n_heads=20,
            n_kv_heads=5,
            d_model=1280,
            d_ff=3584
        )
    }

    if name not in presets:
        raise ValueError(
            f"Unknown preset: {name}"
        )

    return presets[name]


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    train_parser = subparsers.add_parser(
        "train"
    )

    train_parser.add_argument(
        "--preset",
        default="tiny",
        choices=[
            "tiny",
            "small",
            "medium",
            "large",
            "xl"
        ]
    )

    train_parser.add_argument(
        "--train-file",
        required=True
    )

    train_parser.add_argument(
        "--val-file",
        default=None
    )

    train_parser.add_argument(
        "--output-dir",
        default="checkpoints"
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8
    )

    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=10000
    )

    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4
    )

    train_parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=3e-5
    )

    train_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500
    )

    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1
    )

    train_parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0
    )

    train_parser.add_argument(
        "--eval-interval",
        type=int,
        default=250
    )

    train_parser.add_argument(
        "--eval-steps",
        type=int,
        default=50
    )

    train_parser.add_argument(
        "--save-interval",
        type=int,
        default=500
    )

    train_parser.add_argument(
        "--log-interval",
        type=int,
        default=10
    )

    train_parser.add_argument(
        "--device",
        default="auto"
    )

    train_parser.add_argument(
        "--dtype",
        default="auto",
        choices=[
            "auto",
            "float32",
            "float16",
            "bfloat16"
        ]
    )

    train_parser.add_argument(
        "--compile",
        action="store_true"
    )

    train_parser.add_argument(
        "--resume",
        default=None
    )

    train_parser.add_argument(
        "--context-length",
        type=int,
        default=None
    )

    train_parser.add_argument(
        "--layers",
        type=int,
        default=None
    )

    train_parser.add_argument(
        "--heads",
        type=int,
        default=None
    )

    train_parser.add_argument(
        "--kv-heads",
        type=int,
        default=None
    )

    train_parser.add_argument(
        "--d-model",
        type=int,
        default=None
    )

    train_parser.add_argument(
        "--d-ff",
        type=int,
        default=None
    )

    train_parser.add_argument(
        "--dropout",
        type=float,
        default=None
    )

    generate_parser = subparsers.add_parser(
        "generate"
    )

    generate_parser.add_argument(
        "--checkpoint",
        required=True
    )

    generate_parser.add_argument(
        "--prompt",
        required=True
    )

    generate_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256
    )

    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8
    )

    generate_parser.add_argument(
        "--top-k",
        type=int,
        default=50
    )

    generate_parser.add_argument(
        "--top-p",
        type=float,
        default=0.95
    )

    generate_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05
    )

    generate_parser.add_argument(
        "--device",
        default="auto"
    )

    chat_parser = subparsers.add_parser(
        "chat"
    )

    chat_parser.add_argument(
        "--checkpoint",
        required=True
    )

    chat_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256
    )

    chat_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8
    )

    chat_parser.add_argument(
        "--top-k",
        type=int,
        default=50
    )

    chat_parser.add_argument(
        "--top-p",
        type=float,
        default=0.95
    )

    chat_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05
    )

    chat_parser.add_argument(
        "--device",
        default="auto"
    )

    chat_parser.add_argument(
        "--system",
        default=(
            "You are a helpful, accurate and concise AI assistant."
        )
    )

    instruction_parser = subparsers.add_parser(
        "prepare-instructions"
    )

    instruction_parser.add_argument(
        "--input",
        required=True
    )

    instruction_parser.add_argument(
        "--output",
        required=True
    )

    split_parser = subparsers.add_parser(
        "split"
    )

    split_parser.add_argument(
        "--input",
        required=True
    )

    split_parser.add_argument(
        "--train",
        required=True
    )

    split_parser.add_argument(
        "--val",
        required=True
    )

    split_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01
    )

    split_parser.add_argument(
        "--seed",
        type=int,
        default=1337
    )

    info_parser = subparsers.add_parser(
        "info"
    )

    info_parser.add_argument(
        "--preset",
        default="tiny",
        choices=[
            "tiny",
            "small",
            "medium",
            "large",
            "xl"
        ]
    )

    args = parser.parse_args()

    if args.command == "train":
        model_config = preset_config(
            args.preset
        )

        if args.context_length is not None:
            model_config.context_length = (
                args.context_length
            )

        if args.layers is not None:
            model_config.n_layers = args.layers

        if args.heads is not None:
            model_config.n_heads = args.heads

        if args.kv_heads is not None:
            model_config.n_kv_heads = (
                args.kv_heads
            )

        if args.d_model is not None:
            model_config.d_model = args.d_model

        if args.d_ff is not None:
            model_config.d_ff = args.d_ff

        if args.dropout is not None:
            model_config.dropout = args.dropout

        train_config = TrainConfig(
            train_file=args.train_file,
            val_file=args.val_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            min_learning_rate=(
                args.min_learning_rate
            ),
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            eval_interval=args.eval_interval,
            eval_steps=args.eval_steps,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
            device=args.device,
            dtype=args.dtype,
            compile=args.compile,
            resume=args.resume
        )

        train(
            model_config,
            train_config
        )

    elif args.command == "generate":
        generate_text(
            checkpoint_path=args.checkpoint,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=(
                args.repetition_penalty
            ),
            device_string=args.device
        )

    elif args.command == "chat":
        chat(
            checkpoint_path=args.checkpoint,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=(
                args.repetition_penalty
            ),
            device_string=args.device,
            system_prompt=args.system
        )

    elif args.command == "prepare-instructions":
        build_instruction_dataset(
            args.input,
            args.output
        )

    elif args.command == "split":
        split_dataset(
            args.input,
            args.train,
            args.val,
            args.val_ratio,
            args.seed
        )

    elif args.command == "info":
        config = preset_config(
            args.preset
        )

        count = parameter_count_for_config(
            config
        )

        print(
            json.dumps(
                {
                    "preset": args.preset,
                    "parameters": count,
                    "parameters_human":
                        format_number(count),
                    "config": asdict(config)
                },
                indent=2
            )
        )


if __name__ == "__main__":
    main()