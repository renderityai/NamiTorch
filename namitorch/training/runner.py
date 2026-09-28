import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ..autograd import no_grad
from ..amp import GradScaler, autocast
from ..backends import get_backend
from ..device import Device
from ..data import TokenShardBatcher, TokenShardDataset
from ..models import GPT, GPTConfig
from ..optim import AdamW, WarmupCosine, clip_grad_norm_
from ..random import Generator, get_state, manual_seed, set_state
from ..serialization import load
from ..tokenization import ByteBPETokenizer
from ._io import canonical_json, integer, read_json, real, seed_value, write_json
from .checkpoint import load_checkpoint, save_checkpoint


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    total_steps: int = 1000
    batch_size: int = 4
    grad_accum_steps: int = 1
    sequence_length: int | None = None
    learning_rate: float = 3e-4
    min_lr: float = 0.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    clip_norm: float = 1.0
    warmup_steps: int = 0
    eval_interval: int = 100
    eval_batches: int = 10
    save_interval: int = 100
    seed: int = 1337
    dtype: str = "float32"
    device: str = "cpu"
    amp_enabled: bool = False
    amp_init_scale: float = 65536.0
    amp_growth_factor: float = 2.0
    amp_backoff_factor: float = 0.5
    amp_growth_interval: int = 2000

    def __post_init__(self):
        for name in ("total_steps", "batch_size", "grad_accum_steps", "eval_interval", "eval_batches", "save_interval"):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        if self.sequence_length is not None:
            object.__setattr__(self, "sequence_length", integer(self.sequence_length, "sequence_length", 1))
        object.__setattr__(self, "warmup_steps", integer(self.warmup_steps, "warmup_steps"))
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps.")
        for name in ("learning_rate", "min_lr", "weight_decay", "beta1", "beta2", "eps", "clip_norm"):
            object.__setattr__(self, name, real(getattr(self, name), name))
        if self.beta1 >= 1 or self.beta2 >= 1 or self.eps == 0:
            raise ValueError("AdamW betas must be in [0, 1) and eps must be positive.")
        if self.min_lr > self.learning_rate:
            raise ValueError("min_lr cannot exceed learning_rate.")
        object.__setattr__(self, "seed", seed_value(self.seed))
        if self.dtype not in ("float32", "float64"):
            raise ValueError("Training dtype must be float32 or float64.")
        object.__setattr__(self, "device", str(Device(self.device)))
        if type(self.amp_enabled) is not bool:
            raise TypeError("amp_enabled must be a Python bool.")
        if self.amp_enabled and (Device(self.device).type != "cuda" or self.dtype != "float32"):
            raise ValueError("AMP training requires a CUDA device and float32 master parameters.")
        scaler = GradScaler(self.amp_init_scale, self.amp_growth_factor, self.amp_backoff_factor, self.amp_growth_interval, self.amp_enabled)
        settings = scaler.state_dict()
        for name, source in (("amp_init_scale", "scale"), ("amp_growth_factor", "growth_factor"), ("amp_backoff_factor", "backoff_factor"), ("amp_growth_interval", "growth_interval")):
            object.__setattr__(self, name, settings[source])


def _training_metadata(config):
    values = asdict(config)
    defaults = {"device": "cpu", "amp_enabled": False, "amp_init_scale": 65536.0, "amp_growth_factor": 2.0, "amp_backoff_factor": 0.5, "amp_growth_interval": 2000}
    for name, value in defaults.items():
        if values[name] == value:
            del values[name]
    return values


def decay_parameter_groups(model, weight_decay):
    weight_decay = real(weight_decay, "weight_decay")
    parameters = list(model.parameters())
    groups = []
    for decay, selected in ((weight_decay, [p for p in parameters if p.ndim >= 2]), (0.0, [p for p in parameters if p.ndim < 2])):
        if selected:
            groups.append({"params": selected, "weight_decay": decay})
    return groups


def evaluate(model, batcher, batches, *, amp_enabled=False):
    batches = integer(batches, "batches", 1)
    if batcher.generator is None:
        raise ValueError("Evaluation requires a separate explicit Generator.")
    modes = [(module, module.training) for module in model.modules()]
    validation_state, default_state = batcher.generator.get_state(), get_state()
    losses = []
    device = next(model.parameters()).device
    try:
        model.eval()
        with no_grad():
            for _ in range(batches):
                inputs, targets = batcher.sample_batch()
                inputs, targets = inputs.to(device), targets.to(device)
                with get_backend(device).context(), autocast(device_type=device.type, enabled=amp_enabled):
                    loss = model(inputs, targets).loss.item()
                if not math.isfinite(loss):
                    raise RuntimeError("Validation loss is nonfinite.")
                losses.append(loss)
    finally:
        batcher.generator.set_state(validation_state)
        set_state(default_state)
        for module, mode in modes:
            module.training = mode
    return math.fsum(losses) / batches


def _check_manifest(dataset, split, tokenizer):
    manifest = dataset.manifest
    if manifest["split"] != split:
        raise ValueError(f"Expected {split!r} manifest, received {manifest['split']!r}.")
    if manifest["tokenizer_fingerprint"] != tokenizer.fingerprint:
        raise ValueError(f"{split} manifest tokenizer fingerprint does not match the tokenizer.")
    if "vocab_size" in manifest and (type(manifest["vocab_size"]) is not int or manifest["vocab_size"] != tokenizer.vocab_size):
        raise ValueError(f"{split} manifest vocabulary does not match the tokenizer.")
    if not len(dataset):
        raise ValueError(f"{split} corpus has no windows of sequence_length + 1 tokens; use longer shards or more documents.")
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


def _resume_metadata(path, run, tokens_per_step):
    checkpoint = load(path)
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    keys = {"run", "best_val_loss", "optimizer_updates"} if run["training"].get("amp_enabled", False) else {"run", "best_val_loss"}
    if not isinstance(config, dict) or set(config) != keys or config["run"] != run:
        raise ValueError("Resume checkpoint is incompatible with the model, training configuration, tokenizer or manifests.")
    step = integer(checkpoint.get("step"), "checkpoint step")
    tokens = integer(checkpoint.get("cumulative_tokens"), "checkpoint cumulative_tokens")
    if step > run["training"]["total_steps"] or tokens != step * tokens_per_step:
        raise ValueError("Resume checkpoint counters disagree with the training configuration.")
    scheduler = checkpoint.get("scheduler")
    updates = integer(config.get("optimizer_updates", step), "checkpoint optimizer_updates")
    if updates > step or not isinstance(scheduler, dict) or type(scheduler.get("last_step")) is not int or scheduler["last_step"] != updates:
        raise ValueError("Resume scheduler step disagrees with optimizer update count.")
    best = config["best_val_loss"]
    if best is not None:
        best = real(best, "best_val_loss")
    return step, tokens, best, updates


def _prepare_output(output_dir, run, resume, step):
    output = Path(output_dir).resolve()
    if output.exists():
        if not output.is_dir():
            raise ValueError("Training output must be a directory.")
        if any(output.iterdir()):
            if resume is None:
                raise FileExistsError("Training output is nonempty; use resume or a new output directory.")
            if not (output / "run.json").is_file() or read_json(output / "run.json") != run:
                raise ValueError("Output directory belongs to an incompatible run.")
            log = output / "metrics.jsonl"
            if log.exists():
                import json

                with log.open("r", encoding="utf-8") as stream:
                    previous = 0
                    for line in stream:
                        record = json.loads(line)
                        logged_step = integer(record.get("step"), "logged step", 1)
                        if logged_step <= previous or logged_step > step:
                            raise ValueError("Metrics extend beyond the resume checkpoint or are out of order; use a new output directory.")
                        previous = logged_step
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "run.json").exists():
        write_json(output / "run.json", run)
    return output


def train_gpt(train_manifest, val_manifest, tokenizer, model_config, output_dir, *, training_config=None, resume=None, run_steps=None):
    if not isinstance(model_config, GPTConfig):
        raise TypeError("model_config must be GPTConfig.")
    config = TrainingConfig() if training_config is None else training_config
    if not isinstance(config, TrainingConfig):
        raise TypeError("training_config must be TrainingConfig.")
    if not isinstance(tokenizer, ByteBPETokenizer):
        tokenizer = ByteBPETokenizer.load(tokenizer)
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocab_size must equal tokenizer vocab_size.")
    length = model_config.context_length if config.sequence_length is None else config.sequence_length
    if length > model_config.context_length:
        raise ValueError("sequence_length exceeds model context_length.")
    run_steps = None if run_steps is None else integer(run_steps, "run_steps", 1)
    with TokenShardDataset(train_manifest, length) as train_data, TokenShardDataset(val_manifest, length) as val_data:
        run = {
            "model": asdict(model_config), "training": _training_metadata(config),
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "train_manifest_fingerprint": _check_manifest(train_data, "train", tokenizer),
            "val_manifest_fingerprint": _check_manifest(val_data, "val", tokenizer),
        }
        tokens_per_step = config.batch_size * length * config.grad_accum_steps
        step, cumulative_tokens, best, optimizer_updates = (0, 0, None, 0) if resume is None else _resume_metadata(resume, run, tokens_per_step)
        manual_seed(config.seed)
        model = GPT(model_config, dtype=config.dtype).to(config.device)
        scaler = GradScaler(config.amp_init_scale, config.amp_growth_factor, config.amp_backoff_factor, config.amp_growth_interval, config.amp_enabled)
        if config.amp_enabled:
            with get_backend(config.device).context(), autocast():
                device = model.token_embedding.weight.device
        else:
            device = model.token_embedding.weight.device
        optimizer = AdamW(decay_parameter_groups(model, config.weight_decay), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2), eps=config.eps)
        scheduler = WarmupCosine(optimizer, config.warmup_steps, config.total_steps, min_lr=config.min_lr)
        generators = {"train": Generator((config.seed + 1) % 2 ** 64), "validation": Generator((config.seed + 2) % 2 ** 64)}
        if resume is not None:
            load_checkpoint(resume, model, optimizer, scheduler, generators=generators, scaler=scaler if config.amp_enabled else None)
        output = _prepare_output(output_dir, run, resume, step)
        train_batches = TokenShardBatcher(train_data, config.batch_size, generator=generators["train"])
        val_batches = TokenShardBatcher(val_data, config.batch_size, generator=generators["validation"])
        stop = config.total_steps if run_steps is None else min(config.total_steps, step + run_steps)

        def save_current(name):
            metadata = {"run": run, "best_val_loss": best}
            if config.amp_enabled:
                metadata["optimizer_updates"] = optimizer_updates
            save_checkpoint(
                output / name, model, optimizer, scheduler, step=step,
                cumulative_tokens=cumulative_tokens, config=metadata, generators=generators,
                scaler=scaler if config.amp_enabled else None,
            )

        model.train()
        with (output / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as log:
            while step < stop:
                optimizer.zero_grad(set_to_none=True)
                losses = []
                lr_used = [group["lr"] for group in optimizer.param_groups]
                for _ in range(config.grad_accum_steps):
                    inputs, targets = train_batches.sample_batch()
                    inputs, targets = inputs.to(device), targets.to(device)
                    with get_backend(device).context(), autocast(device_type=device.type, enabled=config.amp_enabled):
                        loss = model(inputs, targets).loss
                    raw_loss = loss.item()
                    if not math.isfinite(raw_loss):
                        raise RuntimeError("Training loss is nonfinite.")
                    losses.append(raw_loss)
                    scaler.scale(loss / config.grad_accum_steps).backward()
                finite = scaler.unscale_(optimizer)
                norm = clip_grad_norm_(model.parameters(), config.clip_norm, error_if_nonfinite=True).item() if finite else None
                updated = scaler.step(optimizer)
                scaler.update()
                if updated:
                    scheduler.step()
                    optimizer_updates += 1
                step += 1
                cumulative_tokens += tokens_per_step
                validation = None
                improved = False
                if step % config.eval_interval == 0 or step == config.total_steps:
                    validation = evaluate(model, val_batches, config.eval_batches, amp_enabled=config.amp_enabled)
                    if best is None or validation < best:
                        best, improved = validation, True
                record = dict(step=step, cumulative_tokens=cumulative_tokens, train_loss=math.fsum(losses) / len(losses), val_loss=validation, best_val_loss=best, grad_norm=norm, lr=lr_used, next_lr=scheduler.get_last_lr())
                if config.amp_enabled:
                    record.update(optimizer_updates=optimizer_updates, skipped=not updated, scale=scaler.get_scale())
                log.write(canonical_json(record).decode("utf-8") + "\n")
                log.flush()
                if improved:
                    save_current("best.nt")
                if step % config.save_interval == 0 or step == stop:
                    save_current("latest.nt")
        if not (output / "latest.nt").exists():
            save_current("latest.nt")
        return dict(step=step, cumulative_tokens=cumulative_tokens, best_val_loss=best, latest_checkpoint=str(output / "latest.nt"), best_checkpoint=str(output / "best.nt") if (output / "best.nt").exists() else None, metrics=str(output / "metrics.jsonl"))


__all__ = ["TrainingConfig", "decay_parameter_groups", "evaluate", "train_gpt"]
