import math
from bisect import bisect_right
from copy import deepcopy
from numbers import Integral, Real

from .optimizer import Optimizer


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative real scalar.")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite.") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _floor_lrs(base_lrs: list[float], minimum: float, fraction: float) -> list[float]:
    if any(minimum > base for base in base_lrs):
        raise ValueError("Minimum learning rate cannot exceed a group's base learning rate.")
    return [minimum + (base - minimum) * fraction for base in base_lrs]


class LRScheduler:
    _config_keys = ()

    def __init__(self, optimizer: Optimizer, last_step: int = 0, **configuration):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("Scheduler optimizer must be a NamiTorch Optimizer.")
        optimizer._registered_parameters()
        step = _integer(last_step, "last_step")
        config = self._normalize_config(configuration)
        base_lrs = []
        for group in optimizer.param_groups:
            if "lr" not in group:
                raise ValueError("Every optimizer parameter group must define lr.")
            _real(group["lr"], "lr")
            base_lrs.append(_real(group.get("initial_lr", group["lr"]), "initial_lr"))
        values = self._checked_lrs(step, base_lrs, config)
        self.optimizer = optimizer
        self.base_lrs = base_lrs
        self.last_step = step
        self._group_signature = self._signature()
        for name, value in config.items():
            setattr(self, name, value)
        self._apply(values)

    @classmethod
    def _normalize_config(cls, config: dict) -> dict:
        if not isinstance(config, dict) or set(config) != set(cls._config_keys):
            raise ValueError(f"{cls.__name__} configuration must contain {cls._config_keys}.")
        return deepcopy(config)

    def _signature(self):
        return tuple(tuple(id(parameter) for parameter in group["params"]) for group in self.optimizer.param_groups)

    def _validate_groups(self) -> None:
        self.optimizer._registered_parameters()
        if self._signature() != self._group_signature or len(self.base_lrs) != len(self.optimizer.param_groups):
            raise RuntimeError("Optimizer parameter groups changed after scheduler construction; create a new scheduler.")

    def _configuration(self) -> dict:
        return self._normalize_config({name: getattr(self, name) for name in self._config_keys})

    def _compute_lrs(self, step: int, base_lrs: list[float], config: dict) -> list[float]:
        return list(base_lrs)

    def _checked_lrs(self, step: int, base_lrs: list[float], config: dict) -> list[float]:
        try:
            values = self._compute_lrs(step, base_lrs, config)
        except OverflowError:
            raise ValueError("Scheduled learning rate exceeds the finite supported range.") from None
        return [_real(value, "scheduled lr") for value in values]

    def _apply(self, values: list[float]) -> None:
        for group, base, value in zip(self.optimizer.param_groups, self.base_lrs, values):
            group["initial_lr"] = base
            group["lr"] = value
        self._last_lr = list(values)

    def get_lr(self) -> list[float]:
        self._validate_groups()
        bases = [_real(value, "base_lr") for value in self.base_lrs]
        return self._checked_lrs(_integer(self.last_step, "last_step"), bases, self._configuration())

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def step(self) -> None:
        self._validate_groups()
        step = _integer(self.last_step, "last_step") + 1
        bases = [_real(value, "base_lr") for value in self.base_lrs]
        values = self._checked_lrs(step, bases, self._configuration())
        self._apply(values)
        self.last_step = step

    def state_dict(self) -> dict:
        self.get_lr()
        return {
            "version": 1,
            "scheduler": type(self).__name__,
            "last_step": int(self.last_step),
            "base_lrs": [_real(value, "base_lr") for value in self.base_lrs],
            "config": self._configuration(),
        }

    def _prepare_load_state_dict(self, state: dict):
        self._validate_groups()
        if not isinstance(state, dict):
            raise TypeError("Scheduler state must be a dictionary.")
        if set(state) != {"version", "scheduler", "last_step", "base_lrs", "config"}:
            raise ValueError("Invalid scheduler state fields.")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("Unsupported scheduler state version.")
        if state["scheduler"] != type(self).__name__:
            raise ValueError("Saved scheduler type does not match the current scheduler.")
        step = _integer(state["last_step"], "last_step")
        if not isinstance(state["base_lrs"], list) or len(state["base_lrs"]) != len(self.optimizer.param_groups):
            raise ValueError("Saved base_lrs must match the optimizer parameter group count.")
        bases = [_real(value, "base_lr") for value in state["base_lrs"]]
        config = self._normalize_config(state["config"])
        values = self._checked_lrs(step, bases, config)
        return step, bases, config, values

    def _apply_loaded_state(self, prepared):
        step, bases, config, values = prepared
        self.base_lrs = bases
        self.last_step = step
        for name, value in config.items():
            setattr(self, name, value)
        self._apply(values)

    def load_state_dict(self, state: dict) -> None:
        self._apply_loaded_state(self._prepare_load_state_dict(state))


class StepLR(LRScheduler):
    _config_keys = ("step_size", "gamma")

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_step: int = 0):
        super().__init__(optimizer, last_step, step_size=step_size, gamma=gamma)

    @classmethod
    def _normalize_config(cls, config: dict) -> dict:
        config = super()._normalize_config(config)
        config["step_size"] = _integer(config["step_size"], "step_size", 1)
        config["gamma"] = _real(config["gamma"], "gamma")
        return config

    def _compute_lrs(self, step: int, base_lrs: list[float], config: dict) -> list[float]:
        factor = config["gamma"] ** (step // config["step_size"])
        return [base * factor for base in base_lrs]


class MultiStepLR(LRScheduler):
    _config_keys = ("milestones", "gamma")

    def __init__(self, optimizer: Optimizer, milestones, gamma: float = 0.1, last_step: int = 0):
        if isinstance(milestones, (str, bytes, dict, set, frozenset)):
            raise TypeError("milestones must be an ordered iterable of integers.")
        try:
            milestones = list(milestones)
        except TypeError:
            raise TypeError("milestones must be an ordered iterable of integers.") from None
        super().__init__(optimizer, last_step, milestones=milestones, gamma=gamma)

    @classmethod
    def _normalize_config(cls, config: dict) -> dict:
        config = super()._normalize_config(config)
        if not isinstance(config["milestones"], (list, tuple)):
            raise TypeError("milestones must be a list or tuple of integers.")
        config["milestones"] = sorted(_integer(value, "milestone") for value in config["milestones"])
        config["gamma"] = _real(config["gamma"], "gamma")
        return config

    def _compute_lrs(self, step: int, base_lrs: list[float], config: dict) -> list[float]:
        factor = config["gamma"] ** bisect_right(config["milestones"], step)
        return [base * factor for base in base_lrs]


class ExponentialLR(LRScheduler):
    _config_keys = ("gamma",)

    def __init__(self, optimizer: Optimizer, gamma: float, last_step: int = 0):
        super().__init__(optimizer, last_step, gamma=gamma)

    @classmethod
    def _normalize_config(cls, config: dict) -> dict:
        config = super()._normalize_config(config)
        config["gamma"] = _real(config["gamma"], "gamma")
        return config

    def _compute_lrs(self, step: int, base_lrs: list[float], config: dict) -> list[float]:
        factor = config["gamma"] ** step
        return [base * factor for base in base_lrs]


class CosineAnnealingLR(LRScheduler):
    _config_keys = ("T_max", "eta_min")

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_step: int = 0):
        super().__init__(optimizer, last_step, T_max=T_max, eta_min=eta_min)

    @classmethod
    def _normalize_config(cls, config: dict) -> dict:
        config = super()._normalize_config(config)
        config["T_max"] = _integer(config["T_max"], "T_max", 1)
        config["eta_min"] = _real(config["eta_min"], "eta_min")
        return config

    def _compute_lrs(self, step: int, base_lrs: list[float], config: dict) -> list[float]:
        progress = min(step, config["T_max"]) / config["T_max"]
        return _floor_lrs(base_lrs, config["eta_min"], (1 + math.cos(math.pi * progress)) / 2)


class LinearLR(LRScheduler):
    _config_keys = ("start_factor", "end_factor", "total_iters")

    def __init__(self, optimizer: Optimizer, start_factor: float = 1 / 3, end_factor: float = 1.0, total_iters: int = 5, last_step: int = 0):
        super().__init__(optimizer, last_step, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)

    @classmethod
    def _normalize_config(cls, config: dict) -> dict:
        config = super()._normalize_config(config)
        config["start_factor"] = _real(config["start_factor"], "start_factor")
        config["end_factor"] = _real(config["end_factor"], "end_factor")
        config["total_iters"] = _integer(config["total_iters"], "total_iters", 1)
        return config

    def _compute_lrs(self, step: int, base_lrs: list[float], config: dict) -> list[float]:
        progress = min(step, config["total_iters"]) / config["total_iters"]
        factor = config["start_factor"] * (1 - progress) + config["end_factor"] * progress
        return [base * factor for base in base_lrs]


class WarmupCosine(LRScheduler):
    _config_keys = ("warmup_steps", "total_steps", "min_lr")

    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0, last_step: int = 0):
        super().__init__(optimizer, last_step, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)

    @classmethod
    def _normalize_config(cls, config: dict) -> dict:
        config = super()._normalize_config(config)
        config["warmup_steps"] = _integer(config["warmup_steps"], "warmup_steps")
        config["total_steps"] = _integer(config["total_steps"], "total_steps", 1)
        if config["warmup_steps"] >= config["total_steps"]:
            raise ValueError("warmup_steps must be smaller than total_steps.")
        config["min_lr"] = _real(config["min_lr"], "min_lr")
        return config

    def _compute_lrs(self, step: int, base_lrs: list[float], config: dict) -> list[float]:
        _floor_lrs(base_lrs, config["min_lr"], 1.0)
        if step < config["warmup_steps"]:
            return [base * (step / config["warmup_steps"]) for base in base_lrs]
        progress = (min(step, config["total_steps"]) - config["warmup_steps"]) / (config["total_steps"] - config["warmup_steps"])
        return _floor_lrs(base_lrs, config["min_lr"], (1 + math.cos(math.pi * progress)) / 2)


__all__ = ["LRScheduler", "StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "LinearLR", "WarmupCosine"]
