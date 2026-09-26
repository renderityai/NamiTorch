from numbers import Integral

from ..autograd import no_grad
from ..nn.module import Module
from ..optim.optimizer import Optimizer
from ..optim.lr_scheduler import LRScheduler
from ..random import Generator, _default_generator
from ..serialization import SerializationError, load, save


_CHECKPOINT_KEYS = {
    "checkpoint_version", "model", "optimizer", "optimizer_type", "optimizer_parameter_names",
    "scheduler", "rng", "step", "epoch", "config", "cumulative_tokens",
}


def _counter(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _components(model, optimizer, scheduler, generators):
    if not isinstance(model, Module):
        raise TypeError("model must be a NamiTorch Module.")
    if optimizer is not None and not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a NamiTorch Optimizer or None.")
    if scheduler is not None:
        if not isinstance(scheduler, LRScheduler):
            raise TypeError("scheduler must be a NamiTorch LRScheduler or None.")
        if optimizer is None or scheduler.optimizer is not optimizer:
            raise ValueError("scheduler must belong to the supplied optimizer.")
    generators = {} if generators is None else generators
    if not isinstance(generators, dict) or any(type(name) is not str or not name for name in generators):
        raise TypeError("generators must be a dictionary of nonempty string names.")
    if any(not isinstance(generator, Generator) for generator in generators.values()):
        raise TypeError("Every named generator must be a NamiTorch Generator.")
    return dict(generators)


def _optimizer_names(model, optimizer):
    if optimizer is None:
        return None
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer._registered_parameters()
    result = []
    for group in optimizer.param_groups:
        if any(id(parameter) not in names for parameter in group["params"]):
            raise ValueError("Checkpoint optimizer parameters must belong to the saved model.")
        result.append([names[id(parameter)] for parameter in group["params"]])
    return result


def _optimizer_type(optimizer):
    return None if optimizer is None else f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"


def _pack_optimizer(optimizer):
    if optimizer is None:
        return None
    state = optimizer.state_dict()
    state["state"] = {str(index): values for index, values in state["state"].items()}
    return state


def _unpack_optimizer(state):
    if not isinstance(state, dict) or not isinstance(state.get("state"), dict):
        raise SerializationError("Invalid checkpoint optimizer state.")
    restored = dict(state)
    values = {}
    for key, value in state["state"].items():
        if type(key) is not str or not key.isascii() or not key.isdecimal() or str(int(key)) != key:
            raise SerializationError("Optimizer state indices must be canonical nonnegative integer strings.")
        values[int(key)] = value
    restored["state"] = values
    return restored


def save_checkpoint(path, model, optimizer=None, scheduler=None, *, step=0, epoch=0, config=None, cumulative_tokens=0, generators=None) -> None:
    generators = _components(model, optimizer, scheduler, generators)
    parameter_names = _optimizer_names(model, optimizer)
    if scheduler is not None and scheduler.get_lr() != [group["lr"] for group in optimizer.param_groups]:
        raise ValueError("Optimizer learning rates disagree with the scheduler state.")
    checkpoint = {
        "checkpoint_version": 1,
        "model": model.state_dict(),
        "optimizer": _pack_optimizer(optimizer),
        "optimizer_type": _optimizer_type(optimizer),
        "optimizer_parameter_names": parameter_names,
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "rng": {
            "default": _default_generator.get_state(),
            "generators": {name: generator.get_state() for name, generator in generators.items()},
        },
        "step": _counter(step, "step"),
        "epoch": _counter(epoch, "epoch"),
        "config": config,
        "cumulative_tokens": _counter(cumulative_tokens, "cumulative_tokens"),
    }
    save(checkpoint, path)


def _prepare_rng(state, generators):
    if type(state) is not dict or set(state) != {"default", "generators"} or type(state["generators"]) is not dict:
        raise SerializationError("Invalid checkpoint RNG fields.")
    if set(state["generators"]) != set(generators):
        raise ValueError("Named generators must match the saved checkpoint exactly when restoring RNG.")
    requests = [(_default_generator, state["default"])]
    requests.extend((generator, state["generators"][name]) for name, generator in generators.items())
    prepared = {}
    for generator, saved in requests:
        validator = Generator(0)
        validator.set_state(saved)
        validated = validator.get_state()
        identity = id(generator)
        if identity in prepared and prepared[identity][1] != validated:
            raise ValueError("Conflicting checkpoint states for aliased generators.")
        prepared[identity] = (generator, validated)
    return tuple(prepared.values())


def load_checkpoint(path, model, optimizer=None, scheduler=None, *, generators=None, strict=True, restore_rng=True, max_array_bytes=None):
    generators = _components(model, optimizer, scheduler, generators)
    if type(strict) is not bool or type(restore_rng) is not bool:
        raise TypeError("strict and restore_rng must be Python bools.")
    checkpoint = load(path, max_array_bytes=max_array_bytes)
    if type(checkpoint) is not dict or set(checkpoint) != _CHECKPOINT_KEYS:
        raise SerializationError("Invalid checkpoint fields.")
    if type(checkpoint["checkpoint_version"]) is not int or checkpoint["checkpoint_version"] != 1:
        raise SerializationError("Unsupported checkpoint_version.")
    info = {name: _counter(checkpoint[name], name) for name in ("step", "epoch", "cumulative_tokens")}
    info["config"] = checkpoint["config"]
    result, model_updates = model._prepare_load_state_dict(checkpoint["model"], strict)
    optimizer_update = None
    if optimizer is not None:
        if checkpoint["optimizer_type"] != _optimizer_type(optimizer):
            raise ValueError("Checkpoint optimizer type does not match the supplied optimizer.")
        if checkpoint["optimizer_parameter_names"] != _optimizer_names(model, optimizer):
            raise ValueError("Checkpoint optimizer parameter names or group order do not match the model.")
        optimizer_update = optimizer._prepare_load_state_dict(_unpack_optimizer(checkpoint["optimizer"]))
    scheduler_update = None
    if scheduler is not None:
        scheduler_update = scheduler._prepare_load_state_dict(checkpoint["scheduler"])
        if [group["lr"] for group in optimizer_update[1]] != scheduler_update[3]:
            raise ValueError("Checkpoint optimizer learning rates disagree with the scheduler state.")
    rng_updates = _prepare_rng(checkpoint["rng"], generators) if restore_rng else ()
    with no_grad():
        for _, target, snapshot in model_updates:
            target.copy_(snapshot)
    if optimizer_update is not None:
        optimizer.defaults, optimizer.param_groups, optimizer.state = optimizer_update
    if scheduler_update is not None:
        scheduler._apply_loaded_state(scheduler_update)
    for generator, state in rng_updates:
        generator.set_state(state)
    info["load_result"] = result
    return info


__all__ = ["save_checkpoint", "load_checkpoint"]
