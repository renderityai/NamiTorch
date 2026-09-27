from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import NamedTuple

from ..backends import is_array, namespace, same_device, transfer, writable
from ..autograd import no_grad
from ..tensor import Tensor
from .parameter import Parameter


_REGISTRIES = ("_parameters", "_modules", "_buffers")
_STATE_ATTRIBUTES = frozenset((*_REGISTRIES, "_non_persistent_buffers_set"))


def _boolean(name: str, value: bool) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a Python bool.")


def _qualified_name(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class LoadStateDictResult(NamedTuple):
    missing_keys: list[str]
    unexpected_keys: list[str]


class Module:
    def to(self, device=None, dtype=None):
        values = list(self.parameters()) + list(self.buffers())
        staged = {}
        with no_grad():
            for value in values:
                if id(value) in staged:
                    continue
                target_dtype = dtype if value.dtype.is_floating_point else None
                converted = value.to(device=device, dtype=target_dtype)
                if value.requires_grad and not converted.dtype.can_require_grad:
                    raise ValueError("Module parameters requiring gradients must keep a floating dtype.")
                gradient = None if value.grad is None else value.grad.to(device=device, dtype=target_dtype)
                staged[id(value)] = (value, converted, gradient)
            for value, converted, gradient in staged.values():
                if value.device == converted.device and value.dtype is converted.dtype:
                    continue
                required = value.requires_grad
                value._version_counter.increment()
                value._initialize(converted._data, required, converted._version_counter)
                value._grad = gradient
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self, index=0):
        from ..device import Device
        return self.to(Device("cuda", index))

    def __init__(self):
        if "_parameters" in self.__dict__:
            raise RuntimeError("Module.__init__() has already been called.")
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_non_persistent_buffers_set", set())
        object.__setattr__(self, "training", True)

    def __getattr__(self, name: str):
        for registry in _REGISTRIES:
            entries = self.__dict__.get(registry, {})
            if name in entries:
                return entries[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}.")

    def __setattr__(self, name: str, value: object) -> None:
        if name in _STATE_ATTRIBUTES:
            raise AttributeError(f"Module registry {name!r} cannot be replaced.")
        if name in self.__dict__.get("_parameters", {}):
            self.register_parameter(name, value)
        elif name in self.__dict__.get("_modules", {}):
            self.add_module(name, value)
        elif name in self.__dict__.get("_buffers", {}):
            self.register_buffer(name, value, persistent=name not in self._non_persistent_buffers_set)
        elif isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        else:
            if name == "training":
                _boolean("training", value)
            object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in _STATE_ATTRIBUTES or name == "training":
            raise AttributeError(f"Module state {name!r} cannot be deleted.")
        for registry in _REGISTRIES:
            entries = self.__dict__.get(registry, {})
            if name in entries:
                del entries[name]
                if registry == "_buffers":
                    self._non_persistent_buffers_set.discard(name)
                return
        object.__delattr__(self, name)

    def _validate_registration(self, name: str, registry: str) -> None:
        if "_parameters" not in self.__dict__:
            raise RuntimeError("Call Module.__init__() before registering parameters, modules or buffers.")
        if not isinstance(name, str):
            raise TypeError("Registered names must be strings.")
        if not name or "." in name:
            raise ValueError("Registered names must be nonempty and cannot contain a dot.")
        if name in _STATE_ATTRIBUTES or name in self.__dict__ or any(name in cls.__dict__ for cls in type(self).__mro__):
            raise KeyError(f"Attribute {name!r} already exists; delete it before registration.")
        if any(name in self.__dict__[other] for other in _REGISTRIES if other != registry):
            raise TypeError(f"Name {name!r} belongs to another registry; delete it before registration.")

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        self._validate_registration(name, "_parameters")
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(f"Parameter {name!r} must be a Parameter or None; delete the registration before assigning another type.")
        if param is not None and not param.is_leaf:
            raise ValueError("Registered parameters must be leaf tensors.")
        self._parameters[name] = param

    def add_module(self, name: str, module: Module | None) -> None:
        self._validate_registration(name, "_modules")
        if module is not None and not isinstance(module, Module):
            raise TypeError(f"Module {name!r} must be a Module or None.")
        if module is not None:
            for descendant in module.modules():
                if descendant is self:
                    raise ValueError(f"Registering module {name!r} would create a module cycle.")
        self._modules[name] = module

    def register_buffer(self, name: str, tensor: Tensor | None, persistent: bool = True) -> None:
        self._validate_registration(name, "_buffers")
        _boolean("persistent", persistent)
        if tensor is not None and (not isinstance(tensor, Tensor) or isinstance(tensor, Parameter)):
            raise TypeError(f"Buffer {name!r} must be a Tensor or None, and cannot be a Parameter.")
        self._buffers[name] = tensor
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

    def named_modules(self, prefix: str = "", remove_duplicate: bool = True):
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string.")
        _boolean("remove_duplicate", remove_duplicate)
        pending = [(prefix, self, False)]
        active, seen = set(), set()
        while pending:
            path, module, exiting = pending.pop()
            identity = id(module)
            if exiting:
                active.remove(identity)
                continue
            if identity in active:
                raise ValueError(f"Module cycle detected at {path!r}.")
            if remove_duplicate and identity in seen:
                continue
            active.add(identity)
            seen.add(identity)
            yield path, module
            pending.append((path, module, True))
            for name, child in reversed(tuple(module._modules.items())):
                if child is not None:
                    if not isinstance(child, Module):
                        raise TypeError(f"Registered child {name!r} is not a Module.")
                    pending.append((_qualified_name(path, name), child, False))

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def named_children(self):
        seen = set()
        for name, module in self._modules.items():
            if module is not None and id(module) not in seen:
                seen.add(id(module))
                yield name, module

    def children(self):
        for _, module in self.named_children():
            yield module

    def _named_members(self, registry: str, prefix: str, recurse: bool, remove_duplicate: bool):
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string.")
        _boolean("recurse", recurse)
        _boolean("remove_duplicate", remove_duplicate)
        modules = self.named_modules(prefix, remove_duplicate) if recurse else ((prefix, self),)
        seen = set()
        for path, module in modules:
            for name, value in getattr(module, registry).items():
                if value is not None and (not remove_duplicate or id(value) not in seen):
                    seen.add(id(value))
                    yield _qualified_name(path, name), value

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        yield from self._named_members("_parameters", prefix, recurse, remove_duplicate)

    def parameters(self, recurse: bool = True):
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def named_buffers(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        yield from self._named_members("_buffers", prefix, recurse, remove_duplicate)

    def buffers(self, recurse: bool = True):
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def _state_members(self):
        for path, module in self.named_modules(remove_duplicate=False):
            for name, parameter in module._parameters.items():
                if parameter is not None:
                    yield _qualified_name(path, name), parameter
            for name, buffer in module._buffers.items():
                if buffer is not None and name not in module._non_persistent_buffers_set:
                    yield _qualified_name(path, name), buffer

    def state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            (name, Tensor._from_array(value._data.copy(), False))
            for name, value in self._state_members()
        )

    def _prepare_load_state_dict(self, state: Mapping, strict: bool = True):
        _boolean("strict", strict)
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping from names to Tensors or NumPy arrays.")
        supplied = OrderedDict(state.items())
        if any(not isinstance(name, str) for name in supplied):
            raise TypeError("State dictionary keys must be strings.")
        targets = OrderedDict(self._state_members())
        missing = [name for name in targets if name not in supplied]
        unexpected = [name for name in supplied if name not in targets]
        errors = []
        if strict and missing:
            errors.append(f"Missing keys: {missing}.")
        if strict and unexpected:
            errors.append(f"Unexpected keys: {unexpected}.")
        updates = {}
        for name, target in targets.items():
            if name not in supplied:
                continue
            source = supplied[name]
            if not isinstance(source, Tensor) and not is_array(source):
                errors.append(f"State {name!r} must be a Tensor or NumPy array.")
                continue
            array = source._data if isinstance(source, Tensor) else source
            if array.shape != target.shape:
                errors.append(f"Shape mismatch for {name!r}: expected {target.shape}, received {array.shape}.")
                continue
            if array.dtype != target.dtype.numpy_dtype:
                errors.append(f"Dtype mismatch for {name!r}: expected {target.dtype.name}, received {array.dtype}; explicit conversion is required.")
                continue
            if not target._writable or not writable(target._data):
                errors.append(f"Cannot load {name!r} into read-only Tensor storage.")
                continue
            snapshot = transfer(array, target.device, copy=True)
            identity = id(target)
            if identity in updates:
                previous_name, _, previous = updates[identity]
                if not namespace(previous).array_equal(previous, snapshot, equal_nan=True):
                    errors.append(f"Conflicting values for aliased state keys {previous_name!r} and {name!r}.")
            else:
                updates[identity] = (name, target, snapshot)
        if errors:
            raise RuntimeError("Cannot load module state:\n" + "\n".join(errors))
        return LoadStateDictResult(missing, unexpected), tuple(updates.values())

    def load_state_dict(self, state: Mapping, strict: bool = True) -> LoadStateDictResult:
        result, updates = self._prepare_load_state_dict(state, strict)
        with no_grad():
            for _, target, snapshot in updates:
                target.copy_(snapshot)
        return result

    def train(self, mode: bool = True) -> Module:
        _boolean("mode", mode)
        modules = list(self.modules())
        for module in modules:
            module.training = mode
        return self

    def eval(self) -> Module:
        return self.train(False)

    def apply(self, fn) -> Module:
        if not callable(fn):
            raise TypeError("Module.apply expects a callable.")
        tuple(self.modules())
        pending = [(self, False)]
        visited, ordered = set(), []
        while pending:
            module, exiting = pending.pop()
            if exiting:
                ordered.append(module)
            elif id(module) not in visited:
                visited.add(id(module))
                pending.append((module, True))
                pending.extend((child, False) for child in reversed(tuple(module.children())))
        for module in ordered:
            fn(module)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        _boolean("set_to_none", set_to_none)
        parameters = list(self.parameters())
        with no_grad():
            for parameter in parameters:
                if set_to_none:
                    parameter._grad = None
                elif parameter.grad is not None:
                    parameter.grad.zero_()

    def requires_grad_(self, requires_grad: bool = True) -> Module:
        _boolean("requires_grad", requires_grad)
        parameters = list(self.parameters())
        if requires_grad and any(not parameter.dtype.can_require_grad for parameter in parameters):
            raise ValueError("requires_grad=True requires floating parameters.")
        for parameter in parameters:
            parameter.requires_grad_(requires_grad)
        return self

    @same_device
    def __call__(self, *args, **kwargs):
        forward = getattr(self, "forward", None)
        if not callable(forward):
            raise TypeError(f"{type(self).__name__} must implement a callable forward method.")
        return forward(*args, **kwargs)


__all__ = ["Module", "LoadStateDictResult"]
