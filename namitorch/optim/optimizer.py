from collections import defaultdict

import numpy as np

from ..autograd import no_grad
from ..backends import ensure_same_device, is_array
from ..dtype import from_numpy_dtype
from ..nn.parameter import Parameter
from ..tensor import Tensor


def _copy_value(value, path: str, parameter_shape: tuple[int, ...] | None = None, active: set[int] | None = None):
    if isinstance(value, Tensor) or is_array(value):
        shape = value.shape
        if parameter_shape is not None and shape not in ((), parameter_shape):
            raise ValueError(f"{path} has shape {shape}; expected scalar state or parameter shape {parameter_shape}.")
        if isinstance(value, Tensor):
            return Tensor(value, device=value.device)
        from_numpy_dtype(value.dtype)
        return value.copy()
    if isinstance(value, np.generic):
        from_numpy_dtype(value.dtype)
        return value.item()
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if not isinstance(value, (dict, list, tuple)):
        raise TypeError(f"{path} contains unsupported optimizer state type {type(value).__name__}.")
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError(f"{path} contains a cyclic optimizer state container.")
    active.add(identity)
    try:
        if isinstance(value, dict):
            copied = {}
            for key, item in value.items():
                if type(key) not in (str, int):
                    raise TypeError(f"{path} dictionary keys must be strings or integers.")
                copied[key] = _copy_value(item, f"{path}[{key!r}]", parameter_shape, active)
            return copied
        copied = [_copy_value(item, f"{path}[{index}]", parameter_shape, active) for index, item in enumerate(value)]
        return tuple(copied) if isinstance(value, tuple) else copied
    finally:
        active.remove(identity)


def _copy_options(options: dict, path: str) -> dict:
    if not isinstance(options, dict):
        raise TypeError(f"{path} must be a dictionary.")
    if any(not isinstance(key, str) or not key for key in options):
        raise TypeError(f"{path} option names must be nonempty strings.")
    if "params" in options:
        raise ValueError(f"{path} cannot contain the reserved option 'params'.")
    return _copy_value(options, path)


def _parameter_list(params, path: str, allow_single: bool = False) -> list[Parameter]:
    if allow_single and isinstance(params, Parameter):
        values = [params]
    else:
        if isinstance(params, (Tensor, dict, str, bytes, set, frozenset)):
            raise TypeError(f"{path} must be an ordered iterable of Parameters.")
        try:
            values = list(params)
        except TypeError:
            raise TypeError(f"{path} must be an ordered iterable of Parameters.") from None
    if not values:
        raise ValueError(f"{path} cannot be empty.")
    identities = set()
    for parameter in values:
        if not isinstance(parameter, Parameter):
            raise TypeError(f"{path} must contain only NamiTorch Parameters, got {type(parameter).__name__}.")
        if not parameter.is_leaf:
            raise ValueError(f"{path} must contain only leaf Parameters.")
        if id(parameter) in identities:
            raise ValueError(f"{path} contains a duplicate Parameter object.")
        identities.add(id(parameter))
    return values


class Optimizer:
    def __init__(self, params, defaults: dict | None = None):
        self.defaults = self._prepare_options({} if defaults is None else defaults, "defaults")
        self.state: dict[int, dict] = defaultdict(dict)
        self.param_groups: list[dict] = []
        if isinstance(params, (Tensor, dict, str, bytes, set, frozenset)):
            raise TypeError("Optimizer params must be an ordered iterable of Parameters or parameter-group dictionaries.")
        try:
            entries = list(params)
        except TypeError:
            raise TypeError("Optimizer params must be an ordered iterable of Parameters or parameter-group dictionaries.") from None
        if not entries:
            raise ValueError("Optimizer cannot receive an empty parameter list.")
        if all(isinstance(entry, dict) for entry in entries):
            groups = entries
        elif any(isinstance(entry, dict) for entry in entries):
            raise TypeError("Optimizer params cannot mix Parameters and parameter-group dictionaries.")
        else:
            groups = [{"params": entries}]
        for group in groups:
            self.add_param_group(group)

    def _prepare_options(self, options: dict, path: str) -> dict:
        return _copy_options(options, path)

    def _prepare_state(self, values: dict, parameter: Parameter, path: str) -> dict:
        return _copy_value(values, path, parameter.shape)

    def _registered_parameters(self) -> list[Parameter]:
        if not self.param_groups:
            raise ValueError("Optimizer must contain at least one parameter group.")
        parameters = []
        identities = set()
        for index, group in enumerate(self.param_groups):
            if not isinstance(group, dict) or "params" not in group:
                raise TypeError(f"Parameter group {index} must be a dictionary containing 'params'.")
            if not isinstance(group["params"], list):
                raise TypeError(f"Parameter group {index} params must remain a materialized list.")
            current = _parameter_list(group["params"], f"param_groups[{index}]['params']")
            for parameter in current:
                if id(parameter) in identities:
                    raise ValueError("A Parameter object cannot belong to more than one parameter group.")
                identities.add(id(parameter))
            parameters.extend(current)
        return parameters

    def add_param_group(self, param_group: dict) -> None:
        if not isinstance(param_group, dict):
            raise TypeError("add_param_group requires a dictionary.")
        if "params" not in param_group:
            raise ValueError("A parameter group must contain 'params'.")
        parameters = _parameter_list(param_group["params"], "param_group['params']", allow_single=True)
        existing = {id(parameter) for parameter in self._registered_parameters()} if self.param_groups else set()
        if any(id(parameter) in existing for parameter in parameters):
            raise ValueError("A Parameter object cannot belong to more than one parameter group.")
        options = _copy_options(self.defaults, "defaults")
        options.update(_copy_options({key: value for key, value in param_group.items() if key != "params"}, "param_group"))
        options = self._prepare_options(options, "param_group")
        options["params"] = parameters
        self.param_groups.append(options)

    def _parameters_with_grad(self):
        self._registered_parameters()
        for group in self.param_groups:
            for parameter in group["params"]:
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                gradient = parameter.grad
                if not isinstance(gradient, Tensor):
                    raise TypeError("Parameter gradients must be NamiTorch Tensors.")
                ensure_same_device(parameter, gradient)
                if gradient.shape != parameter.shape:
                    raise ValueError(f"Gradient shape {gradient.shape} does not match parameter shape {parameter.shape}.")
                if gradient.dtype is not parameter.dtype:
                    raise TypeError(f"Gradient dtype {gradient.dtype} does not match parameter dtype {parameter.dtype}.")
                yield group, parameter, gradient

    def zero_grad(self, set_to_none: bool = True) -> None:
        if type(set_to_none) is not bool:
            raise TypeError("set_to_none must be a Python bool.")
        parameters = self._registered_parameters()
        with no_grad():
            for parameter in parameters:
                if set_to_none:
                    parameter._grad = None
                elif parameter.grad is not None:
                    parameter.grad.zero_()

    def step(self, closure=None):
        raise RuntimeError("Optimizer subclass must implement step.")

    def state_dict(self) -> dict:
        parameters = self._registered_parameters()
        indices = {id(parameter): index for index, parameter in enumerate(parameters)}
        state = {}
        for identity, values in self.state.items():
            if type(identity) is not int or identity not in indices:
                raise ValueError("Optimizer state keys must be identities of registered Parameters.")
            if not isinstance(values, dict):
                raise TypeError("Each parameter's optimizer state must be a dictionary.")
            index = indices[identity]
            state[index] = self._prepare_state(values, parameters[index], f"state[{index}]")
        groups = []
        for index, group in enumerate(self.param_groups):
            copied = self._prepare_options({key: value for key, value in group.items() if key != "params"}, f"param_groups[{index}]")
            copied["params"] = [indices[id(parameter)] for parameter in group["params"]]
            groups.append(copied)
        return {
            "version": 1,
            "defaults": self._prepare_options(self.defaults, "defaults"),
            "param_groups": groups,
            "param_shapes": [list(parameter.shape) for parameter in parameters],
            "state": state,
        }

    def _prepare_load_state_dict(self, state_dict: dict):
        if not isinstance(state_dict, dict):
            raise TypeError("Optimizer state_dict must be a dictionary.")
        expected_keys = {"version", "defaults", "param_groups", "param_shapes", "state"}
        if set(state_dict) != expected_keys:
            raise ValueError("Optimizer state_dict must contain version, defaults, param_groups, param_shapes and state.")
        if type(state_dict["version"]) is not int or state_dict["version"] != 1:
            raise ValueError("Unsupported optimizer state_dict version.")
        parameters = self._registered_parameters()
        groups = state_dict["param_groups"]
        if not isinstance(groups, list) or len(groups) != len(self.param_groups):
            raise ValueError("Loaded optimizer must have the same number of parameter groups.")
        shapes = state_dict["param_shapes"]
        if not isinstance(shapes, list) or len(shapes) != len(parameters):
            raise ValueError("Loaded optimizer must contain one shape for each parameter.")
        defaults = self._prepare_options(state_dict["defaults"], "defaults")
        restored_groups = []
        index_to_parameter = {}
        for group_index, (saved, current) in enumerate(zip(groups, self.param_groups)):
            if not isinstance(saved, dict) or "params" not in saved:
                raise ValueError(f"Loaded parameter group {group_index} must contain 'params'.")
            saved_indices = saved["params"]
            if not isinstance(saved_indices, list) or len(saved_indices) != len(current["params"]):
                raise ValueError(f"Loaded parameter group {group_index} has a different number of parameters.")
            options = self._prepare_options({key: value for key, value in saved.items() if key != "params"}, f"param_groups[{group_index}]")
            if not defaults.keys() <= options.keys():
                raise ValueError(f"Loaded parameter group {group_index} is missing default options.")
            for index, parameter in zip(saved_indices, current["params"]):
                if type(index) is not int or not 0 <= index < len(parameters):
                    raise ValueError("Loaded parameter indices must be integers in the registered parameter range.")
                if index in index_to_parameter:
                    raise ValueError("Loaded parameter groups contain duplicate parameter indices.")
                shape = shapes[index]
                if not isinstance(shape, (list, tuple)) or any(type(dimension) is not int or dimension < 0 for dimension in shape):
                    raise ValueError(f"Invalid saved parameter shape for index {index}.")
                if tuple(shape) != parameter.shape:
                    raise ValueError(f"Saved parameter {index} shape {tuple(shape)} does not match current shape {parameter.shape}.")
                index_to_parameter[index] = parameter
            options["params"] = list(current["params"])
            restored_groups.append(options)
        saved_state = state_dict["state"]
        if not isinstance(saved_state, dict):
            raise TypeError("Loaded optimizer state must be a dictionary keyed by parameter indices.")
        restored_state = defaultdict(dict)
        for index, values in saved_state.items():
            if type(index) is not int or index not in index_to_parameter:
                raise ValueError("Loaded state keys must be registered parameter indices.")
            if not isinstance(values, dict):
                raise TypeError(f"Loaded state[{index}] must be a dictionary.")
            parameter = index_to_parameter[index]
            restored_state[id(parameter)] = self._prepare_state(values, parameter, f"state[{index}]")
        return defaults, restored_groups, restored_state

    def load_state_dict(self, state_dict: dict) -> None:
        self.defaults, self.param_groups, self.state = self._prepare_load_state_dict(state_dict)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(param_groups={len(self.param_groups)})"


__all__ = ["Optimizer"]
