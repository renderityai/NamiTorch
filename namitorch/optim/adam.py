import numpy as np

from ..tensor import Tensor
from ._common import _FirstOrderOptimizer, _apply_update, _buffer, _gradient, _real_option, _scalar


class Adam(_FirstOrderOptimizer):
    _state_buffers = ("exp_avg", "exp_avg_sq")
    _required_buffers = ("exp_avg", "exp_avg_sq")
    _nonnegative_buffers = ("exp_avg_sq",)
    _has_step = True

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0, *, maximize: bool = False):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "maximize": maximize})

    def _prepare_options(self, options: dict, path: str) -> dict:
        prepared = super()._prepare_options(options, path)
        _real_option(prepared, "eps", positive=True)
        betas = prepared.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise TypeError("Adam betas must contain exactly two real scalars.")
        normalized = {"beta1": betas[0], "beta2": betas[1]}
        prepared["betas"] = tuple(_real_option(normalized, name, unit_interval=True) for name in ("beta1", "beta2"))
        return prepared

    def _update_parameter(self, parameter: Tensor, gradient: Tensor, options: dict) -> None:
        grad = _gradient(parameter, gradient, options, self._decoupled_weight_decay)
        state = self.state[id(parameter)]
        first = _buffer(state, "exp_avg", parameter)
        second = _buffer(state, "exp_avg_sq", parameter)
        beta1, beta2 = options["betas"]
        first.copy_(_scalar(parameter, beta1) * first._data + _scalar(parameter, 1 - beta1) * grad)
        second.copy_(_scalar(parameter, beta2) * second._data + _scalar(parameter, 1 - beta2) * grad * grad)
        step = state.get("step", 0) + 1
        state["step"] = step
        first_corrected = first._data / _scalar(parameter, 1 - beta1 ** step)
        second_corrected = second._data / _scalar(parameter, 1 - beta2 ** step)
        denominator = np.sqrt(second_corrected) + _scalar(parameter, options["eps"], "eps")
        _apply_update(parameter, first_corrected / denominator, options, self._decoupled_weight_decay)


class AdamW(Adam):
    _decoupled_weight_decay = True

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2, *, maximize: bool = False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, maximize=maximize)


__all__ = ["Adam", "AdamW"]
