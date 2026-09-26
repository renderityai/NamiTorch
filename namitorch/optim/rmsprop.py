import numpy as np

from ..tensor import Tensor
from ._common import _FirstOrderOptimizer, _apply_update, _bool_option, _buffer, _gradient, _real_option, _scalar


class RMSprop(_FirstOrderOptimizer):
    _state_buffers = ("square_avg", "grad_avg", "momentum_buffer")
    _required_buffers = ("square_avg",)
    _nonnegative_buffers = ("square_avg",)
    _has_step = True

    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8, weight_decay: float = 0.0, momentum: float = 0.0, centered: bool = False, *, maximize: bool = False):
        super().__init__(params, {"lr": lr, "alpha": alpha, "eps": eps, "weight_decay": weight_decay, "momentum": momentum, "centered": centered, "maximize": maximize})

    def _prepare_options(self, options: dict, path: str) -> dict:
        prepared = super()._prepare_options(options, path)
        _real_option(prepared, "alpha", unit_interval=True)
        _real_option(prepared, "eps", positive=True)
        _real_option(prepared, "momentum")
        _bool_option(prepared, "centered")
        return prepared

    def _update_parameter(self, parameter: Tensor, gradient: Tensor, options: dict) -> None:
        grad = _gradient(parameter, gradient, options)
        state = self.state[id(parameter)]
        square_avg = _buffer(state, "square_avg", parameter)
        alpha = _scalar(parameter, options["alpha"])
        complement = _scalar(parameter, 1 - options["alpha"])
        square_avg.copy_(alpha * square_avg._data + complement * grad * grad)
        variance = square_avg._data
        if options["centered"]:
            grad_avg = _buffer(state, "grad_avg", parameter)
            grad_avg.copy_(alpha * grad_avg._data + complement * grad)
            variance = np.maximum(variance - grad_avg._data * grad_avg._data, 0)
        denominator = np.sqrt(variance) + _scalar(parameter, options["eps"], "eps")
        update = grad / denominator
        if options["momentum"] != 0:
            buffer = _buffer(state, "momentum_buffer", parameter)
            buffer.copy_(_scalar(parameter, options["momentum"]) * buffer._data + update)
            update = buffer._data
        state["step"] = state.get("step", 0) + 1
        _apply_update(parameter, update, options)


__all__ = ["RMSprop"]
