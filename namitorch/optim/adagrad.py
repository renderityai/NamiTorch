import numpy as np

from ..tensor import Tensor
from ._common import _FirstOrderOptimizer, _apply_update, _buffer, _gradient, _real_option, _scalar


class Adagrad(_FirstOrderOptimizer):
    _state_buffers = ("sum",)
    _required_buffers = ("sum",)
    _nonnegative_buffers = ("sum",)
    _has_step = True

    def __init__(self, params, lr: float = 1e-2, weight_decay: float = 0.0, eps: float = 1e-10, *, maximize: bool = False):
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay, "eps": eps, "maximize": maximize})

    def _prepare_options(self, options: dict, path: str) -> dict:
        prepared = super()._prepare_options(options, path)
        _real_option(prepared, "eps", positive=True)
        return prepared

    def _update_parameter(self, parameter: Tensor, gradient: Tensor, options: dict) -> None:
        grad = _gradient(parameter, gradient, options)
        state = self.state[id(parameter)]
        total = _buffer(state, "sum", parameter)
        total.copy_(total._data + grad * grad)
        state["step"] = state.get("step", 0) + 1
        denominator = np.sqrt(total._data) + _scalar(parameter, options["eps"], "eps")
        _apply_update(parameter, grad / denominator, options)


__all__ = ["Adagrad"]
