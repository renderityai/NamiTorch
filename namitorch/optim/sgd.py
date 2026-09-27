from ..backends import same_device
from ..tensor import Tensor
from ._common import _FirstOrderOptimizer, _apply_update, _bool_option, _gradient, _real_option, _scalar


class SGD(_FirstOrderOptimizer):
    _state_buffers = ("momentum_buffer",)

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0, dampening: float = 0.0, weight_decay: float = 0.0, nesterov: bool = False, *, maximize: bool = False):
        super().__init__(params, {"lr": lr, "momentum": momentum, "dampening": dampening, "weight_decay": weight_decay, "nesterov": nesterov, "maximize": maximize})

    def _prepare_options(self, options: dict, path: str) -> dict:
        prepared = super()._prepare_options(options, path)
        momentum = _real_option(prepared, "momentum")
        dampening = _real_option(prepared, "dampening")
        if _bool_option(prepared, "nesterov") and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum > 0 and dampening == 0.")
        return prepared

    @same_device
    def _update_parameter(self, parameter: Tensor, gradient: Tensor, options: dict) -> None:
        update = _gradient(parameter, gradient, options)
        if options["momentum"] != 0:
            state = self.state[id(parameter)]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = Tensor(update, dtype=parameter.dtype, device=parameter.device)
            else:
                buffer = state["momentum_buffer"]
                buffer.copy_(_scalar(parameter, options["momentum"]) * buffer._data + _scalar(parameter, 1 - options["dampening"]) * update)
            buffer = state["momentum_buffer"]
            update = update + _scalar(parameter, options["momentum"]) * buffer._data if options["nesterov"] else buffer._data
        _apply_update(parameter, update, options)


__all__ = ["SGD"]
