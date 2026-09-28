from ..backends import get_backend, namespace, same_device
from ..backends._settings import fused_kernels_enabled

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

    @same_device
    def _update_parameter(self, parameter: Tensor, gradient: Tensor, options: dict) -> None:
        xp = namespace(parameter)
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
        denominator = xp.sqrt(second_corrected) + _scalar(parameter, options["eps"], "eps")
        _apply_update(parameter, first_corrected / denominator, options, self._decoupled_weight_decay)


class AdamW(Adam):
    _decoupled_weight_decay = True

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2, *, maximize: bool = False, fused=None):
        _FirstOrderOptimizer.__init__(self, params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "maximize": maximize, "fused": fused})

    def _prepare_options(self, options: dict, path: str) -> dict:
        prepared = super()._prepare_options(options, path)
        fused = prepared.setdefault("fused", None)
        if fused is not None and type(fused) is not bool:
            raise TypeError("AdamW fused must be None, True or False.")
        return prepared

    def _fused_kernel(self, parameter, gradient):
        kernel = getattr(get_backend(parameter), "adamw", None)
        state = self.state.get(id(parameter), {})
        first, second = (state.get(name) for name in self._state_buffers)
        arrays = (parameter._data, gradient._data, None if first is None else first._data, None if second is None else second._data)
        return kernel if kernel is not None and kernel.supports(*arrays) else None

    def _validate_update(self, parameter, gradient, options):
        if options["fused"] is True and self._fused_kernel(parameter, gradient) is None:
            raise RuntimeError("fused=True requires non-overlapping contiguous float32 CUDA parameters, gradients and states on the same device.")

    @same_device
    def _update_parameter(self, parameter: Tensor, gradient: Tensor, options: dict) -> None:
        fused = options["fused"]
        kernel = None if fused is False or (fused is None and not fused_kernels_enabled()) else self._fused_kernel(parameter, gradient)
        if kernel is None:
            if fused is True:
                raise RuntimeError("The parameter, gradient or optimizer state does not support fused AdamW.")
            return super()._update_parameter(parameter, gradient, options)
        state = self.state.get(id(parameter)) or {}
        first = _buffer(state, "exp_avg", parameter)
        second = _buffer(state, "exp_avg_sq", parameter)
        step = state.get("step", 0) + 1
        beta1, beta2 = options["betas"]
        coefficients = tuple(_scalar(parameter, value) for value in (
            options["lr"], beta1, beta2, 1 - beta1, 1 - beta2, options["eps"],
            options["weight_decay"], 1 - options["lr"] * options["weight_decay"],
            1 - beta1 ** step, 1 - beta2 ** step,
        ))
        launched = kernel.run(parameter._data, gradient._data, first._data, second._data, coefficients=coefficients, maximize=options["maximize"], update_parameter=options["lr"] != 0)
        if launched:
            first._version_counter.increment()
            second._version_counter.increment()
            if options["lr"] != 0:
                parameter._version_counter.increment()
        state["step"] = step
        self.state[id(parameter)] = state


__all__ = ["Adam", "AdamW"]
