import math
from numbers import Real

import numpy as np

from ..autograd import no_grad
from ..random import Generator, _get_rng, _validate_generator, rand, randn
from ..tensor import Tensor


def _tensor(tensor: Tensor, floating: bool = False) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError("Initialization requires a NamiTorch Tensor.")
    if floating and not tensor.dtype.is_floating_point:
        raise TypeError("Random initialization requires a floating Tensor.")
    if not tensor._data.flags.writeable:
        raise RuntimeError("Cannot initialize read-only Tensor storage.")
    return tensor


def _real(value: object, name: str, tensor: Tensor) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar.")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite.") from None
    if not math.isfinite(value) or abs(value) > float(np.finfo(tensor.dtype.numpy_dtype).max):
        raise ValueError(f"{name} must be finite and representable in {tensor.dtype.name}.")
    return value


def constant_(tensor: Tensor, value: object, *, generator: Generator | None = None) -> Tensor:
    _tensor(tensor)
    _validate_generator(generator)
    with no_grad():
        return tensor.fill_(value)


def zeros_(tensor: Tensor, *, generator: Generator | None = None) -> Tensor:
    return constant_(tensor, 0, generator=generator)


def ones_(tensor: Tensor, *, generator: Generator | None = None) -> Tensor:
    return constant_(tensor, 1, generator=generator)


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0, *, generator: Generator | None = None) -> Tensor:
    _tensor(tensor, floating=True)
    _validate_generator(generator)
    lower, upper = _real(a, "a", tensor), _real(b, "b", tensor)
    if lower > upper:
        raise ValueError("uniform_ requires a <= b.")
    width = _real(upper - lower, "b - a", tensor)
    if width == 0:
        return constant_(tensor, lower)
    with no_grad():
        values = rand(tensor.shape, dtype=tensor.dtype, generator=generator)
        return tensor.copy_(values * Tensor(width, dtype=tensor.dtype) + Tensor(lower, dtype=tensor.dtype))


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, *, generator: Generator | None = None) -> Tensor:
    _tensor(tensor, floating=True)
    _validate_generator(generator)
    center, deviation = _real(mean, "mean", tensor), _real(std, "std", tensor)
    if deviation <= 0 or tensor.dtype.numpy_dtype.type(deviation) == 0:
        raise ValueError("normal_ requires a positive, representable std.")
    with no_grad():
        values = randn(tensor.shape, dtype=tensor.dtype, generator=generator)
        return tensor.copy_(values * Tensor(deviation, dtype=tensor.dtype) + Tensor(center, dtype=tensor.dtype))


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar.")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite.") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def calculate_gain(nonlinearity: str, param: object = None) -> float:
    if not isinstance(nonlinearity, str):
        raise TypeError("nonlinearity must be a string.")
    if nonlinearity in ("linear", "sigmoid"):
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else _finite_real(param, "negative_slope")
        return math.sqrt(2.0) / math.hypot(1.0, slope)
    raise ValueError(f"Unsupported nonlinearity {nonlinearity!r} for calculate_gain.")


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    if not isinstance(tensor, Tensor):
        raise TypeError("Fan calculation requires a NamiTorch Tensor.")
    if tensor.ndim < 2:
        raise ValueError(f"Fan calculation requires rank >= 2, got shape {tensor.shape}.")
    receptive_size = math.prod(tensor.shape[2:])
    return tensor.shape[1] * receptive_size, tensor.shape[0] * receptive_size


def _xavier_std(tensor: Tensor, gain: object) -> float:
    _tensor(tensor, floating=True)
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    scale = _finite_real(gain, "gain")
    if scale < 0:
        raise ValueError("gain must be nonnegative.")
    if fan_in + fan_out <= 0:
        raise ValueError("Xavier initialization requires a positive fan_in + fan_out.")
    return scale * math.sqrt(2.0 / (fan_in + fan_out))


def xavier_uniform_(tensor: Tensor, gain: float = 1.0, *, generator: Generator | None = None) -> Tensor:
    bound = math.sqrt(3.0) * _xavier_std(tensor, gain)
    return uniform_(tensor, -bound, bound, generator=generator)


def xavier_normal_(tensor: Tensor, gain: float = 1.0, *, generator: Generator | None = None) -> Tensor:
    std = _xavier_std(tensor, gain)
    if std == 0:
        return zeros_(tensor, generator=generator)
    return normal_(tensor, std=std, generator=generator)


def _kaiming_std(tensor: Tensor, a: object, mode: str, nonlinearity: str) -> float:
    _tensor(tensor, floating=True)
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode not in ("fan_in", "fan_out"):
        raise ValueError("mode must be 'fan_in' or 'fan_out'.")
    slope = _finite_real(a, "a")
    gain = calculate_gain(nonlinearity, slope)
    fan = fan_in if mode == "fan_in" else fan_out
    if fan <= 0:
        raise ValueError(f"Kaiming initialization requires positive {mode}.")
    return gain / math.sqrt(fan)


def kaiming_uniform_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu", *, generator: Generator | None = None) -> Tensor:
    bound = math.sqrt(3.0) * _kaiming_std(tensor, a, mode, nonlinearity)
    return uniform_(tensor, -bound, bound, generator=generator)


def kaiming_normal_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu", *, generator: Generator | None = None) -> Tensor:
    std = _kaiming_std(tensor, a, mode, nonlinearity)
    return normal_(tensor, std=std, generator=generator)


def _normal_interval_probability(lower: float, upper: float) -> float:
    scale = math.sqrt(2.0)
    if lower >= 0:
        return 0.5 * (math.erfc(lower / scale) - math.erfc(upper / scale))
    if upper <= 0:
        return 0.5 * (math.erfc(-upper / scale) - math.erfc(-lower / scale))
    return 0.5 * (math.erf(upper / scale) - math.erf(lower / scale))


def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0, *, generator: Generator | None = None) -> Tensor:
    _tensor(tensor, floating=True)
    _validate_generator(generator)
    center, deviation = _real(mean, "mean", tensor), _real(std, "std", tensor)
    lower, upper = _real(a, "a", tensor), _real(b, "b", tensor)
    if deviation <= 0 or tensor.dtype.numpy_dtype.type(deviation) == 0:
        raise ValueError("trunc_normal_ requires a positive, representable std.")
    if lower >= upper:
        raise ValueError("trunc_normal_ requires a < b.")
    acceptance = _normal_interval_probability((lower - center) / deviation, (upper - center) / deviation)
    if acceptance < 1e-6:
        raise ValueError("trunc_normal_ acceptance probability must be at least 1e-6; adjust mean, std or [a, b].")
    lowest = tensor.dtype.numpy_dtype.type(lower)
    if float(lowest) < lower:
        lowest = np.nextafter(lowest, tensor.dtype.numpy_dtype.type(np.inf))
    if float(lowest) > upper:
        raise ValueError("trunc_normal_ interval contains no value representable in the Tensor dtype.")
    count = tensor.numel()
    values = np.empty(count, dtype=tensor.dtype.numpy_dtype)
    rng = _get_rng(generator)
    filled = 0
    remaining_draws = max(10000, math.ceil(10 * count / acceptance))
    while filled < count:
        if remaining_draws <= 0:
            raise RuntimeError("trunc_normal_ exceeded its rejection sampling budget; Tensor storage was not modified.")
        draws = min(65536, remaining_draws, max(32, math.ceil(1.2 * (count - filled) / acceptance)))
        with np.errstate(over="ignore", invalid="ignore"):
            samples = rng.standard_normal(draws) * deviation + center
            accepted = samples[(samples >= lower) & (samples <= upper)].astype(tensor.dtype.numpy_dtype)
        accepted = accepted[(accepted.astype(np.float64) >= lower) & (accepted.astype(np.float64) <= upper)]
        take = min(count - filled, accepted.size)
        values[filled:filled + take] = accepted[:take]
        filled += take
        remaining_draws -= draws
    with no_grad():
        return tensor.copy_(values.reshape(tensor.shape))


__all__ = [
    "calculate_gain", "constant_", "zeros_", "ones_", "uniform_", "normal_",
    "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_", "trunc_normal_",
]
