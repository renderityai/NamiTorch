from .backends._graph_capture import reject_during_capture
import operator
from math import prod
from secrets import randbits
from threading import RLock

import numpy as np

from .backends import get_backend, namespace, same_device

from .device import Device
from .dtype import DType, get_default_dtype, int64, normalize_dtype
from .tensor import Tensor, _validate_requires_grad
from .utils import normalize_shape


def _seed(seed: object) -> int:
    if isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must be an integer, not a boolean.")
    try:
        value = operator.index(seed)
    except TypeError:
        raise TypeError("seed must be an integer.") from None
    if not 0 <= value < 2 ** 64:
        raise ValueError("seed must be in [0, 2**64 - 1].")
    return int(value)


class Generator:
    __slots__ = ("_rng_backend", "_rng")

    def __init__(self, seed: object = None, *, device=None):
        seed = None if seed is None else _seed(seed)
        self._rng_backend = get_backend(Device("cpu" if device is None else device))
        self._rng = self._rng_backend.create_rng(seed)

    @property
    def device(self):
        return self._rng_backend.device

    def manual_seed(self, seed: object):
        self._rng_backend.seed_rng(self._rng, _seed(seed))
        return self

    def get_state(self) -> dict:
        return self._rng_backend.rng_state(self._rng)

    def set_state(self, state: dict) -> None:
        self._rng_backend.restore_rng(self._rng, state)

    def permutation(self, n: object) -> Tensor:
        return permutation(n, generator=self, device=self.device)

    def choice(self, n: object, size: object = (), replace: bool = True) -> Tensor:
        return choice(n, size, replace, generator=self, device=self.device)

    def __repr__(self) -> str:
        if self.device.type == "cpu":
            return "Generator(PCG64)"
        return f"Generator(PHILOX, device={str(self.device)!r})"


_default_generator = Generator()
_default_rng = _default_generator._rng
_cuda_generators = {}
_cuda_base_seed = randbits(64)
_default_lock = RLock()


def _validate_generator(generator: Generator | None, device=None) -> None:
    if generator is not None and not isinstance(generator, Generator):
        raise TypeError("generator must be a NamiTorch Generator or None.")
    if generator is not None and device is not None and generator.device != Device(device):
        raise RuntimeError(f"Generator device {generator.device} does not match output device {Device(device)}.")


def _get_default_generator(device=None):
    target = Device("cpu" if device is None else device)
    if target.type == "cpu":
        return _default_generator
    with _default_lock:
        if target not in _cuda_generators:
            _cuda_generators[target] = Generator(_cuda_base_seed, device=target)
        return _cuda_generators[target]


def _get_rng(generator: Generator | None = None, *, device=None):
    reject_during_capture('Random sampling and dropout')
    target = Device("cpu" if device is None else device)
    _validate_generator(generator, target)
    if generator is not None:
        return generator._rng
    return _default_rng if target.type == "cpu" else _get_default_generator(target)._rng


def _manual_seed_cuda_all(seed):
    global _cuda_base_seed
    value = _seed(seed)
    with _default_lock:
        _cuda_base_seed = value
        for generator in _cuda_generators.values():
            generator.manual_seed(value)


def manual_seed(seed: object) -> Generator:
    value = _seed(seed)
    with _default_lock:
        _default_generator.manual_seed(value)
        _manual_seed_cuda_all(value)
    return _default_generator


def get_state(device=None) -> dict:
    return _get_default_generator(device).get_state()


def set_state(state: dict, device=None) -> None:
    _get_default_generator(device).set_state(state)


def _cuda_states():
    with _default_lock:
        return {
            "version": 1,
            "base_seed": _cuda_base_seed,
            "generators": {str(device): generator.get_state() for device, generator in sorted(_cuda_generators.items(), key=lambda item: item[0].index)},
        }


def _prepare_cuda_states(state):
    if type(state) is not dict or set(state) != {"version", "base_seed", "generators"}:
        raise ValueError("Invalid default CUDA generator state fields.")
    if type(state["version"]) is not int or state["version"] != 1:
        raise ValueError("Unsupported default CUDA generator state version.")
    seed = _seed(state["base_seed"])
    if type(state["generators"]) is not dict:
        raise TypeError("Default CUDA generator states must be a dictionary.")
    restored = {}
    for name, saved in state["generators"].items():
        target = Device(name)
        if target.type != "cuda" or str(target) != name:
            raise ValueError("Default CUDA generator keys must be canonical CUDA device strings.")
        validator = Generator(0, device=target)
        validator.set_state(saved)
        restored[target] = validator
    with _default_lock:
        requests = [(_cuda_generators.get(device, validator), validator.get_state()) for device, validator in restored.items()]
        for device, generator in _cuda_generators.items():
            if device not in restored:
                requests.append((generator, Generator(seed, device=device).get_state()))
    return seed, requests


def _install_cuda_defaults(seed, requests):
    global _cuda_base_seed
    with _default_lock:
        _cuda_base_seed = seed
        for generator, state in requests:
            generator.set_state(state)
            _cuda_generators[generator.device] = generator


def _floating_dtype(dtype: object, requires_grad: bool) -> DType:
    target = get_default_dtype() if dtype is None else normalize_dtype(dtype)
    if not target.is_floating_point:
        raise TypeError("rand and randn require a floating dtype.")
    _validate_requires_grad(target, requires_grad)
    return target


def rand(*shape: object, dtype: object = None, requires_grad: bool = False, generator: Generator | None = None, device=None) -> Tensor:
    dimensions = normalize_shape(*shape)
    target = _floating_dtype(dtype, requires_grad)
    rng = _get_rng(generator, device=device)
    array = get_backend(device).random(rng, "random", dimensions, target.numpy_dtype.type)
    return Tensor._from_array(array, requires_grad)


def randn(*shape: object, dtype: object = None, requires_grad: bool = False, generator: Generator | None = None, device=None) -> Tensor:
    dimensions = normalize_shape(*shape)
    target = _floating_dtype(dtype, requires_grad)
    rng = _get_rng(generator, device=device)
    array = get_backend(device).random(rng, "standard_normal", dimensions, target.numpy_dtype.type)
    return Tensor._from_array(array, requires_grad)


def _integer_bound(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("randint bounds must be integers, not booleans.")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError("randint bounds must be integers.") from None


def randint(
    low: object, high: object = None, size: object = (),
    *, dtype: object = None, requires_grad: bool = False, generator: Generator | None = None, device=None,
) -> Tensor:
    lower = 0 if high is None else _integer_bound(low)
    upper = _integer_bound(low if high is None else high)
    dimensions = normalize_shape(size)
    target = int64 if dtype is None else normalize_dtype(dtype)
    if not target.is_integer:
        raise TypeError("randint requires an integer dtype.")
    _validate_requires_grad(target, requires_grad)
    if lower >= upper:
        raise ValueError("randint requires low < high.")
    limits = np.iinfo(target.numpy_dtype)
    if lower < limits.min or upper - 1 > limits.max:
        raise ValueError(f"randint bounds are outside the range of {target.name}.")
    rng = _get_rng(generator, device=device)
    array = get_backend(device).random(rng, "integers", dimensions, target.numpy_dtype.type, low=lower, high=upper)
    return Tensor._from_array(array, False)


def _population_size(n: object) -> int:
    if isinstance(n, (bool, np.bool_)):
        raise TypeError("Population size must be an integer, not a boolean.")
    try:
        value = operator.index(n)
    except TypeError:
        raise TypeError("Population size must be an integer.") from None
    if not 0 <= value <= np.iinfo(np.intp).max:
        raise ValueError("Population size must be nonnegative and fit in the platform index range.")
    return int(value)


def permutation(n: object, *, generator: Generator | None = None, device=None) -> Tensor:
    count = _population_size(n)
    rng = _get_rng(generator, device=device)
    array = get_backend(device).random(rng, "permutation", count, np.int64)
    return Tensor._from_array(array, False)


def choice(n: object, size: object = (), replace: bool = True, *, generator: Generator | None = None, device=None) -> Tensor:
    count = _population_size(n)
    dimensions = normalize_shape(size)
    if type(replace) is not bool:
        raise TypeError("replace must be a Python bool.")
    samples = prod(dimensions)
    if count == 0 and samples > 0:
        raise ValueError("Cannot sample from an empty population.")
    if not replace and samples > count:
        raise ValueError("Cannot draw more samples than the population without replacement.")
    rng = _get_rng(generator, device=device)
    array = get_backend(device).random(rng, "choice", dimensions, np.int64, n=count, replace=replace)
    return Tensor._from_array(array, False)


@same_device
def categorical(probabilities: Tensor, *, generator: Generator | None = None) -> Tensor:
    _validate_generator(generator)
    if not isinstance(probabilities, Tensor) or not probabilities.dtype.is_floating_point:
        raise TypeError("categorical probabilities must be a floating Tensor.")
    _validate_generator(generator, probabilities.device)
    if probabilities.ndim < 1 or probabilities.shape[-1] == 0:
        raise ValueError("categorical requires a nonempty final category dimension.")
    xp = namespace(probabilities)
    values = xp.asarray(probabilities._data, dtype=np.float64)
    if xp.any(~xp.isfinite(values)) or xp.any(values < 0):
        raise ValueError("categorical probabilities must be finite and nonnegative.")
    maximum = values.max(axis=-1, keepdims=True)
    if xp.any(maximum == 0):
        raise ValueError("Each categorical distribution must have positive total weight.")
    scaled = values / maximum
    normalized = scaled / scaled.sum(axis=-1, keepdims=True)
    cumulative = xp.cumsum(normalized, axis=-1)
    cumulative /= cumulative[..., -1:]
    cumulative[..., -1] = 1.0
    uniforms = get_backend(probabilities).random(_get_rng(generator, device=probabilities.device), "random", values.shape[:-1], np.float64)
    indices = xp.sum(uniforms[..., None] >= cumulative, axis=-1, dtype=np.int64)
    return Tensor._from_array(xp.asarray(indices, dtype=np.int64), False)


__all__ = ["Generator", "manual_seed", "get_state", "set_state", "rand", "randn", "randint", "permutation", "choice", "categorical"]
