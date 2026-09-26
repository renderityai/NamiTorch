import operator
from copy import deepcopy

import numpy as np

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
    __slots__ = ("_rng",)

    def __init__(self, seed: object = None):
        self._rng = np.random.Generator(np.random.PCG64(None if seed is None else _seed(seed)))

    def manual_seed(self, seed: object):
        self._rng.bit_generator.state = np.random.PCG64(_seed(seed)).state
        return self

    def get_state(self) -> dict:
        state = deepcopy(self._rng.bit_generator.state)
        state["version"] = 1
        return state

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise TypeError("Generator state must be a dictionary.")
        expected_keys = {"version", "bit_generator", "state", "has_uint32", "uinteger"}
        if set(state) != expected_keys or state.get("bit_generator") != "PCG64":
            raise ValueError("Expected a versioned PCG64 generator state.")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("Unsupported generator state version.")
        inner = state["state"]
        if not isinstance(inner, dict) or set(inner) != {"state", "inc"}:
            raise ValueError("Invalid PCG64 state and increment fields.")
        for name, value, maximum in (
            ("state", inner["state"], 2 ** 128), ("inc", inner["inc"], 2 ** 128),
            ("has_uint32", state["has_uint32"], 2), ("uinteger", state["uinteger"], 2 ** 32),
        ):
            if type(value) is not int or not 0 <= value < maximum:
                raise ValueError(f"Invalid generator state field {name!r}.")
        if inner["inc"] % 2 != 1:
            raise ValueError("The PCG64 increment must be odd.")
        restored = deepcopy(state)
        del restored["version"]
        self._rng.bit_generator.state = restored

    def permutation(self, n: object) -> Tensor:
        return permutation(n, generator=self)

    def choice(self, n: object, size: object = (), replace: bool = True) -> Tensor:
        return choice(n, size, replace, generator=self)

    def __repr__(self) -> str:
        return "Generator(PCG64)"


_default_generator = Generator()
_default_rng = _default_generator._rng


def _validate_generator(generator: Generator | None) -> None:
    if generator is not None and not isinstance(generator, Generator):
        raise TypeError("generator must be a NamiTorch Generator or None.")


def _get_rng(generator: Generator | None = None) -> np.random.Generator:
    _validate_generator(generator)
    return _default_rng if generator is None else generator._rng


def manual_seed(seed: object) -> Generator:
    return _default_generator.manual_seed(seed)


def get_state() -> dict:
    return _default_generator.get_state()


def set_state(state: dict) -> None:
    _default_generator.set_state(state)


def _floating_dtype(dtype: object, requires_grad: bool) -> DType:
    target = get_default_dtype() if dtype is None else normalize_dtype(dtype)
    if not target.is_floating_point:
        raise TypeError("rand and randn require a floating dtype.")
    _validate_requires_grad(target, requires_grad)
    return target


def rand(*shape: object, dtype: object = None, requires_grad: bool = False, generator: Generator | None = None) -> Tensor:
    dimensions = normalize_shape(*shape)
    target = _floating_dtype(dtype, requires_grad)
    array = _get_rng(generator).random(dimensions, dtype=target.numpy_dtype.type)
    return Tensor._from_array(array, requires_grad)


def randn(*shape: object, dtype: object = None, requires_grad: bool = False, generator: Generator | None = None) -> Tensor:
    dimensions = normalize_shape(*shape)
    target = _floating_dtype(dtype, requires_grad)
    array = _get_rng(generator).standard_normal(dimensions, dtype=target.numpy_dtype.type)
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
    *, dtype: object = None, requires_grad: bool = False, generator: Generator | None = None,
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
    array = _get_rng(generator).integers(lower, upper, size=dimensions, dtype=target.numpy_dtype.type)
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


def permutation(n: object, *, generator: Generator | None = None) -> Tensor:
    count = _population_size(n)
    array = _get_rng(generator).permutation(count).astype(np.int64, copy=False)
    return Tensor._from_array(array, False)


def choice(n: object, size: object = (), replace: bool = True, *, generator: Generator | None = None) -> Tensor:
    count = _population_size(n)
    dimensions = normalize_shape(size)
    if type(replace) is not bool:
        raise TypeError("replace must be a Python bool.")
    array = _get_rng(generator).choice(count, size=dimensions, replace=replace)
    return Tensor._from_array(np.asarray(array, dtype=np.int64), False)


def categorical(probabilities: Tensor, *, generator: Generator | None = None) -> Tensor:
    _validate_generator(generator)
    if not isinstance(probabilities, Tensor) or not probabilities.dtype.is_floating_point:
        raise TypeError("categorical probabilities must be a floating Tensor.")
    if probabilities.ndim < 1 or probabilities.shape[-1] == 0:
        raise ValueError("categorical requires a nonempty final category dimension.")
    values = np.asarray(probabilities._data, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("categorical probabilities must be finite and nonnegative.")
    maximum = values.max(axis=-1, keepdims=True)
    if np.any(maximum == 0):
        raise ValueError("Each categorical distribution must have positive total weight.")
    scaled = values / maximum
    normalized = scaled / scaled.sum(axis=-1, keepdims=True)
    cumulative = np.cumsum(normalized, axis=-1)
    cumulative /= cumulative[..., -1:]
    cumulative[..., -1] = 1.0
    uniforms = _get_rng(generator).random(values.shape[:-1])
    indices = np.sum(uniforms[..., None] >= cumulative, axis=-1, dtype=np.int64)
    return Tensor._from_array(np.asarray(indices, dtype=np.int64), False)


__all__ = ["Generator", "manual_seed", "get_state", "set_state", "rand", "randn", "randint", "permutation", "choice", "categorical"]
