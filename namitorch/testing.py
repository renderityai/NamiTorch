import math
import operator
from numbers import Real

import numpy as np

from .autograd import enable_grad, no_grad
from .dtype import float64
from .tensor import Tensor


class GradcheckError(AssertionError):
    __slots__ = ()


def _tolerance(name: str, value: object, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite.") from None
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        constraint = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {constraint}.")
    return result


def _integer_option(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean.")
    try:
        result = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer.") from None
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _evaluate(fn, inputs: tuple[Tensor, ...], name: str, expected: Tensor | None = None) -> Tensor:
    versions = tuple(value._version for value in inputs)
    output = fn(*inputs)
    for position, (value, version) in enumerate(zip(inputs, versions)):
        if value._version != version:
            raise GradcheckError(f"{name}: input {position} was modified in-place during gradcheck.")
    if not isinstance(output, Tensor) or not output.dtype.is_floating_point:
        raise TypeError(f"{name}: gradcheck requires a floating Tensor output.")
    if expected is not None and (output.shape != expected.shape or output.dtype is not expected.dtype):
        raise GradcheckError(f"{name}: output shape or dtype changed during gradcheck; expected {expected.shape}/{expected.dtype}, got {output.shape}/{output.dtype}.")
    if not np.all(np.isfinite(output.numpy())):
        raise GradcheckError(f"{name}: gradcheck requires finite output values.")
    return output


def gradcheck(
    fn, inputs, eps=1e-6, atol=1e-5, rtol=1e-4, max_elements=None, seed=0,
) -> bool:
    if not callable(fn):
        raise TypeError("fn must be callable.")
    eps = _tolerance("eps", eps, positive=True)
    atol = _tolerance("atol", atol)
    rtol = _tolerance("rtol", rtol)
    seed = _integer_option("seed", seed, 0)
    if max_elements is not None:
        max_elements = _integer_option("max_elements", max_elements, 1)
    name = getattr(fn, "__qualname__", type(fn).__qualname__)
    arguments = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    originals = tuple(value if isinstance(value, Tensor) else Tensor(value) for value in arguments)
    differentiable = tuple(value.dtype.is_floating_point for value in originals)
    if not any(flag and value.numel() for value, flag in zip(originals, differentiable)):
        raise ValueError("gradcheck requires at least one nonempty floating input.")
    arrays = tuple(value.numpy().astype(np.float64) if flag else value.numpy() for value, flag in zip(originals, differentiable))
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("gradcheck requires finite input values.")

    def copies(position=None, index=None, delta=0.0):
        values = []
        for number, (array, flag) in enumerate(zip(arrays, differentiable)):
            data = array.copy()
            if number == position:
                original = float(data[index])
                perturbed = original + delta
                if not math.isfinite(perturbed) or perturbed == original:
                    raise GradcheckError(f"{name}: eps={eps} cannot perturb input {position}, index {index}, value {original} safely in float64.")
                data[index] = perturbed
            values.append(Tensor(data, requires_grad=flag))
        return tuple(values)

    rng = np.random.default_rng(seed)
    analytical_inputs = copies()
    with enable_grad():
        output = _evaluate(fn, analytical_inputs, name)
        if output.numel() == 0:
            raise ValueError("gradcheck requires a nonempty output.")
        if output.grad_fn is not None:
            name = f"{name} [{output.grad_fn.name}]"
        upstream = np.asarray(rng.standard_normal(output.shape), dtype=np.float64)
        if output.requires_grad:
            loss = (output * Tensor(upstream, dtype=float64)).sum()
            loss.backward()
    analytical = tuple(value.grad.numpy() if value.grad is not None else np.zeros(value.shape, dtype=np.float64) for value in analytical_inputs)
    with no_grad():
        baseline = _evaluate(fn, copies(), name, output)
        if not np.array_equal(baseline.numpy(), output.numpy()):
            raise GradcheckError(f"{name}: fn must be deterministic and return the same values with and without grad mode.")
        for position, (array, flag) in enumerate(zip(arrays, differentiable)):
            if not flag:
                continue
            indices = range(array.size) if max_elements is None or max_elements >= array.size else np.sort(rng.choice(array.size, size=max_elements, replace=False))
            for flat_index in indices:
                index = tuple(int(coordinate) for coordinate in np.unravel_index(flat_index, array.shape))
                positive = _evaluate(fn, copies(position, index, eps), name, output).numpy()
                negative = _evaluate(fn, copies(position, index, -eps), name, output).numpy()
                with np.errstate(over="ignore", invalid="ignore"):
                    numerical = float(np.sum((positive - negative) * upstream, dtype=np.float64) / (2 * eps))
                actual = float(analytical[position][index])
                absolute = abs(actual - numerical)
                relative = absolute / max(1.0, abs(actual), abs(numerical))
                if not (math.isfinite(actual) and math.isfinite(numerical)) or (absolute > atol and relative > rtol):
                    raise GradcheckError(
                        f"{name}: input {position}, index {index}: analytical={actual:.17g}, numerical={numerical:.17g}, "
                        f"absolute error={absolute:.17g}, relative error={relative:.17g}; atol={atol:.17g}, rtol={rtol:.17g}."
                    )
    return True


__all__ = ["GradcheckError", "gradcheck"]
