import numpy as np

from ..nn import functional as F
from ..random import Generator, _validate_generator, categorical
from ..tensor import Tensor


def _sampling_options(temperature, top_k, top_p, repetition_penalty, frequency_penalty, presence_penalty):
    temperature = F._real_parameter(temperature, "temperature")
    top_k = None if top_k is None else F._integer(top_k, "top_k")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be nonnegative or None.")
    top_p = F._real_parameter(top_p, "top_p")
    if not 0 <= top_p <= 1:
        raise ValueError("top_p must be in [0, 1].")
    repetition_penalty = F._real_parameter(repetition_penalty, "repetition_penalty")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive.")
    return dict(
        temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
        frequency_penalty=F._real_parameter(frequency_penalty, "frequency_penalty"),
        presence_penalty=F._real_parameter(presence_penalty, "presence_penalty"),
    )


def _probabilities(scores):
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def sample_next_token(
    logits: Tensor, history: Tensor | None = None, *, temperature: float = 1.0,
    top_k: int | None = None, top_p: float = 1.0, repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
    generator: Generator | None = None,
) -> Tensor:
    options = _sampling_options(temperature, top_k, top_p, repetition_penalty, frequency_penalty, presence_penalty)
    _validate_generator(generator)
    if not isinstance(logits, Tensor) or not logits.dtype.is_floating_point:
        raise TypeError("logits must be a floating NamiTorch Tensor.")
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [B, V] with V > 0.")
    batch, vocabulary = logits.shape
    if history is not None:
        if not isinstance(history, Tensor) or not history.dtype.is_integer:
            raise TypeError("history must be an integer NamiTorch Tensor or None.")
        if history.ndim != 2 or history.shape[0] != batch:
            raise ValueError("history must have shape [B, T] matching the logits batch.")
        if np.any(history._data < 0) or np.any(history._data >= vocabulary):
            raise ValueError("history contains token IDs outside the vocabulary.")
    scores = np.array(logits._data, dtype=np.float64, copy=True)
    if np.any(np.isnan(scores) | np.isposinf(scores)) or np.any(~np.any(np.isfinite(scores), axis=-1)):
        raise ValueError("Each logits row must contain a finite value; NaN and positive infinity are forbidden.")
    if history is not None and (options["repetition_penalty"] != 1 or options["frequency_penalty"] != 0 or options["presence_penalty"] != 0):
        for row in range(batch):
            tokens, counts = np.unique(history._data[row], return_counts=True)
            selected = scores[row, tokens]
            with np.errstate(over="ignore", invalid="ignore"):
                adjusted = selected.copy()
                negative = selected < 0
                adjusted[negative] *= options["repetition_penalty"]
                adjusted[~negative] /= options["repetition_penalty"]
                adjusted -= options["frequency_penalty"] * counts
                adjusted -= options["presence_penalty"]
            if np.any(~np.isfinite(adjusted[np.isfinite(selected)])) or np.any(np.isnan(adjusted) | np.isposinf(adjusted)):
                raise ValueError("Sampling penalties produce nonfinite logits.")
            scores[row, tokens] = adjusted
    if options["temperature"] <= 0:
        return Tensor._from_array(np.asarray(np.argmax(scores, axis=-1), dtype=np.int64).reshape(batch, 1), False)
    with np.errstate(over="ignore", under="ignore"):
        if options["temperature"] >= 1:
            scores /= options["temperature"]
            scores -= scores.max(axis=-1, keepdims=True)
        else:
            scores -= scores.max(axis=-1, keepdims=True)
            scores /= options["temperature"]
    if options["top_k"]:
        keep = min(options["top_k"], vocabulary)
        threshold = np.partition(scores, vocabulary - keep, axis=-1)[:, vocabulary - keep: vocabulary - keep + 1]
        scores[scores < threshold] = -np.inf
    if options["top_p"] < 1:
        order = np.argsort(-scores, axis=-1, kind="stable")
        sorted_scores = np.take_along_axis(scores, order, axis=-1)
        exceeded = np.cumsum(_probabilities(sorted_scores), axis=-1) > options["top_p"]
        remove = np.zeros_like(exceeded)
        remove[:, 1:] = exceeded[:, :-1]
        sorted_scores[remove] = -np.inf
        np.put_along_axis(scores, order, sorted_scores, axis=-1)
    probabilities = Tensor._from_array(_probabilities(scores), False)
    sampled = categorical(probabilities, generator=generator)
    return Tensor._from_array(sampled._data.reshape(batch, 1), False)


__all__ = ["sample_next_token"]
