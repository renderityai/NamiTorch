import json
import math
import operator
import os
import tempfile
from numbers import Real
from pathlib import Path


def integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        value = int(operator.index(value))
    except TypeError:
        raise TypeError(f"{name} must be an integer.") from None
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def real(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite.") from None
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}.")
    return value


def seed_value(value):
    value = integer(value, "seed")
    if value >= 2 ** 64:
        raise ValueError("seed must be smaller than 2**64.")
    return value


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Nonfinite JSON constant {value} is forbidden.")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_object, parse_constant=_constant)


def write_json(path, value):
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(canonical_json(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
