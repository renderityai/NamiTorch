from numbers import Integral


def integer(value: object, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def normalize_index(index: object, size: int) -> int:
    value = integer(index, "index")
    if not -size <= value < size:
        raise IndexError(f"Dataset index {value} is out of range for length {size}.")
    return value + size if value < 0 else value


def boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a Python bool.")
    return value
