import operator
import re
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True, slots=True, init=False)
class Device:
    type: str
    index: int | None

    def __init__(self, value="cpu", index=None):
        if isinstance(value, Device):
            if index is not None:
                raise ValueError("A Device cannot be combined with an index override.")
            kind, ordinal = value.type, value.index
        elif isinstance(value, str):
            match = re.fullmatch(r"(cpu|cuda)(?::(0|[1-9][0-9]*))?", value)
            if match is None:
                raise ValueError(f"Unsupported device {value!r}; expected cpu, cuda or cuda:N.")
            kind, embedded = match.groups()
            if embedded is not None and index is not None:
                raise ValueError("Specify the device index only once.")
            ordinal = int(embedded) if embedded is not None else index
            if kind == "cpu":
                if ordinal is not None:
                    raise ValueError("CPU devices do not have an index.")
            else:
                if ordinal is None:
                    ordinal = 0
                if isinstance(ordinal, bool) or not isinstance(ordinal, Integral):
                    raise TypeError("Device index must be a nonnegative integer.")
                try:
                    ordinal = operator.index(ordinal)
                except TypeError:
                    raise TypeError("Device index must be a nonnegative integer.") from None
                if ordinal < 0:
                    raise ValueError("Device index must be nonnegative.")
        else:
            raise TypeError("Device expects a device string or another Device.")
        object.__setattr__(self, "type", kind)
        object.__setattr__(self, "index", ordinal)

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __repr__(self):
        return f"Device({str(self)!r})"


def device(value="cpu", index=None):
    return Device(value, index)


__all__ = ["Device", "device"]
