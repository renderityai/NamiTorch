from ..tensor import Tensor
from ..random import Generator, _validate_generator
from . import functional as F
from .module import Module


class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False, *, generator: Generator | None = None):
        super().__init__()
        F._validate_inplace(inplace)
        self.p = F._dropout_probability(p)
        self.inplace = inplace
        _validate_generator(generator)
        self.generator = generator

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, p=self.p, training=self.training, inplace=self.inplace, generator=self.generator)

    def __repr__(self) -> str:
        generator = "" if self.generator is None else f", generator={self.generator!r}"
        return f"Dropout(p={self.p}, inplace={self.inplace}{generator})"


__all__ = ["Dropout"]
