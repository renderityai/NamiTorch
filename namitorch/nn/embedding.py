from ..autograd import no_grad
from ..creation import empty
from ..dtype import get_default_dtype, normalize_dtype
from ..random import Generator
from ..tensor import Tensor
from . import functional as F
from . import init
from .module import Module
from .parameter import Parameter


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None, *, dtype=None, generator: Generator | None = None):
        super().__init__()
        self.num_embeddings = F._positive_dimension(num_embeddings, "num_embeddings")
        self.embedding_dim = F._positive_dimension(embedding_dim, "embedding_dim")
        self.padding_idx = F._padding_index(padding_idx, self.num_embeddings)
        dtype = get_default_dtype() if dtype is None else normalize_dtype(dtype)
        if not dtype.is_floating_point:
            raise TypeError("Embedding parameters require a floating dtype.")
        self.weight = Parameter(empty(self.num_embeddings, self.embedding_dim, dtype=dtype))
        self.reset_parameters(generator=generator)

    def reset_parameters(self, *, generator: Generator | None = None) -> None:
        init.normal_(self.weight, generator=generator)
        self.reset_padding()

    def reset_padding(self) -> None:
        if self.padding_idx is not None:
            with no_grad():
                self.weight[self.padding_idx].zero_()

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(input, self.weight, self.padding_idx)

    def __repr__(self) -> str:
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx})"


__all__ = ["Embedding"]
