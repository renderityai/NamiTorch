from ..tensor import Tensor
from . import functional as F
from .module import Module


class _Activation(Module):
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ReLU(_Activation):
    def __init__(self, inplace: bool = False):
        super().__init__()
        F._validate_inplace(inplace)
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def __repr__(self) -> str:
        return f"ReLU(inplace={self.inplace})"


class LeakyReLU(_Activation):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        F._validate_inplace(inplace)
        self.negative_slope = F._real_parameter(negative_slope, "negative_slope")
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)

    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope}, inplace={self.inplace})"


class ELU(_Activation):
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        F._validate_inplace(inplace)
        self.alpha = F._real_parameter(alpha, "alpha")
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.elu(input, alpha=self.alpha, inplace=self.inplace)

    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha}, inplace={self.inplace})"


class GELU(_Activation):
    def __init__(self, approximation: str = "exact"):
        super().__init__()
        self.approximation = F._gelu_approximation(approximation)

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximation=self.approximation)

    def __repr__(self) -> str:
        return f"GELU(approximation={self.approximation!r})"


class SiLU(_Activation):
    def __init__(self, inplace: bool = False):
        super().__init__()
        F._validate_inplace(inplace)
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input, inplace=self.inplace)

    def __repr__(self) -> str:
        return f"SiLU(inplace={self.inplace})"


class Sigmoid(_Activation):
    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)


class Tanh(_Activation):
    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)


class Softplus(_Activation):
    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input)


class Softmax(_Activation):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = F._integer(dim, "dim")

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, dim=self.dim)

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class LogSoftmax(_Activation):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = F._integer(dim, "dim")

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, dim=self.dim)

    def __repr__(self) -> str:
        return f"LogSoftmax(dim={self.dim})"


__all__ = ["ReLU", "LeakyReLU", "ELU", "GELU", "SiLU", "Sigmoid", "Tanh", "Softplus", "Softmax", "LogSoftmax"]
