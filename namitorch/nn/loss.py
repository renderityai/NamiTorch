from ..tensor import Tensor
from . import functional as F
from .module import Module


class _Loss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = F._loss_reduction(reduction)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(reduction={self.reduction!r})"


class MSELoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)


class L1Loss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(_Loss):
    def __init__(self, reduction: str = "mean", beta: float = 1.0):
        super().__init__(reduction)
        self.beta = F._loss_scale(beta, "beta", allow_zero=True)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)

    def __repr__(self) -> str:
        return f"SmoothL1Loss(reduction={self.reduction!r}, beta={self.beta})"


class HuberLoss(_Loss):
    def __init__(self, reduction: str = "mean", delta: float = 1.0):
        super().__init__(reduction)
        self.delta = F._loss_scale(delta, "delta")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)

    def __repr__(self) -> str:
        return f"HuberLoss(reduction={self.reduction!r}, delta={self.delta})"


class BCELoss(_Loss):
    def __init__(self, weight: Tensor | None = None, reduction: str = "mean"):
        super().__init__(reduction)
        F._loss_weight(weight, "weight")
        self.register_buffer("weight", weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(_Loss):
    def __init__(self, weight: Tensor | None = None, reduction: str = "mean", pos_weight: Tensor | None = None):
        super().__init__(reduction)
        F._loss_weight(weight, "weight")
        F._loss_weight(pos_weight, "pos_weight")
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target, weight=self.weight, reduction=self.reduction, pos_weight=self.pos_weight)


class NLLLoss(_Loss):
    def __init__(self, weight: Tensor | None = None, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__(reduction)
        self.ignore_index = F._integer(ignore_index, "ignore_index")
        F._class_weight(weight)
        self.register_buffer("weight", weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

    def __repr__(self) -> str:
        return f"NLLLoss(ignore_index={self.ignore_index}, reduction={self.reduction!r})"


class CrossEntropyLoss(_Loss):
    def __init__(self, weight: Tensor | None = None, ignore_index: int = -100, reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__(reduction)
        self.ignore_index = F._integer(ignore_index, "ignore_index")
        self.label_smoothing = F._label_smoothing(label_smoothing)
        F._class_weight(weight)
        self.register_buffer("weight", weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=self.label_smoothing)

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(ignore_index={self.ignore_index}, reduction={self.reduction!r}, label_smoothing={self.label_smoothing})"


__all__ = ["MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss", "BCELoss", "BCEWithLogitsLoss", "NLLLoss", "CrossEntropyLoss"]
