from ..tensor import Tensor


class Parameter(Tensor):
    __slots__ = ()

    def __init__(self, data: object, requires_grad: bool | None = None, *, dtype: object = None):
        super().__init__(data, dtype=dtype)
        self.requires_grad = self.dtype.can_require_grad if requires_grad is None else requires_grad

    def __repr__(self) -> str:
        return f"Parameter({super().__repr__()})"


__all__ = ["Parameter"]
