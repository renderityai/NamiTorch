from ..backends import ensure_same_device, namespace, readonly, writable
from ..autograd import is_grad_enabled
from ..creation import empty
from ..dtype import get_default_dtype, normalize_dtype
from ..ops import cat
from ..tensor import Tensor
from . import functional as F


class LayerKVCache:
    __slots__ = ("_batch_size", "_n_kv_heads", "_head_dim", "_max_seq_len", "_dtype", "_length", "_k", "_v")

    def __init__(self, batch_size: int, n_kv_heads: int, head_dim: int, *, max_seq_len: int | None = None, dtype=None, device=None):
        self._batch_size = F._integer(batch_size, "batch_size")
        if self._batch_size < 0:
            raise ValueError("batch_size must be nonnegative.")
        self._n_kv_heads = F._positive_dimension(n_kv_heads, "n_kv_heads")
        self._head_dim = F._positive_dimension(head_dim, "head_dim")
        self._max_seq_len = None if max_seq_len is None else F._positive_dimension(max_seq_len, "max_seq_len")
        self._dtype = get_default_dtype() if dtype is None else normalize_dtype(dtype)
        if not self._dtype.is_floating_point:
            raise TypeError("LayerKVCache requires a floating dtype.")
        self._length = 0
        capacity = 0 if max_seq_len is None else self._max_seq_len
        self._k = empty(self.batch_size, self.n_kv_heads, capacity, self.head_dim, dtype=self.dtype, device=device)
        self._v = empty(self.batch_size, self.n_kv_heads, capacity, self.head_dim, dtype=self.dtype, device=device)

    @property
    def device(self):
        return self._k.device

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_kv_heads(self):
        return self._n_kv_heads

    @property
    def head_dim(self):
        return self._head_dim

    @property
    def dtype(self):
        return self._dtype

    @property
    def length(self):
        return self._length

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @property
    def capacity(self):
        return self._k.shape[2]

    @property
    def k(self):
        return self.get()[0]

    @property
    def v(self):
        return self.get()[1]

    def validate(self, batch_size: int, n_kv_heads: int, head_dim: int, dtype=None, append_length: int = 0) -> None:
        expected = (self.batch_size, self.n_kv_heads, self.head_dim)
        if (batch_size, n_kv_heads, head_dim) != expected:
            raise ValueError(f"Cache expects batch/heads/head_dim {expected}, got {(batch_size, n_kv_heads, head_dim)}.")
        if dtype is not None and normalize_dtype(dtype) != self.dtype:
            raise TypeError(f"Cache dtype {self.dtype} does not match projected dtype {dtype}.")
        append_length = F._integer(append_length, "append_length")
        if append_length < 0:
            raise ValueError("append_length must be nonnegative.")
        if self.max_seq_len is not None and self.length + append_length > self.max_seq_len:
            raise ValueError(f"Cache capacity {self.max_seq_len} exceeded by requested length {self.length + append_length}.")

    def append(self, k: Tensor, v: Tensor):
        ensure_same_device(self._k, self._v, k, v)
        if is_grad_enabled():
            raise RuntimeError("LayerKVCache.append is inference-only; use no_grad().")
        for name, tensor in (("key", k), ("value", v)):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Cache {name} must be a NamiTorch Tensor.")
            if tensor.ndim != 4:
                raise ValueError(f"Cache {name} must have rank 4, got {tensor.shape}.")
            self.validate(tensor.shape[0], tensor.shape[1], tensor.shape[3], tensor.dtype, tensor.shape[2])
        if k.shape != v.shape:
            raise ValueError("New cache key and value must have identical shapes.")
        if k.shape[2] == 0:
            return self
        end = self.length + k.shape[2]
        if self.max_seq_len is None:
            new_k = cat((self.k, k.detach()), dim=2)
            new_v = cat((self.v, v.detach()), dim=2)
            self._k, self._v = new_k, new_v
        else:
            if not writable(self._k._data) or not writable(self._v._data):
                raise RuntimeError("Cache storage must be writable.")
            if any(namespace(source).shares_memory(source._data, target._data) for source in (k, v) for target in (self._k, self._v)):
                k, v = k.detach().clone(), v.detach().clone()
            self._k[:, :, self.length:end].copy_(k)
            self._v[:, :, self.length:end].copy_(v)
        self._length = end
        return self

    def get(self) -> tuple[Tensor, Tensor]:
        result = []
        for tensor in (self._k, self._v):
            array = tensor._data[:, :, :self.length].view()
            readonly(array)
            view = Tensor._from_array(array, False, tensor._version_counter)
            view._writable = False
            result.append(view)
        return result[0], result[1]

    def reset(self) -> None:
        self._length = 0

    def __repr__(self) -> str:
        return (
            f"LayerKVCache(batch_size={self.batch_size}, n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
            f"length={self.length}, max_seq_len={self.max_seq_len}, dtype={self.dtype})"
        )


__all__ = ["LayerKVCache"]
