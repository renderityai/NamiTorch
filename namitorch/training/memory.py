from ..backends import array_device, is_array
from ..nn.module import Module
from ..optim.optimizer import Optimizer
from ..tensor import Tensor, _storage_owner


_CATEGORIES = ("parameter_bytes", "gradient_bytes", "buffer_bytes", "optimizer_state_bytes")


def _arrays(value, visited):
    if isinstance(value, Tensor):
        yield value._data
    elif is_array(value):
        yield value
    elif isinstance(value, (dict, list, tuple)) and id(value) not in visited:
        visited.add(id(value))
        for item in value.values() if isinstance(value, dict) else value:
            yield from _arrays(item, visited)


def _allocation(array):
    device = array_device(array)
    if device.type == "cuda":
        memory = array.data.mem
        return (str(device), int(memory.ptr)), int(memory.size)
    owner = _storage_owner(array)
    size = owner.nbytes if hasattr(owner, "nbytes") else memoryview(owner).nbytes
    return (str(device), id(owner)), int(size)


def memory_summary(model, optimizer=None):
    if not isinstance(model, Module):
        raise TypeError("memory_summary requires a NamiTorch Module.")
    if optimizer is not None and not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a NamiTorch Optimizer or None.")
    parameters = list(model.parameters())
    groups = (
        parameters, [parameter.grad for parameter in parameters if parameter.grad is not None],
        list(model.buffers()), {} if optimizer is None else optimizer.state,
    )
    totals = dict.fromkeys(_CATEGORIES, 0)
    devices = {}
    seen = set()
    for category, values in zip(_CATEGORIES, groups, strict=True):
        for array in _arrays(values, set()):
            identity, size = _allocation(array)
            if identity in seen:
                continue
            seen.add(identity)
            device = identity[0]
            counts = devices.setdefault(device, dict.fromkeys(_CATEGORIES, 0))
            counts[category] += size
            totals[category] += size
    for counts in devices.values():
        counts["total_bytes"] = sum(counts.values())
    totals["total_bytes"] = sum(totals.values())
    totals["cuda_storage_bytes"] = sum(counts["total_bytes"] for device, counts in devices.items() if device.startswith("cuda:"))
    totals["devices"] = devices
    return totals


__all__ = ["memory_summary"]
