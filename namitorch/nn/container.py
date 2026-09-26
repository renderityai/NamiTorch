from . import functional as F
from .module import Module


class ModuleList(Module):
    def __init__(self, modules=()):
        super().__init__()
        self.extend(modules)

    def _validate_module(self, module):
        if not isinstance(module, Module):
            raise TypeError("ModuleList entries must be Modules.")
        if any(child is self for child in module.modules()):
            raise ValueError("ModuleList registration would create a module cycle.")

    def _index(self, index):
        index = F._integer(index, "index")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("ModuleList index out of range.")
        return index

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ModuleList(tuple(self)[index])
        return self._modules[str(self._index(index))]

    def __setitem__(self, index, module):
        index = self._index(index)
        self._validate_module(module)
        self.add_module(str(index), module)

    def __delitem__(self, index):
        entries = list(self)
        del entries[index if isinstance(index, slice) else self._index(index)]
        self._modules.clear()
        self._modules.update((str(index), module) for index, module in enumerate(entries))

    def append(self, module):
        self._validate_module(module)
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        entries = list(modules)
        for module in entries:
            self._validate_module(module)
        for module in entries:
            self.append(module)
        return self

    def insert(self, index, module):
        index = F._integer(index, "index")
        self._validate_module(module)
        entries = list(self)
        entries.insert(index, module)
        self._modules.clear()
        self._modules.update((str(index), child) for index, child in enumerate(entries))
        return self

    def __repr__(self):
        entries = ", ".join(f"{index}: {module!r}" for index, module in enumerate(self))
        return f"ModuleList({entries})"


__all__ = ["ModuleList"]
