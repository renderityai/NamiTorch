from contextlib import contextmanager
from threading import RLock

from ._graph_capture import current_capture
from ._stream_scope import validate_stream_device


class NamiTorchCUDAOutOfMemoryError(MemoryError):
    def __init__(self, device, requested_bytes, allocated_bytes, reserved_bytes, free_bytes, total_bytes):
        self.device = device
        self.requested_bytes = requested_bytes
        self.allocated_bytes = allocated_bytes
        self.reserved_bytes = reserved_bytes
        self.free_bytes = free_bytes
        self.total_bytes = total_bytes
        fields = (
            ("requested", requested_bytes), ("allocated", allocated_bytes),
            ("reserved", reserved_bytes), ("free", free_bytes), ("total", total_bytes),
        )
        details = ", ".join(f"{name}={value} bytes" if value is not None else f"{name}=unavailable" for name, value in fields)
        super().__init__(f"NamiTorch CUDA out of memory on {device}: {details}.")


class CUDAMemory:
    def __init__(self, module, device):
        self.module = module
        self.device = device
        self.pool = module.cuda.MemoryPool()
        self._lock = RLock()
        self.track_allocation = None
        self.collect_completed = None
        self._peak_allocated = 0
        self._peak_reserved = 0

    def _observe(self):
        allocated = int(self.pool.used_bytes())
        reserved = int(self.pool.total_bytes())
        self._peak_allocated = max(self._peak_allocated, allocated)
        self._peak_reserved = max(self._peak_reserved, reserved)
        return allocated, reserved

    def allocation_error(self, error, requested=None):
        requested = getattr(error, "_size", None) if requested is None else requested
        allocated = reserved = free = total = None
        try:
            allocated, reserved = self._observe()
            free, total = map(int, self.module.cuda.runtime.memGetInfo())
        except Exception:
            return NamiTorchCUDAOutOfMemoryError(self.device, requested, allocated, reserved, free, total)
        return NamiTorchCUDAOutOfMemoryError(self.device, requested, allocated, reserved, free, total)

    def allocate(self, size):
        validate_stream_device(self.device)
        with self.module.cuda.Device(self.device.index), self._lock:
            try:
                session = current_capture()
                pointer = self.pool.malloc(size) if session is None else session.allocate(self, size)
            except self.module.cuda.memory.OutOfMemoryError as error:
                raise self.allocation_error(error, int(size)) from error
            if current_capture() is None or current_capture().phase != "capture":
                self._observe()
            if self.track_allocation is not None:
                self.track_allocation(pointer)
            return pointer

    @contextmanager
    def context(self):
        validate_stream_device(self.device)
        with self.module.cuda.Device(self.device.index):
            try:
                with self.module.cuda.using_allocator(self.allocate):
                    yield
            except self.module.cuda.memory.OutOfMemoryError as error:
                raise self.allocation_error(error) from error

    def stats(self):
        if self.collect_completed is not None:
            self.collect_completed()
        with self.module.cuda.Device(self.device.index), self._lock:
            allocated, reserved = self._observe()
            return allocated, reserved, self._peak_allocated, self._peak_reserved

    def empty_cache(self):
        with self.module.cuda.Device(self.device.index), self._lock:
            self.pool.free_all_blocks()

    def reset_peaks(self):
        with self.module.cuda.Device(self.device.index), self._lock:
            self._peak_allocated = int(self.pool.used_bytes())
            self._peak_reserved = int(self.pool.total_bytes())

    def mem_get_info(self):
        with self.module.cuda.Device(self.device.index):
            return tuple(map(int, self.module.cuda.runtime.memGetInfo()))
