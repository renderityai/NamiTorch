from contextlib import contextmanager

from ..backends._stream_scope import stream_device_scope
from .memory import _backend


class Stream:
    __slots__ = ("_backend", "_stream")

    def __init__(self, device=None, *, non_blocking=False):
        if type(non_blocking) is not bool:
            raise TypeError("non_blocking must be a Python bool.")
        self._backend = _backend(device)
        with self._backend.module.cuda.Device(self.device.index):
            self._stream = self._backend.module.cuda.Stream(non_blocking=non_blocking)

    @classmethod
    def _wrap(cls, backend, native):
        result = cls.__new__(cls)
        result._backend, result._stream = backend, native
        return result

    @property
    def device(self):
        return self._backend.device

    def synchronize(self):
        with self._backend.module.cuda.Device(self.device.index):
            self._stream.synchronize()
            self._backend.collect_transfers()

    def wait_event(self, event):
        if not isinstance(event, Event):
            raise TypeError("wait_event requires a NamiTorch CUDA Event.")
        event._require_recorded()
        if event.device != self.device:
            raise RuntimeError("Stream and event must belong to the same CUDA device.")
        with self._backend.module.cuda.Device(self.device.index):
            self._stream.wait_event(event._event)

    def __eq__(self, other):
        return isinstance(other, Stream) and self.device == other.device and self._stream.ptr == other._stream.ptr

    def __hash__(self):
        return hash((self.device, int(self._stream.ptr)))

    def __repr__(self):
        return f"Stream(device={str(self.device)!r}, id={int(self._stream.ptr)})"


def current_stream(device=None):
    backend = _backend(device)
    with backend.module.cuda.Device(backend.device.index):
        return Stream._wrap(backend, backend.module.cuda.get_current_stream())


def default_stream(device=None):
    backend = _backend(device)
    with backend.module.cuda.Device(backend.device.index):
        return Stream._wrap(backend, backend.module.cuda.Stream.null)


@contextmanager
def stream(value):
    if not isinstance(value, Stream):
        raise TypeError("cuda.stream requires a NamiTorch CUDA Stream.")
    with stream_device_scope(value.device), value._backend.module.cuda.Device(value.device.index), value._stream:
        try:
            yield value
        finally:
            value._backend.collect_transfers()


class Event:
    __slots__ = ("_enable_timing", "_blocking", "_backend", "_event", "_recorded")

    def __init__(self, enable_timing=False, blocking=False):
        if type(enable_timing) is not bool or type(blocking) is not bool:
            raise TypeError("Event enable_timing and blocking must be Python bools.")
        self._enable_timing = enable_timing
        self._blocking = blocking
        self._backend = self._event = None
        self._recorded = False

    @property
    def device(self):
        return None if self._backend is None else self._backend.device

    @property
    def enable_timing(self):
        return self._enable_timing

    @property
    def blocking(self):
        return self._blocking

    def _require_recorded(self):
        if not self._recorded:
            raise RuntimeError("CUDA Event has not been recorded.")

    def record(self, stream=None):
        selected = current_stream() if stream is None else stream
        if not isinstance(selected, Stream):
            raise TypeError("Event.record requires a NamiTorch CUDA Stream or None.")
        if self.device is not None and selected.device != self.device:
            raise RuntimeError("An event cannot be recorded on a different CUDA device.")
        backend = selected._backend
        with backend.module.cuda.Device(selected.device.index):
            if self._event is None:
                self._event = backend.module.cuda.Event(block=self.blocking, disable_timing=not self.enable_timing)
                self._backend = backend
            self._event.record(selected._stream)
            self._recorded = True
        return self

    def synchronize(self):
        if self._recorded:
            with self._backend.module.cuda.Device(self.device.index):
                self._event.synchronize()
                self._backend.collect_transfers()

    def query(self):
        if not self._recorded:
            return True
        with self._backend.module.cuda.Device(self.device.index):
            return bool(self._event.done)

    def __repr__(self):
        return f"Event(device={None if self.device is None else str(self.device)!r}, enable_timing={self.enable_timing}, blocking={self.blocking}, recorded={self._recorded})"


def elapsed_time(event_start, event_end):
    if not isinstance(event_start, Event) or not isinstance(event_end, Event):
        raise TypeError("elapsed_time requires two NamiTorch CUDA Events.")
    if not event_start.enable_timing or not event_end.enable_timing:
        raise ValueError("elapsed_time requires timing-enabled events.")
    event_start._require_recorded()
    event_end._require_recorded()
    if event_start.device != event_end.device:
        raise RuntimeError("Timing events must belong to the same CUDA device.")
    if not event_start.query() or not event_end.query():
        raise RuntimeError("Timing events are not complete; synchronize the end event first.")
    backend = event_start._backend
    with backend.module.cuda.Device(backend.device.index):
        return float(backend.module.cuda.get_elapsed_time(event_start._event, event_end._event))


__all__ = ["Stream", "Event", "current_stream", "default_stream", "stream", "elapsed_time"]
