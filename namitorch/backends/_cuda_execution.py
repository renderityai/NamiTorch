from contextlib import contextmanager
import sys
from threading import RLock, local


_executions = {}


def synchronize_array(array):
    execution = _executions.get(int(array.device.id))
    if execution is not None:
        execution.synchronize_array(array)


class CUDAExecution:
    def __init__(self, module, device):
        self.module = module
        self.device = device
        self._local = local()
        self._lock = RLock()
        self._pending = []
        self._storage_events = {}
        _executions[device.index] = self

    def _key(self, memory):
        return int(memory.ptr), int(memory.size)

    def allocated(self, pointer):
        frame = getattr(self._local, "frame", None)
        if frame is not None:
            frame[0][self._key(pointer.mem)] = pointer.mem

    def hold(self, resource):
        frame = getattr(self._local, "frame", None)
        if frame is None:
            raise RuntimeError("CUDA resources must be retained inside an execution context.")
        frame[1].append(resource)

    def result(self, value):
        frame = getattr(self._local, "frame", None)
        if frame is not None:
            self._inputs(value, frame[0], set())
        return value

    def _inputs(self, value, memories, visited):
        if isinstance(value, self.module.ndarray):
            if value.device.id == self.device.index:
                memories[self._key(value.data.mem)] = value.data.mem
        elif isinstance(value, (tuple, list, dict)):
            if id(value) in visited:
                return
            visited.add(id(value))
            for item in value.values() if isinstance(value, dict) else value:
                self._inputs(item, memories, visited)
        else:
            array = getattr(value, "_data", None)
            if isinstance(array, self.module.ndarray):
                self._inputs(array, memories, visited)

    @contextmanager
    def context(self, operands):
        previous = getattr(self._local, "frame", None)
        frame = ({}, [])
        self._inputs(operands, frame[0], set())
        stream = self.module.cuda.get_current_stream()
        self._local.frame = frame
        try:
            yield
        finally:
            self._local.frame = previous
            if frame[0] or frame[1]:
                operation_error = sys.exc_info()[1]
                try:
                    event = self.module.cuda.Event(disable_timing=True)
                    event.record(stream)
                except Exception as error:
                    self._retain(stream, stream, frame)
                    if operation_error is None:
                        raise
                    operation_error.add_note(f"CUDA event recording also failed: {error}")
                else:
                    self._retain(event, stream, frame)

    def _retain(self, completion, stream, frame):
        with self._lock:
            self._pending.append((completion, stream, frame))
            for key in frame[0]:
                self._storage_events.setdefault(key, {})[int(stream.ptr)] = completion

    def collect(self):
        with self._lock:
            pending = []
            for event, stream, frame in self._pending:
                if not event.done:
                    pending.append((event, stream, frame))
                    continue
                for key in frame[0]:
                    events = self._storage_events.get(key)
                    if events is not None and events.get(int(stream.ptr)) is event:
                        del events[int(stream.ptr)]
                        if not events:
                            del self._storage_events[key]
            self._pending = pending

    def synchronize_array(self, array):
        with self.module.cuda.Device(self.device.index):
            with self._lock:
                events = tuple(self._storage_events.get(self._key(array.data.mem), {}).values())
            for event in events:
                event.synchronize()
            self.collect()
