from contextlib import contextmanager
from contextvars import ContextVar


_current = ContextVar('namitorch_cuda_graph_region', default=None)


def current_capture():
    return _current.get()


def reject_during_capture(operation):
    if current_capture() is not None:
        raise RuntimeError(f'{operation} is not supported in CUDA graph preparation or capture; only deterministic FP32 inference is supported.')


@contextmanager
def capture_region(session):
    if current_capture() is not None:
        raise RuntimeError('Nested CUDA graph capture is not supported.')
    token = _current.set(session)
    try:
        yield
    finally:
        _current.reset(token)


def array_signature(array):
    return (tuple(array.shape), array.dtype.str, tuple(array.strides), int(array.device.id), int(array.data.ptr))


class CaptureSession:
    def __init__(self, backend, stream):
        self.backend = backend
        self.stream = stream
        self.phase = 'warmup'
        self.allocations = []
        self.cursor = 0
        self.memories = {}
        self.resources = []
        self.tensors = {}

    def observe(self, value):
        if isinstance(value, (tuple, list, dict)):
            for item in value.values() if isinstance(value, dict) else value:
                self.observe(item)
            return
        array = getattr(value, '_data', value)
        if not isinstance(array, self.backend.array_type):
            return
        if array.device.id != self.backend.device.index:
            raise RuntimeError('CUDA graph operands must belong to its capture device.')
        if array.dtype.kind == 'f' and array.dtype.name != 'float32':
            raise RuntimeError('CUDA graphs currently support only FP32 floating storage.')
        if array is not value:
            self.tensors[id(value)] = (value, array_signature(array))
        memory = array.data.mem
        self.memories[(int(memory.ptr), int(memory.size))] = memory

    def allocate(self, memory, size):
        if memory.device != self.backend.device:
            raise RuntimeError('CUDA graph allocations cannot span devices.')
        if self.phase == 'capture':
            if self.cursor >= len(self.allocations) or self.allocations[self.cursor][0] != size:
                raise RuntimeError('CUDA graph allocation sequence changed after warmup; dynamic allocations are forbidden.')
            pointer = self.allocations[self.cursor][1]
            self.cursor += 1
            return pointer
        pointer = memory.pool.malloc(size)
        if self.phase == 'prepare':
            self.allocations.append((size, pointer))
        return pointer

    def validate(self):
        for tensor, signature in self.tensors.values():
            if array_signature(tensor._data) != signature:
                raise RuntimeError('A captured Tensor changed shape, dtype, device, strides or storage address; recapture the graph.')
