from threading import RLock

from ..autograd import no_grad
from ..backends._graph_capture import CaptureSession, array_signature, capture_region, reject_during_capture
from ..tensor import Tensor
from .memory import _backend
from .streams import Stream


def _supported(backend):
    native = backend.module.cuda.Stream
    return not getattr(backend.module.cuda.runtime, 'is_hip', False) and all(callable(getattr(native, name, None)) for name in ('begin_capture', 'end_capture'))


def is_available(device=None):
    try:
        return _supported(_backend(device))
    except (RuntimeError, AttributeError):
        return False


def _tensors(values, name):
    if isinstance(values, Tensor):
        return (values,)
    if not isinstance(values, (tuple, list)) or not values or not all(isinstance(value, Tensor) for value in values):
        raise TypeError(f'{name} must be a Tensor or a nonempty tuple/list of Tensors.')
    return tuple(values)


class CUDAGraph:
    def __init__(self, device=None):
        reject_during_capture('Constructing another CUDA graph')
        self._backend = _backend(device)
        if not _supported(self._backend):
            raise RuntimeError('CUDA graphs require a CUDA CuPy/runtime with Stream.begin_capture and Stream.end_capture support.')
        self._stream = Stream(self._backend.device, non_blocking=True)
        self._lock = RLock()
        self._graph = None
        self._session = None
        self._inputs = ()
        self._outputs = None
        self._output_signatures = ()

    @property
    def device(self):
        return self._backend.device

    @property
    def static_inputs(self):
        return self._inputs

    @property
    def outputs(self):
        self._require_captured()
        return self._outputs

    def _require_captured(self):
        if self._graph is None:
            raise RuntimeError('CUDA graph has not been captured or was closed.')

    def _validate_values(self, values):
        for value in values:
            if value.device != self.device:
                raise RuntimeError(f'CUDA graph inputs must already be on {self.device}; transfer them explicitly.')
            if value.dtype.is_floating_point and value.dtype.name != 'float32':
                raise TypeError('CUDA graph floating inputs must have dtype float32.')

    def _run(self, function, session):
        session.observe(self._inputs)
        result = function(*self._inputs)
        values = _tensors(result, 'CUDA graph output')
        self._validate_values(values)
        if any(value.requires_grad or value.grad_fn is not None for value in values):
            raise RuntimeError('CUDA graphs currently support inference only; do not enable autograd in the captured function.')
        session.observe(values)
        return result, tuple(array_signature(value._data) for value in values)

    def capture(self, function, inputs, *, warmup=3):
        reject_during_capture('Nested CUDA graph capture')
        if not callable(function):
            raise TypeError('CUDA graph capture requires a callable receiving static Tensor inputs.')
        if type(warmup) is not int or warmup < 2:
            raise ValueError('CUDA graph warmup must be an integer >= 2.')
        values = _tensors(inputs, 'CUDA graph inputs')
        self._validate_values(values)
        with self._lock:
            if self._graph is not None:
                raise RuntimeError('Close the existing CUDA graph before capturing another callable.')
            backend = self._backend
            native = self._stream._stream
            session = CaptureSession(backend, native)
            with backend.module.cuda.Device(self.device.index), no_grad():
                backend.module.cuda.runtime.deviceSynchronize()
                self._inputs = tuple(value.detach().clone() for value in values)
                backend.module.cuda.get_current_stream().synchronize()
                try:
                    with native, backend.memory.context():
                        for _ in range(warmup):
                            with capture_region(session):
                                self._run(function, session)
                            native.synchronize()
                            session.memories.clear()
                            session.resources.clear()
                            session.tensors.clear()
                        session.phase = 'prepare'
                        with capture_region(session):
                            prepared, signatures = self._run(function, session)
                        native.synchronize()
                        session.phase = 'capture'
                        with capture_region(session):
                            native.begin_capture()
                            try:
                                outputs, captured_signatures = self._run(function, session)
                                if session.cursor != len(session.allocations) or captured_signatures != signatures:
                                    raise RuntimeError('CUDA graph outputs or allocation sequence changed after warmup.')
                            except BaseException as error:
                                try:
                                    native.end_capture()
                                except Exception as cleanup_error:
                                    error.add_note(f'Ending failed CUDA capture: {cleanup_error}')
                                raise
                            graph = native.end_capture()
                        self._graph, self._session = graph, session
                        self._outputs, self._output_signatures = outputs, captured_signatures
                        del prepared
                    self.replay()
                    self._stream.synchronize()
                except BaseException as error:
                    try:
                        native.synchronize()
                    except Exception as cleanup_error:
                        error.add_note(f'Synchronizing failed CUDA capture: {cleanup_error}')
                    self._graph = self._session = self._outputs = None
                    self._inputs = ()
                    if isinstance(error, Exception):
                        raise RuntimeError('CUDA graph capture failed. Use deterministic FP32 inference with fixed allocations, no RNG, host reads, optimizer, backward or data-dependent CPU branching.') from error
                    raise
        return self

    def _validate_storage(self):
        self._require_captured()
        self._session.validate()
        signatures = tuple(array_signature(value._data) for value in _tensors(self._outputs, 'CUDA graph output'))
        if signatures != self._output_signatures:
            raise RuntimeError('CUDA graph output storage changed; recapture the graph.')

    def copy_inputs(self, values):
        reject_during_capture('Updating graph inputs')
        values = _tensors(values, 'New CUDA graph inputs')
        self._validate_values(values)
        with self._lock:
            self._validate_storage()
            if len(values) != len(self._inputs):
                raise ValueError('New CUDA graph inputs must match the captured input count.')
            for value, target in zip(values, self._inputs):
                if value.shape != target.shape or value.dtype is not target.dtype:
                    raise ValueError('New CUDA graph inputs must preserve captured shapes and dtypes.')
            self._wait_dependencies(values)
            with self._backend.module.cuda.Device(self.device.index), self._stream._stream, no_grad():
                for value, target in zip(values, self._inputs):
                    target.copy_(value)
            self._publish_completion()
        return self

    def _publish_completion(self):
        backend = self._backend
        with backend.module.cuda.Device(self.device.index):
            completion = backend.module.cuda.Event(disable_timing=True)
            completion.record(self._stream._stream)
            backend.module.cuda.get_current_stream().wait_event(completion)

    def _wait_dependencies(self, values=()):
        backend = self._backend
        memories = dict(self._session.memories)
        backend.execution._inputs(values, memories, set())
        with backend.module.cuda.Device(self.device.index):
            incoming = backend.module.cuda.Event(disable_timing=True)
            incoming.record(backend.module.cuda.get_current_stream())
            self._stream._stream.wait_event(incoming)
            backend.execution.wait_for_memories(memories, self._stream._stream)

    def replay(self, inputs=None):
        reject_during_capture('CUDA graph replay')
        with self._lock:
            self._validate_storage()
            if inputs is not None:
                self.copy_inputs(inputs)
            backend = self._backend
            self._wait_dependencies()
            with backend.module.cuda.Device(self.device.index), self._stream._stream:
                with backend.context(self._inputs, self._outputs):
                    backend.execution.hold((self._graph, self._session))
                    self._graph.launch(stream=self._stream._stream)
                    backend.result(self._outputs)
                counters = {id(value._version_counter): value._version_counter for value in _tensors(self._outputs, 'CUDA graph output')}
                for counter in counters.values():
                    counter.increment()
            self._publish_completion()
            return self._outputs

    def synchronize(self):
        reject_during_capture('CUDA graph synchronization')
        with self._lock:
            self._stream.synchronize()

    def close(self):
        reject_during_capture('Closing a CUDA graph')
        with self._lock:
            self._stream.synchronize()
            self._graph = self._session = self._outputs = None
            self._inputs = ()
            self._output_signatures = ()


__all__ = ['CUDAGraph', 'is_available']
