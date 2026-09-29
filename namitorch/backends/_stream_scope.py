from contextlib import contextmanager
from contextvars import ContextVar


_explicit_stream_device = ContextVar('namitorch_explicit_stream_device', default=None)


def validate_stream_device(device):
    active = _explicit_stream_device.get()
    if active is not None and device.type == 'cuda' and device != active:
        raise RuntimeError(f'The active CUDA stream belongs to {active}, but the operation targets {device}; select a stream on the target device.')


@contextmanager
def stream_device_scope(device):
    token = _explicit_stream_device.set(device)
    try:
        yield
    finally:
        _explicit_stream_device.reset(token)
