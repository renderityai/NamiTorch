from numbers import Integral

from ..device import Device
from ._cuda_runtime import load_cupy
from ._graph_capture import reject_during_capture


def _device(value):
    target = Device('cuda', value) if isinstance(value, Integral) else Device(value)
    if target.type != 'cuda':
        raise ValueError('Peer access requires CUDA devices.')
    return target


def _runtime_call(module, name, *args):
    function = getattr(module.cuda.runtime, name, None)
    if not callable(function):
        raise RuntimeError(f'The installed CuPy/runtime does not support {name}.')
    try:
        return function(*args)
    except Exception as error:
        raise RuntimeError(f'CUDA peer request {name} failed: {error}') from error


def _pair(src, dst):
    source, destination = _device(src), _device(dst)
    module = load_cupy()
    count = int(_runtime_call(module, 'getDeviceCount'))
    if source.index >= count or destination.index >= count:
        raise RuntimeError(f'Peer access requires existing CUDA devices; found {count}, requested {source} and {destination}.')
    return module, source, destination


def can_access_peer(src, dst):
    reject_during_capture('Querying CUDA peer access')
    module, source, destination = _pair(src, dst)
    return source != destination and bool(_runtime_call(module, 'deviceCanAccessPeer', source.index, destination.index))


def _set_peer_access(src, dst, enabled):
    reject_during_capture('Changing CUDA peer access')
    module, source, destination = _pair(src, dst)
    if source == destination:
        raise ValueError('Peer access requires two different CUDA devices.')
    if enabled and not _runtime_call(module, 'deviceCanAccessPeer', source.index, destination.index):
        raise RuntimeError(f'CUDA peer access from {source} to {destination} is unavailable.')
    return set_peer_access(module, source, destination, enabled)


def set_peer_access(module, source, destination, enabled):
    name = 'deviceEnablePeerAccess' if enabled else 'deviceDisablePeerAccess'
    function = getattr(module.cuda.runtime, name, None)
    if not callable(function):
        raise RuntimeError(f'The installed CuPy/runtime does not support {name}.')
    with module.cuda.Device(source.index):
        try:
            function(destination.index)
        except Exception as error:
            expected = getattr(module.cuda.runtime, 'errorPeerAccessAlreadyEnabled' if enabled else 'errorPeerAccessNotEnabled', 704 if enabled else 705)
            if getattr(error, 'status', None) != expected:
                raise RuntimeError(f'CUDA peer access from {source} to {destination} failed: {error}') from error
            _runtime_call(module, 'getLastError')
            return False
    return True


def enable_peer_access(src, dst):
    return _set_peer_access(src, dst, True)


def disable_peer_access(src, dst):
    return _set_peer_access(src, dst, False)
