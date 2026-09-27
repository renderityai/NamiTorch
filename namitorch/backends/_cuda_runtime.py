from importlib import import_module


def load_cupy():
    try:
        return import_module("cupy")
    except Exception as error:
        raise RuntimeError("CUDA backend is unavailable: install an optional CuPy build compatible with your CUDA driver/runtime. CPU operations remain available.") from error


def runtime_call(name, *args):
    module = load_cupy()
    try:
        return getattr(module.cuda.runtime, name)(*args)
    except Exception as error:
        raise RuntimeError(f"CUDA backend is unavailable or the runtime request {name} failed: {error}") from error
