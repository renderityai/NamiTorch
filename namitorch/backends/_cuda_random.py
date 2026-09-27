from math import prod
from secrets import randbits
from threading import RLock

import numpy as np


_SEED_MASK = 2 ** 64 - 1
_ALGORITHM = "cupy-philox-splitmix64-v1"


def _substream_seed(seed, counter):
    value = (seed + (counter + 1) * 0x9E3779B97F4A7C15) & _SEED_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _SEED_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _SEED_MASK
    return value ^ (value >> 31)


class CUDARandom:
    def __init__(self, backend, seed):
        self.backend = backend
        self.device = backend.device
        self._lock = RLock()
        self._seed = randbits(64) if seed is None else seed
        self._counter = 0
        with backend.context():
            self._rng = backend.module.random.RandomState(
                self._seed, method=backend.module.cuda.curand.CURAND_RNG_PSEUDO_PHILOX4_32_10,
            )

    def manual_seed(self, seed):
        with self._lock:
            self._seed = seed
            self._counter = 0

    def get_state(self):
        with self._lock:
            return {
                "version": 1,
                "algorithm": _ALGORITHM,
                "device": str(self.device),
                "cupy_version": self.backend.module.__version__,
                "seed": self._seed,
                "counter": self._counter,
            }

    def set_state(self, state):
        if not isinstance(state, dict):
            raise TypeError("Generator state must be a dictionary.")
        if set(state) != {"version", "algorithm", "device", "cupy_version", "seed", "counter"}:
            raise ValueError("Invalid CUDA generator state fields.")
        if type(state["version"]) is not int or state["version"] != 1 or state["algorithm"] != _ALGORITHM:
            raise ValueError("Unsupported CUDA generator state version or algorithm.")
        if state["device"] != str(self.device):
            raise ValueError(f"CUDA generator state belongs to {state['device']!r}, expected {str(self.device)!r}.")
        if state["cupy_version"] != self.backend.module.__version__:
            raise ValueError("CUDA generator state requires the same CuPy version for exact replay.")
        for name, maximum in (("seed", _SEED_MASK), ("counter", 2 ** 64)):
            if type(state[name]) is not int or not 0 <= state[name] <= maximum:
                raise ValueError(f"Invalid CUDA generator state field {name!r}.")
        with self._lock:
            self._seed = state["seed"]
            self._counter = state["counter"]

    def draw(self, operation, shape, dtype, **options):
        if operation not in ("permutation", "choice", "integers", "random", "standard_normal"):
            raise ValueError(f"Unsupported CUDA random operation {operation!r}.")
        dtype = np.float64 if dtype is None else dtype
        with self._lock, self.backend.context():
            if self._counter == 2 ** 64:
                raise RuntimeError("CUDA generator substreams are exhausted; reseed the generator.")
            module = self.backend.module
            self._rng.seed(_substream_seed(self._seed, self._counter))
            try:
                dimensions = (shape,) if isinstance(shape, int) else shape
                if prod(dimensions) == 0:
                    result = module.empty(dimensions, dtype=dtype)
                elif operation == "permutation":
                    result = self._rng.permutation(shape).astype(dtype, copy=False)
                elif operation == "choice":
                    if options["replace"]:
                        result = self._rng.randint(0, options["n"], size=shape, dtype=dtype)
                    else:
                        result = self._rng.permutation(options["n"])[:prod(dimensions)].reshape(shape).astype(dtype, copy=False)
                elif operation == "integers":
                    result = self._rng.randint(options["low"], options["high"], size=shape, dtype=dtype)
                elif operation == "random":
                    result = self._rng.random_sample(shape, dtype=dtype)
                else:
                    result = self._rng.standard_normal(shape, dtype=dtype)
            finally:
                module.cuda.get_current_stream().synchronize()
            self._counter += 1
            return result
