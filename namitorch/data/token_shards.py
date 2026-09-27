import hashlib
import json
import os
import sys
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path, PurePosixPath

import numpy as np

from ..random import Generator, _get_rng, _validate_generator
from ..tensor import Tensor
from ._utils import integer, normalize_index
from .dataset import Dataset, IterableDataset


TOKEN_CORPUS_FORMAT_VERSION = 1
_STORAGE_DTYPES = {"uint16": np.dtype("<u2"), "uint32": np.dtype("<u4")}


@dataclass(frozen=True)
class _Shard:
    path: Path
    token_count: int
    byte_size: int
    checksum: str | None


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest field {key!r}.")
        result[key] = value
    return result


def _manifest_integer(value, name):
    if type(value) is not int:
        raise TypeError(f"Manifest {name} must be an integer.")
    return integer(value, name, 0)


def _load_manifest(path):
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream, object_pairs_hook=_json_object)
    required = {"format_version", "tokenizer_fingerprint", "dtype", "split", "total_tokens", "shards"}
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise ValueError(f"Corpus manifest must contain {sorted(required)}.")
    version = _manifest_integer(manifest["format_version"], "format_version")
    if version != TOKEN_CORPUS_FORMAT_VERSION:
        raise ValueError(f"Unsupported corpus format_version {version}; expected 1.")
    for name in ("tokenizer_fingerprint", "split"):
        if not isinstance(manifest[name], str) or not manifest[name].strip():
            raise ValueError(f"Manifest {name} must be a nonempty string.")
    dtype = manifest["dtype"]
    if not isinstance(dtype, str) or dtype not in _STORAGE_DTYPES:
        raise ValueError("Corpus dtype must be 'uint16' or 'uint32' (little-endian).")
    total = _manifest_integer(manifest["total_tokens"], "total_tokens")
    if not isinstance(manifest["shards"], list):
        raise TypeError("Manifest shards must be a list.")
    shards = []
    paths = set()
    for index, entry in enumerate(manifest["shards"]):
        if not isinstance(entry, dict) or not {"path", "token_count", "byte_size"} <= entry.keys():
            raise ValueError(f"Shard {index} requires path, token_count and byte_size.")
        name = entry["path"]
        if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
            raise ValueError(f"Shard {index} path must be a relative POSIX file path.")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Shard {index} path must remain inside the manifest directory.")
        shard_path = (path.parent / relative).resolve()
        if not shard_path.is_relative_to(path.parent) or not shard_path.is_file():
            raise ValueError(f"Shard {index} path must reference a file inside the manifest directory: {name!r}.")
        if shard_path in paths:
            raise ValueError(f"Duplicate shard path {name!r}.")
        paths.add(shard_path)
        count = _manifest_integer(entry["token_count"], f"shards[{index}].token_count")
        size = _manifest_integer(entry["byte_size"], f"shards[{index}].byte_size")
        if size != count * _STORAGE_DTYPES[dtype].itemsize or shard_path.stat().st_size != size:
            raise ValueError(f"Shard {name!r} byte_size does not match token_count or file size.")
        checksum = entry.get("checksum")
        if "checksum" in entry:
            if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdefABCDEF" for c in checksum):
                raise ValueError(f"Shard {name!r} checksum must be a 64-character SHA-256 hex digest.")
            checksum = checksum.lower()
        shards.append(_Shard(shard_path, count, size, checksum))
    if sum(shard.token_count for shard in shards) != total:
        raise ValueError("Manifest total_tokens must equal the sum of shard token_count values.")
    return manifest, tuple(shards), _STORAGE_DTYPES[dtype]


class TokenShardDataset(Dataset):
    def __init__(self, manifest_path, context_length):
        self._context_length = integer(context_length, "context_length", 1)
        self._manifest_path = Path(manifest_path).resolve()
        self._manifest, shards, self._storage_dtype = _load_manifest(self._manifest_path)
        self._shards = tuple(shard for shard in shards if shard.token_count > self._context_length)
        self._valid_positions = tuple(shard.token_count - self._context_length for shard in self._shards)
        self._cumulative_positions = tuple(accumulate(self._valid_positions))
        if self._cumulative_positions and self._cumulative_positions[-1] > min(sys.maxsize, np.iinfo(np.int64).max):
            raise ValueError("Corpus has too many valid positions for this platform.")
        self._maps = {}

    @property
    def context_length(self):
        return self._context_length

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    @property
    def valid_positions(self):
        return self._valid_positions

    def __len__(self):
        return self._cumulative_positions[-1] if self._cumulative_positions else 0

    def _map(self, shard_index):
        if shard_index not in self._maps:
            shard = self._shards[shard_index]
            with shard.path.open("rb") as stream:
                if os.fstat(stream.fileno()).st_size != shard.byte_size:
                    raise ValueError(f"Shard {shard.path.name!r} file size changed since manifest loading.")
                if shard.checksum is not None:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                    if digest.hexdigest() != shard.checksum:
                        raise ValueError(f"SHA-256 checksum mismatch for shard {shard.path.name!r}.")
                    stream.seek(0)
                self._maps[shard_index] = np.memmap(stream, dtype=self._storage_dtype, mode="r", shape=(shard.token_count,))
        return self._maps[shard_index]

    def _window(self, position):
        shard_index = bisect_right(self._cumulative_positions, position)
        start = 0 if shard_index == 0 else self._cumulative_positions[shard_index - 1]
        offset = position - start
        return self._map(shard_index)[offset:offset + self._context_length + 1]

    def __getitem__(self, index):
        window = self._window(normalize_index(index, len(self)))
        inputs = np.array(window[:-1], dtype=np.int64, copy=True)
        targets = np.array(window[1:], dtype=np.int64, copy=True)
        return Tensor._from_array(inputs, False), Tensor._from_array(targets, False)

    def close(self):
        for mapping in self._maps.values():
            mapping._mmap.close()
        self._maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class TokenShardBatcher(IterableDataset):
    def __init__(self, dataset, batch_size, *, generator: Generator | None = None, num_batches=None):
        if not isinstance(dataset, TokenShardDataset):
            raise TypeError("TokenShardBatcher requires a TokenShardDataset.")
        self.dataset = dataset
        self.batch_size = integer(batch_size, "batch_size", 1)
        _validate_generator(generator, "cpu")
        self.generator = generator
        self.num_batches = None if num_batches is None else integer(num_batches, "num_batches", 0)
        if not len(dataset):
            raise ValueError("Random batches require at least one shard with context_length + 1 tokens.")

    def sample_batch(self):
        positions = _get_rng(self.generator).integers(0, len(self.dataset), size=self.batch_size)
        shape = (self.batch_size, self.dataset.context_length)
        inputs = np.empty(shape, dtype=np.int64)
        targets = np.empty(shape, dtype=np.int64)
        for row, position in enumerate(positions):
            window = self.dataset._window(int(position))
            inputs[row] = window[:-1]
            targets[row] = window[1:]
        return Tensor._from_array(inputs, False), Tensor._from_array(targets, False)

    def __iter__(self):
        if self.num_batches is None:
            while True:
                yield self.sample_batch()
        else:
            for _ in range(self.num_batches):
                yield self.sample_batch()

    def __len__(self):
        if self.num_batches is None:
            raise TypeError("An unbounded TokenShardBatcher has no length.")
        return self.num_batches
