import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from ..data import TOKEN_CORPUS_FORMAT_VERSION
from ..tokenization import ByteBPETokenizer
from ._io import integer, real, seed_value, write_json


def discover_text_files(inputs):
    if isinstance(inputs, (str, os.PathLike)):
        inputs = (inputs,)
    files = set()
    for value in inputs:
        path = Path(value).resolve()
        if path.is_dir():
            files.update(candidate.resolve() for candidate in path.rglob("*") if candidate.is_file() and candidate.suffix.lower() == ".txt")
        elif path.is_file() and path.suffix.lower() == ".txt":
            files.add(path)
        else:
            raise ValueError(f"Input must be a .txt file or directory: {path}.")
    if not files:
        raise ValueError("No .txt input files found.")
    return sorted(files, key=lambda path: path.as_posix())


def document_split(document: bytes, seed: int, val_ratio: float) -> str:
    if not isinstance(document, bytes):
        raise TypeError("document_split expects document bytes.")
    seed = seed_value(seed)
    val_ratio = real(val_ratio, "val_ratio")
    if val_ratio > 1:
        raise ValueError("val_ratio must be in [0, 1].")
    if val_ratio == 1:
        return "val"
    digest = hashlib.sha256(document + seed.to_bytes(8, "big")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2 ** 64
    return "val" if fraction < val_ratio else "train"


def _documents(path, mode, separator):
    if mode == "file":
        yield path.read_bytes()
    elif mode == "line":
        with path.open("rb") as stream:
            yield from stream
    else:
        with path.open("rb") as stream:
            pending = b""
            for chunk in iter(lambda: stream.read(65536), b""):
                parts = (pending + chunk).split(separator)
                yield from parts[:-1]
                pending = parts[-1]
            yield pending


class _ShardWriter:
    def __init__(self, directory, split, dtype, shard_tokens):
        self.directory, self.split, self.dtype = directory, split, dtype
        self.buffer = np.empty(shard_tokens, dtype=dtype)
        self.used, self.total_tokens = 0, 0
        self.shards = []

    def append(self, tokens):
        offset = 0
        while offset < len(tokens):
            count = min(len(tokens) - offset, len(self.buffer) - self.used)
            self.buffer[self.used:self.used + count] = tokens[offset:offset + count]
            self.used += count
            offset += count
            if self.used == len(self.buffer):
                self.flush()

    def flush(self):
        if not self.used:
            return
        name = f"{self.split}-{len(self.shards):06d}.bin"
        data = self.buffer[:self.used].tobytes(order="C")
        with (self.directory / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.shards.append({"path": name, "token_count": self.used, "byte_size": len(data), "checksum": hashlib.sha256(data).hexdigest()})
        self.total_tokens += self.used
        self.used = 0


def prepare_corpus(inputs, tokenizer, output_dir, *, shard_tokens=1000000, val_ratio=0.01, seed=1337, document_mode="line", separator=None, dtype="auto", append_eos=True, allowed_special=()):
    if not isinstance(tokenizer, ByteBPETokenizer):
        tokenizer = ByteBPETokenizer.load(tokenizer)
    shard_tokens = integer(shard_tokens, "shard_tokens", 1)
    seed = seed_value(seed)
    val_ratio = real(val_ratio, "val_ratio")
    if val_ratio > 1:
        raise ValueError("val_ratio must be in [0, 1].")
    if document_mode not in ("file", "line", "separator"):
        raise ValueError("document_mode must be file, line or separator.")
    if document_mode == "separator":
        if not isinstance(separator, str) or not separator:
            raise ValueError("separator mode requires a nonempty separator string.")
        separator_bytes = separator.encode("utf-8")
    else:
        if separator is not None:
            raise ValueError("separator is only valid in separator mode.")
        separator_bytes = None
    if type(append_eos) is not bool:
        raise TypeError("append_eos must be a Python bool.")
    allowed = sorted(tokenizer._special_selection(allowed_special, "allowed_special"))
    dtype = ("uint16" if tokenizer.vocab_size <= 2 ** 16 else "uint32") if dtype == "auto" else dtype
    if dtype not in ("uint16", "uint32"):
        raise ValueError("dtype must be auto, uint16 or uint32.")
    storage_dtype = np.dtype("<u2" if dtype == "uint16" else "<u4")
    if tokenizer.vocab_size - 1 > np.iinfo(storage_dtype).max:
        raise ValueError(f"Tokenizer vocabulary does not fit {dtype}.")
    files = discover_text_files(inputs)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Corpus output directory already exists: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        writers = {split: _ShardWriter(temporary, split, storage_dtype, shard_tokens) for split in ("train", "val")}
        counts = {"train": 0, "val": 0}
        for path in files:
            for document in _documents(path, document_mode, separator_bytes):
                if not document:
                    continue
                split = document_split(document, seed, val_ratio)
                tokens = tokenizer.encode(document.decode("utf-8", errors="strict"), allowed_special=allowed, add_eos=append_eos and tokenizer.eos_token_id is not None)
                writers[split].append(tokens)
                counts[split] += 1
        preparation = dict(seed=seed, val_ratio=val_ratio, document_mode=document_mode, separator=separator, allowed_special=allowed, eos_token_id=tokenizer.eos_token_id if append_eos else None)
        fingerprint = tokenizer.fingerprint
        for split, writer in writers.items():
            writer.flush()
            manifest = {
                "format_version": TOKEN_CORPUS_FORMAT_VERSION, "tokenizer_fingerprint": fingerprint,
                "vocab_size": tokenizer.vocab_size, "dtype": dtype, "split": split,
                "total_tokens": writer.total_tokens, "documents": counts[split], "shards": writer.shards,
                "preparation": preparation,
            }
            write_json(temporary / f"{split}.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {split: destination / f"{split}.json" for split in ("train", "val")}


__all__ = ["discover_text_files", "document_split", "prepare_corpus"]
