import codecs
import operator
import os
import re
from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a bool.")
    try:
        result = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer.") from None
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(result)


def _replace_pair(sequence, pair, token_id):
    result = []
    position = 0
    while position < len(sequence):
        if position + 1 < len(sequence) and (sequence[position], sequence[position + 1]) == pair:
            result.append(token_id)
            position += 2
        else:
            result.append(sequence[position])
            position += 1
    return result


def _utf8_prefix(data):
    return codecs.getincrementaldecoder("utf-8")("strict").decode(data, final=False)


def _file_documents(paths, max_bytes):
    remaining = max_bytes
    for path in paths:
        if remaining == 0:
            return
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("File paths must be strings or path-like objects.")
        with open(path, "rb") as stream:
            while remaining is None or remaining > 0:
                data = stream.readline(-1 if remaining is None else remaining)
                if not data:
                    break
                if remaining is not None:
                    remaining -= len(data)
                yield _utf8_prefix(data) if remaining == 0 else data.decode("utf-8", errors="strict")
        if remaining == 0:
            return


class ByteBPETokenizer:
    def __init__(self, special_tokens=()):
        self._special_tokens = {}
        self._special_tokens_by_id = {}
        self._special_token_roles = {}
        self._vocabulary = self._base_vocabulary()
        self._merges = {}
        self._merge_ranks = {}
        self._training_bytes = 0
        self._training_documents = 0
        self.add_special_tokens(special_tokens)

    def add_special_tokens(self, special_tokens):
        if isinstance(special_tokens, (str, bytes)):
            raise TypeError("special_tokens must be an ordered iterable of strings or a role-to-string mapping.")
        if isinstance(special_tokens, (set, frozenset)):
            raise TypeError("special_tokens must have a deterministic order, such as a list or tuple.")
        roles = dict(special_tokens) if isinstance(special_tokens, Mapping) else {}
        tokens = list(roles.values()) if isinstance(special_tokens, Mapping) else list(special_tokens)
        unique = {}
        for token in tokens:
            if not isinstance(token, str) or not token:
                raise ValueError("Special tokens must be nonempty strings.")
            if token in unique and not isinstance(special_tokens, Mapping):
                raise ValueError(f"Duplicate special token {token!r}.")
            token.encode("utf-8", errors="strict")
            unique[token] = None
        for role, token in roles.items():
            if not isinstance(role, str) or not role.strip():
                raise ValueError("Special token roles must be nonempty strings.")
            role.encode("utf-8", errors="strict")
            if role in self._special_token_roles and self._special_token_roles[role] != token:
                raise ValueError(f"Special token role {role!r} is already configured.")
        added = 0
        for token in unique:
            if token not in self._special_tokens:
                token_id = self.vocab_size
                self._special_tokens[token] = token_id
                self._special_tokens_by_id[token_id] = token
                added += 1
        self._special_token_roles.update(roles)
        return added

    def _base_vocabulary(self):
        return {index: bytes((index,)) for index in range(256)}

    @property
    def vocab_size(self):
        return len(self._vocabulary) + len(self._special_tokens)

    @property
    def bpe_vocab_size(self):
        return len(self._vocabulary)

    @property
    def vocabulary(self):
        return MappingProxyType(self._vocabulary)

    @property
    def special_tokens(self):
        return MappingProxyType(self._special_tokens)

    @property
    def special_tokens_by_id(self):
        return MappingProxyType(self._special_tokens_by_id)

    @property
    def special_token_roles(self):
        return MappingProxyType(self._special_token_roles)

    def _role_id(self, role):
        return self._special_tokens.get(self._special_token_roles.get(role))

    @property
    def bos_token_id(self):
        return self._role_id("bos")

    @property
    def eos_token_id(self):
        return self._role_id("eos")

    @property
    def pad_token_id(self):
        return self._role_id("pad")

    @property
    def merges(self):
        return MappingProxyType(self._merges)

    @property
    def merge_ranks(self):
        return MappingProxyType(self._merge_ranks)

    @property
    def training_bytes(self):
        return self._training_bytes

    @property
    def training_documents(self):
        return self._training_documents

    def _special_selection(self, value, name):
        if isinstance(value, str):
            if value != "all":
                raise TypeError(f"{name} must be an iterable of token strings or 'all'.")
            selected = set(self._special_tokens)
        else:
            selected = set(value)
        if any(not isinstance(token, str) or token not in self._special_tokens for token in selected):
            raise ValueError(f"{name} contains an unregistered special token.")
        return selected

    def _special_policy(self, allowed_special, disallowed_special):
        allowed = self._special_selection(allowed_special, "allowed_special")
        disallowed = self._special_selection(disallowed_special, "disallowed_special") - allowed
        alternatives = sorted(allowed | disallowed, key=lambda token: (-len(token), token))
        pattern = re.compile("|".join(re.escape(token) for token in alternatives)) if alternatives else None
        return pattern, allowed

    def _segments(self, text, pattern, allowed):
        if pattern is None:
            yield False, text
            return
        position = 0
        for match in pattern.finditer(text):
            token = match.group()
            if token not in allowed:
                raise ValueError(f"Disallowed special token {token!r} at character {match.start()}.")
            yield False, text[position:match.start()]
            yield True, token
            position = match.end()
        yield False, text[position:]

    def train(self, texts, target_vocab_size: int, *, min_frequency: int = 2, max_training_bytes: int | None = None, allowed_special=(), disallowed_special=()):
        target_vocab_size = _integer(target_vocab_size, "target_vocab_size", 256 + len(self._special_tokens))
        min_frequency = _integer(min_frequency, "min_frequency", 1)
        budget = None if max_training_bytes is None else _integer(max_training_bytes, "max_training_bytes")
        if isinstance(texts, (str, bytes)):
            raise TypeError("Training input must be an iterable of strings; wrap a single document in a list.")
        pattern, allowed = self._special_policy(allowed_special, disallowed_special)
        sequences, training_bytes = [], 0
        documents = iter(texts)
        while budget is None or budget > 0:
            try:
                text = next(documents)
            except StopIteration:
                break
            if not isinstance(text, str):
                raise TypeError("Every training document must be a string.")
            data = (text if budget is None else text[:budget]).encode("utf-8", errors="strict")
            if budget is not None:
                consumed = min(len(data), budget)
                text = _utf8_prefix(data[:consumed])
                budget -= consumed
            training_bytes += len(text.encode("utf-8", errors="strict"))
            sequence = []
            for special, segment in self._segments(text, pattern, allowed):
                if special:
                    sequence.append(None)
                else:
                    sequence.extend(segment.encode("utf-8", errors="strict"))
            sequences.append(sequence)
        vocabulary, merges, ranks = self._base_vocabulary(), {}, {}
        while len(vocabulary) + len(self._special_tokens) < target_vocab_size:
            counts = Counter()
            for sequence in sequences:
                counts.update(
                    (left, right) for left, right in zip(sequence, sequence[1:])
                    if left is not None and right is not None
                )
            eligible = (pair for pair, count in counts.items() if count >= min_frequency)
            pair = min(eligible, key=lambda candidate: (-counts[candidate], candidate), default=None)
            if pair is None:
                break
            token_id = len(vocabulary)
            vocabulary[token_id] = vocabulary[pair[0]] + vocabulary[pair[1]]
            merges[pair] = token_id
            ranks[pair] = len(ranks)
            sequences = [_replace_pair(sequence, pair, token_id) for sequence in sequences]
        self._vocabulary, self._merges, self._merge_ranks = vocabulary, merges, ranks
        self._special_tokens = {token: len(vocabulary) + index for index, token in enumerate(self._special_tokens)}
        self._special_tokens_by_id = {token_id: token for token, token_id in self._special_tokens.items()}
        self._training_bytes, self._training_documents = training_bytes, len(sequences)
        return self

    def train_from_files(self, paths, target_vocab_size: int, max_bytes: int | None = None, *, min_frequency: int = 2, allowed_special=(), disallowed_special=()):
        max_bytes = None if max_bytes is None else _integer(max_bytes, "max_bytes")
        if isinstance(paths, (str, os.PathLike)):
            paths = (paths,)
        documents = _file_documents(paths, max_bytes)
        try:
            return self.train(documents, target_vocab_size, min_frequency=min_frequency, allowed_special=allowed_special, disallowed_special=disallowed_special)
        finally:
            documents.close()

    def _encode_bytes(self, data):
        sequence = list(data)
        while len(sequence) > 1:
            pair = min(
                ((left, right) for left, right in zip(sequence, sequence[1:]) if (left, right) in self._merge_ranks),
                key=self._merge_ranks.__getitem__, default=None,
            )
            if pair is None:
                break
            sequence = _replace_pair(sequence, pair, self._merges[pair])
        return sequence

    def encode(self, text: str, *, allowed_special=(), disallowed_special=(), add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("encode expects a string.")
        if type(add_bos) is not bool or type(add_eos) is not bool:
            raise TypeError("add_bos and add_eos must be Python bools.")
        if add_bos and self.bos_token_id is None:
            raise ValueError("add_bos requires a configured 'bos' special token role.")
        if add_eos and self.eos_token_id is None:
            raise ValueError("add_eos requires a configured 'eos' special token role.")
        pattern, allowed = self._special_policy(allowed_special, disallowed_special)
        sequence = [self.bos_token_id] if add_bos else []
        for special, segment in self._segments(text, pattern, allowed):
            if special:
                sequence.append(self._special_tokens[segment])
            else:
                sequence.extend(self._encode_bytes(segment.encode("utf-8", errors="strict")))
        if add_eos:
            sequence.append(self.eos_token_id)
        return sequence

    def _validated_ids(self, token_ids):
        for value in token_ids:
            token_id = _integer(value, "token_id")
            if token_id >= self.vocab_size:
                raise ValueError(f"Unknown token ID {token_id}; vocabulary size is {self.vocab_size}.")
            yield token_id

    def decode_bytes(self, token_ids, *, skip_special_tokens: bool = False) -> bytes:
        if type(skip_special_tokens) is not bool:
            raise TypeError("skip_special_tokens must be a Python bool.")
        pieces = []
        for token_id in self._validated_ids(token_ids):
            if token_id in self._special_tokens_by_id:
                if not skip_special_tokens:
                    raise ValueError("decode_bytes requires skip_special_tokens=True for special IDs; use decode for literal special strings.")
            else:
                pieces.append(self._vocabulary[token_id])
        return b"".join(pieces)

    def decode(self, token_ids, *, skip_special_tokens: bool = False) -> str:
        if type(skip_special_tokens) is not bool:
            raise TypeError("skip_special_tokens must be a Python bool.")
        pieces, buffer = [], bytearray()
        for token_id in self._validated_ids(token_ids):
            if token_id in self._special_tokens_by_id:
                if not skip_special_tokens:
                    pieces.append(buffer.decode("utf-8", errors="strict"))
                    buffer.clear()
                    pieces.append(self._special_tokens_by_id[token_id])
            else:
                buffer.extend(self._vocabulary[token_id])
        pieces.append(buffer.decode("utf-8", errors="strict"))
        return "".join(pieces)

    @property
    def fingerprint(self):
        from ._serialization import tokenizer_fingerprint

        return tokenizer_fingerprint(self)

    def save(self, path) -> None:
        from ._serialization import save_tokenizer

        save_tokenizer(self, path)

    @classmethod
    def load(cls, path):
        from ._serialization import load_tokenizer

        return load_tokenizer(cls, path)

    def __repr__(self):
        return f"ByteBPETokenizer(vocab_size={self.vocab_size}, merges={len(self._merges)}, special_tokens={tuple(self._special_tokens)})"


__all__ = ["ByteBPETokenizer"]
