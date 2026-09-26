import base64
import binascii
import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path


FORMAT_VERSION = 1


def _canonical(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _fingerprint(payload):
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _payload(tokenizer):
    return {
        "format_version": FORMAT_VERSION,
        "vocab": [base64.b64encode(tokenizer.vocabulary[index]).decode("ascii") for index in range(tokenizer.bpe_vocab_size)],
        "merges": [[*pair, tokenizer.merges[pair]] for pair in sorted(tokenizer.merges, key=tokenizer.merge_ranks.__getitem__)],
        "special_tokens": dict(tokenizer.special_tokens),
        "config": {
            "encoding": "utf-8",
            "special_token_roles": dict(tokenizer.special_token_roles),
            "training_bytes": tokenizer.training_bytes,
            "training_documents": tokenizer.training_documents,
        },
    }


def tokenizer_fingerprint(tokenizer):
    return _fingerprint(_payload(tokenizer))


def save_tokenizer(tokenizer, path):
    payload = _payload(tokenizer)
    payload["fingerprint"] = _fingerprint(payload)
    encoded = _canonical(payload) + b"\n"
    destination = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate tokenizer JSON field {name!r}.")
        result[name] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Tokenizer JSON cannot contain {value}.")


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Tokenizer {name} must be a nonnegative JSON integer.")
    return value


def load_tokenizer(tokenizer_class, path):
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    fields = {"format_version", "vocab", "merges", "special_tokens", "config", "fingerprint"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError(f"Tokenizer JSON must contain exactly {sorted(fields)}.")
    if type(payload["format_version"]) is not int or payload["format_version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported tokenizer format_version {payload['format_version']!r}.")
    fingerprint = payload.pop("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
        raise ValueError("Tokenizer fingerprint must be a lowercase SHA256 hex digest.")
    if not hmac.compare_digest(fingerprint, _fingerprint(payload)):
        raise ValueError("Tokenizer fingerprint mismatch.")
    raw_vocab = payload["vocab"]
    if not isinstance(raw_vocab, list) or len(raw_vocab) < 256:
        raise ValueError("Tokenizer vocab must be a contiguous list with at least 256 byte tokens.")
    vocabulary = {}
    for token_id, encoded in enumerate(raw_vocab):
        if not isinstance(encoded, str):
            raise ValueError(f"Vocab entry {token_id} must be a base64 string.")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError(f"Invalid base64 at vocab entry {token_id}.") from None
        if not data or base64.b64encode(data).decode("ascii") != encoded:
            raise ValueError(f"Vocab entry {token_id} must contain nonempty canonical base64 bytes.")
        if token_id < 256 and data != bytes((token_id,)):
            raise ValueError(f"Base byte token {token_id} has incorrect bytes.")
        vocabulary[token_id] = data
    raw_merges = payload["merges"]
    if not isinstance(raw_merges, list) or len(raw_merges) != len(vocabulary) - 256:
        raise ValueError("Merge count must match the number of vocabulary entries after the 256 base bytes.")
    merges, ranks = {}, {}
    for rank, entry in enumerate(raw_merges):
        if not isinstance(entry, list) or len(entry) != 3:
            raise ValueError("Every merge must be [left, right, new_id].")
        left, right, new_id = (_integer(value, "merge ID") for value in entry)
        if new_id != 256 + rank:
            raise ValueError("Merge child IDs must be contiguous in rank order, starting at 256.")
        if left >= new_id or right >= new_id:
            raise ValueError("Both merge parents must exist before the child.")
        pair = left, right
        if pair in merges:
            raise ValueError(f"Duplicate merge pair {pair}.")
        if vocabulary[new_id] != vocabulary[left] + vocabulary[right]:
            raise ValueError(f"Vocab bytes for merge {new_id} do not match parent concatenation.")
        merges[pair], ranks[pair] = new_id, rank
    special = payload["special_tokens"]
    if not isinstance(special, dict):
        raise ValueError("special_tokens must map strings to IDs.")
    for token, token_id in special.items():
        if not isinstance(token, str) or not token:
            raise ValueError("Special token strings must be nonempty.")
        token.encode("utf-8", errors="strict")
        _integer(token_id, "special token ID")
    if sorted(special.values()) != list(range(len(vocabulary), len(vocabulary) + len(special))):
        raise ValueError("Special token IDs must be unique and contiguous after the BPE vocabulary.")
    config = payload["config"]
    if not isinstance(config, dict) or set(config) != {"encoding", "special_token_roles", "training_bytes", "training_documents"}:
        raise ValueError("Tokenizer config has invalid fields.")
    if config["encoding"] != "utf-8":
        raise ValueError("Tokenizer encoding must be utf-8.")
    roles = config["special_token_roles"]
    if not isinstance(roles, dict) or any(not isinstance(role, str) or not role.strip() or not isinstance(token, str) or token not in special for role, token in roles.items()):
        raise ValueError("Special token roles must reference registered special strings.")
    training_bytes = _integer(config["training_bytes"], "training_bytes")
    training_documents = _integer(config["training_documents"], "training_documents")
    tokenizer = tokenizer_class()
    tokenizer._vocabulary, tokenizer._merges, tokenizer._merge_ranks = vocabulary, merges, ranks
    tokenizer.add_special_tokens(sorted(special, key=special.__getitem__))
    tokenizer.add_special_tokens(roles)
    tokenizer._training_bytes, tokenizer._training_documents = training_bytes, training_documents
    return tokenizer
