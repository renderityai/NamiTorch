import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .dtype import from_numpy_dtype
from .tensor import Tensor


FORMAT_VERSION = 1
_MAX_DEPTH = 128
_MAX_METADATA_BYTES = 16 * 1024 * 1024


class SerializationError(ValueError):
    __slots__ = ()


def _dtype(value):
    if not isinstance(value, str):
        raise SerializationError("Array dtype must be a string.")
    dtype = np.dtype(value)
    sizes = {"b": (1,), "i": (1, 2, 4, 8), "u": (1, 2, 4, 8), "f": (2, 4, 8)}
    if dtype.kind not in sizes or dtype.itemsize not in sizes[dtype.kind] or dtype.fields is not None or dtype.subdtype is not None:
        raise SerializationError(f"Unsupported archive array dtype {value!r}; object and complex arrays are forbidden.")
    if dtype.str != value:
        raise SerializationError("Array dtype must use its canonical NumPy dtype string.")
    return dtype


def _encode(value, arrays, active, depth=0):
    if depth > _MAX_DEPTH:
        raise SerializationError(f"Archive nesting exceeds {_MAX_DEPTH} levels.")
    if value is None:
        return {"type": "none"}
    if type(value) in (bool, int, str):
        return {"type": type(value).__name__, "value": value}
    if type(value) is float:
        return {"type": "float", "value": value.hex()}
    if isinstance(value, (Tensor, np.ndarray)):
        array = value._data if isinstance(value, Tensor) else value
        if array.dtype.metadata is not None:
            raise SerializationError("Array dtypes with custom metadata are unsupported.")
        _dtype(array.dtype.str)
        identifier = f"a{len(arrays)}"
        arrays.append((identifier, array))
        node = {"type": "tensor" if isinstance(value, Tensor) else "ndarray", "id": identifier}
        if isinstance(value, Tensor):
            node["requires_grad"] = value.requires_grad
        return node
    if not isinstance(value, (list, tuple, dict)):
        raise TypeError(f"Cannot serialize {type(value).__name__}; only primitive containers, Tensor and ndarray are supported.")
    identity = id(value)
    if identity in active:
        raise SerializationError("Cannot serialize cyclic containers.")
    active.add(identity)
    try:
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise TypeError("Archive dictionaries require string keys.")
            items = [[key, _encode(item, arrays, active, depth + 1)] for key, item in value.items()]
            return {"type": "dict", "items": items}
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode(item, arrays, active, depth + 1) for item in value],
        }
    finally:
        active.remove(identity)


def save(obj, path) -> None:
    arrays = []
    tree = _encode(obj, arrays, set())
    metadata = {
        "format_version": FORMAT_VERSION,
        "tree": tree,
        "arrays": [
            {"id": identifier, "dtype": array.dtype.str, "shape": list(array.shape), "nbytes": array.nbytes}
            for identifier, array in arrays
        ],
    }
    encoded = json.dumps(metadata, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_METADATA_BYTES:
        raise SerializationError("Archive metadata exceeds the 16 MiB format limit.")
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                archive.writestr("metadata.json", encoded)
                with archive.open("arrays.npz", "w", force_zip64=True) as payload:
                    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as npz:
                        for identifier, array in arrays:
                            snapshot = np.array(array, copy=True, order="C")
                            with npz.open(f"{identifier}.npy", "w", force_zip64=True) as entry:
                                np.lib.format.write_array(entry, snapshot, version=(1, 0), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SerializationError(f"Duplicate JSON field {key!r}.")
        result[key] = value
    return result


def _constant(value):
    raise SerializationError(f"Nonstandard JSON constant {value!r} is forbidden.")


def _fields(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise SerializationError(f"Invalid {name} fields; expected {sorted(expected)}.")


def _entries(archive, expected):
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise SerializationError("Archive contains duplicate, missing or unexpected entries.")
    for entry in entries:
        if entry.compress_type != zipfile.ZIP_STORED or entry.flag_bits & 1:
            raise SerializationError("Archive entries must be uncompressed and unencrypted.")
    return {entry.filename: entry for entry in entries}


def _array_records(records, max_array_bytes):
    if type(records) is not list:
        raise SerializationError("Array records must be a list.")
    result = {}
    total = 0
    for index, record in enumerate(records):
        _fields(record, {"id", "dtype", "shape", "nbytes"}, "array record")
        if record["id"] != f"a{index}":
            raise SerializationError("Array IDs must be unique consecutive identifiers.")
        dtype = _dtype(record["dtype"])
        shape = record["shape"]
        if type(shape) is not list or any(type(size) is not int or not 0 <= size <= np.iinfo(np.intp).max for size in shape):
            raise SerializationError("Array shape must contain nonnegative platform-sized integers.")
        nbytes = math.prod(shape) * dtype.itemsize
        if type(record["nbytes"]) is not int or record["nbytes"] != nbytes or nbytes > np.iinfo(np.intp).max:
            raise SerializationError("Array nbytes does not match its shape and dtype or exceeds the platform limit.")
        total += nbytes
        if max_array_bytes is not None and total > max_array_bytes:
            raise SerializationError("Archive arrays exceed max_array_bytes.")
        result[record["id"]] = (dtype, tuple(shape), nbytes)
    return result


def _read_array(npz, identifier, record):
    dtype, shape, nbytes = record
    entry = npz.getinfo(f"{identifier}.npy")
    with npz.open(entry) as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise SerializationError("Only NPY version 1.0 is supported by this archive format.")
        actual_shape, fortran_order, actual_dtype = np.lib.format.read_array_header_1_0(stream)
        if actual_shape != shape or actual_dtype != dtype or fortran_order:
            raise SerializationError(f"NPY header disagrees with metadata for {identifier!r}.")
        if entry.file_size - stream.tell() != nbytes:
            raise SerializationError(f"Invalid array payload size for {identifier!r}.")
        stream.seek(0)
        array = np.lib.format.read_array(stream, allow_pickle=False)
    return array


def _decode(node, npz, records, used, depth=0):
    if depth > _MAX_DEPTH:
        raise SerializationError(f"Archive nesting exceeds {_MAX_DEPTH} levels.")
    if type(node) is not dict or type(node.get("type")) is not str:
        raise SerializationError("Every tree node must have a string type tag.")
    kind = node["type"]
    if kind == "none":
        _fields(node, {"type"}, "none node")
        return None
    if kind in ("bool", "int", "str", "float"):
        _fields(node, {"type", "value"}, "scalar node")
        value = node["value"]
        if type(value) is not {"bool": bool, "int": int, "str": str, "float": str}[kind]:
            raise SerializationError(f"Invalid scalar value for type {kind!r}.")
        if kind == "float":
            converted = float.fromhex(value)
            if converted.hex() != value:
                raise SerializationError("Float values must use canonical hexadecimal strings.")
            return converted
        return value
    if kind in ("list", "tuple", "dict"):
        _fields(node, {"type", "items"}, "container node")
        if type(node["items"]) is not list:
            raise SerializationError("Container items must be a list.")
        if kind == "dict":
            result = {}
            for pair in node["items"]:
                if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or pair[0] in result:
                    raise SerializationError("Dictionary entries must have unique string keys.")
                result[pair[0]] = _decode(pair[1], npz, records, used, depth + 1)
            return result
        values = [_decode(item, npz, records, used, depth + 1) for item in node["items"]]
        return tuple(values) if kind == "tuple" else values
    if kind in ("tensor", "ndarray"):
        expected = {"type", "id", "requires_grad"} if kind == "tensor" else {"type", "id"}
        _fields(node, expected, "array node")
        identifier = node["id"]
        if type(identifier) is not str or identifier not in records or identifier in used:
            raise SerializationError("Array reference must name a unique existing array ID.")
        used.add(identifier)
        if kind == "tensor":
            dtype = from_numpy_dtype(records[identifier][0])
            if type(node["requires_grad"]) is not bool or (node["requires_grad"] and not dtype.can_require_grad):
                raise SerializationError("Invalid requires_grad flag for serialized Tensor dtype.")
        array = _read_array(npz, identifier, records[identifier])
        if kind == "tensor":
            return Tensor._from_array(array.astype(dtype.numpy_dtype, copy=False), node["requires_grad"])
        return array
    raise SerializationError(f"Unknown archive tree node type {kind!r}.")


def load(path, *, max_array_bytes=None):
    if max_array_bytes is not None and (type(max_array_bytes) is not int or max_array_bytes < 0):
        raise TypeError("max_array_bytes must be a nonnegative Python integer or None.")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            entries = _entries(archive, {"metadata.json", "arrays.npz"})
            if entries["metadata.json"].file_size > _MAX_METADATA_BYTES:
                raise SerializationError("Archive metadata exceeds the 16 MiB format limit.")
            metadata = json.loads(archive.read("metadata.json"), object_pairs_hook=_object, parse_constant=_constant)
            _fields(metadata, {"format_version", "tree", "arrays"}, "metadata")
            if type(metadata["format_version"]) is not int or metadata["format_version"] != FORMAT_VERSION:
                raise SerializationError("Unsupported archive format_version.")
            records = _array_records(metadata["arrays"], max_array_bytes)
            with archive.open("arrays.npz") as payload:
                with zipfile.ZipFile(payload, "r") as npz:
                    _entries(npz, {f"{identifier}.npy" for identifier in records})
                    used = set()
                    result = _decode(metadata["tree"], npz, records, used)
                    if used != set(records):
                        raise SerializationError("Archive contains unreferenced arrays.")
                    return result
    except SerializationError:
        raise
    except (zipfile.BadZipFile, ValueError, TypeError, KeyError, EOFError, OverflowError, RecursionError) as error:
        raise SerializationError(f"Invalid NamiTorch archive: {error}") from error


__all__ = ["FORMAT_VERSION", "SerializationError", "save", "load"]
