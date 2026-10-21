"""Binary serialization and compression for federated model messages."""

import math
import struct
from collections.abc import Sequence

import numpy as np

TOP_K_FRACTION = 0.25
GATE_THRESHOLD = 0.0005

_MODEL_HEADER = struct.Struct("!4sI")
_ARRAY_HEADER = struct.Struct("!I")
_UPDATE_HEADER = struct.Struct("!4sBII f")
_MODEL_MAGIC = b"CFM1"
_UPDATE_MAGIC = b"CFU1"


def flatten_parameters(parameters: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate model tensors into one float32 vector."""
    if not parameters:
        raise ValueError("At least one parameter tensor is required.")
    return np.concatenate(
        [np.asarray(parameter, dtype=np.float32).reshape(-1) for parameter in parameters]
    )


def unflatten_parameters(
    values: np.ndarray, shapes: Sequence[tuple[int, ...]]
) -> list[np.ndarray]:
    """Restore a flattened vector to the supplied model tensor shapes."""
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    expected_size = sum(math.prod(shape) for shape in shapes)
    if flat.size != expected_size:
        raise ValueError(f"Expected {expected_size} values, got {flat.size}.")
    parameters = []
    offset = 0
    for shape in shapes:
        size = math.prod(shape)
        parameters.append(flat[offset : offset + size].reshape(shape).copy())
        offset += size
    return parameters


def encode_model(parameters: Sequence[np.ndarray]) -> bytes:
    """Serialize float32 model tensors in their supplied order."""
    if not parameters:
        raise ValueError("At least one parameter tensor is required.")
    chunks = [_MODEL_HEADER.pack(_MODEL_MAGIC, len(parameters))]
    for parameter in parameters:
        array = np.ascontiguousarray(parameter, dtype="<f4")
        raw = array.tobytes()
        chunks.append(_ARRAY_HEADER.pack(len(raw)))
        chunks.append(raw)
    return b"".join(chunks)


def decode_model(
    payload: bytes, expected_shapes: Sequence[tuple[int, ...]]
) -> list[np.ndarray]:
    """Deserialize and validate a model payload against the local model shape."""
    if len(payload) < _MODEL_HEADER.size:
        raise ValueError("Model payload is shorter than its header.")
    magic, count = _MODEL_HEADER.unpack_from(payload)
    if magic != _MODEL_MAGIC:
        raise ValueError("Invalid model payload magic.")
    if count != len(expected_shapes):
        raise ValueError(f"Expected {len(expected_shapes)} tensors, got {count}.")

    parameters = []
    offset = _MODEL_HEADER.size
    for shape in expected_shapes:
        if len(payload) - offset < _ARRAY_HEADER.size:
            raise ValueError("Model payload ended before an array header.")
        (byte_count,) = _ARRAY_HEADER.unpack_from(payload, offset)
        offset += _ARRAY_HEADER.size
        expected_bytes = math.prod(shape) * np.dtype("<f4").itemsize
        if byte_count != expected_bytes or len(payload) - offset < byte_count:
            raise ValueError("Model tensor payload has an invalid size.")
        array = np.frombuffer(payload, dtype="<f4", count=math.prod(shape), offset=offset)
        parameters.append(array.astype(np.float32, copy=True).reshape(shape))
        offset += byte_count
    if offset != len(payload):
        raise ValueError("Model payload has trailing bytes.")
    return parameters


def encode_update(
    update: np.ndarray,
    model_norm: float,
    top_k_fraction: float = TOP_K_FRACTION,
    gate_threshold: float = GATE_THRESHOLD,
) -> bytes:
    """Encode a model delta using relative-norm gating and top-k int8 values."""
    if not 0 < top_k_fraction <= 1:
        raise ValueError("top_k_fraction must be greater than 0 and at most 1.")
    if gate_threshold < 0:
        raise ValueError("gate_threshold must be non-negative.")

    values = np.asarray(update, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all() or not math.isfinite(model_norm):
        raise ValueError("Updates and model norm must be finite.")
    update_norm = float(np.linalg.norm(values))
    relative_norm = (
        update_norm / model_norm
        if model_norm > 0
        else (0.0 if update_norm == 0 else math.inf)
    )
    if relative_norm < gate_threshold or values.size == 0:
        return _UPDATE_HEADER.pack(_UPDATE_MAGIC, 1, values.size, 0, 0.0)

    count = min(values.size, max(1, math.ceil(values.size * top_k_fraction)))
    if count == values.size:
        indexes = np.arange(values.size)
    else:
        indexes = np.argpartition(np.abs(values), values.size - count)[-count:]
    indexes = np.sort(indexes)
    selected = values[indexes]
    maximum = float(np.max(np.abs(selected)))
    if maximum == 0:
        return _UPDATE_HEADER.pack(_UPDATE_MAGIC, 1, values.size, 0, 0.0)

    scale = maximum / 127.0
    quantized = np.clip(np.rint(selected / scale), -127, 127).astype(np.int8)
    header = _UPDATE_HEADER.pack(
        _UPDATE_MAGIC, 0, values.size, count, scale
    )
    return b"".join(
        (
            header,
            indexes.astype("<u4", copy=False).tobytes(),
            quantized.tobytes(),
        )
    )


def decode_update(payload: bytes, expected_size: int) -> tuple[np.ndarray, bool]:
    """Decode a compressed delta and report whether the client was gated."""
    if len(payload) < _UPDATE_HEADER.size:
        raise ValueError("Update payload is shorter than its header.")
    magic, gated, total_size, count, scale = _UPDATE_HEADER.unpack_from(payload)
    if magic != _UPDATE_MAGIC:
        raise ValueError("Invalid update payload magic.")
    if total_size != expected_size:
        raise ValueError(f"Expected {expected_size} update values, got {total_size}.")
    if gated not in (0, 1):
        raise ValueError("Invalid update gating flag.")
    if gated:
        if count != 0 or scale != 0 or len(payload) != _UPDATE_HEADER.size:
            raise ValueError("Gated update payload contains unexpected data.")
        return np.zeros(expected_size, dtype=np.float32), True
    if count == 0 or count > expected_size or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Compressed update metadata is invalid.")

    expected_bytes = _UPDATE_HEADER.size + count * (
        np.dtype("<u4").itemsize + np.dtype(np.int8).itemsize
    )
    if len(payload) != expected_bytes:
        raise ValueError("Compressed update payload has an invalid size.")
    indexes_start = _UPDATE_HEADER.size
    values_start = indexes_start + count * np.dtype("<u4").itemsize
    indexes = np.frombuffer(payload, dtype="<u4", count=count, offset=indexes_start)
    if np.any(indexes >= expected_size) or np.unique(indexes).size != count:
        raise ValueError("Compressed update indexes are invalid.")
    quantized = np.frombuffer(payload, dtype=np.int8, count=count, offset=values_start)
    decoded = np.zeros(expected_size, dtype=np.float32)
    decoded[indexes] = quantized.astype(np.float32) * scale
    return decoded, False
