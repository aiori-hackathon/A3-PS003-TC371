"""Binary serialization and compression for federated model messages."""

import math
import struct
from collections.abc import Sequence

import numpy as np

TOP_K_FRACTION = 0.25
GATE_THRESHOLD = 0.0005

MODEL_HEADER = struct.Struct("!4sI")
ARRAY_HEADER = struct.Struct("!I")
UPDATE_HEADER = struct.Struct("!4sBII f")
MODEL_MAGIC = b"CFM1"
UPDATE_MAGIC = b"CFU1"

# `flags` bit 0 = gated; bits 1-3 = compression method code.
# 0: dense/fedavg, 1: quantized, 2: top-k sparse, 3: coflow (top-k + quantized).
METHOD_CODES = {
    "fedavg": 0,
    "quantized_fedavg": 1,
    "topk_fedavg": 2,
    "gated_fedavg": 0,
    "coflow_fl": 3,
}


def _method_code(method: str) -> int:
    normalized = method.lower()
    try:
        return METHOD_CODES[normalized]
    except KeyError as exc:  # pragma: no cover - defensive validation
        raise ValueError(f"Unsupported update method: {method!r}") from exc


def _compose_flags(method: str, gated: bool) -> int:
    return ((_method_code(method) << 1) | int(gated))


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
    chunks = [MODEL_HEADER.pack(MODEL_MAGIC, len(parameters))]
    for parameter in parameters:
        array = np.ascontiguousarray(parameter, dtype="<f4")
        raw = array.tobytes()
        chunks.append(ARRAY_HEADER.pack(len(raw)))
        chunks.append(raw)
    return b"".join(chunks)


def decode_model(
    payload: bytes, expected_shapes: Sequence[tuple[int, ...]]
) -> list[np.ndarray]:
    """Deserialize and validate a model payload against the local model shape."""
    if len(payload) < MODEL_HEADER.size:
        raise ValueError("Model payload is shorter than its header.")
    magic, count = MODEL_HEADER.unpack_from(payload)
    if magic != MODEL_MAGIC:
        raise ValueError("Invalid model payload magic.")
    if count != len(expected_shapes):
        raise ValueError(f"Expected {len(expected_shapes)} tensors, got {count}.")

    parameters = []
    offset = MODEL_HEADER.size
    for shape in expected_shapes:
        if len(payload) - offset < ARRAY_HEADER.size:
            raise ValueError("Model payload ended before an array header.")
        (byte_count,) = ARRAY_HEADER.unpack_from(payload, offset)
        offset += ARRAY_HEADER.size
        expected_bytes = math.prod(shape) * np.dtype("<f4").itemsize
        if byte_count != expected_bytes or len(payload) - offset < byte_count:
            raise ValueError("Model tensor payload has an invalid size.")
        array = np.frombuffer(payload, dtype="<f4", count=math.prod(shape), offset=offset)
        parameters.append(array.astype(np.float32, copy=True).reshape(shape))
        offset += byte_count
    if offset != len(payload):
        raise ValueError("Model payload has trailing bytes.")
    return parameters


def _relative_norm(values: np.ndarray, model_norm: float) -> float:
    update_norm = float(np.linalg.norm(values))
    if model_norm > 0:
        return update_norm / model_norm
    return 0.0 if update_norm == 0 else math.inf


def _pack_gated_update(values: np.ndarray, method: str) -> bytes:
    return UPDATE_HEADER.pack(UPDATE_MAGIC, _compose_flags(method, True), values.size, 0, 0.0)


def _select_topk(values: np.ndarray, top_k_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return values, np.empty(0, dtype=np.int64)
    count = min(values.size, max(1, math.ceil(values.size * top_k_fraction)))
    if count == values.size:
        indexes = np.arange(values.size, dtype=np.int64)
    else:
        indexes = np.argpartition(np.abs(values), values.size - count)[-count:]
        indexes = np.sort(indexes)
    return values[indexes], indexes


def _quantize_values(values: np.ndarray, *, scale: float | None = None) -> tuple[np.ndarray, float]:
    if values.size == 0:
        return np.empty(0, dtype=np.int8), 0.0
    maximum = float(np.max(np.abs(values)))
    if maximum == 0.0:
        return np.zeros(values.size, dtype=np.int8), 0.0
    if scale is None:
        scale = maximum / 127.0
    quantized = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
    return quantized, float(scale)


def encode_update(
    update: np.ndarray,
    model_norm: float,
    method: str = "fedavg",
    top_k_fraction: float = TOP_K_FRACTION,
    gate_threshold: float = GATE_THRESHOLD,
) -> bytes:
    """Encode a model delta for the selected federated variant.

    Supported methods are standard dense FedAvg, quantized FedAvg, top-k sparse
    FedAvg, gate-only FedAvg, and a combined CoFlow-FL path. All payloads are
    serialized with a dedicated update header and their transmitted bytes are
    counted precisely in src/fl/cost.py.
    """
    normalized_method = method.lower()
    if not 0 < top_k_fraction <= 1:
        raise ValueError("top_k_fraction must be greater than 0 and at most 1.")
    if gate_threshold < 0:
        raise ValueError("gate_threshold must be non-negative.")

    values = np.asarray(update, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return _pack_gated_update(values, normalized_method)
    if not np.isfinite(values).all() or not math.isfinite(model_norm):
        raise ValueError("Updates and model norm must be finite.")

    relative_norm = _relative_norm(values, model_norm)
    should_gate = normalized_method in {"gated_fedavg", "coflow_fl"} and relative_norm < gate_threshold
    if should_gate or relative_norm == math.inf:
        return _pack_gated_update(values, normalized_method)

    if normalized_method in {"fedavg", "gated_fedavg"}:
        header = UPDATE_HEADER.pack(UPDATE_MAGIC, _compose_flags(normalized_method, False), values.size, values.size, 0.0)
        return header + values.astype("<f4", copy=False).tobytes()

    if normalized_method == "quantized_fedavg":
        quantized, scale = _quantize_values(values)
        if scale == 0.0:
            return _pack_gated_update(values, normalized_method)
        header = UPDATE_HEADER.pack(UPDATE_MAGIC, _compose_flags(normalized_method, False), values.size, values.size, scale)
        return header + quantized.tobytes()

    if normalized_method == "topk_fedavg":
        selected, indexes = _select_topk(values, top_k_fraction)
        if selected.size == 0:
            return _pack_gated_update(values, normalized_method)
        header = UPDATE_HEADER.pack(UPDATE_MAGIC, _compose_flags(normalized_method, False), values.size, selected.size, 0.0)
        return b"".join((header, indexes.astype("<u4", copy=False).tobytes(), selected.astype("<f4", copy=False).tobytes()))

    if normalized_method == "coflow_fl":
        selected, indexes = _select_topk(values, top_k_fraction)
        if selected.size == 0:
            return _pack_gated_update(values, normalized_method)
        quantized, scale = _quantize_values(selected)
        if scale == 0.0:
            return _pack_gated_update(values, normalized_method)
        header = UPDATE_HEADER.pack(UPDATE_MAGIC, _compose_flags(normalized_method, False), values.size, selected.size, scale)
        return b"".join((header, indexes.astype("<u4", copy=False).tobytes(), quantized.tobytes()))

    raise ValueError(f"Unsupported update method: {method!r}")


def decode_update(payload: bytes, expected_size: int) -> tuple[np.ndarray, bool]:
    """Decode a compressed delta and report whether the client skipped or gated the upload."""
    if len(payload) < UPDATE_HEADER.size:
        raise ValueError("Update payload is shorter than its header.")
    magic, flags, total_size, count, scale = UPDATE_HEADER.unpack_from(payload)
    if magic != UPDATE_MAGIC:
        raise ValueError("Invalid update payload magic.")
    if total_size != expected_size:
        raise ValueError(f"Expected {expected_size} update values, got {total_size}.")

    gated = bool(flags & 1)
    method_code = flags >> 1
    if gated:
        if count != 0 or scale != 0.0 or len(payload) != UPDATE_HEADER.size:
            raise ValueError("Gated update payload contains unexpected data.")
        return np.zeros(expected_size, dtype=np.float32), True

    if method_code == 0:
        expected_bytes = UPDATE_HEADER.size + total_size * np.dtype("<f4").itemsize
        if len(payload) != expected_bytes:
            raise ValueError("Dense update payload has an invalid size.")
        values = np.frombuffer(payload, dtype="<f4", count=total_size, offset=UPDATE_HEADER.size)
        return values.astype(np.float32, copy=True), False

    if method_code == 1:
        expected_bytes = UPDATE_HEADER.size + count * np.dtype(np.int8).itemsize
        if len(payload) != expected_bytes or count == 0 or count > expected_size:
            raise ValueError("Quantized update payload has an invalid size.")
        quantized = np.frombuffer(payload, dtype=np.int8, count=count, offset=UPDATE_HEADER.size)
        decoded = np.zeros(expected_size, dtype=np.float32)
        decoded[:] = quantized.astype(np.float32, copy=False) * float(scale)
        return decoded.astype(np.float32, copy=True), False

    if method_code == 2:
        expected_bytes = UPDATE_HEADER.size + count * (np.dtype("<u4").itemsize + np.dtype("<f4").itemsize)
        if len(payload) != expected_bytes or count == 0 or count > expected_size:
            raise ValueError("Top-k update payload has an invalid size.")
        indexes_start = UPDATE_HEADER.size
        values_start = indexes_start + count * np.dtype("<u4").itemsize
        indexes = np.frombuffer(payload, dtype="<u4", count=count, offset=indexes_start)
        values = np.frombuffer(payload, dtype="<f4", count=count, offset=values_start)
        if np.any(indexes >= expected_size) or np.unique(indexes).size != count:
            raise ValueError("Top-k update indexes are invalid.")
        decoded = np.zeros(expected_size, dtype=np.float32)
        decoded[indexes] = values.astype(np.float32, copy=False)
        return decoded.astype(np.float32, copy=True), False

    if method_code == 3:
        expected_bytes = UPDATE_HEADER.size + count * (np.dtype("<u4").itemsize + np.dtype(np.int8).itemsize)
        if len(payload) != expected_bytes or count == 0 or count > expected_size:
            raise ValueError("CoFlow update payload has an invalid size.")
        indexes_start = UPDATE_HEADER.size
        values_start = indexes_start + count * np.dtype("<u4").itemsize
        indexes = np.frombuffer(payload, dtype="<u4", count=count, offset=indexes_start)
        quantized = np.frombuffer(payload, dtype=np.int8, count=count, offset=values_start)
        if np.any(indexes >= expected_size) or np.unique(indexes).size != count:
            raise ValueError("CoFlow update indexes are invalid.")
        decoded = np.zeros(expected_size, dtype=np.float32)
        decoded[indexes] = quantized.astype(np.float32, copy=False) * float(scale)
        return decoded.astype(np.float32, copy=True), False

    raise ValueError(f"Unsupported update method code: {method_code}")
