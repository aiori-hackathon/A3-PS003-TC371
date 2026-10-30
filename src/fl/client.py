"""One junction's local training and compressed update generation."""

from dataclasses import dataclass
from typing import Callable, TypedDict

import numpy as np

from src.data_prep import load_junction
from src.fl.compression import (
    GATE_THRESHOLD,
    TOP_K_FRACTION,
    UPDATE_HEADER,
    _relative_norm,
    _should_gate,
    decode_model,
    encode_update,
    flatten_parameters,
)
from src.model import (
    TrafficGRU,
    evaluate,
    get_model,
    get_parameters,
    set_parameters,
    train,
)


class ClientUpdateMetrics(TypedDict):
    round: int
    client_id: int
    method: str
    global_model_norm: float
    client_update_norm: float
    relative_update_norm: float
    skipped: bool
    gate_threshold: float


@dataclass(frozen=True)
class ClientUpdate:
    payload: bytes
    num_examples: int
    method: str = "fedavg"
    gated: bool = False


class FederatedClient:
    """Train one local model and return a compressed parameter delta."""

    def __init__(self, junction: int) -> None:
        self.junction = junction
        self.data = load_junction(junction)
        self.model: TrafficGRU = get_model()
        self._shapes = [tuple(value.shape) for value in get_parameters(self.model)]

    def fit(
        self,
        global_model_payload: bytes,
        *,
        method: str = "fedavg",
        top_k_fraction: float = TOP_K_FRACTION,
        gate_threshold: float = GATE_THRESHOLD,
        round_number: int | None = None,
        update_metrics_callback: Callable[[ClientUpdateMetrics], None] | None = None,
    ) -> ClientUpdate:
        global_parameters = decode_model(global_model_payload, self._shapes)
        set_parameters(self.model, global_parameters)
        train(self.model, self.data["X_train"], self.data["y_train"])

        local_parameters = get_parameters(self.model)
        delta = [
            local - global_value
            for local, global_value in zip(local_parameters, global_parameters)
        ]
        global_vector = flatten_parameters(global_parameters)
        delta_vector = flatten_parameters(delta)
        global_norm = float(np.linalg.norm(global_vector))
        update_norm = float(np.linalg.norm(delta_vector))
        relative_norm = _relative_norm(delta_vector, global_norm)
        skipped = _should_gate(method, relative_norm, gate_threshold)
        if update_metrics_callback is not None:
            if round_number is None:
                raise ValueError("round_number is required when logging client update metrics.")
            update_metrics_callback(
                {
                    "round": round_number,
                    "client_id": self.junction,
                    "method": method,
                    "global_model_norm": global_norm,
                    "client_update_norm": update_norm,
                    "relative_update_norm": relative_norm,
                    "skipped": skipped,
                    "gate_threshold": gate_threshold,
                }
            )
        payload = encode_update(
            delta_vector,
            global_norm,
            method=method,
            top_k_fraction=top_k_fraction,
            gate_threshold=gate_threshold,
        )
        _, flags, _, _, _ = UPDATE_HEADER.unpack_from(payload)
        return ClientUpdate(
            payload=payload,
            num_examples=len(self.data["X_train"]),
            method=method,
            gated=bool(flags & 1),
        )

    def evaluate(self, model: TrafficGRU) -> dict[str, float]:
        """Evaluate a global model on this junction's held-out data."""
        return evaluate(
            model,
            self.data["X_test"],
            self.data["y_test"],
            float(self.data["mean"]),
            float(self.data["std"]),
        )
