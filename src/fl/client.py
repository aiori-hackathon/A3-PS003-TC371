"""One junction's local training and compressed update generation."""

from dataclasses import dataclass

import numpy as np

from src.data_prep import load_junction
from src.fl.compression import (
    GATE_THRESHOLD,
    TOP_K_FRACTION,
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
        payload = encode_update(
            delta_vector,
            float(np.linalg.norm(global_vector)),
            method=method,
            top_k_fraction=top_k_fraction,
            gate_threshold=gate_threshold,
        )
        gated = False
        if len(payload) >= 17 and payload[:4] == b"CFU1":
            magic, flags, total_size, count, scale = __import__("struct").unpack_from("!4sBII f", payload)
            gated = bool(flags & 1)
        return ClientUpdate(payload=payload, num_examples=len(self.data["X_train"]), method=method, gated=gated)

    def evaluate(self, model: TrafficGRU) -> dict[str, float]:
        """Evaluate a global model on this junction's held-out data."""
        return evaluate(
            model,
            self.data["X_test"],
            self.data["y_test"],
            float(self.data["mean"]),
            float(self.data["std"]),
        )
