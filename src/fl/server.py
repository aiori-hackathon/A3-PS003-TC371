"""Plain-Python federated orchestration for all junction clients."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.data_prep import NUM_JUNCTIONS
from src.fl.client import ClientUpdateMetrics, FederatedClient
from src.fl.compression import (
    GATE_THRESHOLD,
    TOP_K_FRACTION,
    decode_update,
    encode_model,
    flatten_parameters,
    unflatten_parameters,
)
from src.fl.cost import RoundCost
from src.model import TrafficGRU, get_model, get_parameters, set_parameters

ROUNDS = 15


def _run_experiment(
    method: str,
    rounds: int = ROUNDS,
    *,
    top_k_fraction: float = TOP_K_FRACTION,
    gate_threshold: float = GATE_THRESHOLD,
    update_metrics_callback: Callable[[ClientUpdateMetrics], None] | None = None,
) -> tuple[pd.DataFrame, TrafficGRU]:
    """Train all junctions with a configurable federated update method."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1.")
    if method not in {"fedavg", "quantized_fedavg", "topk_fedavg", "gated_fedavg", "coflow_fl"}:
        raise ValueError(f"Unsupported federated method: {method!r}")

    clients = [FederatedClient(junction) for junction in range(1, NUM_JUNCTIONS + 1)]
    model = get_model()
    parameters = get_parameters(model)
    shapes = [tuple(value.shape) for value in parameters]
    global_vector = flatten_parameters(parameters)
    rows: list[dict[str, float | int | str]] = []
    cumulative_bytes_sent = 0

    for round_number in range(1, rounds + 1):
        cost = RoundCost()
        global_payload = encode_model(parameters)
        cost.record_downlink(global_payload, recipients=len(clients))

        client_updates = [
            client.fit(
                global_payload,
                method=method,
                top_k_fraction=top_k_fraction,
                gate_threshold=gate_threshold,
                round_number=round_number,
                update_metrics_callback=update_metrics_callback,
            )
            for client in clients
        ]
        total_examples = sum(update.num_examples for update in client_updates)
        weighted_update = np.zeros_like(global_vector)
        participating_examples = 0
        skipped_uploads = 0
        for update in client_updates:
            delta, gated = decode_update(update.payload, global_vector.size)
            if gated:
                skipped_uploads += 1
                cost.record_uplink(update.payload, was_skipped=True)
                continue
            cost.record_uplink(update.payload, was_skipped=False)
            weighted_update += delta * update.num_examples
            participating_examples += update.num_examples

        if participating_examples:
            global_vector = global_vector + weighted_update / participating_examples
            parameters = unflatten_parameters(global_vector, shapes)
            set_parameters(model, parameters)

        metrics = [client.evaluate(model) for client in clients]
        mae = float(np.mean([metric["MAE"] for metric in metrics]))
        rmse = float(np.mean([metric["RMSE"] for metric in metrics]))
        cumulative_bytes_sent += cost.bytes_sent
        row = {
            "method": method,
            "round": round_number,
            "MAE": mae,
            "RMSE": rmse,
            "upload_bytes": cost.uplink_bytes,
            "download_bytes": cost.downlink_bytes,
            "uplink_bytes": cost.uplink_bytes,
            "downlink_bytes": cost.downlink_bytes,
            "bytes_sent": cost.bytes_sent,
            "cumulative_bytes": cumulative_bytes_sent,
            "communication_MB": cost.bytes_sent / 1_000_000,
            "cumulative_MB": cumulative_bytes_sent / 1_000_000,
            "num_clients": len(clients),
            "attempted_uploads": cost.attempted_uploads,
            "skipped_uploads": skipped_uploads,
            "gate_threshold": gate_threshold,
            "training_examples": total_examples,
            "participating_clients": len(clients) - skipped_uploads,
            "participating_examples": participating_examples,
        }
        rows.append(row)
        print(
            f"{method} round {round_number}: "
            f"MAE={mae:.4f}, RMSE={rmse:.4f}, bytes_sent={cost.bytes_sent}"
        )

    return pd.DataFrame(rows), model


def run_fedavg(
    rounds: int = ROUNDS,
    *,
    top_k_fraction: float = TOP_K_FRACTION,
    gate_threshold: float = GATE_THRESHOLD,
) -> tuple[pd.DataFrame, TrafficGRU]:
    """Train all junctions with standard FedAvg."""
    return _run_experiment(
        "fedavg",
        rounds=rounds,
        top_k_fraction=top_k_fraction,
        gate_threshold=gate_threshold,
    )


def run_quantized_fedavg(
    rounds: int = ROUNDS,
    *,
    top_k_fraction: float = TOP_K_FRACTION,
    gate_threshold: float = GATE_THRESHOLD,
) -> tuple[pd.DataFrame, TrafficGRU]:
    """Train all junctions with 8-bit quantized update messages."""
    return _run_experiment(
        "quantized_fedavg",
        rounds=rounds,
        top_k_fraction=top_k_fraction,
        gate_threshold=gate_threshold,
    )


def run_topk_fedavg(
    rounds: int = ROUNDS,
    *,
    top_k_fraction: float = TOP_K_FRACTION,
    gate_threshold: float = GATE_THRESHOLD,
) -> tuple[pd.DataFrame, TrafficGRU]:
    """Train all junctions with top-k sparse model deltas."""
    return _run_experiment(
        "topk_fedavg",
        rounds=rounds,
        top_k_fraction=top_k_fraction,
        gate_threshold=gate_threshold,
    )


def run_gated_fedavg(
    rounds: int = ROUNDS,
    *,
    top_k_fraction: float = TOP_K_FRACTION,
    gate_threshold: float = GATE_THRESHOLD,
) -> tuple[pd.DataFrame, TrafficGRU]:
    """Train all junctions with relative-norm update gating."""
    return _run_experiment(
        "gated_fedavg",
        rounds=rounds,
        top_k_fraction=top_k_fraction,
        gate_threshold=gate_threshold,
    )


def run_coflow_fl(
    rounds: int = ROUNDS,
    *,
    top_k_fraction: float = TOP_K_FRACTION,
    gate_threshold: float = GATE_THRESHOLD,
    update_metrics_callback: Callable[[ClientUpdateMetrics], None] | None = None,
) -> tuple[pd.DataFrame, TrafficGRU]:
    """Train all junctions with the combined CoFlow-FL stack."""
    return _run_experiment(
        "coflow_fl",
        rounds=rounds,
        top_k_fraction=top_k_fraction,
        gate_threshold=gate_threshold,
        update_metrics_callback=update_metrics_callback,
    )
