"""Plain-Python FedAvg orchestration for all junction clients."""

import numpy as np
import pandas as pd

from src.data_prep import NUM_JUNCTIONS
from src.fl.client import FederatedClient
from src.fl.compression import (
    decode_update,
    encode_model,
    flatten_parameters,
    unflatten_parameters,
)
from src.fl.cost import RoundCost
from src.model import TrafficGRU, get_model, get_parameters, set_parameters

ROUNDS = 15


def run_fedavg(
    rounds: int = ROUNDS,
) -> tuple[pd.DataFrame, TrafficGRU]:
    """Train all junctions with weighted FedAvg and return per-round results."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1.")

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
            client.fit(global_payload)
            for client in clients
        ]
        total_examples = sum(update.num_examples for update in client_updates)
        weighted_update = np.zeros_like(global_vector)
        participating_examples = 0
        gated_clients = 0
        for update in client_updates:
            cost.record_uplink(update.payload)
            delta, gated = decode_update(update.payload, global_vector.size)
            if gated:
                gated_clients += 1
                continue
            weighted_update += delta * update.num_examples
            participating_examples += update.num_examples

        if participating_examples:
            global_vector = global_vector + weighted_update / participating_examples
            parameters = unflatten_parameters(global_vector, shapes)
            set_parameters(model, parameters)

        metrics = [client.evaluate(model) for client in clients]
        cumulative_bytes_sent += cost.bytes_sent
        rows.append(
            {
                "method": "federated",
                "round": round_number,
                "MAE": float(np.mean([metric["MAE"] for metric in metrics])),
                "RMSE": float(np.mean([metric["RMSE"] for metric in metrics])),
                "bytes_sent": cost.bytes_sent,
                "uplink_bytes": cost.uplink_bytes,
                "downlink_bytes": cost.downlink_bytes,
                "gated_clients": gated_clients,
                "participating_clients": len(clients) - gated_clients,
                "training_examples": total_examples,
                "cumulative_bytes_sent": cumulative_bytes_sent,
                "cumulative_mb": cumulative_bytes_sent / 1_000_000,
            }
        )
        print(
            f"federated round {round_number}: "
            f"MAE={rows[-1]['MAE']:.4f}, RMSE={rows[-1]['RMSE']:.4f}, "
            f"bytes_sent={cost.bytes_sent}"
        )

    return pd.DataFrame(rows), model
