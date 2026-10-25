"""Prepare data, run federated training, and save dashboard-compatible results."""

from pathlib import Path

import pandas as pd
import torch

from src.data_prep import NUM_JUNCTIONS, PROCESSED_DIR, prepare_data
from src.fl.server import ROUNDS, run_fedavg

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "results" / "federated_results.csv"
MODEL_PATH = ROOT / "models" / "federated.pt"


def _ensure_prepared_data() -> None:
    missing = [
        PROCESSED_DIR / f"junction_{junction}.npz"
        for junction in range(1, NUM_JUNCTIONS + 1)
        if not (PROCESSED_DIR / f"junction_{junction}.npz").exists()
    ]
    if missing:
        print("Prepared data not found; preparing it now.")
        prepare_data()


def run_federated(rounds: int = ROUNDS) -> pd.DataFrame:
    """Run federated training and persist its metrics and final model."""
    _ensure_prepared_data()
    results, model = run_fedavg(rounds=rounds)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved federated results to {RESULTS_PATH}")
    print(f"Saved federated model to {MODEL_PATH}")
    return results


def main() -> None:
    run_federated()


if __name__ == "__main__":
    main()
