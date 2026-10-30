"""Prepare data, run federated training, and save dashboard-compatible results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data_prep import NUM_JUNCTIONS, PROCESSED_DIR, prepare_data
from src.fl.server import (
    ROUNDS,
    run_coflow_fl,
    run_fedavg,
    run_gated_fedavg,
    run_quantized_fedavg,
    run_topk_fedavg,
)
from src.fl.cost import add_communication_savings

ROOT = Path(__file__).resolve().parents[2]
FEDAVG_RESULTS_PATH = ROOT / "results" / "fedavg_results.csv"
FEDERATED_RESULTS_PATH = ROOT / "results" / "federated_results.csv"
COMPARISON_PATH = ROOT / "results" / "federated_comparison.csv"
PLOT_PATH = ROOT / "results" / "federated_comparison.png"
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


def _summarise_communication(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[results["method"] == "fedavg"].copy()
    if baseline.empty:
        return results
    result = results.copy()
    result["compression_ratio"] = 1.0
    result["communication_savings_percent"] = 0.0
    result = add_communication_savings(result)
    result["communication_savings_percent"] = result[
        "total_communication_savings_percent"
    ]
    baseline_by_round = baseline.set_index("round")["bytes_sent"]
    result["compression_ratio"] = [
        1.0
        if method == "fedavg" or bytes_sent == 0
        else baseline_by_round.at[round_number] / bytes_sent
        for method, round_number, bytes_sent in zip(
            result["method"], result["round"], result["bytes_sent"]
        )
    ]
    return result


def run_federated(rounds: int = ROUNDS) -> pd.DataFrame:
    """Run standard federated training and persist its metrics and final model."""
    _ensure_prepared_data()
    results, model = run_fedavg(rounds=rounds)
    results = _summarise_communication(results)
    FEDAVG_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(FEDAVG_RESULTS_PATH, index=False)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved FedAvg results to {FEDAVG_RESULTS_PATH}")
    print(f"Saved federated model to {MODEL_PATH}")
    return results


def run_comparison(rounds: int = ROUNDS) -> pd.DataFrame:
    """Run the configured FL variants and save a comparison snapshot."""
    _ensure_prepared_data()
    method_runners = {
        "fedavg": run_fedavg,
        "quantized_fedavg": run_quantized_fedavg,
        "topk_fedavg": run_topk_fedavg,
        "gated_fedavg": run_gated_fedavg,
        "coflow_fl": run_coflow_fl,
    }
    frames = []
    for method, runner in method_runners.items():
        result, _ = runner(rounds=rounds)
        frames.append(result)

    comparison = _summarise_communication(pd.concat(frames, ignore_index=True))
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)
    for method in method_runners:
        method_results = comparison[comparison["method"] == method].copy()
        method_path = ROOT / "results" / f"{method}_results.csv"
        method_results.to_csv(method_path, index=False)
        if method == "coflow_fl":
            method_results.to_csv(FEDERATED_RESULTS_PATH, index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    summary = comparison.groupby("method", as_index=False).tail(1).sort_values("MAE")
    ax.bar(
        summary["method"],
        summary["total_communication_savings_percent"],
        color="steelblue",
    )
    ax.set_title("Final communication savings vs. FedAvg baseline")
    ax.set_xlabel("Method")
    ax.set_ylabel("Savings (%)")
    ax.set_ylim(-10, 100)
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    print(f"Saved comparison results to {COMPARISON_PATH}")
    print(f"Saved comparison plot to {PLOT_PATH}")
    return comparison


def main() -> None:
    run_federated()


if __name__ == "__main__":
    main()
