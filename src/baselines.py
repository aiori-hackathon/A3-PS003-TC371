"""Train centralized and local-only traffic forecasting baselines."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data_prep import NUM_JUNCTIONS, load_junction, prepare_data
from src.model import evaluate, get_model, train

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "baseline_results.csv"
MODELS_DIR = ROOT / "models"
EPOCHS = 15


def _load_all() -> list[dict[str, np.ndarray]]:
    try:
        return [load_junction(i) for i in range(1, NUM_JUNCTIONS + 1)]
    except FileNotFoundError:
        print("Prepared data not found; preparing it now.")
        prepare_data()
        return [load_junction(i) for i in range(1, NUM_JUNCTIONS + 1)]


def _average_metrics(model: torch.nn.Module, datasets: list[dict[str, np.ndarray]]) -> tuple[float, float]:
    metrics = [
        evaluate(model, item["X_test"], item["y_test"], float(item["mean"]), float(item["std"]))
        for item in datasets
    ]
    return (
        float(np.mean([metric["MAE"] for metric in metrics])),
        float(np.mean([metric["RMSE"] for metric in metrics])),
    )


def run_baselines() -> pd.DataFrame:
    datasets = _load_all()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    centralized = get_model()
    x_all = np.concatenate([item["X_train"] for item in datasets])
    y_all = np.concatenate([item["y_train"] for item in datasets])
    centralized_bytes = int(x_all.nbytes + y_all.nbytes)
    for epoch in range(1, EPOCHS + 1):
        train(centralized, x_all, y_all)
        mae, rmse = _average_metrics(centralized, datasets)
        rows.append({"method": "centralized", "round": epoch, "MAE": mae, "RMSE": rmse, "bytes_sent": centralized_bytes})
        print(f"centralized round {epoch}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    torch.save(centralized.state_dict(), MODELS_DIR / "centralized.pt")

    local_models = [get_model() for _ in range(NUM_JUNCTIONS)]
    for epoch in range(1, EPOCHS + 1):
        for model, item in zip(local_models, datasets):
            train(model, item["X_train"], item["y_train"])
        metrics = [
            evaluate(model, item["X_test"], item["y_test"], float(item["mean"]), float(item["std"]))
            for model, item in zip(local_models, datasets)
        ]
        mae = float(np.mean([metric["MAE"] for metric in metrics]))
        rmse = float(np.mean([metric["RMSE"] for metric in metrics]))
        rows.append({"method": "local_only", "round": epoch, "MAE": mae, "RMSE": rmse, "bytes_sent": 0})
        print(f"local_only round {epoch}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    for index, model in enumerate(local_models, start=1):
        torch.save(model.state_dict(), MODELS_DIR / f"local_{index}.pt")

    results = pd.DataFrame(rows, columns=["method", "round", "MAE", "RMSE", "bytes_sent"])
    results.to_csv(RESULTS_PATH, index=False)
    print("\nFinal comparison:")
    print(results.groupby("method", as_index=False).tail(1).to_string(index=False))
    return results


if __name__ == "__main__":
    run_baselines()
