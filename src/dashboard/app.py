"""Streamlit dashboard for predictions and communication-cost comparisons."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data_prep import DATA_KIND, load_junction
from src.model import get_model

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def _load_model(choice: str, junction: int):
    if choice == "centralized":
        path = MODELS_DIR / "centralized.pt"
    elif choice == "local":
        path = MODELS_DIR / f"local_{junction}.pt"
    else:
        path = MODELS_DIR / "federated.pt"
    if not path.exists():
        return None
    model = get_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _prediction_chart(model, data: dict[str, np.ndarray]) -> None:
    with torch.no_grad():
        predicted = model(torch.from_numpy(data["X_test"])).numpy()
    predicted = predicted * float(data["std"]) + float(data["mean"])
    actual = data["y_test"] * float(data["std"]) + float(data["mean"])
    count = min(200, len(actual))
    figure, axis = plt.subplots()
    axis.plot(actual[-count:, 0], label="Actual")
    axis.plot(predicted[-count:, 0], label="Predicted")
    axis.set_title("Actual vs predicted traffic (first future step)")
    axis.set_xlabel("Test window")
    axis.set_ylabel(DATA_KIND)
    axis.legend()
    st.pyplot(figure)
    plt.close(figure)


def _show_alert(model, data: dict[str, np.ndarray], junction: int) -> None:
    train_values = np.concatenate(
        [data["X_train"].reshape(-1), data["y_train"].reshape(-1)]
    )
    train_values = train_values * float(data["std"]) + float(data["mean"])
    percentile = 85 if DATA_KIND == "flow" else 15
    threshold = float(np.percentile(train_values, percentile))
    with torch.no_grad():
        latest = model(torch.from_numpy(data["X_test"][-1:])).numpy()[0]
    values = latest * float(data["std"]) + float(data["mean"])
    is_congested = np.any(values > threshold) if DATA_KIND == "flow" else np.any(values < threshold)
    comparison = "above" if DATA_KIND == "flow" else "below"
    if is_congested:
        st.error(
            f"CONGESTION WARNING - junction {junction}: predicted values "
            f"{np.round(values, 2).tolist()} are {comparison} threshold {threshold:.2f}."
        )
    else:
        st.success(f"Junction {junction} is below the congestion threshold ({threshold:.2f}).")


def _show_costs() -> None:
    files = list(RESULTS_DIR.glob("*.csv"))
    if not files:
        st.info("No baseline or federated results are available yet.")
        return
    frames = [pd.read_csv(path) for path in files]
    results = pd.concat(frames, ignore_index=True)
    line = results.sort_values(["method", "round"])
    st.subheader("MAE by round")
    st.line_chart(line, x="round", y="MAE", color="method")
    final = line.groupby("method", as_index=False).tail(1).copy()
    final["total_MB"] = final["bytes_sent"] / (1024 * 1024)
    final["accuracy-per-MB"] = final.apply(
        lambda row: "n/a"
        if row["total_MB"] == 0
        else f"{1 / row['MAE'] / row['total_MB']:.4f}",
        axis=1,
    )
    st.subheader("Final communication cost")
    st.bar_chart(final.set_index("method")["total_MB"])
    st.dataframe(final[["method", "MAE", "RMSE", "total_MB", "accuracy-per-MB"]], hide_index=True)
    if not (RESULTS_DIR / "federated_results.csv").exists():
        st.info("Federated results are still pending.")


st.set_page_config(page_title="CoFlow-FL Dashboard", layout="wide")
st.title("CoFlow-FL traffic-flow dashboard")
junction = st.sidebar.selectbox("Junction", range(1, 11))
model_options = ["centralized", "local"]
if (MODELS_DIR / "federated.pt").exists():
    model_options.append("federated")
choice = st.sidebar.selectbox("Model", model_options)
data = load_junction(junction)
model = _load_model(choice, junction)
if model is None:
    st.info(f"The {choice} model is not available yet. Run the baseline training first.")
else:
    _prediction_chart(model, data)
    _show_alert(model, data, junction)
_show_costs()
