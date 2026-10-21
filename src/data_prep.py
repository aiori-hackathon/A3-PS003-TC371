"""Prepare per-junction train and test windows for traffic forecasting."""

from pathlib import Path

import numpy as np
import pandas as pd

NUM_JUNCTIONS = 10
INPUT_STEPS = 12
OUTPUT_STEPS = 3
TRAIN_FRACTION = 0.8
DATA_KIND = "flow"
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "traffic.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_raw() -> pd.DataFrame:
    """Load the supplied traffic file or create a reproducible demo dataset."""
    if RAW_PATH.exists():
        data = pd.read_csv(RAW_PATH).iloc[:, :NUM_JUNCTIONS]
        if data.shape[1] != NUM_JUNCTIONS:
            raise ValueError(
                f"{RAW_PATH} must contain at least {NUM_JUNCTIONS} sensor columns."
            )
        print(f"Using traffic data from {RAW_PATH}")
        return data.astype("float32")

    print("No data/raw/traffic.csv found; using synthetic traffic data.")
    rng = np.random.default_rng(SEED)
    steps_per_day = 24 * 12
    total_steps = 30 * steps_per_day
    time = np.arange(total_steps)
    hour = (time % steps_per_day) / 12.0
    day_number = time // steps_per_day
    weekday = (day_number % 7) < 5
    data = np.empty((total_steps, NUM_JUNCTIONS), dtype=np.float32)

    for junction in range(NUM_JUNCTIONS):
        # Different junctions have different normal levels and peak times.
        base = 25 + junction * 3
        amplitude = 35 + (junction % 4) * 9
        morning_peak = 7.0 + (junction % 3) * 0.4
        evening_peak = 17.0 + (junction % 4) * 0.35
        morning = np.exp(-0.5 * ((hour - morning_peak) / 1.4) ** 2)
        evening = np.exp(-0.5 * ((hour - evening_peak) / 1.8) ** 2)
        weekday_factor = np.where(weekday, 1.0, 0.68)
        daily_pattern = amplitude * (0.9 * morning + evening)
        noise = rng.normal(0, 3.0, total_steps)
        data[:, junction] = np.clip(
            base + weekday_factor * daily_pattern + noise, 0, None
        )

    return pd.DataFrame(data, columns=[f"junction_{i}" for i in range(1, NUM_JUNCTIONS + 1)])


def _make_windows(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make windows without crossing the boundary of a chronological split."""
    count = len(series) - INPUT_STEPS - OUTPUT_STEPS + 1
    if count <= 0:
        return (
            np.empty((0, INPUT_STEPS, 1), dtype=np.float32),
            np.empty((0, OUTPUT_STEPS), dtype=np.float32),
        )
    x = np.array(
        [series[i : i + INPUT_STEPS] for i in range(count)], dtype=np.float32
    )[:, :, None]
    y = np.array(
        [
            series[i + INPUT_STEPS : i + INPUT_STEPS + OUTPUT_STEPS]
            for i in range(count)
        ],
        dtype=np.float32,
    )
    return x, y


def prepare_data() -> None:
    """Split, normalize, window, and save every junction."""
    np.random.seed(SEED)
    raw = load_raw()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for junction in range(NUM_JUNCTIONS):
        series = raw.iloc[:, junction].to_numpy(dtype=np.float32)
        split = int(len(series) * TRAIN_FRACTION)
        train_series = series[:split]
        test_series = series[split:]
        mean = float(train_series.mean())
        std = float(train_series.std())
        if std == 0:
            std = 1.0

        x_train, y_train = _make_windows((train_series - mean) / std)
        x_test, y_test = _make_windows((test_series - mean) / std)
        output = PROCESSED_DIR / f"junction_{junction + 1}.npz"
        np.savez(
            output,
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            mean=np.float32(mean),
            std=np.float32(std),
        )
        print(
            f"Junction {junction + 1}: X_train={x_train.shape}, "
            f"y_train={y_train.shape}, X_test={x_test.shape}, y_test={y_test.shape}"
        )


def load_junction(junction: int) -> dict[str, np.ndarray]:
    """Load one prepared junction, using one-based junction numbering."""
    if not 1 <= junction <= NUM_JUNCTIONS:
        raise ValueError(f"junction must be between 1 and {NUM_JUNCTIONS}")
    path = PROCESSED_DIR / f"junction_{junction}.npz"
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def main() -> None:
    prepare_data()


if __name__ == "__main__":
    main()
