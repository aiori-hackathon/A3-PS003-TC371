"""Create a compact traffic CSV from the METR-LA sensor data."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "metr-la.h5"
OUTPUT_PATH = ROOT / "data" / "raw" / "traffic.csv"
NUM_SENSORS = 10


def make_real_csv() -> pd.DataFrame:
    """Select, fill, and save the ten least-missing METR-LA sensors."""
    data = pd.read_hdf(INPUT_PATH)
    missing = data.isna() | data.eq(0)
    missing_counts = missing.sum()

    selected = (
        missing_counts.sort_values(kind="mergesort")
        .head(NUM_SENSORS)
        .index
    )
    selected = [column for column in data.columns if column in selected]
    selected_data = data.loc[:, selected].mask(missing.loc[:, selected])
    selected_data = selected_data.ffill().bfill()
    selected_data.to_csv(OUTPUT_PATH, index=False)

    print(f"Selected sensors: {', '.join(map(str, selected))}")
    print(f"Rows: {len(selected_data)}")
    print("Original missing percentage per selected sensor:")
    for column in selected:
        print(f"  {column}: {missing[column].mean() * 100:.2f}%")

    return selected_data


if __name__ == "__main__":
    make_real_csv()
