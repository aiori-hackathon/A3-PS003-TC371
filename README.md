# AIORI HACKATHON 2026

## Team Introduction Video

**Team Code:** A3-PS003-TC371

**YouTube Video Link:** [Watch Our Team Introduction Video](https://youtu.be/MuOc0o7OQpo?si=avS2p9sLyD6hjKM3)

## CoFlow-FL

This project forecasts traffic at 10 junctions and compares centralized,
local-only, and federated learning communication costs.

Run from the repository root:

```text
python -m src.data_prep
python -m src.baselines
python -m src.fl.run
streamlit run src/dashboard/app.py
```

The `src/` data preparation, model, baseline, and dashboard code is owned by
Aryas. Yuthika owns the federated learning code in `src/fl/` and writes
`results/federated_results.csv`.

Federated training uses a plain-Python FedAvg loop with 10 junction clients and
15 rounds. Client updates use 8-bit quantization, retain the largest 25% of
update values, and are gated when their L2 norm is below 0.05% of the global
model's L2 norm. `bytes_sent` counts the serialized model broadcasts and client
updates for each round; it excludes network-layer overhead.
