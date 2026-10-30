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
Aryas. Yuthika owns the federated learning code in `src/fl/`.
`run_federated()` writes FedAvg results to `results/fedavg_results.csv`.
`run_comparison()` writes the five-method comparison to
`results/federated_comparison.csv` and writes its CoFlow-FL compatibility
output to `results/federated_results.csv`.

Federated training uses a plain-Python FedAvg loop with 10 junction clients and
15 rounds. Client updates use 8-bit quantization, retain the largest 25% of
update values, and are gated when their L2 norm is below 0.05% of the global
model's L2 norm. Federated `bytes_sent` is the per-round sum of serialized
client-update payloads (uplink) and model payloads broadcast once per recipient
(downlink). Serialized application headers and compression metadata are
included; network transport overhead such as TCP/IP headers and retransmissions
is excluded. `cumulative_bytes_sent` is the running total across rounds, and
`cumulative_mb` uses decimal MB (bytes / 1,000,000).

The centralized baseline's `bytes_sent` records the in-memory size of its
training input and target arrays per epoch, not network communication. Do not
compare it directly with federated network traffic. Local-only has no exchanged
updates, while a standard uncompressed FedAvg result is not currently available.
