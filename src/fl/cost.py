"""Count serialized protocol payload bytes, including their application headers.

The values in this module represent the application-layer payloads actually
transmitted by the federated learning loop. Broadcast bytes are counted once per
recipient, and transport-layer overhead such as TCP/IP framing and packet
retransmissions is intentionally excluded. The totals therefore reflect the
serialized payload sizes that are relevant for comparing compression strategies,
not physical network traffic or operating-system-level packet sizes.
"""

from dataclasses import dataclass

import pandas as pd


def savings_percent(method_bytes: int, fedavg_bytes: int) -> float:
    """Return savings against FedAvg, using zero when the baseline is zero."""
    if fedavg_bytes == 0:
        return 0.0
    return 100.0 * (1.0 - method_bytes / fedavg_bytes)


def add_communication_savings(results: pd.DataFrame) -> pd.DataFrame:
    """Add per-round uplink and total-byte savings against plain FedAvg."""
    required = {"method", "round", "upload_bytes", "bytes_sent"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(
            f"Cannot calculate communication savings; missing columns: {sorted(missing)}"
        )

    baseline = results.loc[
        results["method"] == "fedavg", ["round", "upload_bytes", "bytes_sent"]
    ]
    if baseline.empty:
        raise ValueError("Cannot calculate communication savings without FedAvg results.")
    if baseline["round"].duplicated().any():
        raise ValueError("FedAvg results contain duplicate rounds.")

    baseline_by_round = baseline.set_index("round")
    missing_rounds = set(results["round"]) - set(baseline_by_round.index)
    if missing_rounds:
        raise ValueError(
            f"FedAvg baseline is missing rounds: {sorted(missing_rounds)}"
        )

    result = results.copy()
    result["uplink_savings_percent"] = [
        0.0
        if method == "fedavg"
        else savings_percent(
            upload_bytes,
            baseline_by_round.at[round_number, "upload_bytes"],
        )
        for method, round_number, upload_bytes in zip(
            result["method"], result["round"], result["upload_bytes"]
        )
    ]
    result["total_communication_savings_percent"] = [
        0.0
        if method == "fedavg"
        else savings_percent(
            bytes_sent,
            baseline_by_round.at[round_number, "bytes_sent"],
        )
        for method, round_number, bytes_sent in zip(
            result["method"], result["round"], result["bytes_sent"]
        )
    ]
    return result


@dataclass
class RoundCost:
    """Serialized uplink and downlink payload totals for a single FL round."""

    uplink_bytes: int = 0
    downlink_bytes: int = 0
    attempted_uploads: int = 0
    skipped_uploads: int = 0

    @property
    def bytes_sent(self) -> int:
        """Total simulated network payload bytes for one round."""
        return self.uplink_bytes + self.downlink_bytes

    @property
    def communication_mb(self) -> float:
        """Decimal MB using 1 MB = 1,000,000 bytes."""
        return self.bytes_sent / 1_000_000

    def record_uplink(self, payload: bytes, *, was_skipped: bool = False) -> None:
        """Record a client upload payload; a gated/skipped upload still counts once."""
        self.uplink_bytes += len(payload)
        self.attempted_uploads += 1
        if was_skipped:
            self.skipped_uploads += 1

    def record_downlink(self, payload: bytes, recipients: int = 1) -> None:
        if recipients < 0:
            raise ValueError("recipients must be non-negative.")
        self.downlink_bytes += len(payload) * recipients

    def as_dict(self) -> dict[str, int | float]:
        return {
            "upload_bytes": self.uplink_bytes,
            "download_bytes": self.downlink_bytes,
            "bytes_sent": self.bytes_sent,
            "communication_MB": self.communication_mb,
            "attempted_uploads": self.attempted_uploads,
            "skipped_uploads": self.skipped_uploads,
        }
