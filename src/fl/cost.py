"""Count serialized protocol payload bytes, including their application headers.

The values in this module represent the application-layer payloads actually
transmitted by the federated learning loop. Broadcast bytes are counted once per
recipient, and transport-layer overhead such as TCP/IP framing and packet
retransmissions is intentionally excluded. The totals therefore reflect the
serialized payload sizes that are relevant for comparing compression strategies,
not physical network traffic or operating-system-level packet sizes.
"""

from dataclasses import dataclass


@dataclass
class RoundCost:
    """Serialized uplink and downlink payload totals for a single FL round."""

    uplink_bytes: int = 0
    downlink_bytes: int = 0
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
            "skipped_uploads": self.skipped_uploads,
        }
