"""Count serialized protocol payload bytes, including their application headers.

Broadcast bytes are counted once per recipient. Transport and network-layer
overhead (for example TCP/IP headers and retransmissions) is not included.
"""

from dataclasses import dataclass


@dataclass
class RoundCost:
    """Serialized client/server payload totals for one round.

    A gated client's notification is counted if its serialized payload is
    recorded as an uplink, just like a normal compressed update.
    """

    uplink_bytes: int = 0
    downlink_bytes: int = 0

    @property
    def bytes_sent(self) -> int:
        return self.uplink_bytes + self.downlink_bytes

    def record_uplink(self, payload: bytes) -> None:
        self.uplink_bytes += len(payload)

    def record_downlink(self, payload: bytes, recipients: int = 1) -> None:
        if recipients < 0:
            raise ValueError("recipients must be non-negative.")
        self.downlink_bytes += len(payload) * recipients
