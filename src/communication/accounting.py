"""Communication accounting from the messages that are actually exchanged.

Sizes are the serialized payload sizes reported by Flower for every record in
a message (``record.count_bytes()``), summed per direction over the clients
that really sent or received a message in that round. Transport framing (gRPC,
TLS) is not included and no compression is applied; both facts are stated
wherever these numbers are reported.
"""
from __future__ import annotations


def message_bytes(message) -> int:
    """Payload bytes of one Flower message (0 for error replies)."""
    if not message.has_content():
        return 0
    return int(sum(record.count_bytes() for record in message.content.values()))


class CommLedger:
    """Per-round and cumulative uplink/downlink bytes for one run."""

    def __init__(self):
        self.rounds: list[dict] = []
        self.setup = {"downlink_bytes": 0, "uplink_bytes": 0}
        self.evaluation = {"downlink_bytes": 0, "uplink_bytes": 0}
        self._cum_down = 0
        self._cum_up = 0

    def record_round(self, rnd: int, down: list[int], up: list[int]) -> dict:
        row = {
            "round": rnd,
            "downlink_clients": len(down),
            "uplink_clients": len(up),
            "downlink_bytes": sum(down),
            "uplink_bytes": sum(up),
            "downlink_bytes_per_client": sum(down) / len(down) if down else 0.0,
            "uplink_bytes_per_client": sum(up) / len(up) if up else 0.0,
        }
        self._cum_down += row["downlink_bytes"]
        self._cum_up += row["uplink_bytes"]
        row["cum_downlink_bytes"] = self._cum_down
        row["cum_uplink_bytes"] = self._cum_up
        row["cum_total_bytes"] = self._cum_down + self._cum_up
        self.rounds.append(row)
        return row

    def record_other(self, kind: str, down: list[int], up: list[int]) -> None:
        bucket = getattr(self, kind)
        bucket["downlink_bytes"] += sum(down)
        bucket["uplink_bytes"] += sum(up)

    def totals(self) -> dict:
        return {
            "training_downlink_bytes": self._cum_down,
            "training_uplink_bytes": self._cum_up,
            "training_total_bytes": self._cum_down + self._cum_up,
            "setup": dict(self.setup),
            "evaluation": dict(self.evaluation),
        }
