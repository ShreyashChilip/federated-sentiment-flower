"""Convergence-to-target measurements."""
from __future__ import annotations


def rounds_to_target(history: list[dict], key: str, target: float, patience: int = 1):
    """First round at which ``key`` is >= target for ``patience`` evaluations in a row.

    Returns ``None`` if the target is never reached; that outcome is reported
    as such rather than replaced by the last round.
    """
    streak = 0
    for i, row in enumerate(history):
        value = row.get(key)
        streak = streak + 1 if value is not None and value >= target else 0
        if streak >= patience:
            return history[i - patience + 1]["round"]
    return None


def bytes_to_target(history: list[dict], key: str, target: float, patience: int = 1):
    """Cumulative uplink+downlink bytes spent up to the round the target is reached."""
    reached = rounds_to_target(history, key, target, patience)
    if reached is None:
        return None
    for row in history:
        if row["round"] == reached:
            return row["cum_total_bytes"]
    return None
