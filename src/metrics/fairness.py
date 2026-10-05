"""Dispersion of a metric across clients."""
from __future__ import annotations

import numpy as np


def client_dispersion(values) -> dict:
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=np.float64)
    if len(v) == 0:
        return {"num_clients": 0}
    return {
        "num_clients": int(len(v)),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "p10": float(np.percentile(v, 10)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
        "worst": float(v.min()),
        "best": float(v.max()),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "spread": float(v.max() - v.min()),
    }
