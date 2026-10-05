"""Mechanical evaluation of the pre-registered screening rules.

Implements section 6 of docs/BENCHMARK_PROTOCOL.md exactly; written before
any benchmark result. Input: the per-run rows of ``analysis.run_row``.
"""
from __future__ import annotations

import numpy as np

FL = ["fedavg", "fedprox", "fednova", "scaffold", "fedadam", "fedadagrad", "fedyogi"]
# Absolute differences on the 0-1 metric scale (0.02 = 2 percentage points), the
# same for every regime; not relative margins (protocol section 6).
DELTA_MEAN, DELTA_TAIL = 0.02, 0.05

# (regime, metric, control regime, control metric, is_tail)
CELLS = [
    ("yelp_dir01", "held_macro_f1", "yelp_iid", "held_macro_f1", False),
    ("yelp_dir01", "ct_client_min5_f1_mean", "yelp_iid", "ct_client_min5_f1_mean", False),
    ("yelp_dir01", "ct_client_min5_f1_p10", "yelp_iid", "ct_client_min5_f1_p10", True),
    ("yelp_dir01", "ct_client_min5_f1_worst", "yelp_iid", "ct_client_min5_f1_worst", True),
    ("amazon_vg", "held_macro_f1", "amazon_vg", "ct_macro_f1", False),
    ("amazon_vg", "held_client_min5_f1_mean", "amazon_vg", "ct_client_min5_f1_mean", False),
    ("amazon_vg", "held_client_min5_f1_p10", "amazon_vg", "ct_client_min5_f1_p10", True),
    ("amazon_vg", "held_client_min5_f1_worst", "amazon_vg", "ct_client_min5_f1_worst", True),
]


def _by_seed(runs, regime, method, metric) -> dict:
    return {r["seed"]: r[metric] for r in runs if r["regime"] == regime and r["method"] == method
            and r.get(metric) is not None and np.isfinite(r[metric])}


def shortfalls(runs, regime, metric) -> dict:
    """method -> {seed: centralized - method}."""
    central = _by_seed(runs, regime, "centralized", metric)
    out = {}
    for m in FL:
        vals = _by_seed(runs, regime, m, metric)
        out[m] = {s: central[s] - v for s, v in vals.items() if s in central}
    return out


def evaluate_cell(runs, regime, metric, c_regime, c_metric, tail) -> dict | None:
    delta = DELTA_TAIL if tail else DELTA_MEAN
    sf = shortfalls(runs, regime, metric)
    sf = {m: v for m, v in sf.items() if v}
    if not sf:
        return None
    means = {m: float(np.mean(list(v.values()))) for m, v in sf.items()}
    best = min(means, key=means.get)
    ctrl = shortfalls(runs, c_regime, c_metric).get(best) or {}
    seeds = sorted(set(sf[best]) & set(ctrl))
    excess = float(np.mean([sf[best][s] - ctrl[s] for s in seeds])) if seeds else None
    c1 = means[best] >= delta
    c2 = all(v > 0 for v in sf[best].values())
    c3 = excess is not None and excess >= delta
    c4 = all(v >= delta / 2 for v in means.values())
    return {"regime": regime, "metric": metric, "control": f"{c_regime}:{c_metric}", "delta": delta,
            "best_fl_method": best, "best_mean_shortfall": means[best],
            "best_shortfall_per_seed": sf[best], "control_shortfall_mean": float(np.mean(list(ctrl.values()))) if ctrl else None,
            "heterogeneity_excess": excess, "mean_shortfall_by_method": means,
            "c1_material": c1, "c2_consistent": c2, "c3_heterogeneity_specific": c3, "c4_unsolved": c4,
            "candidate": bool(c1 and c2 and c3 and c4)}


def failure_patterns(runs) -> list[dict]:
    """Per FL method vs FedAvg: average-up/tail-down and seen-only improvement (all seeds same direction)."""
    out = []
    for regime in sorted({r["regime"] for r in runs}):
        tails = [("held_macro_f1", "held_client_min5_f1_p10"), ("held_macro_f1", "ct_client_min5_f1_p10")]
        for m in FL[1:]:
            for pooled, tail in tails:
                a_p, b_p = _by_seed(runs, regime, m, pooled), _by_seed(runs, regime, "fedavg", pooled)
                a_t, b_t = _by_seed(runs, regime, m, tail), _by_seed(runs, regime, "fedavg", tail)
                seeds = sorted(set(a_p) & set(b_p) & set(a_t) & set(b_t))
                if seeds:
                    up = all(a_p[s] > b_p[s] for s in seeds)
                    down = all(a_t[s] < b_t[s] for s in seeds)
                    out.append({"regime": regime, "method": m, "pattern": f"average_up_tail_down:{pooled}/{tail}",
                                "holds": bool(up and down), "seeds": len(seeds)})
            a_s, b_s = _by_seed(runs, regime, m, "ct_client_min5_f1_mean"), _by_seed(runs, regime, "fedavg", "ct_client_min5_f1_mean")
            a_u, b_u = _by_seed(runs, regime, m, "held_client_min5_f1_mean"), _by_seed(runs, regime, "fedavg", "held_client_min5_f1_mean")
            seeds = sorted(set(a_s) & set(b_s) & set(a_u) & set(b_u))
            if seeds:
                out.append({"regime": regime, "method": m, "pattern": "seen_only_improvement",
                            "holds": bool(all(a_s[s] > b_s[s] for s in seeds) and not all(a_u[s] > b_u[s] for s in seeds)),
                            "seeds": len(seeds)})
    return out


def screen(runs) -> dict:
    cells = [c for c in (evaluate_cell(runs, *spec) for spec in CELLS) if c is not None]
    ranked = sorted([c for c in cells if c["candidate"]], key=lambda c: -c["heterogeneity_excess"])
    return {"cells": cells, "candidates_ranked": ranked, "failure_patterns": failure_patterns(runs),
            "outcome": "candidate" if ranked else "no_candidate"}


def screen_stage(stage_kind: str, runs) -> dict:
    """Apply the screening rules only to a scientific evaluation stage.

    Pilot runs (seed 0, a few rounds) and tuning runs (seed 7, validation only)
    are not screening evidence; for them the outcome is ``not_applicable`` and
    no cell or candidate is computed.
    """
    if stage_kind != "evaluate":
        return {"outcome": "not_applicable", "stage_kind": stage_kind, "cells": [], "candidates_ranked": [],
                "failure_patterns": [],
                "reason": "screening rules apply only to evaluation stages (screening/confirmation)"}
    return {**screen(runs), "stage_kind": stage_kind}
