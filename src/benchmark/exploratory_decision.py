"""Pre-registered decision rules of docs/EXPLORATORY_CCE_PROTOCOL.md, section 5.

Pure functions over final validation macro-F1 values (seed 7). Written and
tested before any exploratory run; thresholds must not be edited afterwards.
"""
from __future__ import annotations

H1_SCALAR_ENOUGH = 0.55
H2_COVERAGE_MARGIN = 0.03
H3_R_MARGIN = 0.01
COMPETITIVE_MARGIN = 0.005
ABANDON_MARGIN = 0.02
STABILITY_STD = 0.02
SENTINEL_F1_TOL = 0.002
SPECIFICITY_GAIN = 0.10

OFFICIAL_SHA = {
    "amazon_vg": "d83ab0edceb35ff6aad68fb49659602ca171dae01c2fd3fb109b6a21bc88e149",
    "yelp_dir01": "cc0126c555f46f4d4ab1b2244e5110f0c5beab75c2bf4ad2f5a33bdfb0caf25e",
}
OFFICIAL_V = {  # official seed-7 tuning runs (validation macro-F1)
    "amazon_vg": {"fedavg": 0.175341, "fedyogi": 0.575171},
    "yelp_dir01": {"fedavg": 0.748031, "fedyogi": 0.940801},
}


def integrity(regime: str, sentinel_sha: str | None, v_fedavg1: float | None, v_unit: float | None) -> dict:
    s1 = sentinel_sha == OFFICIAL_SHA[regime]
    s2 = v_fedavg1 is not None and v_unit is not None and abs(v_unit - v_fedavg1) <= SENTINEL_F1_TOL
    return {"S1_sentinel_bit_identical": bool(s1), "S2_unit_matches_fedavg": bool(s2), "ok": bool(s1 and s2)}


def amazon_decision(v: dict, stable: dict) -> dict:
    """v: {'fedavg_slr': [...], 'fedexp': [...], 'cav', 'cce', 'fedyogi_lr3'}; stable: {'cav': bool, 'cce': bool}."""
    c1 = max(list(v["fedavg_slr"]) + list(v["fedexp"]))
    yogi = max(OFFICIAL_V["amazon_vg"]["fedyogi"], v["fedyogi_lr3"])
    h1 = c1 >= H1_SCALAR_ENOUGH
    h2 = v["cav"] - c1 >= H2_COVERAGE_MARGIN
    diff = v["cce"] - v["cav"]
    h3 = "adds_value" if diff >= H3_R_MARGIN else ("harmful" if diff <= -H3_R_MARGIN else "no_added_value")
    best_name = "cce" if v["cce"] > v["cav"] else "cav"
    best = v[best_name]
    competitive = best >= yogi - COMPETITIVE_MARGIN
    is_stable = bool(stable.get(best_name, False))
    if h1 or not h2 or best < yogi - ABANDON_MARGIN or not is_stable:
        outcome = "abandon_method_line"
    elif competitive:
        outcome = "proceed_cav_and_cce" if h3 == "adds_value" else "proceed_cav_only_mechanism_study"
    else:
        outcome = "inconclusive"
    return {"C1_best": c1, "FedYogi_best": yogi, "H1_global_step_suffices": h1, "H2_coverage_dilution": h2,
            "H3_r_factor": h3, "cce_minus_cav": diff, "best_candidate": best_name, "competitive": competitive,
            "stable": is_stable, "outcome": outcome}


def yelp_specificity(v_cav: float, v_cce: float) -> dict:
    base = OFFICIAL_V["yelp_dir01"]["fedavg"]
    gain = max(v_cav, v_cce) - base
    return {"gain_over_fedavg": gain, "contradicts_coverage_explanation": gain >= SPECIFICITY_GAIN}
