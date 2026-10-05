"""Screening rules: the thresholds are absolute differences on the 0-1 scale."""
from src.benchmark import screening as S


def runs_for(central, best, control_central, control_best, regime="amazon_vg",
             metric="held_client_min5_f1_mean", c_metric="ct_client_min5_f1_mean"):
    rows = []
    for seed in (42, 123, 456):
        rows.append({"regime": regime, "method": "centralized", "seed": seed, metric: central, c_metric: control_central})
        for m in S.FL:
            rows.append({"regime": regime, "method": m, "seed": seed, metric: best, c_metric: control_best})
    return rows


def test_thresholds_are_absolute_on_the_metric_scale():
    assert (S.DELTA_MEAN, S.DELTA_TAIL) == (0.02, 0.05)
    # Low-valued 5-class metric: a 0.019 absolute shortfall (~6.3% relative) is NOT material ...
    cell = S.evaluate_cell(runs_for(0.300, 0.281, 0.5, 0.5), "amazon_vg", "held_client_min5_f1_mean",
                           "amazon_vg", "ct_client_min5_f1_mean", False)
    assert cell["delta"] == 0.02 and not cell["c1_material"]
    # ... while a 0.021 absolute shortfall is, whatever it is in relative terms.
    cell = S.evaluate_cell(runs_for(0.900, 0.879, 0.5, 0.5), "amazon_vg", "held_client_min5_f1_mean",
                           "amazon_vg", "ct_client_min5_f1_mean", False)
    assert cell["c1_material"] and cell["c3_heterogeneity_specific"] and cell["candidate"]
