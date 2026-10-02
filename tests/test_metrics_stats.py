import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.communication.accounting import CommLedger
from src.metrics.classification import confusion_matrix, metrics_from_confusion
from src.metrics.convergence import bytes_to_target, rounds_to_target
from src.metrics.fairness import client_dispersion
from src.statistics.stats import holm_correction, paired_comparison, summarize
from src.strategies.server import draw_events, select_clients
from src.utils.config import load_config
from src.utils.seeding import derive_seed


def test_metrics_match_scikit_learn():
    rng = np.random.default_rng(0)
    for k in (2, 5):
        y, p = rng.integers(0, k, 500), rng.integers(0, k, 500)
        m = metrics_from_confusion(confusion_matrix(y, p, k))
        assert np.isclose(m["accuracy"], accuracy_score(y, p))
        assert np.isclose(m["macro_f1"], f1_score(y, p, average="macro"))
        assert np.isclose(m["macro_precision"], precision_score(y, p, average="macro", zero_division=0))
        assert np.isclose(m["macro_recall"], recall_score(y, p, average="macro"))


def test_macro_f1_on_single_class_client_uses_present_classes_only():
    m = metrics_from_confusion(confusion_matrix([1, 1, 1, 1], [1, 1, 1, 0], 2))
    assert np.isclose(m["macro_f1"], 2 * 1.0 * 0.75 / 1.75)


def test_client_dispersion():
    d = client_dispersion([0.9, 0.8, 0.5, float("nan"), None])
    assert d["num_clients"] == 3 and d["worst"] == 0.5 and np.isclose(d["spread"], 0.4)


def test_rounds_and_bytes_to_target():
    hist = [{"round": r, "f1": f, "cum_total_bytes": 100 * r} for r, f in enumerate([0.5, 0.7, 0.65, 0.72, 0.8], 1)]
    assert rounds_to_target(hist, "f1", 0.7) == 2
    assert rounds_to_target(hist, "f1", 0.7, patience=2) == 4
    assert rounds_to_target(hist, "f1", 0.95) is None
    assert bytes_to_target(hist, "f1", 0.7) == 200 and bytes_to_target(hist, "f1", 0.95) is None


def test_comm_ledger_counts_participants_and_directions():
    ledger = CommLedger()
    row = ledger.record_round(1, down=[100, 100, 100], up=[90, 90])  # one client dropped
    assert row["downlink_bytes"] == 300 and row["uplink_bytes"] == 180
    assert row["downlink_clients"] == 3 and row["uplink_clients"] == 2
    ledger.record_round(2, down=[100], up=[90])
    assert ledger.totals()["training_total_bytes"] == 300 + 180 + 100 + 90
    ledger.record_other("evaluation", [50], [5])
    assert ledger.totals()["training_total_bytes"] == 670  # evaluation kept separate


def test_client_selection_is_seeded_and_sized():
    eligible = list(range(100))
    a = select_clients(eligible, 0.25, 1, seed=42, rnd=3)
    assert a == select_clients(eligible, 0.25, 1, seed=42, rnd=3)
    assert a != select_clients(eligible, 0.25, 1, seed=42, rnd=4)
    assert len(a) == 25 and len(set(a)) == 25
    assert len(select_clients(eligible, 1.0, 1, 42, 1)) == 100
    assert len(select_clients([1, 2, 3], 0.1, 2, 42, 1)) == 2


def test_dropout_rate_is_close_to_p_and_zero_when_disabled():
    selected = list(range(1000))
    assert draw_events(selected, 0.0, "dropout", 1, 1) == set()
    rate = np.mean([len(draw_events(selected, 0.1, "dropout", 1, r)) for r in range(20)]) / 1000
    assert abs(rate - 0.1) < 0.01


def test_summarize_mean_std_ci():
    s = summarize([0.80, 0.82, 0.84, 0.86, 0.88])
    assert np.isclose(s["mean"], 0.84) and np.isclose(s["std"], np.std([0.80, 0.82, 0.84, 0.86, 0.88], ddof=1))
    # t(0.975, df=4) = 2.776
    assert np.isclose(s["ci95_high"] - s["mean"], 2.776 * s["std"] / np.sqrt(5), atol=1e-3)
    assert summarize([0.5])["std"] is None


def test_paired_comparison_reports_wilcoxon_floor_for_five_seeds():
    a = [0.81, 0.83, 0.85, 0.87, 0.89]
    b = [0.80, 0.82, 0.84, 0.86, 0.88]
    out = paired_comparison(a, b)
    assert np.isclose(out["mean_diff"], 0.01)
    # all five differences positive: the smallest two-sided exact p is 2 / 2^5
    assert np.isclose(out["wilcoxon_p"], 0.0625)
    assert np.isclose(out["wilcoxon_min_p"], 0.0625)
    assert out["n"] == 5
    same = paired_comparison(a, a)
    assert same["wilcoxon_p"] is None and same["mean_diff"] == 0


def test_holm_correction():
    adj = holm_correction([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_config_inheritance_and_seed_derivation():
    cfg = load_config("smoke.yaml", overrides={"fl.mu": 0.1})
    assert cfg["fl"]["mu"] == 0.1 and cfg["partition"]["num_clients"] == 3
    assert cfg["fl"]["batch_size"] == 32  # inherited from base.yaml
    assert derive_seed("a", 1) == derive_seed("a", 1) != derive_seed("a", 2)


def test_laptop_is_never_reported_as_kaggle(monkeypatch):
    from src.utils import env

    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    assert env.platform_kind() == "local"
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr(env.sys, "platform", "win32")
    assert env.platform_kind() == "local"
    monkeypatch.setattr(env.sys, "platform", "linux")
    assert env.platform_kind() == "kaggle"
