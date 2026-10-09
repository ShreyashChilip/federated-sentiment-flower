"""EXPLORATORY algorithms (docs/EXPLORATORY_CCE_PROTOCOL.md): mathematical properties,
edge cases and engine integration. SYNTHETIC data only; no number here is a result."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.benchmark import algorithms as A
from src.benchmark import exploratory_algorithms as X  # noqa: F401  (registers the algorithms)
from src.benchmark import plan as P
from src.utils.config import deep_merge, load_config

C, F = 3, 6          # classes, features
LR, LAM = 0.3, 1e-3  # a large weight decay makes the decomposition visible in tests


def fl(name, **kw):
    return {"algorithm": name, "lr": LR, "weight_decay": LAM, "momentum": 0.0, "local_epochs": 1, **kw}


def x0(seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(C, F)).astype(np.float32), rng.normal(size=C).astype(np.float32)]


def omega(steps):
    return float(np.expm1(steps * np.log1p(-LR * LAM)))


def client(x, steps, support, dW=None, db=None):
    """Client output: weight decay on every coordinate + a data part on its support (and bias)."""
    om = omega(steps)
    W = x[0].astype(np.float64) * (1 + om)
    b = x[1].astype(np.float64) * (1 + om)
    if dW is not None:
        mask = np.zeros(F, bool)
        mask[support] = True
        W = W + np.asarray(dW, np.float64) * mask
    if db is not None:
        b = b + np.asarray(db, np.float64)
    return {"weights": [W.astype(np.float32), b.astype(np.float32)], "steps": steps, "support": np.asarray(support)}


def run(name, x, clients, ns, **kw):
    algo = A.make_algorithm(fl(name, **kw), x, num_clients=len(clients))
    algo.begin_round(x, [(i, n, c["steps"]) for i, (c, n) in enumerate(zip(clients, ns))])
    for i, (c, n) in enumerate(zip(clients, ns)):
        algo.add_client(i, n, c)
    return algo, algo.finish_round(x)


def fedavg_step(x, clients, ns):
    p = np.asarray(ns, float) / sum(ns)
    return [sum(pi * (c["weights"][k].astype(np.float64) - x[k].astype(np.float64)) for pi, c in zip(p, clients)) for k in (0, 1)]


def step(new, x):
    return [n.astype(np.float64) - o.astype(np.float64) for n, o in zip(new, x)]


# ---- FedExP ---------------------------------------------------------------------
def test_fedexp_matches_the_published_rule_with_size_weights():
    x = x0()
    rng = np.random.default_rng(1)
    outs = [{"weights": [x[0] + rng.normal(size=(C, F)).astype(np.float32), x[1] + rng.normal(size=C).astype(np.float32)], "steps": 3}
            for _ in range(3)]
    ns, eps = [10, 20, 30], 0.01
    algo, new = run("fedexp", x, outs, ns, fedexp_eps=eps)
    p = np.array(ns) / 60
    d = [[o["weights"][k].astype(np.float64) - x[k] for k in (0, 1)] for o in outs]
    dbar = [sum(pi * di[k] for pi, di in zip(p, d)) for k in (0, 1)]
    num = sum(pi * sum(float((dk ** 2).sum()) for dk in di) for pi, di in zip(p, d))
    eta = max(1.0, num / (2 * (sum(float((v ** 2).sum()) for v in dbar) + eps)))
    assert algo.last_eta == pytest.approx(eta)
    for got, xx, v in zip(new, x, dbar):
        assert np.allclose(got, xx + eta * v, atol=1e-5)


def test_fedexp_reduces_to_fedavg_for_large_eps_and_reports_two_iterate_average():
    x = x0()
    outs = [{"weights": [x[0] + 0.1, x[1] - 0.1], "steps": 2}, {"weights": [x[0] - 0.3, x[1] + 0.2], "steps": 2}]
    algo, new = run("fedexp", x, outs, [1, 1], fedexp_eps=1e9)
    assert algo.last_eta == 1.0
    fa = fedavg_step(x, outs, [1, 1])
    for k in (0, 1):
        assert np.allclose(new[k], x[k] + fa[k], atol=1e-6)
    avg = algo.evaluation_weights(new)
    for k in (0, 1):
        assert np.allclose(avg[k], (new[k].astype(np.float64) + x[k]) / 2, atol=1e-6)


# ---- coverage decomposition: properties ----------------------------------------
def test_disjoint_equal_magnitude_coverage_gives_the_covering_clients_update():
    x = x0()
    a = client(x, 4, [0, 1], dW=np.full((C, F), 0.5), db=np.zeros(C))
    b = client(x, 4, [2, 3], dW=np.full((C, F), -0.5), db=np.zeros(C))
    ns = [10, 30]
    Om = omega(4)
    for name in ("cav", "cce"):
        algo, new = run(name, x, [a, b], ns)
        s = step(new, x)[0]
        assert np.allclose(s[:, [0, 1]], Om * x[0][:, [0, 1]] + 0.5, atol=1e-6)   # full update, not 1/4 of it
        assert np.allclose(s[:, [2, 3]], Om * x[0][:, [2, 3]] - 0.5, atol=1e-6)
        assert np.allclose(s[:, [4, 5]], Om * x[0][:, [4, 5]], atol=1e-7)         # absent: weight decay only
    _, new = run("cce", x, [a, b], ns)
    # eta = 1 / coverage when magnitudes are equal: coverage of features 0,1 is 0.25
    assert np.allclose(step(new, x)[0][:, 0] - Om * x[0][:, 0], 0.5, atol=1e-6)


def test_full_agreement_gives_fedavg():
    x = x0()
    cl = [client(x, 4, list(range(F)), dW=np.full((C, F), 0.2), db=np.full(C, 0.1)) for _ in range(3)]
    fa = fedavg_step(x, cl, [1, 2, 3])
    for name in ("cav", "cce", "cce_unit"):
        _, new = run(name, x, cl, [1, 2, 3])
        for k in (0, 1):
            assert np.allclose(step(new, x)[k], fa[k], atol=1e-6)


def test_equal_magnitude_conflict_is_not_amplified():
    x = x0()
    a = client(x, 4, [0], dW=np.full((C, F), 0.4), db=np.zeros(C))
    b = client(x, 4, [0], dW=np.full((C, F), -0.4), db=np.zeros(C))
    o = client(x, 4, [5], dW=np.zeros((C, F)), db=np.zeros(C))
    for name in ("cav", "cce"):
        _, new = run(name, x, [a, b, o], [1, 1, 2])
        assert np.allclose(step(new, x)[0][:, 0], omega(4) * x[0][:, 0], atol=1e-6)   # mean among covering = 0


def test_magnitude_dispersion_is_what_cce_adds_beyond_cav():
    """Two covering clients, same sign, |d| = 1 and 3 (equal weights), half of the mass absent:
    CAV -> mean 2; CCE -> r * mean with r = E[d^2]/(E|d|)^2 = 5/4 -> 2.5; bound max|d| = 3."""
    x = x0()
    a = client(x, 4, [1], dW=np.full((C, F), 1.0), db=np.zeros(C))
    b = client(x, 4, [1], dW=np.full((C, F), 3.0), db=np.zeros(C))
    o = client(x, 4, [5], dW=np.zeros((C, F)), db=np.zeros(C))
    data = lambda new: step(new, x)[0][:, 1] - omega(4) * x[0][:, 1]
    _, cav = run("cav", x, [a, b, o], [1, 1, 2])
    algo, cce = run("cce", x, [a, b, o], [1, 1, 2])
    assert np.allclose(data(cav), 2.0, atol=1e-5)
    assert np.allclose(data(cce), 2.5, atol=1e-5)
    assert algo.summary()["r_median"] is not None


def test_cce_step_never_exceeds_the_largest_client_update():
    rng = np.random.default_rng(3)
    x = x0()
    for trial in range(20):
        cl = [client(x, int(rng.integers(1, 9)), sorted(rng.choice(F, size=rng.integers(1, F), replace=False)),
                     dW=rng.normal(size=(C, F)) * rng.lognormal(size=(C, F)), db=rng.normal(size=C)) for _ in range(5)]
        ns = list(rng.integers(1, 50, size=5))
        _, new = run("cce", x, cl, ns)
        s = step(new, x)
        Om = sum(n * omega(c["steps"]) for c, n in zip(cl, ns)) / sum(ns)
        dmax = [np.max([np.abs(c["weights"][k].astype(np.float64) - x[k] - omega(c["steps"]) * x[k]) for c in cl], axis=0) for k in (0, 1)]
        for k in (0, 1):
            assert (np.abs(s[k] - Om * x[k].astype(np.float64)) <= dmax[k] + 1e-5).all()


def test_zero_denominator_and_absent_features_are_safe():
    x = x0()
    a = client(x, 2, [0, 1], dW=np.zeros((C, F)), db=np.zeros(C))   # covered but zero data update -> A = 0
    for name in ("cav", "cce"):
        algo, new = run(name, x, [a], [5])
        s = step(new, x)
        assert all(np.isfinite(v).all() for v in s)
        assert np.allclose(s[0], omega(2) * x[0], atol=1e-6)


def test_bias_cav_equals_fedavg_and_cce_scales_the_bias_by_r():
    x = x0()
    a = client(x, 4, [0], dW=np.zeros((C, F)), db=np.full(C, 1.0))
    b = client(x, 4, [1], dW=np.zeros((C, F)), db=np.full(C, 3.0))
    fa = fedavg_step(x, [a, b], [1, 1])[1]
    _, cav = run("cav", x, [a, b], [1, 1])
    _, cce = run("cce", x, [a, b], [1, 1])
    assert np.allclose(step(cav, x)[1], fa, atol=1e-6)
    om_b = omega(4) * x[1].astype(np.float64)
    assert np.allclose(step(cce, x)[1] - om_b, 1.25 * 2.0, atol=1e-5)   # pure form: bias multiplied by r = 1.25


def test_single_client_is_fedavg():
    x = x0()
    c = client(x, 3, [0, 2, 4], dW=np.random.default_rng(5).normal(size=(C, F)), db=np.ones(C))
    fa = fedavg_step(x, [c], [7])
    for name in ("cav", "cce", "cce_unit"):
        _, new = run(name, x, [c], [7])
        for k in (0, 1):
            assert np.allclose(step(new, x)[k], fa[k], atol=1e-6)


def test_weight_decay_is_removed_exactly_on_coordinates_without_data_gradient():
    """Real torch SGD: absent features move only by weight decay. CAV/CCE apply weight decay at
    the FedAvg rate there and treat the coordinate as uncovered (no float32 residue leaks in)."""
    x = x0()
    outs = []
    for steps, sup in ((5, [0, 1]), (40, [2])):
        p = torch.nn.Parameter(torch.tensor(x[0].copy()))
        q = torch.nn.Parameter(torch.tensor(x[1].copy()))
        opt = torch.optim.SGD([p, q], lr=LR, momentum=0.0, weight_decay=LAM)
        g = torch.zeros(C, F)
        g[:, sup] = 0.05
        for _ in range(steps):
            opt.zero_grad()
            p.grad, q.grad = g.clone(), torch.full((C,), 0.01)
            opt.step()
        outs.append({"weights": [p.detach().numpy().copy(), q.detach().numpy().copy()], "steps": steps, "support": np.array(sup)})
    fa = fedavg_step(x, outs, [1, 1])
    for name in ("cav", "cce", "cce_unit"):
        _, new = run(name, x, outs, [1, 1])
        s = step(new, x)
        Om = (omega(5) + omega(40)) / 2
        assert np.allclose(s[0][:, [3, 4, 5]], Om * x[0][:, [3, 4, 5]], rtol=0, atol=1e-6)   # never touched by data
        assert np.allclose(s[0][:, [3, 4, 5]], fa[0][:, [3, 4, 5]], atol=1e-6)               # same as FedAvg there


def test_cce_unit_equals_fedavg_on_random_updates():
    rng = np.random.default_rng(7)
    x = x0()
    cl = [client(x, int(rng.integers(1, 30)), sorted(rng.choice(F, size=3, replace=False)), dW=rng.normal(size=(C, F)) * 0.1,
                 db=rng.normal(size=C) * 0.1) for _ in range(6)]
    ns = list(rng.integers(5, 60, size=6))
    algo, new = run("cce_unit", x, cl, ns)
    fa = fedavg_step(x, cl, ns)
    for k in (0, 1):
        assert np.allclose(step(new, x)[k], fa[k], atol=1e-6)
    assert algo.summary()["max_abs_gap_to_fedavg_step"] < 1e-6


def test_guards():
    x = x0()
    with pytest.raises(ValueError, match="momentum"):
        A.make_algorithm(fl("cce", momentum=0.9), x, 2)
    with pytest.raises(ValueError, match="epoch"):
        A.make_algorithm(fl("cav", local_epochs=2), x, 2)
    algo = A.make_algorithm(fl("cav"), x, 2)
    algo.begin_round(x, [(0, 5, 2)])
    with pytest.raises(KeyError, match="support"):
        algo.add_client(0, 5, {"weights": x, "steps": 2})


# ---- engine integration (SYNTHETIC natural clients) -------------------------------
@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    from tests.synthetic_reviews import write_reviews

    tmp = tmp_path_factory.mktemp("explore")
    path = tmp / "Synthetic.jsonl"
    write_reviews(path, num_users=80, num_items=12)
    c = load_config("exploratory/amazon_vg_tuned.yaml", overrides={
        "data.local_path": str(path), "data.category": "Synthetic", "data.cache_dir": str(tmp / "cache"),
        "data.client_filter.min_reviews": 10, "features.max_features": 300, "features.min_df": 1,
        "partition.dir": str(tmp / "partitions"), "device": "cpu", "fl.rounds": 3, "fl.fraction_fit": 0.5,
        "eval.test": False, "eval.full_every": 2})
    from src.analysis.exp0 import cell_config

    return cell_config(c, "natural") | {"_tmp": str(tmp)}


def engine(cfg, name, **flkw):
    from src.benchmark.engine import run_federated
    from src.utils.runtime import prepare_run

    c = deep_merge({k: v for k, v in cfg.items() if k != "_tmp"}, {"fl": {"algorithm": name, **flkw}})
    run_dir = Path(cfg["_tmp"]) / "runs" / (name + "_".join(map(str, flkw.values())))
    if (run_dir / "final.json").exists():          # finished runs are immutable: reuse within the module
        return json.loads((run_dir / "final.json").read_text()), run_dir
    prepare_run(c, 7, run_dir, create_partition=True)
    return run_federated(run_dir), run_dir


def test_engine_runs_every_exploratory_algorithm(cfg):
    base, _ = engine(cfg, "fedavg")
    for name, kw in (("cav", {}), ("cce", {}), ("fedexp", {"fedexp_eps": 0.01})):
        final, run_dir = engine(cfg, name, **kw)
        hist = json.loads((run_dir / "history.json").read_text())
        assert final["rounds_completed"] == 3 and not final["diverged"]
        assert final["communication"]["uplink_bytes_per_message"] >= base["communication"]["uplink_bytes_per_message"]
        if name == "fedexp":
            assert final["reported_model"] == "average_of_last_two_iterates"
            assert "val_macro_f1_last_iterate" in hist[0] and (run_dir / "last_iterate_weights.npz").exists()
            assert hist[0]["algo_fedexp_eta_g"] >= 1.0
        else:
            assert (run_dir / "algo_diagnostics" / "round001.npz").exists()
            assert hist[0]["algo_features_covered"] > 0 and hist[0]["algo_r_median"] >= 1.0


def test_engine_cce_unit_matches_fedavg(cfg):
    _, a = engine(cfg, "fedavg")
    _, b = engine(cfg, "cce_unit")
    wa, wb = np.load(a / "final_weights.npz"), np.load(b / "final_weights.npz")
    for k in wa.files:
        assert np.allclose(wa[k], wb[k], atol=1e-5)
    hist = json.loads((b / "history.json").read_text())
    assert max(h["algo_max_abs_gap_to_fedavg_step"] for h in hist) < 1e-5


# ---- protocol isolation -------------------------------------------------------------
def test_exploratory_plan_is_isolated_and_validation_only():
    ex = P.load_benchmark("exploratory/cce_v1.yaml")
    specs = P.plan_stage(ex, "explore")
    assert len(specs) == 28 and {s["seed"] for s in specs} == {7}
    assert all(s["cfg"]["eval"]["test"] is False for s in specs)
    assert all("exploratory" in s["run_dir"] and "explore_cce_v1" in s["cfg"]["experiment"]["name"] for s in specs)
    assert 7 not in P.evaluation_seeds(ex)
    official = P.load_benchmark("benchmark.yaml")
    assert official["algorithm_grids"]["fedavg"] == {} and "fedexp" not in official["algorithm_grids"]
    assert official["results_dir"] == "results/benchmark"


def test_exploratory_regimes_use_the_official_data_and_hyperparameters():
    from src.data.prepare import bundle_dir

    ex, off = P.load_benchmark("exploratory/cce_v1.yaml"), P.load_benchmark("benchmark.yaml")
    for reg in ("amazon_vg", "yelp_dir01"):
        e, o = P.regime_config(ex, reg), P.regime_config(off, reg, untuned=True)
        if reg == "amazon_vg":
            o = deep_merge(o, {"fl": {"lr": 0.3, "weight_decay": 1e-6}})   # official tune_base selection
        assert bundle_dir(e) == bundle_dir(o)
        for section in ("data", "features", "partition", "model", "fl", "eval"):
            assert e[section] == o[section], section


# ---- pre-registered decision rules (docs/EXPLORATORY_CCE_PROTOCOL.md, section 5) -------
def _v(slr=(0.2, 0.25, 0.3, 0.28), exp=(0.3,) * 5, cav=0.57, cce=0.57, yogi3=0.56):
    return {"fedavg_slr": list(slr), "fedexp": list(exp), "cav": cav, "cce": cce, "fedyogi_lr3": yogi3}


def test_decision_rules_cover_every_outcome():
    from src.benchmark import exploratory_decision as D

    ok = {"cav": True, "cce": True}
    assert D.amazon_decision(_v(slr=(0.56, 0.3, 0.3, 0.3)), ok)["outcome"] == "abandon_method_line"      # H1: global step suffices
    assert D.amazon_decision(_v(exp=(0.55,) * 5), ok)["H1_global_step_suffices"]
    assert D.amazon_decision(_v(slr=(0.55 - 1e-9,) * 4, cav=0.565, cce=0.565), ok)["outcome"] == "abandon_method_line"  # H2 fails
    assert D.amazon_decision(_v(cav=0.50, cce=0.50), ok)["outcome"] == "abandon_method_line"          # far below FedYogi*
    assert D.amazon_decision(_v(), {"cav": False, "cce": False})["outcome"] == "abandon_method_line"  # unstable
    r = D.amazon_decision(_v(cav=0.572, cce=0.575), ok)
    assert r["H3_r_factor"] == "no_added_value" and r["outcome"] == "proceed_cav_only_mechanism_study"
    r = D.amazon_decision(_v(cav=0.565, cce=0.576), ok)
    assert r["H3_r_factor"] == "adds_value" and r["outcome"] == "proceed_cav_and_cce"
    assert D.amazon_decision(_v(cav=0.575, cce=0.560), ok)["H3_r_factor"] == "harmful"
    assert D.amazon_decision(_v(cav=0.562, cce=0.562), ok)["outcome"] == "inconclusive"               # not competitive, not abandoned
    assert D.amazon_decision(_v(yogi3=0.60, cav=0.59, cce=0.59), ok)["FedYogi_best"] == 0.60


def test_integrity_and_specificity_rules():
    from src.benchmark import exploratory_decision as D

    assert D.integrity("amazon_vg", D.OFFICIAL_SHA["amazon_vg"], 0.1753, 0.1760)["ok"]
    assert not D.integrity("amazon_vg", "0" * 64, 0.1753, 0.1753)["ok"]
    assert not D.integrity("yelp_dir01", D.OFFICIAL_SHA["yelp_dir01"], 0.748, 0.752)["ok"]
    assert D.yelp_specificity(0.85, 0.80)["contradicts_coverage_explanation"]
    assert not D.yelp_specificity(0.76, 0.70)["contradicts_coverage_explanation"]
