"""Benchmark engine: equivalence with the Flower path, algorithm rules, evaluation.

Uses the SYNTHETIC review file; no number here is a result.
"""
import json

import numpy as np
import pytest

from src.benchmark import algorithms as A
from src.benchmark.engine import run_centralized, run_federated, run_local_only
from src.benchmark.evaluation import per_client_scores
from src.benchmark.state_store import ClientStateStore
from src.metrics.classification import metrics_from_confusion
from src.utils.config import deep_merge, load_config
from src.utils.runtime import prepare_run
from tests.synthetic_reviews import write_reviews


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bench")
    path = tmp / "Synthetic.jsonl"
    write_reviews(path, num_users=80, num_items=12)
    return load_config("exp0b_product.yaml", overrides={
        "data.local_path": str(path), "data.category": "Synthetic", "data.cache_dir": str(tmp / "cache"),
        "data.client_filter.min_reviews": 10, "features.max_features": 300, "features.min_df": 1,
        "partition.dir": str(tmp / "partitions"), "experiment.results_dir": str(tmp / "results"),
        "device": "cpu", "fl.rounds": 3, "fl.fraction_fit": 0.5, "fl.lr": 0.3, "fl.weight_decay": 1e-5,
        "eval.full_every": 2, "centralized.epochs": 2, "local_only.epochs": 2,
        "simulation.ray_cpus": 2,
    }) | {"_tmp": str(tmp)}


def engine_run(cfg, algorithm, name, **fl):
    c = deep_merge({k: v for k, v in cfg.items() if k != "_tmp"}, {"fl": {"algorithm": algorithm, **fl}})
    run_dir = prepare_run(c, 42, f"{cfg['_tmp']}/engine/{name}", create_partition=True)
    return run_federated(run_dir), run_dir


@pytest.mark.parametrize("algorithm", ["fedavg", "scaffold"])
def test_engine_reproduces_the_flower_path_bit_for_bit(cfg, algorithm):
    from src.runner import run_federated as flower_run

    ours, _ = engine_run(cfg, algorithm, f"eq_{algorithm}")
    c = deep_merge({k: v for k, v in cfg.items() if k != "_tmp"}, {"fl": {"algorithm": algorithm}})
    theirs = flower_run(c, 42, f"{cfg['_tmp']}/flower/{algorithm}", create_partition=True)
    assert ours["weights_sha256"] == theirs["weights_sha256"]


def test_every_algorithm_runs_and_writes_the_artifacts(cfg):
    for algorithm, extra in [("fedprox", {"mu": 0.1}), ("fednova", {}), ("fedadam", {"server_lr": 0.1}),
                             ("fedadagrad", {"server_lr": 0.1}), ("fedyogi", {"server_lr": 0.1})]:
        final, run_dir = engine_run(cfg, algorithm, algorithm, **extra)
        assert final["rounds_completed"] == 3
        hist = json.loads((run_dir / "history.json").read_text())
        assert "val_macro_f1" in hist[0] and "unseen_macro_f1" in hist[1] and "unseen_macro_f1" in hist[-1]
        assert final["unseen"]["n"] > 0 and "seen_unseen_gap" in final
        assert "p25" in final["unseen_client_macro_f1"]
        unseen = json.loads((run_dir / "unseen_clients.json").read_text())
        assert sum(u["n"] for u in unseen) == final["unseen"]["n"]


def test_tuning_run_never_evaluates_held_out_data(cfg):
    final, run_dir = engine_run(deep_merge(cfg, {"eval": {"test": False}}), "fedavg", "tuning")
    hist = json.loads((run_dir / "history.json").read_text())
    assert "unseen" not in final and "client_test" not in final
    assert not any(k.startswith(("unseen", "client_test", "test")) for h in hist for k in h)


def test_baselines(cfg):
    c = {k: v for k, v in cfg.items() if k != "_tmp"}
    central = run_centralized(prepare_run(c, 42, f"{cfg['_tmp']}/central", create_partition=True))
    assert central["unseen"]["n"] > 0 and central["epochs"] == 2
    local = run_local_only(prepare_run(c, 42, f"{cfg['_tmp']}/local", create_partition=True))
    assert local["val"]["n"] > 0 and local["unseen_note"]


# ---- update rules on hand-made numbers ------------------------------------------
X = [np.array([[1.0, -2.0]], np.float32), np.array([0.5], np.float32)]
Y1 = [np.array([[2.0, -1.0]], np.float32), np.array([1.5], np.float32)]
Y2 = [np.array([[0.0, 0.0]], np.float32), np.array([0.0], np.float32)]


def one_round(name, fl, steps=(4, 4), n=(30, 10)):
    algo = A.make_algorithm({"algorithm": name, **fl}, X, num_clients=10)
    algo.begin_round(X, [(0, n[0], steps[0]), (1, n[1], steps[1])])
    algo.add_client(0, n[0], {"weights": Y1, "steps": steps[0]})
    algo.add_client(1, n[1], {"weights": Y2, "steps": steps[1]})
    return algo, algo.finish_round(X)


def test_fedavg_and_fednova_coincide_for_equal_steps_and_differ_otherwise():
    _, avg = one_round("fedavg", {})
    _, nova = one_round("fednova", {})
    for a, b in zip(avg, nova):
        assert np.allclose(a, b, atol=1e-6)
    _, nova2 = one_round("fednova", {}, steps=(8, 2))
    p, tau = np.array([0.75, 0.25]), np.array([8, 2])
    expect = [x + (p * tau).sum() * (p[0] * (y1 - x) / 8 + p[1] * (y2 - x) / 2) for x, y1, y2 in zip(X, Y1, Y2)]
    for a, b in zip(nova2, expect):
        assert np.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("name", ["fedadam", "fedyogi", "fedadagrad"])
def test_fedopt_first_step_matches_reddi_et_al(name):
    tau, eta = 1e-3, 0.1
    _, new = one_round(name, {"server_lr": eta})
    b1, b2 = (0.0, 0.0) if name == "fedadagrad" else (0.9, 0.99)
    for x, y1, y2, got in zip(X, Y1, Y2, new):
        d = 0.75 * (y1.astype(np.float64) - x) + 0.25 * (y2.astype(np.float64) - x)
        m = (1 - b1) * d
        v0 = np.full_like(d, tau ** 2)
        v = {"fedadam": b2 * v0 + (1 - b2) * d * d, "fedadagrad": v0 + d * d,
             "fedyogi": v0 - (1 - b2) * d * d * np.sign(v0 - d * d)}[name]
        assert np.allclose(got, x + eta * m / (np.sqrt(v) + tau), atol=1e-6)


def test_client_state_store_round_trip(tmp_path):
    store = ClientStateStore(tmp_path / "s.f32", 5, [(2, 3), (2,)])
    assert all((a == 0).all() for a in store.get(3))
    arrays = [np.arange(6, dtype=np.float32).reshape(2, 3), np.array([7, 8], np.float32)]
    store.put(3, arrays)
    for a, b in zip(store.get(3), arrays):
        assert np.array_equal(a, b)
    assert store.num_written == 1
    store.close()
    assert not (tmp_path / "s.f32").exists()


def test_vectorized_client_scores_match_the_reference():
    rng = np.random.default_rng(0)
    cms = rng.integers(0, 5, size=(20, 5, 5))
    cms[3] = 0
    cms[4, 1:, :] = 0  # single-class client
    scores = per_client_scores(cms)
    for k in range(20):
        if cms[k].sum() == 0:
            assert np.isnan(scores["macro_f1"][k])
            continue
        ref = metrics_from_confusion(cms[k])
        assert np.isclose(scores["macro_f1"][k], ref["macro_f1"]) and np.isclose(scores["accuracy"][k], ref["accuracy"])


def test_client_state_file_layout_is_fixed(tmp_path):
    """Row cid = the client's arrays flattened in order, float32, at byte cid * width * 4."""
    shapes = [(2, 3), (2,)]
    store = ClientStateStore(tmp_path / "s.f32", 4, shapes)
    rows = {1: [np.arange(6, dtype=np.float32).reshape(2, 3), np.array([6, 7], np.float32)],
            3: [np.full((2, 3), -1.5, np.float32), np.array([0.25, 9], np.float32)]}
    for cid, arrays in rows.items():
        store.put(cid, arrays)
    raw = np.fromfile(tmp_path / "s.f32", dtype=np.float32).reshape(4, 8)
    assert (raw[0] == 0).all() and (raw[2] == 0).all()
    for cid, arrays in rows.items():
        assert np.array_equal(raw[cid], np.concatenate([a.ravel() for a in arrays]))
        assert all(np.array_equal(a, b) for a, b in zip(store.get(cid), arrays))
    store.put(1, rows[3])                                    # overwrite in place
    assert np.array_equal(np.fromfile(tmp_path / "s.f32", dtype=np.float32).reshape(4, 8)[1], raw[3])
    store.close()


def test_client_state_is_not_held_in_process_memory(tmp_path):
    """Regression (Job 2, SCAFFOLD on Amazon): a memory-mapped store kept every
    written row resident (peak 8.75 GB after 3 rounds). 200 rows at Amazon width
    (~200 MB) must not grow the process by more than a few rows."""
    import psutil

    shapes = [(5, 50_000), (5,)]
    proc = psutil.Process()
    store = ClientStateStore(tmp_path / "s.f32", 200, shapes)
    base = proc.memory_info().rss
    rng = np.random.default_rng(0)
    for cid in range(200):
        store.put(cid, [rng.standard_normal(s).astype(np.float32) for s in shapes])
        store.get(cid)
    growth_mb = (proc.memory_info().rss - base) / 2**20
    store.close()
    assert growth_mb < 60, f"process grew by {growth_mb:.0f} MB while writing ~190 MB of client state"
