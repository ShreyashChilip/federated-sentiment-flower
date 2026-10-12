"""Tests of the exploratory DistilBERT FL script with a tiny CPU backend (no transformers needed)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("distilbert_fl", ROOT / "experiments" / "distilbert_fl.py")
m = importlib.util.module_from_spec(spec)
sys.modules["distilbert_fl"] = m
spec.loader.exec_module(m)

from src.data.roles import CLIENT_TEST, TRAIN, VAL, assign_roles  # noqa: E402

POS, NEG, FILL = ["good", "great", "love", "fine"], ["bad", "awful", "hate", "poor"], [f"w{i}" for i in range(60)]


def synth(n, seed, k=2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    texts = [" ".join(list(rng.choice(POS if t else NEG, 2)) + list(rng.choice(FILL, rng.integers(1, 8)))) for t in y]
    return texts, y.astype(np.int64)


def yelp_like(n=1200):
    texts, y = synth(n, 0)
    tt, ty = synth(200, 1)
    return {"texts": texts, "labels": y, "roles": assign_roles(y, [0.8, 0.1, 0.1], 3), "codes": None,
            "test_texts": tt, "test_labels": ty, "test_rows": None, "num_classes": 2, "meta": {"synthetic": True}}


def amazon_like(n=900, clients=30):
    texts, y = synth(n, 2)
    codes = np.sort(np.random.default_rng(3).integers(0, clients, n))
    roles = np.empty(n, np.int8)
    for c in range(clients):
        rows = np.flatnonzero(codes == c)
        roles[rows] = m.UNSEEN if c >= int(clients * 0.8) else assign_roles(y[rows], [0.8, 0.1, 0.1], c)
    return {"texts": texts, "labels": y, "roles": roles, "codes": codes, "test_texts": None,
            "test_rows": np.flatnonzero(roles == m.UNSEEN), "test_labels": y[roles == m.UNSEEN], "num_classes": 2,
            "meta": {"synthetic": True}}


def study(**over):
    s = {"results_dir": "unused", "model": "tiny", "max_len": 64, "lr": 0.05, "weight_decay": 0.01,
         "regimes": {"yelp_iid": {"dataset": "yelp", "config": "exp0a_tuned.yaml", "cell": "iid"},
                     "yelp_dir01": {"dataset": "yelp", "config": "exp0a_tuned.yaml", "cell": "label_skew"},
                     "amazon_vg": {"dataset": "amazon", "config": "exp0b_frozen.yaml", "cell": "natural"}},
         "centralized": {"yelp": "yelp_iid", "amazon": "amazon_vg"}, "centralized_epochs": 2, "centralized_batch_size": 16,
         "centralized_checkpoint_steps": 2, "eval_val_every": 1, "eval_test_rounds": [2], "checkpoint_every": 1,
         "mu_grid": [0.001, 0.01], "tuning_seed": 7, "seeds": [42]}
    s.update(over)
    return s


@pytest.fixture
def ctx(tmp_path):
    c = m.load_context(study(), tmp_path / "cache", backend=m.TinyBackend(2), raw={"yelp": yelp_like(), "amazon": amazon_like()})
    for R in c["regimes"].values():          # small synthetic data: fewer clients, rounds and larger batches
        R["fl"] = {**R["fl"], "rounds": 3, "batch_size": 16}
        if R["partition"]["scheme"] != "natural":
            R["partition"] = {**R["partition"], "num_clients": 10}
    return c


def strip_time(hist):
    return [{k: v for k, v in h.items() if not k.endswith("wall_s")} for h in hist]


def test_context_cache_roundtrip(tmp_path, ctx):
    again = m.load_context(study(), tmp_path / "cache", backend=m.TinyBackend(2))   # no raw data: from cache only
    for ds in ("yelp", "amazon"):
        for k in ("labels", "roles", "test_labels"):
            assert np.array_equal(again["data"][ds][k], ctx["data"][ds][k])
        assert np.array_equal(again["tokens"][ds].flat, ctx["tokens"][ds].flat)
    assert again["data"]["yelp"]["test_rows"] is None and len(again["test_tokens"]["yelp"]) == 200
    assert np.array_equal(again["data"]["amazon"]["codes"], ctx["data"]["amazon"]["codes"])


def test_natural_clients_exclude_unseen(ctx):
    d = ctx["data"]["amazon"]
    clients = m.make_clients(d, ctx["regimes"]["amazon_vg"]["partition"], 42)
    rows = np.concatenate(clients)
    assert len(clients) == 24 and len(np.unique(rows)) == len(rows)
    assert not np.any(d["roles"][rows] == m.UNSEEN)
    assert set(rows.tolist()) | set(d["test_rows"].tolist()) == set(range(len(d["labels"])))
    for c in clients:                            # one product per client
        assert len(np.unique(d["codes"][c])) == 1


def test_streaming_aggregation_is_weighted_fedavg(ctx, tmp_path):
    ctx["backend"] = ctx["backends"]["yelp"]
    R = ctx["regimes"]["yelp_iid"]
    R["fl"] = {**R["fl"], "rounds": 1, "fraction_fit": 1.0}
    m.federated_run(ctx, "yelp_iid", 42, "fedavg", 0.0, tmp_path / "run", eval_test=False)
    # recompute by hand from the same initial model and seeds
    data, tokens, b = ctx["data"]["yelp"], ctx["tokens"]["yelp"], ctx["backend"]
    clients = m.make_clients(data, R["partition"], 42)
    train = [c[data["roles"][c] == TRAIN] for c in clients]
    torch.manual_seed(m.derive_seed("init", 42))
    model = b.new_model()
    keys = m.float_keys(model)
    g0 = m.get_state(model, keys)
    total = sum(len(t) for t in train)
    avg = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in g0.items()}
    for cid, t in enumerate(train):
        m.set_state(model, g0)
        m.local_train(b, model, tokens, data["labels"], t, epochs=1, batch_size=16, lr=0.05, weight_decay=0.01,
                      seed=m.derive_seed("batch", 42, cid, 1))
        for k in keys:
            avg[k] += model.state_dict()[k].double() * (len(t) / total)
    m.set_state(model, {k: v.float() for k, v in avg.items()})
    val = np.flatnonzero(data["roles"] == VAL)
    expected = m.scores(data["labels"][val], m.predict(b, model, tokens, val), 2)["macro_f1"]
    got = json.loads((tmp_path / "run" / "final.json").read_text())["val"]["macro_f1"]
    assert got == pytest.approx(expected, abs=1e-12)


def test_fedprox_pulls_towards_global(ctx):
    b, data, tokens = ctx["backends"]["yelp"], ctx["data"]["yelp"], ctx["tokens"]["yelp"]
    rows = np.flatnonzero(data["roles"] == TRAIN)[:200]
    dist = {}
    for mu in (0.0, 5.0):
        torch.manual_seed(0)
        model = b.new_model()
        keys = m.float_keys(model)
        g = m.get_state(model, keys)
        m.local_train(b, model, tokens, data["labels"], rows, epochs=1, batch_size=16, lr=0.05, weight_decay=0.0, seed=1, mu=mu, anchor=g)
        dist[mu] = sum(float(((model.state_dict()[k].float() - g[k]) ** 2).sum()) for k in keys)
    assert dist[5.0] < dist[0.0]


def test_federated_resume_is_identical(ctx, tmp_path, monkeypatch):
    ctx["backend"] = ctx["backends"]["amazon"]
    full = m.federated_run(ctx, "amazon_vg", 42, "fedprox", 0.01, tmp_path / "a")
    calls = {"n": 0}

    def deadline_after_two_rounds():
        calls["n"] += 1
        return calls["n"] > 2                    # one check per round -> stop before round 3
    monkeypatch.setattr(m, "past_deadline", deadline_after_two_rounds)
    with pytest.raises(m.Paused):
        m.federated_run(ctx, "amazon_vg", 42, "fedprox", 0.01, tmp_path / "b")
    assert (tmp_path / "b" / "checkpoint.pt").exists() and not (tmp_path / "b" / "COMPLETE").exists()
    monkeypatch.setattr(m, "past_deadline", lambda: False)
    resumed = m.federated_run(ctx, "amazon_vg", 42, "fedprox", 0.01, tmp_path / "b")
    assert resumed["val"] == full["val"] and resumed["held_out"] == full["held_out"]
    ha = json.loads((tmp_path / "a" / "history.json").read_text())
    hb = json.loads((tmp_path / "b" / "history.json").read_text())
    assert strip_time(ha) == strip_time(hb)
    assert not (tmp_path / "b" / "checkpoint.pt").exists()


def test_centralized_resume_mid_epoch_is_identical(ctx, tmp_path, monkeypatch):
    ctx["backend"] = ctx["backends"]["yelp"]
    part = ctx["regimes"]["yelp_iid"]["partition"]
    full = m.centralized_run(ctx, "yelp", 42, tmp_path / "a", part)
    monkeypatch.setattr(m, "past_deadline", lambda: True)
    with pytest.raises(m.Paused):
        m.centralized_run(ctx, "yelp", 42, tmp_path / "b", part)
    saved = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)
    assert saved["epoch"] == 1 and saved["step"] == 2
    monkeypatch.setattr(m, "past_deadline", lambda: False)
    resumed = m.centralized_run(ctx, "yelp", 42, tmp_path / "b", part)
    assert resumed["val"] == full["val"] and resumed["held_out"] == full["held_out"]


def test_held_out_sources(ctx, tmp_path):
    for reg, n in (("yelp_dir01", 200), ("amazon_vg", len(ctx["data"]["amazon"]["test_rows"]))):
        ctx["backend"] = ctx["backends"][ctx["regimes"][reg]["dataset"]]
        fin = m.federated_run(ctx, reg, 42, "fedavg", 0.0, tmp_path / reg)
        assert fin["held_out"]["n"] == n
        hist = json.loads((tmp_path / reg / "history.json").read_text())
        assert [h["round"] for h in hist if "held_macro_f1" in h] == [2, 3]   # eval_test_rounds + last round
        assert fin["communication"]["training_support_bytes_derived"] < fin["communication"]["training_dense_bytes"]


def test_support_bytes_and_token_support():
    acc = {"dense_message_bytes": 4 * 1000, "word_embedding_rows": 50, "hidden": 4}
    # 1000 floats, of which 200 are embedding rows; 3 support rows -> 800 + 12 floats + min(12, ceil(50/8)=7) bytes
    assert m.support_message_bytes(acc, 3) == 4 * 800 + 4 * 12 + 7
    ts = m.TokenStore(np.array([1, 2, 2, 5, 1, 9], np.int32), np.array([0, 3, 4, 6]))
    assert ts.support_size([0, 1]) == 3 and ts.support_size([2]) == 2 and ts.support_size([]) == 0


def test_select_mu_ties_to_smaller(tmp_path):
    s = study(regimes={"r": {}}, mu_grid=[0.001, 0.01, 0.1])
    for mu, f1 in ((0.001, 0.8), (0.01, 0.9), (0.1, 0.9)):
        d = tmp_path / f"tune/r/fedprox_mu{mu}/seed7"
        d.mkdir(parents=True)
        (d / "final.json").write_text(json.dumps({"val": {"macro_f1": f1}}))
    sel = m.select_mu(s, tmp_path)
    assert sel["r"]["mu"] == 0.01 and not sel["r"]["on_grid_edge"]


def test_run_list_counts():
    s = study(seeds=[42, 123, 456], mu_grid=[0.001, 0.01, 0.1])
    assert len(m.run_list(s, "tune")) == 9
    sel = {r: {"mu": 0.01} for r in s["regimes"]}
    assert len(m.run_list(s, "evaluate", sel)) == 3 * (3 * 2 + 2)


def test_full_scale_config_matches_benchmark_regimes():
    s = m.load_study("exploratory/extend_v1/bert.yaml")
    R = m.resolve_regimes(s)
    assert {k: (v["fl"]["rounds"], v["fl"]["fraction_fit"], v["fl"]["batch_size"], v["fl"]["local_epochs"]) for k, v in R.items()} == {
        "yelp_iid": (50, 0.25, 32, 1), "yelp_dir01": (50, 0.25, 32, 1), "amazon_vg": (50, 0.1, 32, 1)}
    assert R["yelp_dir01"]["partition"]["alpha"] == 0.1 and R["amazon_vg"]["partition"]["scheme"] == "natural"
    assert m.cache_tag(s, R["yelp_iid"]["cfg"]).startswith("yelp_polarity_7355dff6b72f")
    assert m.cache_tag(s, R["amazon_vg"]["cfg"]).startswith("amazon2023_86405fdce84c")


@pytest.mark.parametrize("bundle", ["amazon2023_3fce98a2fa5e"])
def test_amazon_reconstruction_against_local_bundle(bundle):
    b = ROOT / "data_cache" / "features" / bundle
    if not (b / "client_codes.npy").exists():
        pytest.skip("local synthetic Amazon bundle not available")
    meta = json.loads((b / "meta.json").read_text())
    d = m.amazon_data({"data": meta["data_config"]}, b)
    assert len(d["texts"]) == len(np.load(b / "y_train.npy"))


def test_cli_two_workers_tune_select_evaluate_and_deadline(tmp_path):
    """End to end through the CLI: cache prepared from synthetic data, two worker processes, tuning,
    selection, evaluation; then a rerun finds everything complete. A zero deadline stops cleanly."""
    import subprocess
    import yaml

    s = study(backend="tiny", seeds=[42, 123], test_overrides={"fl": {"rounds": 2, "batch_size": 16}, "partition": {"num_clients": 10}},
              centralized_epochs=1, eval_test_rounds=[2])
    cfg = tmp_path / "study.yaml"
    cfg.write_text(yaml.safe_dump(s))
    m.load_context(s, tmp_path / "cache", backend=m.TinyBackend(2), raw={"yelp": yelp_like(), "amazon": amazon_like()})
    base = [sys.executable, str(ROOT / "experiments" / "distilbert_fl.py"), "--config", str(cfg), "--out", str(tmp_path / "out"),
            "--cache", str(tmp_path / "cache")]
    r = subprocess.run(base + ["--phase", "tune", "--gpus", "2", "--deadline-hours", "0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "deadline reached" in r.stdout and not list((tmp_path / "out").rglob("COMPLETE"))
    r = subprocess.run(base + ["--phase", "all", "--gpus", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    done = sorted(str(p.parent.relative_to(tmp_path / "out")).replace("\\", "/") for p in (tmp_path / "out").rglob("COMPLETE"))
    assert len(done) == 3 * 2 + 2 * (3 * 2 + 2)          # tuning + evaluation (2 seeds)
    assert not list((tmp_path / "out").rglob("CLAIM")) and not list((tmp_path / "out").rglob("checkpoint.pt"))
    sel = json.loads((tmp_path / "out" / "selections.json").read_text())
    for reg in s["regimes"]:
        mu = json.loads((tmp_path / "out" / f"evaluate/{reg}/fedprox/seed42/final.json").read_text())["mu"]
        assert mu == sel[reg]["mu"]
    r = subprocess.run(base + ["--phase", "all", "--gpus", "2"], capture_output=True, text=True)
    assert r.returncode == 0 and "[worker" not in r.stdout                    # nothing re-run


def test_pilot_projection(ctx, tmp_path):
    res = m.pilot(ctx, tmp_path)
    assert set(res["regimes"]) == {"yelp_iid", "yelp_dir01", "amazon_vg"}
    for v in res["regimes"].values():
        assert v["projected_hours_per_federated_run"] > 0 and v["selected"] >= 1
    assert (tmp_path / "pilot" / "pilot.json").exists()
