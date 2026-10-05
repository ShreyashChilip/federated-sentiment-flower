"""Sequential in-process federated simulation used by the benchmark.

Why not the Flower/Ray runtime of Phase 0: with natural clients the run has
tens of thousands of clients. Flower's simulation registers one node per
client and its final federated evaluation sends one full-model message per
client, so serialized messages alone would need tens of GB of RAM on Kaggle.
This engine runs the same computation one client at a time in one process:

* client sampling, dropout and straggler draws: ``select_clients`` /
  ``draw_events`` of ``src.strategies.server`` (same seeded streams);
* local training: ``src.clients.trainer.local_train`` (same optimizer, batch
  order seed ``derive_seed("batch", seed, client, round)``, losses);
* aggregation: ``src.benchmark.algorithms`` (streaming; identical arithmetic
  to ``ServerState`` for FedAvg and SCAFFOLD).

For FedAvg the final weights are bit-identical to the Flower path
(``tests/test_benchmark_engine.py``). Communication is the payload of the
arrays a real deployment would exchange (float32 model-sized arrays per
message, per direction, for every client that exchanged a message); Flower's
record framing (a few hundred bytes per message) is not included.

Training never loads the held-out split. The seen-client validation rows are
evaluated every round; client-test and held-out rows only at the full
evaluation points (``eval.full_every`` and the last round), and never in a
tuning run (``eval.test: false``).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.benchmark.algorithms import make_algorithm
from src.benchmark.evaluation import Evaluator, compact
from src.clients.trainer import local_train, planned_steps
from src.metrics.fairness import client_dispersion
from src.models.registry import build_model, get_weights, parameter_counts, set_weights
from src.resources.monitor import rss_mb
from src.strategies.server import draw_events, select_clients, weights_sha256
from src.utils.runtime import load_runtime, resolve_device
from src.utils.seeding import derive_seed, set_seed


class RowView:
    """Rows of a CSR matrix addressed by local position, without copying them.

    ``local_train`` only ever does ``x[batch]``; this returns
    ``base[rows[batch]]``, which equals ``base[rows][batch]``.
    """

    def __init__(self, base: sp.csr_matrix, rows: np.ndarray):
        self.base, self.rows = base, np.asarray(rows)
        self.shape = (len(self.rows), base.shape[1])

    def __getitem__(self, idx):
        return self.base[self.rows[idx]]

    def __len__(self):
        return len(self.rows)


def peak_rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB
    except Exception:  # Windows
        try:
            import psutil

            return psutil.Process(os.getpid()).memory_info().peak_wset / 2**20
        except Exception:
            return float("nan")


def _test_loader(bundle_path: Path):
    def load():
        return sp.load_npz(bundle_path / "x_test.npz").tocsr(), np.load(bundle_path / "y_test.npy")
    return load


def _dump(run_dir: Path, name: str, obj) -> None:
    tmp = run_dir / f"{name}.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, run_dir / f"{name}.json")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _clean(values) -> list:
    return [None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else v for v in values]


def _client_table(evals: dict, n_train: list[int], participation=None, local=None) -> list[dict]:
    """One row per seen client: sizes and the per-client val/client-test scores."""
    rows = []
    for cid in range(len(n_train)):
        row = {"client_id": cid, "n_train": int(n_train[cid])}
        if participation is not None:
            row["participation"] = int(participation[cid])
        for role in ("val", "client_test"):
            entry = evals.get(role)
            if entry and entry["clients"] is not None:
                sc = entry["clients"]
                row[f"{role}_n"] = int(sc["n"][cid])
                if sc["n"][cid] > 0:
                    row[f"{role}_accuracy"] = float(sc["accuracy"][cid])
                    row[f"{role}_macro_f1"] = float(sc["macro_f1"][cid])
                    row[f"{role}_classes_present"] = int(sc["classes_present"][cid])
        if local is not None:
            row.update(local[cid])
        rows.append(row)
    return rows


def _unseen_table(evaluator: Evaluator, evals: dict) -> list[dict] | None:
    entry = evals.get("unseen")
    if not entry or entry["clients"] is None:
        return None
    sc = entry["clients"]
    return [{"unseen_index": i, "client_code": int(code), "n": int(sc["n"][i]),
             "accuracy": float(sc["accuracy"][i]), "macro_f1": float(sc["macro_f1"][i]),
             "classes_present": int(sc["classes_present"][i])}
            for i, code in enumerate(evaluator.unseen_keys)]


def _final_block(evals: dict, held: str | None) -> dict:
    out = {}
    for name, entry in evals.items():
        out[name] = entry["pooled"]
        if entry.get("dispersion") is not None:
            out[f"{name}_client_macro_f1"] = entry["dispersion"]
    if held == "unseen" and "unseen" in evals and "client_test" in evals:
        seen, unseen = evals["client_test"], evals["unseen"]
        out["seen_unseen_gap"] = {
            "pooled_macro_f1": seen["pooled"]["macro_f1"] - unseen["pooled"]["macro_f1"],
            "mean_client_macro_f1": (seen.get("dispersion") or {}).get("mean", float("nan"))
            - (unseen.get("dispersion") or {}).get("mean", float("nan")),
            "median_client_macro_f1": (seen.get("dispersion") or {}).get("median", float("nan"))
            - (unseen.get("dispersion") or {}).get("median", float("nan")),
        }
    return out


def _resources(t0: float, cpu0: float, extra: dict) -> dict:
    return {"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - cpu0,
            "rss_mb_end": rss_mb(), "peak_rss_mb": peak_rss_mb(), **extra}


def _setup(run_dir: Path):
    rt = load_runtime(run_dir, include_test=False)
    cfg = rt.cfg
    import torch

    torch.set_num_threads(int(cfg.get("client_threads", 1)))
    evaluator = Evaluator(rt, _test_loader(rt.bundle.path))
    return rt, cfg, resolve_device(cfg), evaluator


def eval_names(cfg: dict, evaluator: Evaluator, full: bool) -> tuple:
    if not bool(cfg["eval"].get("test", True)):
        return ("val",)
    return ("val", "client_test", evaluator.held_out_group) if full else ("val",)


# ---------------------------------------------------------------------------
def run_federated(run_dir) -> dict:
    run_dir = Path(run_dir)
    t0, cpu0 = time.perf_counter(), time.process_time()
    rt, cfg, device, evaluator = _setup(run_dir)
    fl, seed, b = cfg["fl"], rt.seed, rt.bundle
    n_train = [len(rt.client_rows(c, "train")) for c in range(rt.num_clients)]
    eligible = [c for c in range(rt.num_clients) if n_train[c] > 0]
    batch = int(fl["batch_size"])

    set_seed(derive_seed("init", seed))
    model = build_model(cfg["model"], b.input_dim, b.num_classes)
    x = get_weights(model)
    # Scratch space (SCAFFOLD client state) is an execution detail, not part of the config.
    scratch = Path(os.environ.get("FEDBENCH_SCRATCH") or (run_dir / "_scratch"))
    algo = make_algorithm(fl, x, rt.num_clients, scratch / run_dir.name)
    model_bytes = int(sum(w.nbytes for w in x))
    down_msg = model_bytes * algo.downlink_model_copies
    up_msg = model_bytes * algo.uplink_model_copies

    rounds = int(fl["rounds"])
    full_every = int(cfg["eval"].get("full_every", 10) or rounds)
    every = int(cfg["eval"].get("every", 1))
    participation = np.zeros(rt.num_clients, dtype=np.int64)
    history, cum_down, cum_up = [], 0, 0
    train_wall = train_cpu = agg_wall = eval_wall = 0.0
    diverged_round = None
    try:
        for rnd in range(1, rounds + 1):
            t_round = time.perf_counter()
            selected = select_clients(eligible, float(fl["fraction_fit"]), int(fl.get("min_fit_clients", 1)), seed, rnd)
            dropped = draw_events(selected, float(fl.get("dropout_prob", 0.0)), "dropout", seed, rnd)
            stragglers = draw_events(selected, float(fl.get("straggler_prob", 0.0)), "straggler", seed, rnd) - dropped
            if fl.get("straggler_policy", "partial") == "drop":
                dropped, stragglers = dropped | stragglers, set()
            reporting = [c for c in selected if c not in dropped]
            lr = float(fl["lr"]) * float(fl.get("lr_decay", 1.0)) ** (rnd - 1)
            epochs_of = {c: float(fl["local_epochs"]) * (float(fl.get("straggler_work", 0.5)) if c in stragglers else 1.0)
                         for c in reporting}
            plan = [(c, n_train[c], planned_steps(n_train[c], batch, epochs_of[c])) for c in reporting]
            algo.begin_round(x, plan)
            losses, steps_total = [], 0
            for cid in reporting:
                rows = rt.client_rows(cid, "train")
                set_weights(model, x)
                out = local_train(
                    model, RowView(b.x_train, rows), rt.y_train[rows], epochs=epochs_of[cid], batch_size=batch, lr=lr,
                    momentum=float(fl.get("momentum", 0.0)), weight_decay=float(fl.get("weight_decay", 0.0)),
                    seed=derive_seed("batch", seed, cid, rnd), device=device,
                    algorithm=algo.local_rule, **algo.client_kwargs(cid),
                )
                ta = time.perf_counter()
                algo.add_client(cid, int(out["num_examples"]), out)
                agg_wall += time.perf_counter() - ta
                train_wall += out["train_wall_s"]
                train_cpu += out["train_cpu_s"]
                losses.append((out["train_loss"], out["num_examples"]))
                steps_total += out["steps"]
                participation[cid] += 1
                del out
            ta = time.perf_counter()
            x = algo.finish_round(x)
            agg_wall += time.perf_counter() - ta
            down = down_msg * len(selected)          # dropped clients received the model too
            up = up_msg * len(reporting)
            cum_down, cum_up = cum_down + down, cum_up + up
            row = {
                "round": rnd, "lr": lr, "selected": len(selected), "dropped": len(dropped),
                "stragglers": len(stragglers), "reported": len(reporting),
                "train_loss": float(np.average([l for l, _ in losses], weights=[n for _, n in losses])) if losses else None,
                "client_steps": int(steps_total),
                "downlink_bytes": down, "uplink_bytes": up,
                "cum_downlink_bytes": cum_down, "cum_uplink_bytes": cum_up, "cum_total_bytes": cum_down + cum_up,
                **{f"algo_{k}": v for k, v in algo.summary().items()},
            }
            finite = all(np.isfinite(w).all() for w in x)
            if not finite and diverged_round is None:
                diverged_round = rnd
            if finite and (rnd % every == 0 or rnd == rounds):
                te = time.perf_counter()
                full = rnd % full_every == 0 and rnd != rounds
                evals = evaluator.evaluate(x, eval_names(cfg, evaluator, full))
                for name, entry in evals.items():
                    row.update(compact(entry, name))
                eval_wall += time.perf_counter() - te
            row["round_wall_s"] = time.perf_counter() - t_round
            history.append(row)
            if rnd == 1 or rnd % 10 == 0 or rnd == rounds:
                print(f"  round {rnd}/{rounds} val_f1={row.get('val_macro_f1')} rss={rss_mb():.0f}MB "
                      f"wall={row['round_wall_s']:.1f}s", flush=True)
            if not finite:
                break   # a diverged model is recorded as such; later rounds would only repeat NaN
    finally:
        algo.close()

    te = time.perf_counter()
    finite = all(np.isfinite(w).all() for w in x)
    evals = evaluator.evaluate(x, eval_names(cfg, evaluator, True)) if finite else {}
    eval_wall += time.perf_counter() - te
    if history and finite:
        for name, entry in evals.items():   # the last history row carries the full final evaluation
            history[-1].update(compact(entry, name))
    clients = _client_table(evals, n_train, participation)
    unseen = _unseen_table(evaluator, evals)
    final = {
        "kind": "federated", "engine": "benchmark_sequential_v1", "algorithm": fl["algorithm"], "seed": seed,
        "rounds": rounds, "rounds_completed": len(history), "diverged": diverged_round is not None,
        "diverged_round": diverged_round, "test_evaluated": bool(cfg["eval"].get("test", True)),
        "held_out_group": evaluator.held_out_group,
        **_final_block(evals, evaluator.held_out_group),
        "participation": client_dispersion(participation[eligible].astype(float)),
        "clients_never_selected": int((participation[eligible] == 0).sum()),
        "communication": {
            "model_bytes_float32": model_bytes, "downlink_bytes_per_message": down_msg,
            "uplink_bytes_per_message": up_msg, "training_downlink_bytes": cum_down,
            "training_uplink_bytes": cum_up, "training_total_bytes": cum_down + cum_up,
            "mean_total_bytes_per_round": (cum_down + cum_up) / max(len(history), 1),
            "note": "float32 array payload only; no compression; transport framing excluded",
        },
        "resources": _resources(t0, cpu0, {
            "client_train_wall_s": train_wall, "client_train_cpu_s": train_cpu, "aggregation_wall_s": agg_wall,
            "evaluation_wall_s": eval_wall, "mean_round_wall_s": float(np.mean([h["round_wall_s"] for h in history])),
            "device": device,
        }),
        "model": parameter_counts(model),
        "weights_sha256": weights_sha256(x),
    }
    np.savez_compressed(run_dir / "final_weights.npz", *x)
    _dump(run_dir, "history", history)
    _dump(run_dir, "clients", clients)
    if unseen is not None:
        _dump(run_dir, "unseen_clients", unseen)
    _dump(run_dir, "final", final)
    return final


# ---------------------------------------------------------------------------
def run_centralized(run_dir) -> dict:
    """Pooled training on every seen client's TRAIN rows; one history row per epoch."""
    run_dir = Path(run_dir)
    t0, cpu0 = time.perf_counter(), time.process_time()
    rt, cfg, device, evaluator = _setup(run_dir)
    fl, seed, b = cfg["fl"], rt.seed, rt.bundle
    set_seed(derive_seed("init", seed))
    model = build_model(cfg["model"], b.input_dim, b.num_classes)
    rows = np.sort(np.concatenate([rt.client_rows(c, "train") for c in range(rt.num_clients)]))
    epochs = int(cfg["centralized"]["epochs"])
    history = []
    for epoch in range(1, epochs + 1):
        lr = float(fl["lr"]) * float(fl.get("lr_decay", 1.0)) ** (epoch - 1)
        out = local_train(model, RowView(b.x_train, rows), rt.y_train[rows], epochs=1, lr=lr,
                          batch_size=int(fl["batch_size"]), momentum=float(fl.get("momentum", 0.0)),
                          weight_decay=float(fl.get("weight_decay", 0.0)),
                          seed=derive_seed("batch", seed, "central", epoch), device=device)
        evals = evaluator.evaluate(get_weights(model), ("val",))
        history.append({"round": epoch, "lr": lr, "train_loss": out["train_loss"], "steps": out["steps"],
                        "train_wall_s": out["train_wall_s"], **compact(evals["val"], "val")})
        print(f"  epoch {epoch}/{epochs} val_f1={history[-1].get('val_macro_f1')}", flush=True)
    x = get_weights(model)
    evals = evaluator.evaluate(x, eval_names(cfg, evaluator, True))
    n_train = [len(rt.client_rows(c, "train")) for c in range(rt.num_clients)]
    final = {
        "kind": "centralized", "engine": "benchmark_sequential_v1", "algorithm": "centralized", "seed": seed,
        "epochs": epochs, "train_rows": int(len(rows)), "held_out_group": evaluator.held_out_group,
        "test_evaluated": bool(cfg["eval"].get("test", True)),
        **_final_block(evals, evaluator.held_out_group),
        "communication": {"training_total_bytes": 0, "note": "centralized: no federated communication"},
        "resources": _resources(t0, cpu0, {"device": device}),
        "model": parameter_counts(model), "weights_sha256": weights_sha256(x),
    }
    np.savez_compressed(run_dir / "final_weights.npz", *x)
    _dump(run_dir, "history", history)
    _dump(run_dir, "clients", _client_table(evals, n_train))
    unseen = _unseen_table(evaluator, evals)
    if unseen is not None:
        _dump(run_dir, "unseen_clients", unseen)
    _dump(run_dir, "final", final)
    return final


# ---------------------------------------------------------------------------
def run_local_only(run_dir) -> dict:
    """Every seen client trains its own model on its own TRAIN rows, no communication.

    Each model is scored on its own client's validation and client-test rows.
    Pooled numbers add the per-client confusion matrices. Held-out clients
    have no local model, so local-only has no unseen-client result; for a
    standard bundle each local model is also scored on the official test
    split (mean over clients reported).
    """
    from src.benchmark.evaluation import linear_logits
    from src.metrics.classification import confusion_matrix, metrics_from_confusion

    run_dir = Path(run_dir)
    t0, cpu0 = time.perf_counter(), time.process_time()
    rt, cfg, device, evaluator = _setup(run_dir)
    fl, seed, b = cfg["fl"], rt.seed, rt.bundle
    c = b.num_classes
    epochs = float(cfg["local_only"]["epochs"])
    use_test = bool(cfg["eval"].get("test", True))
    pooled = {"val": np.zeros((c, c), np.int64), "client_test": np.zeros((c, c), np.int64)}
    per_client = []
    global_test = []
    x_test = y_test = None
    if use_test and b.kind != "natural":
        x_test, y_test = _test_loader(b.path)()
    n_train = [len(rt.client_rows(cid, "train")) for cid in range(rt.num_clients)]
    for cid in range(rt.num_clients):
        rows = rt.client_rows(cid, "train")
        entry = {}
        if len(rows):
            set_seed(derive_seed("init", seed))
            model = build_model(cfg["model"], b.input_dim, b.num_classes)
            local_train(model, RowView(b.x_train, rows), rt.y_train[rows], epochs=epochs, lr=float(fl["lr"]),
                        batch_size=int(fl["batch_size"]), momentum=float(fl.get("momentum", 0.0)),
                        weight_decay=float(fl.get("weight_decay", 0.0)),
                        seed=derive_seed("batch", seed, cid, "local_only"), device=device)
            w = get_weights(model)
            roles = ("val", "client_test") if use_test else ("val",)
            for role in roles:
                r = rt.client_rows(cid, role)
                if len(r):
                    pred = linear_logits(b.x_train[r], w).argmax(axis=1)
                    cm = confusion_matrix(b.y_train[r], pred, c)
                    pooled[role] += cm
                    m = metrics_from_confusion(cm)
                    entry.update({f"{role}_n": int(len(r)), f"{role}_accuracy": m["accuracy"], f"{role}_macro_f1": m["macro_f1"]})
            if x_test is not None:
                pred = linear_logits(x_test, w).argmax(axis=1)
                global_test.append(metrics_from_confusion(confusion_matrix(y_test, pred, c))["macro_f1"])
        per_client.append(entry)
        if cid % 2000 == 0:
            print(f"  local-only client {cid}/{rt.num_clients}", flush=True)
    clients = [{"client_id": cid, "n_train": int(n_train[cid]), **per_client[cid]} for cid in range(rt.num_clients)]
    final = {
        "kind": "local_only", "engine": "benchmark_sequential_v1", "algorithm": "local_only", "seed": seed,
        "epochs": epochs, "held_out_group": evaluator.held_out_group, "test_evaluated": use_test,
        "val": metrics_from_confusion(pooled["val"]),
        "val_client_macro_f1": client_dispersion([e.get("val_macro_f1") for e in per_client]),
        "unseen_note": "local-only has no model for held-out clients" if b.kind == "natural" else None,
        "communication": {"training_total_bytes": 0, "note": "local-only: no communication"},
        "resources": _resources(t0, cpu0, {"device": device}),
    }
    if use_test:
        final["client_test"] = metrics_from_confusion(pooled["client_test"])
        final["client_test_client_macro_f1"] = client_dispersion([e.get("client_test_macro_f1") for e in per_client])
        if global_test:
            final["global_test_macro_f1_over_local_models"] = client_dispersion(global_test)
    _dump(run_dir, "history", [])
    _dump(run_dir, "clients", clients)
    _dump(run_dir, "final", final)
    return final


RUNNERS = {"federated": run_federated, "centralized": run_centralized, "local_only": run_local_only}
