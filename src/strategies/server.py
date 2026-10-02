"""Flower ServerApp: the federated training loop.

The loop is written against Flower's Message API so that client sampling,
dropout, straggler handling and communication accounting are explicit, seeded
and logged. The aggregation rules themselves live in ``aggregation.py``.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from flwr.app import ArrayRecord, Context, Message, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp

from src.clients.flower_client import config_record
from src.clients.trainer import evaluate_weights
from src.communication.accounting import CommLedger, message_bytes
from src.metrics.classification import metrics_from_confusion
from src.metrics.convergence import bytes_to_target, rounds_to_target
from src.metrics.fairness import client_dispersion
from src.models.registry import build_model, get_weights, parameter_counts
from src.resources.monitor import energy_proxy_joules, rss_mb
from src.strategies.aggregation import ServerState
from src.utils.runtime import load_runtime, resolve_device
from src.utils.seeding import derive_seed, set_seed


def select_clients(eligible: list[int], fraction: float, min_clients: int, seed: int, rnd: int) -> list[int]:
    """Uniform sampling without replacement of max(min_clients, round(C * K)) clients."""
    k = min(len(eligible), max(min_clients, int(round(fraction * len(eligible)))))
    rng = np.random.default_rng(derive_seed("sample", seed, rnd))
    return sorted(rng.choice(np.asarray(eligible), size=k, replace=False).tolist())


def draw_events(selected: list[int], prob: float, kind: str, seed: int, rnd: int) -> set[int]:
    """Independent per-client Bernoulli(prob) events (dropout or straggling)."""
    if prob <= 0:
        return set()
    rng = np.random.default_rng(derive_seed(kind, seed, rnd))
    return {c for c, u in zip(selected, rng.random(len(selected))) if u < prob}


def weights_sha256(weights) -> str:
    h = hashlib.sha256()
    for w in weights:
        h.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    return h.hexdigest()


def _check(replies, expected: int, what: str) -> list[Message]:
    replies = list(replies)
    errors = [r for r in replies if r.has_error()]
    if errors:
        raise RuntimeError(f"{what}: {len(errors)} client error(s); first: {errors[0].error}")
    if len(replies) != expected:
        raise RuntimeError(f"{what}: expected {expected} replies, received {len(replies)}")
    return replies


def _wait_for_nodes(grid: Grid, expected: int, timeout: float = 120.0) -> list[int]:
    deadline = time.time() + timeout
    while True:
        nodes = sorted(grid.get_node_ids())
        if len(nodes) >= expected:
            return nodes
        if time.time() > deadline:
            raise RuntimeError(f"only {len(nodes)} of {expected} nodes connected")
        time.sleep(0.5)


def federated_evaluation(grid, node_of, weights, run_dir, group, ledger, num_classes) -> list[dict]:
    """Ask every client to evaluate ``weights`` on its own held-out rows."""
    msgs = [
        Message(
            content=RecordDict({"arrays": ArrayRecord(numpy_ndarrays=weights), "config": config_record(run_dir)}),
            message_type=MessageType.EVALUATE, dst_node_id=node, group_id=group,
        )
        for _, node in sorted(node_of.items())
    ]
    down = [message_bytes(m) for m in msgs]
    replies = _check(grid.send_and_receive(msgs), len(msgs), "evaluation")
    ledger.record_other("evaluation", down, [message_bytes(r) for r in replies])
    rows = []
    for reply in replies:
        m = reply.content["metrics"]
        row = {"client_id": int(m["client_id"])}
        for role in ("val", "client_test"):
            row[f"{role}_n"] = int(m[f"{role}_n"])
            if row[f"{role}_n"] > 0:
                cm = np.asarray(m[f"{role}_confusion"]).reshape(num_classes, num_classes)
                stats = metrics_from_confusion(cm)
                row[f"{role}_loss"] = float(m[f"{role}_loss"])
                row[f"{role}_accuracy"] = stats["accuracy"]
                row[f"{role}_macro_f1"] = stats["macro_f1"]
                row[f"{role}_confusion"] = stats["confusion_matrix"]
        rows.append(row)
    return sorted(rows, key=lambda r: r["client_id"])


def run_server(grid: Grid, run_dir) -> dict:
    run_dir = Path(run_dir)
    rt = load_runtime(run_dir)
    cfg, fl, seed = rt.cfg, rt.cfg["fl"], rt.seed
    algorithm = fl["algorithm"]
    device = resolve_device(cfg)
    num_classes = rt.bundle.num_classes
    ledger = CommLedger()

    # ---- registration: map Flower node ids to client (partition) ids --------
    nodes = _wait_for_nodes(grid, rt.num_clients)
    msgs = [
        Message(content=RecordDict({"config": config_record(run_dir)}), message_type=MessageType.QUERY,
                dst_node_id=node, group_id="register")
        for node in nodes
    ]
    down = [message_bytes(m) for m in msgs]
    replies = _check(grid.send_and_receive(msgs), len(msgs), "registration")
    ledger.record_other("setup", down, [message_bytes(r) for r in replies])
    node_of, n_train = {}, {}
    class_counts = np.zeros(num_classes, dtype=np.float64)
    for reply in replies:
        cid = int(reply.content["metrics"]["client_id"])
        node_of[cid] = reply.metadata.src_node_id
        n_train[cid] = int(reply.content["metrics"]["n_train"])
        class_counts += np.asarray(reply.content["metrics"]["class_counts"], dtype=np.float64)
    round_extra = {}
    if algorithm == "fedavg_la" and fl.get("prior_reference", "none") == "global":
        round_extra["global_prior"] = ((class_counts + 1.0) / (class_counts + 1.0).sum()).tolist()
    if sorted(node_of) != list(range(rt.num_clients)):
        raise RuntimeError("client ids reported by the nodes do not match the partition")
    eligible = [c for c in sorted(node_of) if n_train[c] > 0]

    # ---- global model -------------------------------------------------------
    set_seed(derive_seed("init", seed))
    model = build_model(cfg["model"], rt.bundle.input_dim, num_classes)
    state = ServerState(get_weights(model), fl, rt.num_clients)

    val_rows = rt.pooled_rows("val")
    x_val, y_val = rt.bundle.x_train[val_rows], rt.bundle.y_train[val_rows]
    eval_every = int(cfg["eval"].get("every", 1))
    rounds = int(fl["rounds"])
    history: list[dict] = []
    totals = {"client_wall_s": 0.0, "client_cpu_s": 0.0, "client_gpu_s": 0.0, "agg_wall_s": 0.0,
              "peak_client_rss_mb": 0.0, "peak_client_vram_mb": 0.0}

    for rnd in range(1, rounds + 1):
        t_round = time.perf_counter()
        selected = select_clients(eligible, float(fl["fraction_fit"]), int(fl.get("min_fit_clients", 1)), seed, rnd)
        dropped = draw_events(selected, float(fl.get("dropout_prob", 0.0)), "dropout", seed, rnd)
        stragglers = draw_events(selected, float(fl.get("straggler_prob", 0.0)), "straggler", seed, rnd) - dropped
        if fl.get("straggler_policy", "partial") == "drop":
            dropped, stragglers = dropped | stragglers, set()
        reporting = [c for c in selected if c not in dropped]

        lr = float(fl["lr"]) * float(fl.get("lr_decay", 1.0)) ** (rnd - 1)
        base = {"arrays": ArrayRecord(numpy_ndarrays=state.weights)}
        if algorithm == "scaffold":
            base["c_global"] = ArrayRecord(numpy_ndarrays=state.c_global)

        def make(cid: int) -> Message:
            epochs = float(fl["local_epochs"]) * (float(fl.get("straggler_work", 0.5)) if cid in stragglers else 1.0)
            content = dict(base)
            content["config"] = config_record(run_dir, round=rnd, lr=lr, epochs=epochs, **round_extra)
            return Message(content=RecordDict(content), message_type=MessageType.TRAIN,
                           dst_node_id=node_of[cid], group_id=str(rnd))

        # A dropped client is modelled as failing after it received the global
        # model: its downlink is charged, it is not trained, nothing comes back.
        down = [message_bytes(make(cid)) for cid in dropped]
        msgs = [make(cid) for cid in reporting]
        down += [message_bytes(m) for m in msgs]
        replies = _check(grid.send_and_receive(msgs), len(msgs), f"round {rnd}") if msgs else []
        replies.sort(key=lambda r: int(r.content["metrics"]["client_id"]))
        comm = ledger.record_round(rnd, down, [message_bytes(r) for r in replies])

        t_agg = time.perf_counter()
        client_weights = [r.content["arrays"].to_numpy_ndarrays() for r in replies]
        examples = [int(r.content["metrics"]["num_examples"]) for r in replies]
        c_deltas = [r.content["c_delta"].to_numpy_ndarrays() for r in replies] if algorithm == "scaffold" else None
        state.apply(client_weights, examples, c_deltas)
        agg_wall = time.perf_counter() - t_agg

        metrics = [r.content["metrics"] for r in replies]
        wall = sum(float(m["train_wall_s"]) for m in metrics)
        totals["client_wall_s"] += wall
        totals["client_cpu_s"] += sum(float(m["train_cpu_s"]) for m in metrics)
        totals["client_gpu_s"] += sum(float(m["train_wall_s"]) for m in metrics if int(m["on_gpu"]))
        totals["agg_wall_s"] += agg_wall
        for m in metrics:
            totals["peak_client_rss_mb"] = max(totals["peak_client_rss_mb"], float(m["rss_mb"]))
            totals["peak_client_vram_mb"] = max(totals["peak_client_vram_mb"], float(m["vram_mb"]))

        row = {
            "round": rnd, "lr": lr, "selected": len(selected), "dropped": len(dropped),
            "stragglers": len(stragglers), "reported": len(replies),
            "train_loss": float(np.average([float(m["train_loss"]) for m in metrics], weights=examples)) if metrics else None,
            "client_steps": int(sum(int(m["steps"]) for m in metrics)),
            "client_train_wall_s": wall, "agg_wall_s": agg_wall, **comm,
        }
        if algorithm == "scaffold":
            row["max_c_local_norm"] = max((float(m["c_local_norm"]) for m in metrics), default=0.0)
        if rnd % eval_every == 0 or rnd == rounds:
            val = evaluate_weights(model, state.weights, x_val, y_val, num_classes, device)
            test = evaluate_weights(model, state.weights, rt.bundle.x_test, rt.bundle.y_test, num_classes, device)
            for name, out in (("val", val), ("test", test)):
                for key in ("loss", "accuracy", "macro_f1", "macro_precision", "macro_recall"):
                    row[f"{name}_{key}"] = out[key]
        row["round_wall_s"] = time.perf_counter() - t_round
        history.append(row)

    # ---- final evaluation ---------------------------------------------------
    final_val = evaluate_weights(model, state.weights, x_val, y_val, num_classes, device)
    final_test = evaluate_weights(model, state.weights, rt.bundle.x_test, rt.bundle.y_test, num_classes, device)
    per_client = federated_evaluation(grid, node_of, state.weights, run_dir, "final_eval", ledger, num_classes)
    for row in per_client:
        row["n_train"] = n_train[row["client_id"]]

    res = cfg.get("resources", {})
    target = cfg["eval"].get("target_f1")
    final = {
        "kind": "federated",
        "algorithm": algorithm,
        "seed": seed,
        "rounds": rounds,
        "test": final_test,
        "val": final_val,
        "client_test_macro_f1": client_dispersion([r.get("client_test_macro_f1") for r in per_client]),
        "client_test_accuracy": client_dispersion([r.get("client_test_accuracy") for r in per_client]),
        "communication": ledger.totals(),
        "target_val_macro_f1": target,
        "rounds_to_target": rounds_to_target(history, "val_macro_f1", target) if target else None,
        "bytes_to_target": bytes_to_target(history, "val_macro_f1", target) if target else None,
        "resources": {
            **totals,
            "server_rss_mb": rss_mb(),
            "total_round_wall_s": sum(r["round_wall_s"] for r in history),
            "energy_proxy_joules": energy_proxy_joules(
                totals["client_cpu_s"], float(res.get("cpu_watts") or 0.0),
                totals["client_gpu_s"], float(res.get("gpu_watts") or 0.0),
            ),
            "energy_proxy_inputs": res,
        },
        "model": parameter_counts(model),
        "weights_sha256": weights_sha256(state.weights),
    }
    np.savez_compressed(run_dir / "final_weights.npz", *state.weights)
    for name, obj in (("history", history), ("clients", per_client), ("final", final)):
        with open(run_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)
    return final


def make_server_app(run_dir) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:  # noqa: ARG001
        run_server(grid, run_dir)

    return app
