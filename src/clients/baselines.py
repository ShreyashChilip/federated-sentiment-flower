"""Non-federated reference points: centralized and local-only training.

Both use the same features, model, optimizer, batch size and learning rate as
the federated runs (``src.clients.trainer.local_train``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.clients.trainer import evaluate, local_train
from src.metrics.fairness import client_dispersion
from src.models.registry import build_model, get_weights, parameter_counts
from src.resources.monitor import energy_proxy_joules, rss_mb
from src.strategies.server import weights_sha256
from src.utils.runtime import load_runtime, resolve_device
from src.utils.seeding import derive_seed, set_seed


def _train_kwargs(cfg: dict) -> dict:
    fl = cfg["fl"]
    return {
        "batch_size": int(fl["batch_size"]),
        "momentum": float(fl.get("momentum", 0.0)),
        "weight_decay": float(fl.get("weight_decay", 0.0)),
    }


def _save(run_dir: Path, **objects) -> None:
    for name, obj in objects.items():
        with open(run_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)


def run_centralized(run_dir) -> dict:
    """Train on the pooled TRAIN rows; one history row per epoch."""
    run_dir = Path(run_dir)
    rt = load_runtime(run_dir)
    cfg, seed, b = rt.cfg, rt.seed, rt.bundle
    device = resolve_device(cfg)
    set_seed(derive_seed("init", seed))
    model = build_model(cfg["model"], b.input_dim, b.num_classes)

    rows, val_rows = rt.pooled_rows("train"), rt.pooled_rows("val")
    x, y = b.x_train[rows], rt.y_train[rows]
    x_val, y_val = b.x_train[val_rows], b.y_train[val_rows]
    c = cfg["centralized"]
    history, cpu, wall = [], 0.0, 0.0
    for epoch in range(1, int(c["epochs"]) + 1):
        lr = float(cfg["fl"]["lr"]) * float(cfg["fl"].get("lr_decay", 1.0)) ** (epoch - 1)
        out = local_train(model, x, y, epochs=1, lr=lr, seed=derive_seed("batch", seed, "central", epoch),
                          device=device, **_train_kwargs(cfg))
        cpu, wall = cpu + out["train_cpu_s"], wall + out["train_wall_s"]
        val = evaluate(model, x_val, y_val, b.num_classes, device)
        test = evaluate(model, b.x_test, b.y_test, b.num_classes, device)
        history.append({
            "round": epoch, "lr": lr, "train_loss": out["train_loss"], "steps": out["steps"],
            **{f"val_{k}": val[k] for k in ("loss", "accuracy", "macro_f1")},
            **{f"test_{k}": test[k] for k in ("loss", "accuracy", "macro_f1", "macro_precision", "macro_recall")},
        })
    res = cfg.get("resources", {})
    on_gpu = device == "cuda"
    final = {
        "kind": "centralized", "seed": seed, "epochs": int(c["epochs"]),
        "test": evaluate(model, b.x_test, b.y_test, b.num_classes, device),
        "val": evaluate(model, x_val, y_val, b.num_classes, device),
        "resources": {
            "train_wall_s": wall, "train_cpu_s": cpu, "rss_mb": rss_mb(),
            "energy_proxy_joules": energy_proxy_joules(cpu, float(res.get("cpu_watts") or 0.0),
                                                       wall if on_gpu else 0.0, float(res.get("gpu_watts") or 0.0)),
            "energy_proxy_inputs": res,
        },
        "model": parameter_counts(model),
        "weights_sha256": weights_sha256(get_weights(model)),
    }
    np.savez_compressed(run_dir / "final_weights.npz", *get_weights(model))
    _save(run_dir, history=history, final=final)
    return final


def run_local_only(run_dir) -> dict:
    """Every client trains its own model with no communication at all."""
    run_dir = Path(run_dir)
    rt = load_runtime(run_dir)
    cfg, seed, b = rt.cfg, rt.seed, rt.bundle
    device = resolve_device(cfg)
    epochs = float(cfg["local_only"]["epochs"])
    clients, t0 = [], time.perf_counter()
    for cid in range(rt.num_clients):
        rows = rt.client_rows(cid, "train")
        row = {"client_id": cid, "n_train": int(len(rows))}
        if len(rows) > 0:
            set_seed(derive_seed("init", seed))
            model = build_model(cfg["model"], b.input_dim, b.num_classes)
            out = local_train(model, b.x_train[rows], rt.y_train[rows], epochs=epochs, lr=float(cfg["fl"]["lr"]),
                              seed=derive_seed("batch", seed, cid, "local_only"), device=device, **_train_kwargs(cfg))
            test = evaluate(model, b.x_test, b.y_test, b.num_classes, device)
            own_rows = rt.client_rows(cid, "client_test")
            own = evaluate(model, b.x_train[own_rows], b.y_train[own_rows], b.num_classes, device)
            row.update({
                "global_test_accuracy": test["accuracy"], "global_test_macro_f1": test["macro_f1"],
                "client_test_n": own["n"], "client_test_accuracy": own["accuracy"], "client_test_macro_f1": own["macro_f1"],
                "train_cpu_s": out["train_cpu_s"], "train_wall_s": out["train_wall_s"],
            })
        clients.append(row)
    trained = [r for r in clients if "global_test_macro_f1" in r]
    final = {
        "kind": "local_only", "seed": seed, "epochs": epochs,
        # How well a model trained by one client alone does on the global test set.
        "global_test_macro_f1": client_dispersion([r["global_test_macro_f1"] for r in trained]),
        "global_test_accuracy": client_dispersion([r["global_test_accuracy"] for r in trained]),
        # How well each client's own model does on its own held-out rows.
        "client_test_macro_f1": client_dispersion([r["client_test_macro_f1"] for r in trained if r["client_test_n"] > 0]),
        "resources": {
            "train_cpu_s": float(np.sum([r["train_cpu_s"] for r in trained])),
            "wall_s": time.perf_counter() - t0,
        },
    }
    _save(run_dir, clients=clients, final=final)
    return final
