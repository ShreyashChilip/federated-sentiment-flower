"""Validation-only hyperparameter tuning for Experiment 0A.

Protocol (EXPERIMENT_PLAN.md amendment A8, frozen before any result):

* Tuned: local learning rate and weight decay, on the grid in the config
  (3 x 3 = 9 runs). Everything else is fixed a priori in ``exp0a.yaml``.
* What is run: FedAvg on the label-skew cell with the tuning seed, which is
  neither an official seed nor the pilot seed. The tuning seed has its own
  partition, so no official partition is touched.
* Criterion: pooled validation macro-F1 after the last round. Ties: smaller
  learning rate first, then larger weight decay.
* The test split is never evaluated (``eval.test: false``); the code asserts
  that no tuning run contains a test metric.
* No diagnostic (coefficient bias, exposure, correlations) is computed, so
  tuning cannot be steered towards an outcome of the experiment.
* The selected values apply unchanged to all three conditions.
* A selection on the edge of the grid is recorded as such. The grid is not
  extended automatically; extending it is a protocol change for CHANGELOG.md.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import yaml

from src.analysis import exp0
from src.runner import is_complete, run_federated
from src.utils.config import PROJECT_ROOT, deep_merge
from src.utils.env import collect_environment


def tuning_config(cfg: dict, lr: float, weight_decay: float) -> dict:
    t = cfg["tuning"]
    out = exp0.cell_config(cfg, t["cell"])
    return deep_merge(out, {
        "experiment": {"name": cfg["experiment"]["name"] + "_tuning"},
        "fl": {"algorithm": t["algorithm"], "lr": lr, "weight_decay": weight_decay},
        "eval": {"test": False},
    })


def grid_points(cfg: dict) -> list[tuple[float, float]]:
    g = cfg["tuning"]["grid"]
    return list(itertools.product(g["lr"], g["weight_decay"]))


def run_dir_for(cfg: dict, lr: float, weight_decay: float) -> Path:
    run_cfg = tuning_config(cfg, lr, weight_decay)
    return exp0.experiment_dir(run_cfg) / f"lr{lr:g}_wd{weight_decay:g}" / f"seed{cfg['tuning']['seed']}"


def run_grid(cfg: dict, create_partitions: bool = True) -> None:
    seed = cfg["tuning"]["seed"]
    if seed in (cfg["experiment"].get("seeds") or []) or seed == cfg["pilot"]["seed"]:
        raise ValueError("the tuning seed must differ from every official seed and from the pilot seed")
    for lr, wd in grid_points(cfg):
        run_dir = run_dir_for(cfg, lr, wd)
        if is_complete(run_dir):
            print(f"skip  lr={lr:g} wd={wd:g} (finished)", flush=True)
            continue
        print(f"tune  lr={lr:g} wd={wd:g}", flush=True)
        run_federated(tuning_config(cfg, lr, wd), seed, run_dir, create_partitions)


def select(cfg: dict) -> dict:
    """Apply the selection rule to the finished grid and write the outputs."""
    g = cfg["tuning"]["grid"]
    rows = []
    for lr, wd in grid_points(cfg):
        run_dir = run_dir_for(cfg, lr, wd)
        if not is_complete(run_dir):
            raise RuntimeError(f"tuning run missing: {run_dir}")
        final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        # Hard guarantee that tuning did not look at the test split.
        if final.get("test_evaluated") or final.get("test") or any("test_macro_f1" in h for h in history):
            raise RuntimeError(f"{run_dir} contains test metrics; a tuning run must never evaluate the test split")
        rows.append({
            "lr": lr, "weight_decay": wd, "val_macro_f1": final["val"]["macro_f1"],
            "val_accuracy": final["val"]["accuracy"], "val_loss": final["val"]["loss"],
            "val_macro_f1_5_rounds_earlier": history[-6]["val_macro_f1"] if len(history) >= 6 else None,
            "rounds": final["rounds"], "seed": meta["seed"], "run_id": meta["run_id"],
            "config_hash": meta["config_hash"], "partition_sha256": meta["partition"]["sha256"],
            "official_environment": meta["official_environment"], "git_commit": meta["environment"]["git"].get("commit"),
            "git_dirty": meta["environment"]["git"].get("dirty"),
        })
    # criterion: highest validation macro-F1; ties -> smaller lr, then larger weight decay
    best = sorted(rows, key=lambda r: (-r["val_macro_f1"], r["lr"], -r["weight_decay"]))[0]
    out_dir = exp0.experiment_dir(tuning_config(cfg, best["lr"], best["weight_decay"]))
    with open(out_dir / "tuning.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table_sha = hashlib.sha256((out_dir / "tuning.csv").read_bytes()).hexdigest()
    official = all(r["official_environment"] and r["git_commit"] and not r["git_dirty"] for r in rows)
    selected = {
        "lr": best["lr"], "weight_decay": best["weight_decay"], "val_macro_f1": best["val_macro_f1"],
        "criterion": cfg["tuning"]["criterion"], "tie_break": cfg["tuning"]["tie_break"],
        "grid": g, "tuning_seed": cfg["tuning"]["seed"], "cell": cfg["tuning"]["cell"],
        "algorithm": cfg["tuning"]["algorithm"],
        "on_grid_edge": {"lr": best["lr"] in (min(g["lr"]), max(g["lr"])),
                         "weight_decay": best["weight_decay"] in (min(g["weight_decay"]), max(g["weight_decay"]))},
        "test_split_evaluated": False,
        "official_environment": official,
        "git_commits": sorted({str(r["git_commit"]) for r in rows}),
        "tuning_csv_sha256": table_sha,
        "environment": collect_environment(PROJECT_ROOT),
    }
    with open(out_dir / "selected.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=1)
    tuned = {
        "inherits": "exp0a.yaml",
        "fl": {"lr": best["lr"], "weight_decay": best["weight_decay"]},
        "protocol": {
            # The official run is unlocked only by a tuning pass made on Kaggle from a clean commit.
            "frozen": official,
            "tuning": {
                "selected": {"lr": best["lr"], "weight_decay": best["weight_decay"]},
                "official_environment": official, "tuning_csv_sha256": table_sha,
                "git_commits": selected["git_commits"], "tuning_seed": cfg["tuning"]["seed"],
            },
        },
    }
    header = ("# Written by experiments/exp0a_tune.py. Do not edit by hand.\n"
              "# Copy to configs/exp0a_tuned.yaml, commit, and run the official experiment with\n"
              "#   --config exp0a_tuned.yaml\n")
    (out_dir / "exp0a_tuned.yaml").write_text(header + yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
    return selected
