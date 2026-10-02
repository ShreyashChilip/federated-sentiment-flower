"""Results schema: what every run must record, and flat tables built from it.

The source of truth stays in each run directory (``run_metadata.json``,
``history.json``, ``clients.json``, ``final.json``). This module flattens those
files into three tables per experiment, joined by ``run_id``:

* ``runs.csv``    one row per run (identity, configuration, final metrics),
* ``rounds.csv``  one row per run and round,
* ``clients.csv`` one row per run and client.

``validate_run`` is the gate: a run with a missing field is reported, never
silently accepted.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# Must be present AND non-null for every run.
REQUIRED_NON_NULL = (
    "experiment_id", "run_id", "dataset", "dataset_hash", "vocabulary_sha256", "partition_type",
    "partition_sha256", "partition_seed", "client_count", "seed", "algorithm", "model", "learning_rate",
    "config_hash", "official_environment", "timestamp_utc", "python", "torch",
)
# Must be present; may be null where the field does not apply (alpha for an
# IID split, mu for FedAvg, client_definition for synthetic clients, ...).
REQUIRED_KEYS = REQUIRED_NON_NULL + (
    "dataset_version", "alpha", "client_definition", "client_filter_rule", "local_epochs", "client_fraction",
    "dropout", "mu", "weight_decay", "batch_size", "rounds", "git_commit", "git_dirty", "gpu", "cuda",
    "test_accuracy", "test_macro_f1", "val_accuracy", "val_macro_f1", "uplink_bytes", "downlink_bytes",
    "wall_clock_s", "cpu_time_s", "memory_mb",
)
ROUND_KEYS = ("round", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "test_loss", "test_accuracy",
              "test_macro_f1", "uplink_bytes", "downlink_bytes", "cum_total_bytes", "selected", "reported",
              "dropped", "stragglers", "round_wall_s", "client_train_wall_s", "agg_wall_s", "lr")


def _read(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_record(run_dir) -> dict:
    """Flat description of one finished run."""
    run_dir = Path(run_dir)
    meta, final = _read(run_dir / "run_metadata.json"), _read(run_dir / "final.json")
    cfg, ds, part, env = meta["config"], meta["dataset"], meta["partition"], meta["environment"]
    fl, hw = cfg["fl"], env["hardware"]
    kind = final["kind"]
    federated = kind == "federated"
    res = final.get("resources", {})
    comm = final.get("communication", {})
    test, val = final.get("test", {}), final.get("val", {})
    fingerprint = "|".join(str(ds.get(k)) for k in ("train_fingerprint", "test_fingerprint", "source_sha256") if ds.get(k))
    return {
        "experiment_id": meta.get("experiment_id"),
        "run_id": meta.get("run_id"),
        "run_dir": str(run_dir),
        "dataset": ds.get("dataset"),
        "dataset_version": ds.get("revision"),
        "dataset_hash": fingerprint or None,
        "vocabulary_sha256": meta.get("vocabulary_sha256"),
        "num_features": ds.get("num_features"),
        "partition_type": part.get("type"),
        "alpha": part.get("alpha"),
        "client_count": part.get("client_count"),
        "client_definition": part.get("client_definition"),
        "client_filter_rule": json.dumps(part["client_filter_rule"], sort_keys=True) if part.get("client_filter_rule") else None,
        "partition_sha256": part.get("sha256"),
        "partition_seed": part.get("partition_seed"),
        "seed": meta.get("seed"),
        "algorithm": final.get("algorithm") if federated else kind,
        "model": cfg["model"]["name"],
        "learning_rate": fl.get("lr"),
        "lr_decay": fl.get("lr_decay"),
        "local_epochs": fl.get("local_epochs") if federated else None,
        "client_fraction": fl.get("fraction_fit") if federated else None,
        "dropout": fl.get("dropout_prob") if federated else None,
        "mu": fl.get("mu") if final.get("algorithm") == "fedprox" else None,
        "weight_decay": fl.get("weight_decay"),
        "batch_size": fl.get("batch_size"),
        "rounds": final.get("rounds", final.get("epochs")),
        "config_hash": meta.get("config_hash"),
        "git_commit": env["git"].get("commit"),
        "git_dirty": env["git"].get("dirty"),
        "official_environment": meta.get("official_environment"),
        "platform": env.get("platform_kind"),
        "timestamp_utc": env.get("timestamp_utc"),
        "python": hw.get("python"),
        "torch": env["packages"].get("torch"),
        "flwr": env["packages"].get("flwr"),
        "gpu": "; ".join(g["name"] for g in hw.get("gpus", [])) or None,
        "cuda": hw.get("cuda_version"),
        "test_accuracy": test.get("accuracy"),
        "test_macro_f1": test.get("macro_f1"),
        "val_accuracy": val.get("accuracy"),
        "val_macro_f1": val.get("macro_f1"),
        "worst_client_macro_f1": (final.get("client_test_macro_f1") or {}).get("worst"),
        "client_macro_f1_std": (final.get("client_test_macro_f1") or {}).get("std"),
        "uplink_bytes": comm.get("training_uplink_bytes"),
        "downlink_bytes": comm.get("training_downlink_bytes"),
        "wall_clock_s": res.get("total_round_wall_s", res.get("train_wall_s", res.get("wall_s"))),
        "cpu_time_s": res.get("client_cpu_s", res.get("train_cpu_s")),
        "memory_mb": res.get("peak_client_rss_mb", res.get("rss_mb")),
        "vram_mb": res.get("peak_client_vram_mb"),
        "energy_proxy_joules": res.get("energy_proxy_joules"),
        "weights_sha256": final.get("weights_sha256"),
    }


def validate_run(run_dir) -> list[str]:
    """Problems with one run directory; an empty list means the run is complete."""
    run_dir = Path(run_dir)
    problems = [f"missing file {name}" for name in ("run_metadata.json", "final.json") if not (run_dir / name).exists()]
    if problems:
        return problems
    rec = run_record(run_dir)
    problems += [f"field '{k}' not recorded" for k in REQUIRED_KEYS if k not in rec]
    problems += [f"field '{k}' is empty" for k in REQUIRED_NON_NULL if rec.get(k) is None]
    if rec.get("official_environment") and (not rec.get("git_commit") or rec.get("git_dirty")):
        problems.append("official run without a clean git commit")
    if rec["algorithm"] not in ("centralized", "local_only"):
        for name in ("history.json", "clients.json"):
            if not (run_dir / name).exists():
                problems.append(f"missing file {name}")
        for k in ("uplink_bytes", "downlink_bytes", "local_epochs", "client_fraction"):
            if rec.get(k) is None:
                problems.append(f"field '{k}' is empty")
        if str(rec["partition_type"]).startswith("dirichlet") and rec.get("alpha") is None:
            problems.append("alpha not recorded for a Dirichlet partition")
    if not _read(run_dir / "run_metadata.json")["partition"].get("client_class_counts"):
        problems.append("realized client class counts not saved")
    return problems


def _write(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect_tables(experiment_dir) -> dict:
    """Write runs.csv, rounds.csv and clients.csv for every finished run below a directory."""
    experiment_dir = Path(experiment_dir)
    runs, rounds, clients, problems = [], [], [], {}
    for final_path in sorted(experiment_dir.rglob("final.json")):
        run_dir = final_path.parent
        issues = validate_run(run_dir)
        if issues:
            problems[str(run_dir)] = issues
        rec = run_record(run_dir)
        runs.append(rec)
        ident = {"run_id": rec["run_id"], "seed": rec["seed"], "algorithm": rec["algorithm"]}
        if (run_dir / "history.json").exists():
            for row in _read(run_dir / "history.json"):
                rounds.append({**ident, **{k: row.get(k) for k in ROUND_KEYS}})
        if (run_dir / "clients.json").exists():
            for row in _read(run_dir / "clients.json"):
                clients.append({**ident, **{k: v for k, v in row.items() if not isinstance(v, (list, dict))}})
    for name, rows in (("runs", runs), ("rounds", rounds), ("clients", clients)):
        if rows:
            _write(experiment_dir / f"{name}.csv", rows)
    with open(experiment_dir / "schema_validation.json", "w", encoding="utf-8") as f:
        json.dump({"runs": len(runs), "runs_with_problems": len(problems), "problems": problems}, f, indent=1)
    return {"runs": len(runs), "problems": problems}
