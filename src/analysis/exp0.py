"""Experiment 0 (diagnostics): run layout, pre-run guards, consistency checks, analysis.

Layout (one directory per run, never overwritten):

    results/<experiment>/<cell>/<condition>/seed<k>/
        cell       "iid", "label_skew" (0A) or "natural" (0B)
        condition  "centralized", "fedavg", "fedavg_la"
    results/<experiment>/<cell>/diagnostics/seed<k>/
        checks.json  diagnostic.json  feature_confounding.csv

The centralized model does not depend on how samples are assigned to clients,
so in 0A it is trained once per seed (under the ``iid`` cell) and reused.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.analysis import diagnostics as dg
from src.clients.trainer import to_tensor
from src.metrics.classification import confusion_matrix, metrics_from_confusion
from src.metrics.fairness import client_dispersion
from src.models.registry import build_model, set_weights
from src.statistics.stats import summarize
from src.utils.config import PROJECT_ROOT, deep_merge
from src.utils.runtime import load_runtime

CONDITIONS = ("centralized", "fedavg", "fedavg_la")
FEDERATED = ("fedavg", "fedavg_la")
# Settings that must be identical in every condition of a cell.
SHARED_SETTINGS = (("fl", "lr"), ("fl", "lr_decay"), ("fl", "batch_size"), ("fl", "weight_decay"),
                   ("fl", "momentum"), ("model", "name"))


def experiment_dir(cfg: dict) -> Path:
    root = Path(cfg["experiment"].get("results_dir") or "results")
    return (root if root.is_absolute() else PROJECT_ROOT / root) / cfg["experiment"]["name"]


def run_dir_for(cfg: dict, cell: str, condition: str, seed: int) -> Path:
    return experiment_dir(cfg) / cell / condition / f"seed{seed}"


def cell_config(cfg: dict, cell: str) -> dict:
    """Config of one cell: the base config with that cell's partition settings."""
    out = deep_merge(cfg, {})
    # The cell replaces the partition section (only the storage folder is kept),
    # so no parameter of another scheme, such as alpha, leaks into an IID cell.
    out["partition"] = {**cfg["cells"][cell], "dir": cfg["partition"].get("dir", "partitions")}
    return out


def official_preconditions(cfg: dict, environment: dict, seeds: list[int]) -> list[str]:
    """Reasons why a run may NOT be recorded as official; empty means it may."""
    problems = []
    if environment["platform_kind"] != "kaggle":
        problems.append("not running on Kaggle")
    git = environment["git"]
    if not git.get("commit"):
        problems.append("no git commit")
    if git.get("dirty"):
        problems.append("git working tree has uncommitted changes")
    protocol = cfg.get("protocol", {})
    if not protocol.get("frozen"):
        problems.append("protocol.frozen is false: the hyperparameters marked PROVISIONAL in the config have not been approved")
    want = protocol.get("required_seed_count")
    official = cfg["experiment"].get("seeds") or []
    if want and len(official) != want:
        problems.append(f"experiment.seeds holds {len(official)} seeds; the protocol requires {want}")
    if sorted(seeds) != sorted(official):
        problems.append("the seeds requested differ from experiment.seeds")
    if len(set(official)) != len(official):
        problems.append("experiment.seeds contains duplicates")
    if not cfg["data"].get("revision"):
        problems.append("data.revision is not pinned")
    return problems


# --------------------------------------------------------------------------- checks

def _meta(run_dir: Path) -> dict:
    with open(run_dir / "run_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def consistency_checks(run_dirs: dict) -> list[dict]:
    """Verify that the conditions of one cell are comparable and fully recorded."""
    metas = {c: _meta(Path(d)) for c, d in run_dirs.items()}
    checks = []

    def add(name: str, passed: bool, detail="") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    def same(name: str, getter, among=None) -> None:
        values = {c: getter(m) for c, m in metas.items() if among is None or c in among}
        add(name, len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) == 1,
            next(iter(values.values())) if len(values) else "")

    same("same vocabulary (feature space) in every condition", lambda m: m["vocabulary_sha256"])
    same("same feature bundle and preprocessing", lambda m: (m["bundle_dir"], m["dataset"].get("features_config")))
    same("same training split", lambda m: m["dataset"].get("train_fingerprint"))
    same("same evaluation test set", lambda m: (m["dataset"].get("test_fingerprint"), m["dataset"].get("test_size")))
    same("same seed", lambda m: m["seed"])
    for section, key in SHARED_SETTINGS:
        same(f"identical {section}.{key}", lambda m, s=section, k=key: m["config"][s].get(k))
    same("federated conditions use the same saved partition", lambda m: m["partition"]["sha256"], among=FEDERATED)
    fed = [m for c, m in metas.items() if c in FEDERATED]
    add("partition seed recorded", all(m["partition"].get("partition_seed") is not None for m in fed),
        [m["partition"].get("partition_seed") for m in fed])
    add("realized client class counts saved",
        all(len(m["partition"].get("client_class_counts") or []) == m["partition"]["client_count"] for m in fed))
    add("alpha recorded", all(not m["partition"]["type"].startswith("dirichlet") or m["partition"].get("alpha") is not None for m in fed),
        [m["partition"].get("alpha") for m in fed])
    add("client count recorded", all(m["partition"].get("client_count") for m in fed), [m["partition"].get("client_count") for m in fed])
    add("seed recorded", all(m.get("seed") is not None for m in metas.values()))
    add("configuration hash recorded", all(m.get("config_hash") for m in metas.values()))
    add("official/non-official status recorded", all(isinstance(m.get("official_environment"), bool) for m in metas.values()),
        sorted({m.get("official_environment") for m in metas.values()}))
    official = all(m.get("official_environment") for m in metas.values())
    commits = sorted({str(m["environment"]["git"].get("commit")) for m in metas.values()})
    # A commit is mandatory for official runs; for laptop runs it is recorded as null.
    add("git commit recorded", all("commit" in m["environment"]["git"] for m in metas.values())
        and (not official or all(m["environment"]["git"].get("commit") for m in metas.values())), commits)
    return checks


# --------------------------------------------------------------------------- analysis

@torch.no_grad()
def predict(weights, x, input_dim: int, num_classes: int, model_cfg: dict, batch_size: int = 8192) -> np.ndarray:
    model = build_model(model_cfg, input_dim, num_classes)
    set_weights(model, weights)
    model.eval()
    out = [model(to_tensor(x[lo:lo + batch_size], "cpu")).argmax(dim=1).numpy() for lo in range(0, x.shape[0], batch_size)]
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def load_weights(run_dir: Path) -> list[np.ndarray]:
    data = np.load(Path(run_dir) / "final_weights.npz")
    return [data[k] for k in sorted(data.files, key=lambda n: int(n.split("_")[1]))]


def analyze_cell(run_dirs: dict, out_dir, rule: dict, seed: int) -> dict:
    """Checks, feature-level decomposition and model comparison for one cell and seed."""
    run_dirs = {c: Path(d) for c, d in run_dirs.items()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = consistency_checks(run_dirs)
    with open(out_dir / "checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=1)
    failed = [c["check"] for c in checks if not c["passed"]]
    if failed:
        raise RuntimeError(f"conditions are not comparable, refusing to analyse: {failed}")

    rt = load_runtime(run_dirs["fedavg"])
    b = rt.bundle
    num_classes, d = b.num_classes, b.input_dim
    terms = b.vocabulary()
    if len(terms) != d:
        raise RuntimeError("vocabulary length does not match the feature dimension")

    # client id of every row that belongs to a training client
    client_of = np.full(b.x_train.shape[0], -1, dtype=np.int64)
    for cid, rows in enumerate(rt.clients):
        client_of[rows] = cid
    train_rows = np.concatenate([rt.client_rows(c, "train") for c in range(rt.num_clients)])
    groups = client_of[train_rows]
    # clients without training rows would be empty groups: re-code to 0..K'-1
    _, groups = np.unique(groups, return_inverse=True)
    x_tr, score = b.x_train[train_rows], rt.y_train[train_rows].astype(np.float64)

    decomp = dg.association_decomposition(x_tr, score, groups)
    repeats = int(rule["null_repeats"])
    null_mean, null_sd = dg.permutation_null(x_tr, score, groups, repeats, seed)
    flags = dg.flag_candidates(decomp, null_mean, null_sd, repeats, rule)

    weights = {c: load_weights(p) for c, p in run_dirs.items()}
    direction = {c: dg.sentiment_direction(w[0]) for c, w in weights.items()}
    columns = {
        "document_frequency": decomp["document_frequency"], "client_frequency": decomp["client_frequency"],
        "global_association": decomp["beta_pooled"], "within_client_association": decomp["beta_within"],
        "across_client_association": decomp["beta_between"], "lambda_within": decomp["lambda_within"],
        "between_client_variance": decomp["between_client_variance"], "client_exposure": decomp["exposure"],
        "client_gap": decomp["gap"], "gap_null_mean": null_mean, "gap_null_sd": null_sd, "gap_z": flags["gap_z"], "gap_q": flags["gap_q"],
        "supported": flags["supported"], "client_structure": flags["client_structure"], "gap_type": flags["gap_type"],
        "candidate_confounded": flags["candidate"],
        **{f"w_{c}": direction[c] for c in CONDITIONS},
        **{f"coefficient_bias_{c}": direction[c] - direction["centralized"] for c in FEDERATED},
    }
    dg.write_feature_table(out_dir / "feature_confounding.csv", terms, columns)

    report = {
        "seed": seed,
        "num_features": d,
        "num_supported_features": int(flags["supported"].sum()),
        "num_client_structure_features": int(flags["client_structure"].sum()),
        "gap_type_counts": {k: int((flags["gap_type"] == k).sum()) for k in ("inflated", "sign_reversed", "attenuated")},
        "num_candidate_features": int(flags["candidate"].sum()),
        "candidate_rule": rule,
        "bias_vs_exposure": {c: dg.bias_statistics(direction[c], direction["centralized"], decomp["exposure"], flags["supported"])
                             for c in FEDERATED},
        "bias_vs_exposure_candidates_only": {
            c: dg.bias_statistics(direction[c], direction["centralized"], decomp["exposure"], flags["candidate"]) for c in FEDERATED},
        "mean_abs_weight_on_candidates": {c: float(np.abs(direction[c][flags["candidate"]]).mean()) if flags["candidate"].any() else None
                                          for c in CONDITIONS},
        "performance": {},
    }

    # model comparison on identical rows for all three conditions
    seen_rows = np.concatenate([rt.client_rows(c, "client_test") for c in range(rt.num_clients)])
    for cond in CONDITIONS:
        pred_seen = predict(weights[cond], b.x_train[seen_rows], d, num_classes, rt.cfg["model"])
        pred_test = predict(weights[cond], b.x_test, d, num_classes, rt.cfg["model"])
        if b.kind == "natural":
            perf = dg.generalization_report(pred_seen, b.y_train[seen_rows], client_of[seen_rows],
                                            pred_test, b.y_test, b.test_client_codes, num_classes)
        else:
            seen = metrics_from_confusion(confusion_matrix(b.y_train[seen_rows], pred_seen, num_classes))
            test = metrics_from_confusion(confusion_matrix(b.y_test, pred_test, num_classes))
            per_client = dg.per_group_metrics(pred_seen, b.y_train[seen_rows], client_of[seen_rows], num_classes)
            perf = {
                "global_test": {k: test[k] for k in ("n", "accuracy", "macro_f1")},
                "seen": {k: seen[k] for k in ("n", "accuracy", "macro_f1")},
                "seen_client_macro_f1": client_dispersion([r["macro_f1"] for r in per_client]),
                "per_client": {"seen": per_client},
            }
        report["performance"][cond] = perf
    with open(out_dir / "diagnostic.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return report


def aggregate(exp_dir, cells: list[str]) -> dict:
    """Across-seed summary of the per-seed diagnostics. Purely mechanical."""
    exp_dir = Path(exp_dir)
    out = {"cells": {}}
    for cell in cells:
        reports = []
        for path in sorted((exp_dir / cell / "diagnostics").glob("seed*/diagnostic.json")):
            with open(path, "r", encoding="utf-8") as f:
                reports.append(json.load(f))
        if not reports:
            continue
        entry = {"seeds": [r["seed"] for r in reports], "bias_vs_exposure": {}}
        for cond in FEDERATED:
            entry["bias_vs_exposure"][cond] = {}
            for stat in ("pearson", "spearman", "partial_pearson"):
                vals = [r["bias_vs_exposure"][cond][stat] for r in reports if r["bias_vs_exposure"][cond][stat] is not None]
                entry["bias_vs_exposure"][cond][stat] = {
                    "per_seed": vals, **(summarize(vals) if vals else {}),
                    "mean_abs": float(np.mean(np.abs(vals))) if vals else None,
                    "same_sign_all_seeds": bool(vals) and len(vals) == len(reports) and (all(v > 0 for v in vals) or all(v < 0 for v in vals)),
                }
        entry["num_candidate_features"] = [r["num_candidate_features"] for r in reports]
        out["cells"][cell] = entry
    return out


def decision_rule_0a(summary: dict, skew_cell: str, control_cell: str = "iid", margin: float = 0.10) -> dict:
    """Mechanical evaluation of the decision rule frozen in EXPERIMENT_PLAN.md section 5.

    An effect is declared for a condition only if the Spearman correlation of
    coefficient bias with label exposure under the skewed partition has the
    same sign in every seed AND its mean absolute value exceeds the IID
    control's mean absolute value by at least ``margin``. This describes label
    heterogeneity only; it is not evidence about natural-client confounding.
    """
    out = {}
    for cond in FEDERATED:
        try:
            skew = summary["cells"][skew_cell]["bias_vs_exposure"][cond]["spearman"]
            ctrl = summary["cells"][control_cell]["bias_vs_exposure"][cond]["spearman"]
        except KeyError:
            out[cond] = {"evaluated": False}
            continue
        excess = None if skew["mean_abs"] is None or ctrl["mean_abs"] is None else skew["mean_abs"] - ctrl["mean_abs"]
        out[cond] = {
            "evaluated": True, "same_sign_all_seeds": skew["same_sign_all_seeds"],
            "mean_abs_skew": skew["mean_abs"], "mean_abs_control": ctrl["mean_abs"], "excess": excess,
            "effect_declared": bool(skew["same_sign_all_seeds"] and excess is not None and excess >= margin),
        }
    return out
