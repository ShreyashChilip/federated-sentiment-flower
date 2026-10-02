"""Smoke test of the Experiment 0 pipelines (pilot mode, toy scale, through Flower).

  0A  centralized / FedAvg / FedAvg + label-prior correction on a 6,000-row
      Yelp subset, 5 clients, IID control and a Dirichlet cell.
  0B  the same three conditions on a SYNTHETIC review file, once with
      user clients and once with product clients, with held-out clients.

It verifies plumbing only: files written, checks passed, schema complete, no
client overlap, feature table mapped to vocabulary terms. Everything is stored
under results/_smoke_* (git-ignored). Nothing produced here is a result.

Usage:  python experiments/00b_exp0_smoke.py
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import exp0, exp0_driver  # noqa: E402
from src.data.prepare import build_bundle, load_bundle  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from tests.synthetic_reviews import write_reviews  # noqa: E402

SMALL_YELP = {
    "experiment.name": "_smoke_exp0a", "device": "cpu", "data.max_train": 6000, "data.max_test": 2000,
    "features.max_features": 5000, "features.min_df": 2, "partition.dir": "partitions/_smoke",
    "cells.iid.num_clients": 5, "cells.label_skew.num_clients": 5, "cells.label_skew.alpha": 0.5,
    "fl.fraction_fit": 1.0, "fl.lr": 0.5, "simulation.ray_cpus": 2,
    "diagnostic.min_document_frequency": 20, "diagnostic.null_repeats": 5,
}
RESULTS = PROJECT_ROOT / "results"
checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)


def run(config: str, overrides: dict) -> tuple[dict, Path]:
    cfg, seeds = exp0_driver.resolve(load_config(config, overrides=overrides), "pilot", None)
    exp_dir = exp0.experiment_dir(cfg)
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    exp0_driver.run_conditions(cfg, seeds, create_partitions=True)
    exp0_driver.analyse(cfg, seeds)
    return cfg, exp_dir


def verify(tag: str, cfg: dict, exp_dir: Path, natural: bool) -> None:
    seed = cfg["pilot"]["seed"]
    summary = json.loads((exp_dir / "summary.json").read_text())
    check(f"{tag}: every run passes schema validation", summary["schema"]["problems"] == {}, f"{summary['schema']['runs']} runs")
    expected_runs = 1 + 2 * len(cfg["cells"])
    check(f"{tag}: one centralized run and two federated runs per cell", summary["schema"]["runs"] == expected_runs)
    for cell in cfg["cells"]:
        diag_dir = exp_dir / cell / "diagnostics" / f"seed{seed}"
        cell_checks = json.loads((diag_dir / "checks.json").read_text())
        check(f"{tag}/{cell}: all consistency checks pass", all(c["passed"] for c in cell_checks), f"{len(cell_checks)} checks")
        report = json.loads((diag_dir / "diagnostic.json").read_text())
        meta = json.loads((exp0.run_dir_for(cfg, cell, "fedavg", seed) / "run_metadata.json").read_text())
        bundle = load_bundle(meta["bundle_dir"])
        with open(diag_dir / "feature_confounding.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        vocab = bundle.vocabulary()
        check(f"{tag}/{cell}: feature table has one row per vocabulary term, in order",
              [r["term"] for r in rows] == vocab and len(rows) == report["num_features"])
        needed = {"global_association", "within_client_association", "across_client_association", "between_client_variance",
                  "document_frequency", "client_frequency", "client_exposure", "candidate_confounded",
                  "w_centralized", "w_fedavg", "w_fedavg_la", "coefficient_bias_fedavg", "coefficient_bias_fedavg_la"}
        check(f"{tag}/{cell}: feature table has all diagnostic columns", needed <= set(rows[0]))
        perf = report["performance"]
        check(f"{tag}/{cell}: three conditions compared", set(perf) == set(exp0.CONDITIONS))
        if natural:
            check(f"{tag}/{cell}: seen, unseen and gap reported for each condition",
                  all({"seen", "unseen", "generalization_gap_macro_f1", "seen_client_macro_f1", "unseen_client_macro_f1"} <= set(p) for p in perf.values()))
            hold = meta["dataset"]["holdout"]
            check(f"{tag}/{cell}: zero overlap between training and held-out clients", hold["overlap"] == 0,
                  f"{hold['seen_clients']} seen, {hold['unseen_clients']} unseen")
            check(f"{tag}/{cell}: one Flower client per seen client", meta["partition"]["client_count"] == hold["seen_clients"])
        else:
            check(f"{tag}/{cell}: global test and per-client metrics reported",
                  all({"global_test", "seen_client_macro_f1"} <= set(p) for p in perf.values()))
    check(f"{tag}: summary.md marks the run as pilot and not official",
          "PILOT" in (exp_dir / "summary.md").read_text() and "NOT AN OFFICIAL RESULT" in (exp_dir / "summary.md").read_text())
    for name in ("runs.csv", "rounds.csv", "clients.csv"):
        check(f"{tag}: {name} written", (exp_dir / name).exists())


def main() -> int:
    cfg, exp_dir = run("exp0a.yaml", SMALL_YELP)
    verify("0A", cfg, exp_dir, natural=False)
    check("0A: decision rule evaluated mechanically for both federated conditions",
          set(json.loads((exp_dir / "summary.json").read_text())["decision_rule_0a"]) == set(exp0.FEDERATED))

    review_file = RESULTS / "_smoke_exp0b_data" / "Synthetic.jsonl"
    review_file.parent.mkdir(parents=True, exist_ok=True)
    write_reviews(review_file, num_users=80, num_items=30, seed=1)
    for definition in ("user", "product"):
        overrides = {
            "experiment.name": f"_smoke_exp0b_{definition}", "device": "cpu", "data.local_path": str(review_file),
            "data.category": "Synthetic", "data.client_filter.min_reviews": 10, "features.max_features": 500,
            "features.min_df": 1, "partition.dir": "partitions/_smoke", "fl.fraction_fit": 0.5, "fl.lr": 0.5,
            "simulation.ray_cpus": 2, "diagnostic.min_document_frequency": 10, "diagnostic.min_client_frequency": 3,
            "diagnostic.null_repeats": 5,
        }
        cfg, exp_dir = run(f"exp0b_{definition}.yaml", overrides)
        verify(f"0B-{definition}", cfg, exp_dir, natural=True)
        expected = "user_id" if definition == "user" else "parent_asin"
        bundle = load_bundle(build_bundle(exp0.cell_config(cfg, "natural")))
        check(f"0B-{definition}: client definition is {expected}", bundle.meta["client_definition"] == expected)

    ok = all(p for _, p, _ in checks)
    out = RESULTS / "_smoke_exp0_report.json"
    out.write_text(json.dumps({"all_passed": ok, "checks": [{"check": n, "passed": p, "detail": d} for n, p, d in checks]}, indent=1))
    print("\nEXPERIMENT 0 SMOKE TEST", "PASSED" if ok else "FAILED", "->", out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
