"""Phase 0 smoke test: 3 clients, 2 rounds, small Yelp subset.

Verifies, end to end and through Flower:
  1. client training, aggregation and evaluation run for every algorithm
     (IID split, so that "the model learns" is a fair assertion),
  2. results serialize to disk,
  3. two runs with the same seed give bit-identical global weights and metrics,
  4. a different seed gives a different result,
  5. the same mechanics hold on a Dirichlet label-skew split (no accuracy
     assertion: with 3 nearly single-class clients and 2 rounds, FedAvg is
     expected to do badly, and that observation is recorded, not hidden).

Usage:  python experiments/00_smoke_test.py [--algorithms fedavg fedprox ...]

Nothing produced here is a paper result.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runner import run_baseline, run_federated  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402

OVERRIDES = {
    "fedavg": {},
    "fedprox": {"fl.mu": 0.1},
    "fedadam": {"fl.server_lr": 0.1},
    "scaffold": {},
}
# Fields that must be identical between two runs with the same seed.
DETERMINISTIC = ("weights_sha256", "test", "val", "client_test_macro_f1", "communication")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", default=list(OVERRIDES))
    parser.add_argument("--keep", action="store_true", help="keep earlier smoke results")
    args = parser.parse_args()

    out_root = PROJECT_ROOT / "results" / "_smoke"
    if out_root.exists() and not args.keep:
        shutil.rmtree(out_root)
    report, ok = {"checks": []}, True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        report["checks"].append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

    for algorithm in args.algorithms:
        cfg = load_config("smoke.yaml", overrides={"partition.scheme": "iid", "fl.algorithm": algorithm, **OVERRIDES[algorithm]})
        runs = {}
        for tag, seed in (("a", 42), ("b", 42), ("c", 123)):
            run_dir = out_root / algorithm / f"seed{seed}_{tag}"
            runs[tag] = run_federated(cfg, seed, run_dir, create_partition=True)
            for name in ("run_metadata.json", "history.json", "clients.json", "final.json", "final_weights.npz"):
                if not (run_dir / name).exists():
                    check(f"{algorithm}: {name} written", False)
        a, b, c = runs["a"], runs["b"], runs["c"]
        same = all(a[k] == b[k] for k in DETERMINISTIC)
        check(f"{algorithm}: same seed reproduces exactly", same, f"sha {a['weights_sha256'][:12]} vs {b['weights_sha256'][:12]}")
        check(f"{algorithm}: different seed differs", a["weights_sha256"] != c["weights_sha256"])
        history = json.loads((out_root / algorithm / "seed42_a" / "history.json").read_text())
        check(f"{algorithm}: all clients reported each round", all(r["reported"] == r["selected"] == 3 for r in history))
        check(f"{algorithm}: learns on an IID split", a["val"]["accuracy"] > 0.65,
              f"test acc {a['test']['accuracy']:.4f}, macro-F1 {a['test']['macro_f1']:.4f}")
        check(f"{algorithm}: uplink and downlink bytes recorded",
              a["communication"]["training_uplink_bytes"] > 0 and a["communication"]["training_downlink_bytes"] > 0,
              f"up {a['communication']['training_uplink_bytes']} B, down {a['communication']['training_downlink_bytes']} B")
        if algorithm == "scaffold":
            check("scaffold: client control variates persist across rounds", history[1]["max_c_local_norm"] > 0,
                  f"max ||c_i|| entering round 2 = {history[1]['max_c_local_norm']:.4g}")
        report[algorithm] = {"seed42": a, "seed123": c}

    cfg = load_config("smoke.yaml")  # Dirichlet alpha = 0.5, 3 clients
    skew = [run_federated(cfg, 42, out_root / "fedavg_dirichlet" / f"seed42_{t}", create_partition=True) for t in "ab"]
    check("dirichlet: same seed reproduces exactly", all(skew[0][k] == skew[1][k] for k in DETERMINISTIC))
    report["fedavg_dirichlet"] = skew[0]
    print(f"[INFO] dirichlet FedAvg after 2 rounds: test acc {skew[0]['test']['accuracy']:.4f} (recorded, not asserted)")

    # Robustness paths in one run: partial participation, dropout, stragglers,
    # LR decay, unequal client sizes and 2% label noise.
    stress = load_config("smoke.yaml", overrides={
        "partition.scheme": "quantity_skew", "partition.beta": 0.5, "partition.num_clients": 6,
        "fl.algorithm": "fedprox", "fl.mu": 0.01, "fl.rounds": 4, "fl.fraction_fit": 0.5,
        "fl.dropout_prob": 0.3, "fl.straggler_prob": 0.5, "fl.lr_decay": 0.9, "data.label_noise": 0.02,
    })
    runs = [run_federated(stress, 42, out_root / "stress" / f"seed42_{t}", create_partition=True) for t in "ab"]
    hist = json.loads((out_root / "stress" / "seed42_a" / "history.json").read_text())
    meta = json.loads((out_root / "stress" / "seed42_a" / "run_metadata.json").read_text())
    check("stress: same seed reproduces exactly", all(runs[0][k] == runs[1][k] for k in DETERMINISTIC))
    check("stress: C = 0.5 selects 3 of 6 clients every round", all(r["selected"] == 3 for r in hist))
    check("stress: dropout occurs and dropped clients send nothing",
          sum(r["dropped"] for r in hist) > 0
          and all(r["reported"] == r["selected"] - r["dropped"] == r["uplink_clients"] for r in hist)
          and all(r["downlink_clients"] == r["selected"] for r in hist),
          f"dropped per round {[r['dropped'] for r in hist]}")
    check("stress: stragglers occur", sum(r["stragglers"] for r in hist) > 0, f"per round {[r['stragglers'] for r in hist]}")
    check("stress: local LR decays by gamma each round", abs(hist[1]["lr"] - 0.5 * 0.9) < 1e-12 and abs(hist[3]["lr"] - 0.5 * 0.9 ** 3) < 1e-12)
    check("stress: 2% of training-role labels flipped", meta["label_noise"]["num_flipped"] == round(0.02 * meta["dataset"]["role_counts"][0]),
          f"{meta['label_noise']['num_flipped']} labels")
    sizes = meta["partition"]["client_sizes"]
    check("stress: client sizes are unequal", max(sizes) > 2 * min(sizes), f"sizes {sizes}")
    report["stress"] = runs[0]

    cfg = load_config("smoke.yaml")
    for kind in ("centralized", "local_only"):
        first = run_baseline(cfg, 42, kind, out_root / kind / "seed42_a")
        second = run_baseline(cfg, 42, kind, out_root / kind / "seed42_b")
        key = "weights_sha256" if kind == "centralized" else "global_test_macro_f1"
        check(f"{kind}: same seed reproduces exactly", first[key] == second[key])
        report[kind] = first

    report["all_passed"] = ok
    with open(out_root / "smoke_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("\nSMOKE TEST", "PASSED" if ok else "FAILED", "->", out_root / "smoke_report.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
