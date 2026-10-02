"""Phase 1 baseline: centralized vs local-only vs FedAvg.

Default setting: Yelp Polarity, TF-IDF + LR, 10 clients, Dirichlet alpha = 0.1,
3 seeds. Finished runs are skipped, so the script can be re-launched after a
Kaggle session timeout.

Usage:
  python experiments/make_partitions.py --config yelp.yaml --clients 10 --alphas 0.1
  python experiments/01_baseline.py --config yelp.yaml
  python experiments/01_baseline.py --config smoke.yaml --seeds 42 123 --set fl.rounds=3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.summary import write_summary  # noqa: E402
from src.runner import is_complete, run_baseline, run_directory, run_federated  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config, parse_override  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="yelp.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--create-partitions", action="store_true",
                        help="allow creating missing partitions (otherwise run make_partitions.py first)")
    args = parser.parse_args()

    overrides = dict(parse_override(item) for item in args.set)
    overrides.setdefault("experiment.name", "01_baseline")
    cfg = load_config(args.config, overrides={**overrides, "fl.algorithm": "fedavg"})

    for seed in args.seeds:
        for kind in ("centralized", "local_only", "fedavg"):
            run_dir = run_directory(cfg, seed, kind)
            if is_complete(run_dir):
                print(f"skip   {kind} seed {seed} (finished)")
                continue
            print(f"run    {kind} seed {seed} -> {run_dir}")
            if kind == "fedavg":
                run_federated(cfg, seed, run_dir, args.create_partitions)
            else:
                run_baseline(cfg, seed, kind, run_dir, args.create_partitions)

    experiment_dir = PROJECT_ROOT / cfg["experiment"]["results_dir"] / cfg["experiment"]["name"]
    print("summary ->", write_summary(experiment_dir))


if __name__ == "__main__":
    main()
