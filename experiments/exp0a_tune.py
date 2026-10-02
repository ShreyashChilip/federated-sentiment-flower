"""Validation-only tuning pass for Experiment 0A (learning rate and weight decay).

Runs FedAvg on the label-skew cell with the tuning seed for every grid point,
selects by pooled validation macro-F1, and writes

  results/exp0a_tuning/tuning.csv
  results/exp0a_tuning/selected.json
  results/exp0a_tuning/exp0a_tuned.yaml   <- copy to configs/ and commit

The test split is never evaluated. No diagnostic is computed. The protocol is
described in src/analysis/tuning.py and EXPERIMENT_PLAN.md (amendment A8).

Usage:
  python experiments/exp0a_tune.py
  python experiments/exp0a_tune.py --select-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import tuning  # noqa: E402
from src.utils.config import load_config, parse_override  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="exp0a.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="smoke tests only")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=dict(parse_override(item) for item in args.set))
    if not args.select_only:
        tuning.run_grid(cfg)
    selected = tuning.select(cfg)
    print("selected:", {k: selected[k] for k in ("lr", "weight_decay", "on_grid_edge", "official_environment")})
    if not selected["official_environment"]:
        print("NOT OFFICIAL: this tuning pass cannot unlock the official run (protocol.frozen stays false).")


if __name__ == "__main__":
    main()
