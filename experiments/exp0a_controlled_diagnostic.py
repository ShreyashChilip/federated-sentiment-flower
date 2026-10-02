"""Experiment 0A: controlled label-heterogeneity diagnostic (Yelp Polarity).

Three conditions (centralized, FedAvg, FedAvg + local label-prior correction)
on an IID control and a Dirichlet(0.1) partition with 100 clients.

Scope: 0A describes ordinary label heterogeneity. A Dirichlet label partition
does not create natural-client vocabulary confounding, so nothing produced
here is evidence about lexical/client confounding.

Usage:
  python experiments/exp0a_controlled_diagnostic.py --mode pilot --create-partitions
  python experiments/exp0a_controlled_diagnostic.py --mode full
  python experiments/exp0a_controlled_diagnostic.py --mode full --seeds 42 123     # resume in chunks
  python experiments/exp0a_controlled_diagnostic.py --mode full --analyze-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import exp0, exp0_driver  # noqa: E402
from src.utils.config import load_config, parse_override  # noqa: E402


def main(default_config: str = "exp0a.yaml", argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="subset of the official seeds (full mode)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides; each one is stored in the run metadata and changes the config hash")
    parser.add_argument("--create-partitions", action="store_true", help="create missing partitions instead of failing")
    parser.add_argument("--analyze-only", action="store_true", help="skip training; analyse finished runs")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=dict(parse_override(item) for item in args.set))
    cfg, seeds = exp0_driver.resolve(cfg, args.mode, args.seeds)
    if not args.analyze_only:
        exp0_driver.run_conditions(cfg, seeds, args.create_partitions)
    exp0_driver.analyse(cfg, seeds)
    print("summary ->", exp0.experiment_dir(cfg) / "summary.md")


if __name__ == "__main__":
    main()
