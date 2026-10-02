"""Experiment 0B, step 1: profile the natural clients. No model is trained.

Reports, for one category and one client definition: number of raw clients,
client-size distribution (min, percentiles, median, max), per-client label
distribution, global class distribution, and how many clients and records
survive each candidate minimum-size threshold.

The client-size rule is chosen from this report and written into the config
BEFORE any training; it is never revisited after model results exist.

Usage:
  python experiments/exp0b_profile_clients.py --definition user --set data.category=<Category>
  python experiments/exp0b_profile_clients.py --definition product --set data.category=<Category>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.profile import profile_and_save  # noqa: E402
from src.utils.config import load_config, parse_override  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--definition", choices=("user", "product"), required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(f"exp0b_{args.definition}.yaml", overrides=dict(parse_override(item) for item in args.set))
    out, report = profile_and_save(cfg)
    for key in ("num_records", "num_raw_clients", "client_size", "global_class_distribution",
                "clients_with_single_class", "retention_by_min_reviews"):
        print(key, "=", report[key])
    print("profile ->", out)


if __name__ == "__main__":
    main()
