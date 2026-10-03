"""Run exactly one validation-only tuning point in an isolated process."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runner import run_federated  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--create-partition", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config_json.read_text(encoding="utf-8"))
    run_federated(cfg, args.seed, args.run_dir, args.create_partition)


if __name__ == "__main__":
    main()