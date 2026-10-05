"""Tables, statistics and figures for a benchmark stage (src/benchmark/analysis.py).

Usage
  python experiments/analyze_benchmark.py --stages screening
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import analysis, plan as P  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="benchmark.yaml")
    ap.add_argument("--stages", nargs="+", required=True)
    args = ap.parse_args()
    bench = P.load_benchmark(args.bench)
    for stage in args.stages:
        print(json.dumps(analysis.analyze_stage(bench, stage), indent=1))


if __name__ == "__main__":
    main()
