"""Model-free client descriptors for a benchmark regime (src/benchmark/client_features.py).

Writes results/benchmark/diagnostics/<regime>/seed<k>/client_features.csv. For
natural regimes the client population does not depend on the seed; one seed
is enough. Uses no model and no metric.

Usage
  python experiments/client_diagnostics.py --regimes amazon_vg --seeds 42
  python experiments/client_diagnostics.py --regimes yelp_iid yelp_dir01 --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import client_features, plan as P  # noqa: E402
from src.data.prepare import cache_root  # noqa: E402
from src.utils.config import PROJECT_ROOT, config_hash  # noqa: E402
from src.utils.env import collect_environment  # noqa: E402
from src.utils.runtime import load_runtime, prepare_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="benchmark.yaml")
    ap.add_argument("--regimes", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--no-raw-text", action="store_true")
    args = ap.parse_args()
    bench = P.load_benchmark(args.bench)
    for regime in args.regimes:
        cfg = P.regime_config(bench, regime, allow_untuned=True)
        for seed in args.seeds:
            out = P.results_root(bench) / "diagnostics" / regime / f"seed{seed}"
            if (out / "client_features.csv").exists():
                print(f"skip {regime}/seed{seed} (exists)")
                continue
            work = out / "_prepare"
            if work.exists():
                shutil.rmtree(work)
            prepare_run(cfg, seed, work, create_partition=True)
            rt = load_runtime(work, include_test=False)
            store = None
            if rt.bundle.kind == "natural" and not args.no_raw_text:
                from src.data import amazon2023

                store, _ = amazon2023.open_review_store(cfg["data"], cache_root(cfg))
            rows = client_features.compute(rt, store, cfg["data"].get("client_definition"))
            run_meta = json.loads((work / "run_metadata.json").read_text(encoding="utf-8"))
            meta = {"regime": regime, "seed": seed, "config_hash": config_hash(cfg), "bundle": str(rt.bundle.path),
                    "partition_sha256": run_meta["partition"]["sha256"], "raw_text_measures": store is not None,
                    "environment": collect_environment(PROJECT_ROOT)}
            print(client_features.write(rows, out, meta), f"{len(rows)} clients")
            shutil.rmtree(work)


if __name__ == "__main__":
    main()
