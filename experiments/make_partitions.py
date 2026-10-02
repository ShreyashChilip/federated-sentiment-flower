"""Create and save the client partitions for a dataset, once.

Experiments only load partitions. Run this script first (it is idempotent:
existing partitions are verified and left untouched).

Usage:
  python experiments/make_partitions.py --config yelp.yaml
  python experiments/make_partitions.py --config smoke.yaml --clients 3 --alphas 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prepare import build_bundle, load_bundle  # noqa: E402
from src.partitioning.store import get_partition  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.runtime import partitions_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="yelp.yaml")
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.1, 0.5, 1.0, 10.0])
    parser.add_argument("--clients", nargs="*", type=int, default=[10, 50, 100])
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--no-iid", action="store_true")
    parser.add_argument("--cells", action="store_true",
                        help="create exactly the partitions listed under 'cells' in the config (Experiment 0)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds or cfg["experiment"]["seeds"]
    bundle = load_bundle(build_bundle(cfg))
    min_size = cfg["partition"].get("min_client_size", 10)
    schemes = [{"scheme": "dirichlet", "alpha": a, "min_client_size": min_size} for a in args.alphas]
    if not args.no_iid:
        schemes.append({"scheme": "iid"})
    grid = [{**scheme, "num_clients": n} for n in args.clients for scheme in schemes]
    if args.cells:
        grid = [dict(cell) for cell in cfg["cells"].values()]
    for part_cfg in grid:
        for seed in seeds:
            _, summary, created = get_partition(
                partitions_dir(cfg), cfg["data"]["dataset"], bundle.y_train, bundle.roles,
                bundle.num_classes, part_cfg, seed, create=True,
            )
            sizes = [c["num_samples"] for c in summary["clients"]]
            print(f"{'created ' if created else 'verified'} {summary['name']}  sha256={summary['sha256'][:12]}  "
                  f"sizes min/median/max = {min(sizes)}/{sorted(sizes)[len(sizes) // 2]}/{max(sizes)}")


if __name__ == "__main__":
    main()
