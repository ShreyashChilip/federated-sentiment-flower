"""Experiment 0B: natural-client diagnostic (Amazon Reviews 2023, 5 classes).

One client definition per invocation; the two are never mixed:
  --definition user     one client = one user_id      (configs/exp0b_user.yaml)
  --definition product  one client = one parent_asin  (configs/exp0b_product.yaml)

Requires, in the config, the category and the client-size rule, both fixed
after exp0b_profile_clients.py and before any training.

Usage:
  python experiments/exp0b_natural_clients.py --definition user --mode pilot --create-partitions
  python experiments/exp0b_natural_clients.py --definition user --mode full
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main() -> None:
    argv = sys.argv[1:]
    if "--definition" not in argv or argv.index("--definition") + 1 >= len(argv):
        raise SystemExit("--definition user|product is required")
    i = argv.index("--definition")
    definition = argv[i + 1]
    if definition not in ("user", "product"):
        raise SystemExit("--definition must be user or product")
    spec = importlib.util.spec_from_file_location("exp0a_script", HERE / "exp0a_controlled_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(default_config=f"exp0b_{definition}.yaml", argv=argv[:i] + argv[i + 2:])


if __name__ == "__main__":
    main()
