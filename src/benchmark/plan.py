"""Turn configs/benchmark.yaml into concrete run specifications.

A spec is everything a worker needs: the fully resolved config (snapshotted
into run_metadata.json by ``prepare_run``), the seed, the run kind and the run
directory. Specs of later stages depend on the selections written by earlier
stages (tuned hyperparameters); planning such a stage before its dependency
exists raises ``MissingSelection`` rather than guessing a value.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from src.analysis.exp0 import cell_config
from src.utils.config import PROJECT_ROOT, config_hash, deep_merge, load_config

BASELINES = ("centralized", "local_only")
# Placeholders used ONLY by the engineering pilot before any tuning exists.
PILOT_PLACEHOLDERS = {"mu": 0.01, "server_lr": 0.1}


class MissingSelection(RuntimeError):
    pass


def load_benchmark(name: str = "benchmark.yaml") -> dict:
    return load_config(name)["benchmark"]


def results_root(bench: dict) -> Path:
    root = Path(bench["results_dir"])
    return root if root.is_absolute() else PROJECT_ROOT / root


def stage_dir(bench: dict, stage: str) -> Path:
    return results_root(bench) / stage


def selection_path(bench: dict, kind: str, regime: str) -> Path:
    return results_root(bench) / ("tune_base" if kind == "base" else "tune_algorithms") / regime / "selected.json"


def read_selection(bench: dict, kind: str, regime: str) -> dict:
    path = selection_path(bench, kind, regime)
    if not path.exists():
        raise MissingSelection(f"{path} does not exist: run the {('tune_base' if kind == 'base' else 'tune_algorithms')} stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def regime_config(bench: dict, regime: str, *, untuned: bool = False) -> dict:
    """Frozen experiment config of the regime + engine settings + tuned lr/wd.

    ``untuned=True`` (pilot, base tuning, model-free diagnostics) never reads a
    tuning selection, so those specs are identical before and after tuning.
    Otherwise a regime tuned by ``tune_base`` requires its selection.
    """
    spec = bench["regimes"][regime]
    cfg = cell_config(load_config(spec["config"]), spec["cell"])
    cfg = deep_merge(cfg, bench.get("common", {}))
    cfg["benchmark"] = {"protocol": bench["name"], "regime": regime}
    if spec["base_tuning"] == "fedavg_grid":
        if untuned:
            cfg["benchmark"]["base_tuning"] = {"source": "UNTUNED_PILOT_PLACEHOLDER"}
        else:
            sel = read_selection(bench, "base", regime)
            cfg = deep_merge(cfg, {"fl": {"lr": sel["lr"], "weight_decay": sel["weight_decay"]}})
            cfg["benchmark"]["base_tuning"] = {"source": "tune_base", "lr": sel["lr"], "weight_decay": sel["weight_decay"]}
    else:
        cfg["benchmark"]["base_tuning"] = {"source": spec["config"], "lr": cfg["fl"]["lr"],
                                           "weight_decay": cfg["fl"]["weight_decay"]}
    return cfg


def _spec(bench, stage, regime, method, seed, cfg, variant="") -> dict:
    kind = method if method in BASELINES else "federated"
    if kind == "federated":
        cfg = deep_merge(cfg, {"fl": {"algorithm": method}})
    cfg = deep_merge(cfg, {"experiment": {"name": f"{bench['name']}_{stage}"}})
    tag = method + (f"__{variant}" if variant else "")
    run_dir = stage_dir(bench, stage) / regime / tag / f"seed{seed}"
    return {"key": f"{stage}/{regime}/{tag}/seed{seed}", "stage": stage, "regime": regime, "method": method,
            "variant": variant, "seed": int(seed), "kind": kind, "cfg": cfg, "config_hash": config_hash(cfg),
            "run_dir": str(run_dir)}


def _grid(grid: dict) -> list[dict]:
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _variant(point: dict) -> str:
    return "_".join(f"{k}{v:g}" for k, v in sorted(point.items()))


def plan_stage(bench: dict, stage: str) -> list[dict]:
    st = bench["stages"][stage]
    kind = st["kind"]
    specs = []
    if kind == "pilot":
        for regime in st["regimes"]:
            cfg = deep_merge(regime_config(bench, regime, untuned=True), st.get("overrides", {}))
            for method in st["methods"]:
                c = deep_merge(cfg, {"fl": {k: v for k, v in PILOT_PLACEHOLDERS.items()}})
                for seed in st["seeds"]:
                    specs.append(_spec(bench, stage, regime, method, seed, c))
        return specs
    seed_t = int(bench["tuning_seed"])
    if kind == "tune_base":
        for regime in st["regimes"]:
            cfg = regime_config(bench, regime, untuned=True)
            cfg = deep_merge(cfg, {"eval": {"test": False}})
            for point in _grid(bench["base_grid"]):
                c = deep_merge(cfg, {"fl": {"lr": point["lr"], "weight_decay": point["weight_decay"]}})
                c["benchmark"]["base_tuning"] = {"source": "grid_point", **point}
                specs.append(_spec(bench, stage, regime, "fedavg", seed_t, c, _variant(point)))
        return specs
    if kind == "tune_algorithms":
        for regime in st["regimes"]:
            cfg = deep_merge(regime_config(bench, regime), {"eval": {"test": False}})
            for method, grid in bench["algorithm_grids"].items():
                points = _grid(grid) if grid else [{}]
                for point in points:
                    c = deep_merge(cfg, {"fl": point})
                    specs.append(_spec(bench, stage, regime, method, seed_t, c, _variant(point) if point else "reference"))
        return specs
    if kind == "evaluate":
        for regime in st["regimes"]:
            cfg = regime_config(bench, regime)
            selected = read_selection(bench, "algorithms", regime)["algorithms"]
            for method in st["methods"]:
                c = cfg
                if method not in BASELINES:
                    c = deep_merge(cfg, {"fl": selected[method]["hyperparameters"]})
                    c = deep_merge(c, st.get("fl_overrides", {}))
                c = deep_merge(c, st.get("overrides", {}))
                for seed in st["seeds"]:
                    specs.append(_spec(bench, stage, regime, method, seed, c))
        return specs
    raise ValueError(f"unknown stage kind {kind!r}")


def stage_dir_from_spec(spec: dict) -> Path:
    """<results>/<stage>/<regime>/<method>/seed<k> -> <results>/<stage>."""
    return Path(spec["run_dir"]).parents[2]
