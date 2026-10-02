"""Entry points that turn a resolved config and a seed into a finished run."""
from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import PROJECT_ROOT, config_hash
from src.utils.runtime import prepare_run


def run_directory(cfg: dict, seed: int, kind: str, root=None) -> Path:
    root = Path(root or cfg["experiment"].get("results_dir") or "results")
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    name = cfg["experiment"]["name"]
    return root / name / f"{kind}_{config_hash(cfg, 8)}" / f"seed{seed}"


def is_complete(run_dir: Path) -> bool:
    return (Path(run_dir) / "final.json").exists()


def run_federated(cfg: dict, seed: int, run_dir=None, create_partition: bool = False) -> dict:
    """Run one federated experiment in the Flower simulation runtime."""
    from flwr.simulation import run_simulation

    from src.clients.flower_client import app as client_app
    from src.strategies.server import make_server_app

    run_dir = Path(run_dir or run_directory(cfg, seed, cfg["fl"]["algorithm"]))
    prepare_run(cfg, seed, run_dir, create_partition)
    sim = cfg.get("simulation", {})
    backend = {"client_resources": {"num_cpus": float(sim.get("client_cpus", 1)), "num_gpus": float(sim.get("client_gpus", 0.0))}}
    if sim.get("ray_cpus"):
        backend["init_args"] = {"num_cpus": int(sim["ray_cpus"])}
    run_simulation(
        server_app=make_server_app(run_dir),
        client_app=client_app,
        num_supernodes=int(cfg["partition"]["num_clients"]),
        backend_config=backend,
    )
    if not is_complete(run_dir):
        raise RuntimeError(f"simulation ended without results in {run_dir}; see the log above")
    with open(run_dir / "final.json", "r", encoding="utf-8") as f:
        return json.load(f)


def run_baseline(cfg: dict, seed: int, kind: str, run_dir=None, create_partition: bool = False) -> dict:
    from src.clients.baselines import run_centralized, run_local_only

    run_dir = Path(run_dir or run_directory(cfg, seed, kind))
    prepare_run(cfg, seed, run_dir, create_partition)
    return {"centralized": run_centralized, "local_only": run_local_only}[kind](run_dir)
