"""Per-run state that the server and every simulated client load from disk.

``prepare_run`` writes everything a run needs into its run directory;
``load_runtime`` reads it back. Clients receive only the run directory in each
message, so server and clients are guaranteed to use the same configuration,
partition and labels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.prepare import Bundle, build_bundle, load_bundle
from src.data.roles import CLIENT_TEST, TRAIN, VAL
from src.partitioning.noise import flip_labels
from src.partitioning.store import get_partition, load_partition
from src.utils.config import PROJECT_ROOT
from src.utils.env import collect_environment
from src.utils.seeding import derive_seed

ROLE_BY_NAME = {"train": TRAIN, "val": VAL, "client_test": CLIENT_TEST}


@dataclass
class Runtime:
    run_dir: Path
    cfg: dict
    seed: int
    bundle: Bundle
    clients: list
    y_train: np.ndarray  # training labels after optional label noise

    @property
    def num_clients(self) -> int:
        return len(self.clients)

    def client_rows(self, client_id: int, role: str) -> np.ndarray:
        idx = self.clients[client_id]
        return idx[self.bundle.roles[idx] == ROLE_BY_NAME[role]]

    def pooled_rows(self, role: str) -> np.ndarray:
        return np.flatnonzero(self.bundle.roles == ROLE_BY_NAME[role])


def partitions_dir(cfg: dict) -> Path:
    path = Path(cfg["partition"].get("dir") or "partitions")
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_run(cfg: dict, seed: int, run_dir, create_partition: bool = False) -> Path:
    """Materialize bundle, partition and labels; write run metadata."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(build_bundle(cfg))
    part_cfg = {k: v for k, v in cfg["partition"].items() if k != "dir"}
    clients, summary, created = get_partition(
        partitions_dir(cfg), cfg["data"]["dataset"], bundle.y_train, bundle.roles,
        bundle.num_classes, part_cfg, seed, create=create_partition,
    )
    noise_rate = float(cfg["data"].get("label_noise", 0.0))
    y_train, flipped = flip_labels(
        bundle.y_train, np.flatnonzero(bundle.roles == TRAIN), noise_rate,
        bundle.num_classes, derive_seed("label_noise", seed),
    )
    np.save(run_dir / "train_labels.npy", y_train)

    metadata = {
        "seed": seed,
        "config": cfg,
        "bundle_dir": str(bundle.path),
        "dataset": bundle.meta,
        "partition": {
            "dir": str(partitions_dir(cfg)),
            "name": summary["name"],
            "sha256": summary["sha256"],
            "created_in_this_run": created,
            "client_sizes": [c["num_samples"] for c in summary["clients"]],
        },
        "label_noise": {"rate": noise_rate, "num_flipped": int(len(flipped))},
        "environment": collect_environment(PROJECT_ROOT),
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=1, default=str)
    return run_dir


_CACHE: dict[str, Runtime] = {}


def load_runtime(run_dir) -> Runtime:
    key = str(run_dir)
    if key not in _CACHE:
        run_dir = Path(run_dir)
        with open(run_dir / "run_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        clients, _ = load_partition(Path(meta["partition"]["dir"]), meta["partition"]["name"])
        _CACHE[key] = Runtime(
            run_dir=run_dir,
            cfg=meta["config"],
            seed=meta["seed"],
            bundle=load_bundle(meta["bundle_dir"]),
            clients=clients,
            y_train=np.load(run_dir / "train_labels.npy"),
        )
    return _CACHE[key]


def resolve_device(cfg: dict) -> str:
    import torch

    device = cfg.get("device", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
