"""Per-run state that the server and every simulated client load from disk.

``prepare_run`` writes everything a run needs into its run directory;
``load_runtime`` reads it back. Clients receive only the run directory in each
message, so server and clients are guaranteed to use the same configuration,
partition and labels.

Run directories are immutable: a directory that already holds a finished run
is never written to again, and an unfinished one is set aside, not overwritten.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.prepare import Bundle, build_bundle, load_bundle
from src.data.roles import CLIENT_TEST, TRAIN, VAL
from src.partitioning.noise import flip_labels
from src.partitioning.store import get_partition, load_partition
from src.utils.config import PROJECT_ROOT, config_hash
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


def dataset_tag(cfg: dict, bundle: Bundle) -> str:
    """Name used for partition files. Natural bundles depend on the category,
    client definition and filter rule, so their tag carries the bundle hash."""
    if bundle.kind == "natural":
        return bundle.path.name + f"_{bundle.meta['client_definition']}"
    return cfg["data"]["dataset"]


def claim_run_dir(run_dir: Path) -> None:
    """Enforce immutability before anything is written."""
    if (run_dir / "final.json").exists():
        raise FileExistsError(f"{run_dir} already holds a finished run; results are never overwritten")
    if run_dir.exists() and any(run_dir.iterdir()):
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir.rename(run_dir.with_name(f"{run_dir.name}.incomplete-{stamp}"))
    run_dir.mkdir(parents=True, exist_ok=True)


def vocabulary_sha256(bundle: Bundle) -> str:
    return bundle.meta.get("vocabulary_sha256") or hashlib.sha256((bundle.path / "vocabulary.json").read_bytes()).hexdigest()


def prepare_run(cfg: dict, seed: int, run_dir, create_partition: bool = False) -> Path:
    """Materialize bundle, partition and labels; write run metadata."""
    run_dir = Path(run_dir)
    claim_run_dir(run_dir)
    include_test = bool(cfg.get("eval", {}).get("test", True))
    bundle = load_bundle(build_bundle(cfg), include_test=include_test)
    part_cfg = {k: v for k, v in cfg["partition"].items() if k != "dir"}
    groups = None
    if part_cfg["scheme"] == "natural":
        if bundle.kind != "natural":
            raise ValueError("the natural partition scheme needs a dataset with real client metadata")
        from src.data.natural import UNSEEN

        groups = np.where(bundle.roles != UNSEEN, bundle.client_codes, -1)
    clients, summary, created = get_partition(
        partitions_dir(cfg), dataset_tag(cfg, bundle), bundle.y_train, bundle.roles,
        bundle.num_classes, part_cfg, seed, create=create_partition, groups=groups,
    )
    noise_rate = float(cfg["data"].get("label_noise", 0.0))
    y_train, flipped = flip_labels(
        bundle.y_train, np.flatnonzero(bundle.roles == TRAIN), noise_rate,
        bundle.num_classes, derive_seed("label_noise", seed),
    )
    np.save(run_dir / "train_labels.npy", y_train)

    environment = collect_environment(PROJECT_ROOT)
    metadata = {
        "experiment_id": cfg["experiment"]["name"],
        "run_id": uuid.uuid4().hex,
        "seed": seed,
        "config_hash": config_hash(cfg),
        "config": cfg,
        "official_environment": environment["platform_kind"] == "kaggle",
        "bundle_dir": str(bundle.path),
        "dataset": bundle.meta,
        "vocabulary_sha256": vocabulary_sha256(bundle),
        "partition": {
            "dir": str(partitions_dir(cfg)),
            "name": summary["name"],
            "sha256": summary["sha256"],
            "type": part_cfg["scheme"],
            "alpha": part_cfg.get("alpha") if part_cfg["scheme"].startswith("dirichlet") else None,
            "partition_seed": summary["partition_seed"],
            "client_count": summary["num_clients"],
            "client_definition": bundle.meta.get("client_definition"),
            "client_filter_rule": (bundle.meta.get("client_filter") or {}).get("rule"),
            "created_in_this_run": created,
            "client_sizes": [c["num_samples"] for c in summary["clients"]],
            # realized label counts per client, not just the nominal alpha
            "client_class_counts": [c["class_counts"] for c in summary["clients"]],
        },
        "label_noise": {"rate": noise_rate, "num_flipped": int(len(flipped))},
        "environment": environment,
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=1, default=str)
    return run_dir


_CACHE: dict[str, Runtime] = {}


def load_runtime(run_dir, include_test: bool | None = None) -> Runtime:
    run_dir = Path(run_dir)
    with open(run_dir / "run_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if include_test is None:
        include_test = bool(metadata["config"].get("eval", {}).get("test", True))
    key = f"{run_dir}|test={include_test}"
    if key not in _CACHE:
        meta = metadata
        clients, _ = load_partition(Path(meta["partition"]["dir"]), meta["partition"]["name"])
        _CACHE[key] = Runtime(
            run_dir=run_dir,
            cfg=meta["config"],
            seed=meta["seed"],
            bundle=load_bundle(meta["bundle_dir"], include_test=include_test),
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
