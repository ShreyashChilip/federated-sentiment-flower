"""Saving and loading partitions.

A partition is created once, saved, and from then on only loaded. Loading
verifies the stored hash, so an experiment can never silently run on a
different split from the one that is on disk.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.partitioning.partition import make_partition
from src.utils.seeding import derive_seed


def partition_hash(clients: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for cid, idx in enumerate(clients):
        h.update(f"client{cid}:{len(idx)}:".encode())
        h.update(np.asarray(idx, dtype=np.int64).tobytes())
    return h.hexdigest()


def partition_name(dataset: str, n_samples: int, cfg: dict, seed: int) -> str:
    scheme = cfg["scheme"]
    if scheme == "natural":
        # Natural clients come from the data; the partition does not depend on the seed.
        return f"{dataset}_n{n_samples}_natural"
    tag = {"iid": "iid", "dirichlet": f"dir{cfg.get('alpha')}", "dirichlet_client": f"dirclient{cfg.get('alpha')}",
           "quantity_skew": f"qty{cfg.get('beta')}"}[scheme]
    if scheme == "dirichlet" and cfg.get("min_client_size", 10) != 10:
        tag += f"min{cfg['min_client_size']}"  # a non-default size rule is a different partition
    return f"{dataset}_n{n_samples}_{tag}_N{cfg['num_clients']}_seed{seed}"


def describe(clients, labels, roles, num_classes) -> dict:
    labels = np.asarray(labels)
    per_client = []
    for cid, idx in enumerate(clients):
        entry = {
            "client_id": cid,
            "num_samples": int(len(idx)),
            "class_counts": np.bincount(labels[idx], minlength=num_classes).tolist(),
        }
        if roles is not None:
            entry["role_counts"] = np.bincount(roles[idx], minlength=3).tolist()
        per_client.append(entry)
    return {"clients": per_client}


def save_partition(directory: Path, name: str, clients, summary: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    sizes = np.array([len(c) for c in clients], dtype=np.int64)
    np.savez_compressed(
        directory / f"{name}.npz",
        indices=np.concatenate(clients).astype(np.int64),
        offsets=np.concatenate([[0], np.cumsum(sizes)]),
    )
    with open(directory / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return directory / f"{name}.npz"


def load_partition(directory: Path, name: str) -> tuple[list[np.ndarray], dict]:
    with open(directory / f"{name}.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    with np.load(directory / f"{name}.npz") as data:
        # Read each array ONCE: every ``data[key]`` access decompresses the whole
        # array again, and a per-client slice of it keeps that full copy alive, so
        # indexing inside the loop costs (clients x rows) memory (OOM with natural
        # clients). All clients are views into this single array.
        offsets = data["offsets"]
        indices = data["indices"]
    clients = [indices[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    actual = partition_hash(clients)
    if actual != summary["sha256"]:
        raise RuntimeError(f"partition {name}: stored hash {summary['sha256'][:12]} != actual {actual[:12]}")
    return clients, summary


def get_partition(directory, dataset, labels, roles, num_classes, cfg, seed, create=False, groups=None):
    """Load a saved partition; create it only when explicitly asked to.

    Returns ``(clients, summary, created)``.
    """
    directory = Path(directory)
    name = partition_name(dataset, len(labels), cfg, seed)
    if (directory / f"{name}.json").exists():
        clients, summary = load_partition(directory, name)
        if summary["num_samples"] != len(labels):
            raise RuntimeError(f"partition {name} was built for a different dataset size")
        return clients, summary, False
    if not create:
        raise FileNotFoundError(
            f"partition '{name}' not found in {directory}. Partitions are never regenerated "
            "implicitly; run experiments/make_partitions.py or pass create=True."
        )
    partition_seed = derive_seed("partition", seed)
    clients, extra = make_partition(labels, cfg, partition_seed, groups)
    summary = {
        "name": name,
        "dataset": dataset,
        "num_samples": int(len(labels)),
        "num_clients": len(clients),
        "params": {k: v for k, v in cfg.items()},
        "experiment_seed": seed,
        "partition_seed": partition_seed,
        "sha256": partition_hash(clients),
        **extra,
        **describe(clients, labels, roles, num_classes),
    }
    save_partition(directory, name, clients, summary)
    return clients, summary, True
