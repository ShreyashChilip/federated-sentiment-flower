"""Dataset loading.

Only the official train split is ever handed to partitioning, feature fitting
or tuning. The official test split is returned separately and is used for
final reporting only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.config import PROJECT_ROOT
from src.utils.seeding import derive_seed

# name -> (hub id, text column, label column, label offset, number of classes)
REGISTRY = {
    "yelp_polarity": ("fancyzhx/yelp_polarity", "text", "label", 0, 2),
}


@dataclass
class TextDataset:
    name: str
    train_texts: list
    train_labels: np.ndarray
    test_texts: list
    test_labels: np.ndarray
    num_classes: int
    meta: dict = field(default_factory=dict)


def stratified_subset(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Sorted indices of a class-proportional random subset of ``size`` items."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    quota = np.floor(counts / counts.sum() * size).astype(int)
    # hand out the rounding remainder to the largest classes, deterministically
    for i in np.argsort(-counts, kind="stable")[: size - quota.sum()]:
        quota[i] += 1
    picked = [
        rng.choice(np.flatnonzero(labels == c), size=q, replace=False)
        for c, q in zip(classes, quota)
    ]
    return np.sort(np.concatenate(picked))


def _fingerprint(labels: np.ndarray, texts: list) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(labels, dtype=np.int64).tobytes())
    for t in texts[:1000]:
        h.update(t.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def load_text_dataset(cfg: dict) -> TextDataset:
    """Load a dataset described by the ``data`` section of a run config."""
    from datasets import load_dataset

    name = cfg["dataset"]
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset '{name}'; known: {sorted(REGISTRY)}")
    hub_id, text_col, label_col, offset, num_classes = REGISTRY[name]
    cache_dir = Path(cfg.get("cache_dir") or "data_cache")
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    ds = load_dataset(hub_id, revision=cfg.get("revision"), cache_dir=str(cache_dir / "hf"))

    meta = {
        "hub_id": hub_id,
        "revision": cfg.get("revision"),
        "official_train_size": ds["train"].num_rows,
        "official_test_size": ds["test"].num_rows,
    }
    out = {}
    for split, limit_key in (("train", "max_train"), ("test", "max_test")):
        labels = np.asarray(ds[split][label_col], dtype=np.int64) - offset
        limit = cfg.get(limit_key)
        if limit and limit < len(labels):
            # Predefined rule: class-proportional subset with a fixed seed.
            idx = stratified_subset(labels, limit, derive_seed("subset", name, split, cfg.get("subset_seed", 0)))
            texts = ds[split].select(idx.tolist())[text_col]
            labels = labels[idx]
            meta[f"{split}_subset_indices_sha256"] = hashlib.sha256(idx.tobytes()).hexdigest()[:16]
        else:
            texts = ds[split][text_col]
        out[split] = (list(texts), labels)
        meta[f"{split}_size"] = int(len(labels))
        meta[f"{split}_class_counts"] = np.bincount(labels, minlength=num_classes).tolist()
        meta[f"{split}_fingerprint"] = _fingerprint(labels, out[split][0])

    return TextDataset(name, out["train"][0], out["train"][1], out["test"][0], out["test"][1], num_classes, meta)
