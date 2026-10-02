"""Build (once) and load the feature bundle shared by all runs of a dataset.

A bundle holds the feature matrices for the official train and test splits,
the labels and the fixed sample roles. It depends only on the ``data`` and
``features`` config sections, never on the partition, algorithm or seed, so
every method is compared on byte-identical inputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.data.datasets import load_text_dataset
from src.data.roles import TRAIN, assign_roles
from src.preprocessing.tfidf import FederatedTfidf
from src.utils.config import PROJECT_ROOT, config_hash
from src.utils.seeding import derive_seed


@dataclass
class Bundle:
    path: Path
    x_train: sp.csr_matrix
    y_train: np.ndarray
    roles: np.ndarray
    x_test: sp.csr_matrix
    y_test: np.ndarray
    num_classes: int
    meta: dict

    @property
    def input_dim(self) -> int:
        return self.x_train.shape[1]


def bundle_dir(cfg: dict) -> Path:
    cache = Path(cfg["data"].get("cache_dir") or "data_cache")
    if not cache.is_absolute():
        cache = PROJECT_ROOT / cache
    key = config_hash({"data": {k: v for k, v in cfg["data"].items() if k != "cache_dir"}, "features": cfg["features"]})
    return cache / "features" / f"{cfg['data']['dataset']}_{key}"


def build_bundle(cfg: dict) -> Path:
    """Create the bundle if it does not exist yet and return its directory."""
    out = bundle_dir(cfg)
    if (out / "meta.json").exists():
        return out
    if cfg["features"]["type"] != "tfidf":
        raise ValueError("only TF-IDF features are implemented so far")
    out.mkdir(parents=True, exist_ok=True)

    ds = load_text_dataset(cfg["data"])
    role_seed = derive_seed("roles", cfg["data"]["dataset"], cfg["data"].get("role_seed", 0))
    roles = assign_roles(ds.train_labels, cfg["data"]["role_fractions"], role_seed)

    # Vocabulary and IDF come from TRAIN-role documents only.
    train_role = np.flatnonzero(roles == TRAIN)
    vec = FederatedTfidf(cfg["features"]).fit([ds.train_texts[i] for i in train_role])
    sp.save_npz(out / "x_train.npz", vec.transform(ds.train_texts))
    sp.save_npz(out / "x_test.npz", vec.transform(ds.test_texts))
    np.save(out / "y_train.npy", ds.train_labels)
    np.save(out / "y_test.npy", ds.test_labels)
    np.save(out / "roles.npy", roles)
    np.save(out / "idf.npy", vec.idf_)
    np.save(out / "df.npy", vec.df_)
    with open(out / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vec.feature_names, f)
    meta = {
        "dataset": ds.name,
        "num_classes": ds.num_classes,
        "num_features": len(vec.vocabulary_),
        "fit_documents": int(vec.n_docs_),
        "role_seed": role_seed,
        "role_counts": np.bincount(roles, minlength=3).tolist(),
        "data_config": cfg["data"],
        "features_config": cfg["features"],
        **ds.meta,
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return out


_CACHE: dict[str, Bundle] = {}


def load_bundle(path) -> Bundle:
    """Load a bundle; cached per process so simulated clients share one copy."""
    path = Path(path)
    key = str(path)
    if key not in _CACHE:
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        _CACHE[key] = Bundle(
            path=path,
            x_train=sp.load_npz(path / "x_train.npz").tocsr(),
            y_train=np.load(path / "y_train.npy"),
            roles=np.load(path / "roles.npy"),
            x_test=sp.load_npz(path / "x_test.npz").tocsr(),
            y_test=np.load(path / "y_test.npy"),
            num_classes=meta["num_classes"],
            meta=meta,
        )
    return _CACHE[key]
