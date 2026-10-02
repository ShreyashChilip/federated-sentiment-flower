"""Build (once) and load the feature bundle shared by all runs of a dataset.

A bundle holds the feature matrix, the labels and the fixed sample roles. It
depends only on the ``data`` and ``features`` config sections, never on the
algorithm or the experiment seed, so every method is compared on
byte-identical inputs.

Two kinds of bundle exist:

* ``standard`` (Yelp): official train split with train / val / client-test
  roles, plus the official test split.
* ``natural`` (Amazon Reviews 2023): one table of cleaned, size-filtered
  records with a real client key per record. Clients are split into seen and
  unseen; every record of an unseen client has the role ``UNSEEN`` and that
  set is the bundle's test set.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.data.roles import TRAIN, assign_roles
from src.preprocessing.tfidf import FederatedTfidf
from src.utils.config import PROJECT_ROOT, config_hash
from src.utils.seeding import derive_seed

NATURAL_DATASETS = ("amazon2023",)


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
    client_codes: np.ndarray | None = None   # natural bundles: client code per row of x_train
    test_client_codes: np.ndarray | None = None  # natural bundles: client code per test row

    @property
    def input_dim(self) -> int:
        return self.x_train.shape[1]

    @property
    def kind(self) -> str:
        return self.meta.get("kind", "standard")

    def vocabulary(self) -> list[str]:
        """Feature index -> TF-IDF term (position j is column j of the features)."""
        with open(self.path / "vocabulary.json", "r", encoding="utf-8") as f:
            return json.load(f)


def cache_root(cfg: dict) -> Path:
    cache = Path(cfg["data"].get("cache_dir") or "data_cache")
    return cache if cache.is_absolute() else PROJECT_ROOT / cache


def bundle_dir(cfg: dict) -> Path:
    key = config_hash({"data": {k: v for k, v in cfg["data"].items() if k != "cache_dir"}, "features": cfg["features"]})
    return cache_root(cfg) / "features" / f"{cfg['data']['dataset']}_{key}"


def _save_features(out: Path, vec: FederatedTfidf) -> str:
    np.save(out / "idf.npy", vec.idf_)
    np.save(out / "df.npy", vec.df_)
    blob = json.dumps(vec.feature_names, ensure_ascii=False).encode("utf-8")
    (out / "vocabulary.json").write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def _build_standard(cfg: dict, out: Path) -> dict:
    from src.data.datasets import load_text_dataset

    ds = load_text_dataset(cfg["data"])
    role_seed = derive_seed("roles", cfg["data"]["dataset"], cfg["data"].get("role_seed", 0))
    roles = assign_roles(ds.train_labels, cfg["data"]["role_fractions"], role_seed)
    # Vocabulary and IDF come from TRAIN-role documents only.
    vec = FederatedTfidf(cfg["features"]).fit([ds.train_texts[i] for i in np.flatnonzero(roles == TRAIN)])
    sp.save_npz(out / "x_train.npz", vec.transform(ds.train_texts))
    sp.save_npz(out / "x_test.npz", vec.transform(ds.test_texts))
    np.save(out / "y_train.npy", ds.train_labels)
    np.save(out / "y_test.npy", ds.test_labels)
    np.save(out / "roles.npy", roles)
    return {
        "kind": "standard", "dataset": ds.name, "num_classes": ds.num_classes,
        "num_features": len(vec.vocabulary_), "fit_documents": int(vec.n_docs_),
        "vocabulary_sha256": _save_features(out, vec), "role_seed": role_seed,
        "role_counts": np.bincount(roles, minlength=3).tolist(), **ds.meta,
    }


def _build_natural(cfg: dict, out: Path) -> dict:
    from src.data import amazon2023, natural

    data = cfg["data"]
    definition = data.get("client_definition")
    if definition not in amazon2023.CLIENT_DEFINITIONS:
        raise ValueError(f"data.client_definition must be one of {amazon2023.CLIENT_DEFINITIONS}, got {definition!r}")
    table, source = amazon2023.load_reviews(data, cache_root(cfg))
    raw_codes, _ = natural.encode_clients(table[definition].to_numpy())
    keep, filter_report = natural.filter_clients(raw_codes, data["client_filter"])
    table = table.loc[keep].reset_index(drop=True)
    codes, keys = natural.encode_clients(table[definition].to_numpy())
    labels = table["label"].to_numpy().astype(np.int64)

    hold = data["holdout"]
    holdout_seed = derive_seed("holdout", data["dataset"], hold.get("holdout_seed", 0))
    seen, unseen = natural.split_clients(np.arange(len(keys)), float(hold["unseen_fraction"]), holdout_seed)
    role_seed = derive_seed("roles", data["dataset"], data.get("role_seed", 0))
    roles = natural.assign_roles_within_clients(codes, seen, data["role_fractions"], role_seed)
    overlap = natural.assert_no_client_overlap(codes, roles)

    texts = table["text"].tolist()
    vec = FederatedTfidf(cfg["features"]).fit([texts[i] for i in np.flatnonzero(roles == TRAIN)])
    sp.save_npz(out / "x_train.npz", vec.transform(texts))
    np.save(out / "y_train.npy", labels)
    np.save(out / "roles.npy", roles)
    np.save(out / "client_codes.npy", codes)
    with open(out / "client_keys.json", "w", encoding="utf-8") as f:
        json.dump([str(k) for k in keys], f)
    return {
        "kind": "natural", "dataset": data["dataset"], "num_classes": amazon2023.NUM_CLASSES,
        "num_features": len(vec.vocabulary_), "fit_documents": int(vec.n_docs_),
        "vocabulary_sha256": _save_features(out, vec),
        "client_definition": definition, "client_filter": filter_report,
        "holdout": {"unseen_fraction": float(hold["unseen_fraction"]), "holdout_seed": holdout_seed, **overlap},
        "role_seed": role_seed, "role_counts": np.bincount(roles, minlength=4).tolist(),
        "train_size": int(len(labels)), "test_size": overlap["unseen_records"],
        "train_class_counts": np.bincount(labels, minlength=amazon2023.NUM_CLASSES).tolist(),
        "train_fingerprint": hashlib.sha256(labels.tobytes() + codes.tobytes()).hexdigest()[:16],
        "test_fingerprint": hashlib.sha256(np.flatnonzero(roles == natural.UNSEEN).tobytes()).hexdigest()[:16],
        **source,
    }


def build_bundle(cfg: dict) -> Path:
    """Create the bundle if it does not exist yet and return its directory."""
    out = bundle_dir(cfg)
    if (out / "meta.json").exists():
        return out
    if cfg["features"]["type"] != "tfidf":
        raise ValueError("only TF-IDF features are implemented so far")
    out.mkdir(parents=True, exist_ok=True)
    builder = _build_natural if cfg["data"]["dataset"] in NATURAL_DATASETS else _build_standard
    meta = builder(cfg, out)
    meta.update(data_config=cfg["data"], features_config=cfg["features"])
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)
    return out


_CACHE: dict[str, Bundle] = {}


def load_bundle(path) -> Bundle:
    """Load a bundle; cached per process so simulated clients share one copy."""
    path = Path(path)
    key = str(path)
    if key not in _CACHE:
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        x_train = sp.load_npz(path / "x_train.npz").tocsr()
        y_train, roles = np.load(path / "y_train.npy"), np.load(path / "roles.npy")
        if meta.get("kind", "standard") == "natural":
            from src.data.natural import UNSEEN

            codes = np.load(path / "client_codes.npy")
            held = np.flatnonzero(roles == UNSEEN)
            extra = dict(x_test=x_train[held], y_test=y_train[held], client_codes=codes, test_client_codes=codes[held])
        else:
            extra = dict(x_test=sp.load_npz(path / "x_test.npz").tocsr(), y_test=np.load(path / "y_test.npy"))
        _CACHE[key] = Bundle(path=path, x_train=x_train, y_train=y_train, roles=roles,
                             num_classes=meta["num_classes"], meta=meta, **extra)
    return _CACHE[key]
