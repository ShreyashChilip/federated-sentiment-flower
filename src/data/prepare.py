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
import sqlite3
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


def _save_csr_components(out: Path, name: str, matrix: sp.csr_matrix) -> None:
    """Save CSR arrays separately so worker processes can memory-map them."""
    np.save(out / f"{name}.data.npy", matrix.data, allow_pickle=False)
    np.save(out / f"{name}.indices.npy", matrix.indices, allow_pickle=False)
    np.save(out / f"{name}.indptr.npy", matrix.indptr, allow_pickle=False)
    (out / f"{name}.shape.json").write_text(json.dumps(matrix.shape), encoding="utf-8")


def _load_csr_components(path: Path, name: str) -> sp.csr_matrix:
    raw_meta = path / f"{name}.raw.json"
    if raw_meta.exists():
        meta = json.loads(raw_meta.read_text(encoding="utf-8"))
        data = np.memmap(path / f"{name}.data.bin", mode="r", dtype=np.dtype(meta["data_dtype"]), shape=meta["data_shape"])
        indices = np.memmap(path / f"{name}.indices.bin", mode="r", dtype=np.dtype(meta["indices_dtype"]), shape=meta["indices_shape"])
        indptr = np.memmap(path / f"{name}.indptr.bin", mode="r", dtype=np.dtype(meta["indptr_dtype"]), shape=meta["indptr_shape"])
        return sp.csr_matrix((data, indices, indptr), shape=tuple(meta["shape"]), copy=False)
    data = np.load(path / f"{name}.data.npy", mmap_mode="r")
    indices = np.load(path / f"{name}.indices.npy", mmap_mode="r")
    indptr = np.load(path / f"{name}.indptr.npy", mmap_mode="r")
    shape = tuple(json.loads((path / f"{name}.shape.json").read_text(encoding="utf-8")))
    return sp.csr_matrix((data, indices, indptr), shape=shape, copy=False)


def _write_csr_stream(out: Path, name: str, matrices, shape: tuple[int, int]) -> None:
    """Write CSR chunks without retaining the full sparse matrix in RAM."""
    data_path, indices_path, indptr_path = (out / f"{name}.{suffix}.bin" for suffix in ("data", "indices", "indptr"))
    nnz, rows = 0, 0
    data_dtype = indices_dtype = None
    with open(data_path, "wb") as data_file, open(indices_path, "wb") as indices_file, open(indptr_path, "wb") as indptr_file:
        np.asarray([0], dtype=np.int64).tofile(indptr_file)
        for matrix in matrices:
            matrix = matrix.tocsr()
            data_dtype = matrix.data.dtype
            indices_dtype = matrix.indices.dtype
            matrix.data.tofile(data_file)
            matrix.indices.tofile(indices_file)
            (matrix.indptr[1:].astype(np.int64, copy=False) + nnz).tofile(indptr_file)
            nnz += matrix.nnz
            rows += matrix.shape[0]
    if rows != shape[0]:
        raise ValueError(f"streamed CSR row count mismatch for {name}: expected {shape[0]}, got {rows}")
    (out / f"{name}.raw.json").write_text(json.dumps({
        "shape": [rows, shape[1]],
        "data_dtype": str(np.dtype(data_dtype)),
        "indices_dtype": str(np.dtype(indices_dtype)),
        "indptr_dtype": "int64",
        "data_shape": [nnz],
        "indices_shape": [nnz],
        "indptr_shape": [rows + 1],
    }), encoding="utf-8")


def _build_standard(cfg: dict, out: Path) -> dict:
    from src.data.datasets import load_text_dataset

    ds = load_text_dataset(cfg["data"])
    role_seed = derive_seed("roles", cfg["data"]["dataset"], cfg["data"].get("role_seed", 0))
    roles = assign_roles(ds.train_labels, cfg["data"]["role_fractions"], role_seed)
    # Vocabulary and IDF come from TRAIN-role documents only.
    vec = FederatedTfidf(cfg["features"]).fit([ds.train_texts[i] for i in np.flatnonzero(roles == TRAIN)])
    _save_csr_components(out, "x_train", vec.transform(ds.train_texts))
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
    store_path, source = amazon2023.open_review_store(data, cache_root(cfg))
    raw_keys, raw_class_counts = amazon2023.client_counts(store_path, definition)
    raw_sizes = raw_class_counts.sum(axis=1)
    rule = data["client_filter"]
    if rule.get("min_reviews") is None:
        raise ValueError("data.client_filter.min_reviews is not set; run client profiling before training")
    kept_indices = np.flatnonzero(raw_sizes >= int(rule["min_reviews"]))
    filter_rng = np.random.default_rng(int(rule.get("filter_seed", 0)))
    if rule.get("max_clients") and len(kept_indices) > int(rule["max_clients"]):
        kept_indices = np.sort(filter_rng.choice(kept_indices, size=int(rule["max_clients"]), replace=False))
    keys = [raw_keys[i] for i in kept_indices]
    cap = int(rule["max_reviews"]) if rule.get("max_reviews") else None
    clients_capped = int(sum(raw_sizes[i] > cap for i in kept_indices)) if cap else 0
    retained_count = int(sum(min(int(raw_sizes[i]), cap) if cap else int(raw_sizes[i]) for i in kept_indices))
    filter_report = {
        "rule": {k: rule.get(k) for k in ("min_reviews", "max_reviews", "max_clients", "filter_seed")},
        "raw_clients": len(raw_keys), "raw_records": int(source["cleaning_counts"]["kept"]),
        "clients_retained": len(keys), "records_retained": retained_count,
        "record_fraction_retained": retained_count / source["cleaning_counts"]["kept"],
        "clients_capped": clients_capped,
    }

    hold = data["holdout"]
    holdout_seed = derive_seed("holdout", data["dataset"], hold.get("holdout_seed", 0))
    seen, unseen = natural.split_clients(np.arange(len(keys)), float(hold["unseen_fraction"]), holdout_seed)
    role_seed = derive_seed("roles", data["dataset"], data.get("role_seed", 0))
    roles_rng = np.random.default_rng(role_seed)
    n_rows = retained_count
    labels = np.empty(n_rows, dtype=np.int64)
    codes = np.empty(n_rows, dtype=np.int64)
    roles = np.full(n_rows, natural.UNSEEN, dtype=np.int8)

    with sqlite3.connect(store_path) as connection:
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("CREATE TEMP TABLE selected_clients(client_key TEXT PRIMARY KEY, client_code INTEGER)")
        connection.executemany("INSERT INTO selected_clients VALUES(?,?)", ((key, code) for code, key in enumerate(keys)))
        connection.execute("CREATE TEMP TABLE selected_sources(source_row_id INTEGER PRIMARY KEY, client_code INTEGER)")
        if cap is None:
            connection.execute("""INSERT INTO selected_sources
                SELECT r.row_id, c.client_code FROM reviews r JOIN selected_clients c
                ON r.""" + definition + """=c.client_key ORDER BY r.row_id""")
        else:
            for client_code, raw_index in enumerate(kept_indices):
                key = raw_keys[int(raw_index)]
                if raw_sizes[raw_index] > cap:
                    row_ids = np.fromiter((row[0] for row in connection.execute(
                        f"SELECT row_id FROM reviews WHERE {definition}=? ORDER BY row_id", (key,))), dtype=np.int64)
                    selected = np.sort(filter_rng.permutation(row_ids)[:cap])
                    connection.executemany("INSERT INTO selected_sources VALUES(?,?)",
                                           ((int(row_id), client_code) for row_id in selected))
                else:
                    connection.execute(f"""INSERT INTO selected_sources
                        SELECT row_id, ? FROM reviews WHERE {definition}=? ORDER BY row_id""", (client_code, key))
        connection.execute("CREATE TEMP TABLE bundle_rows(seq INTEGER PRIMARY KEY, source_row_id INTEGER UNIQUE, client_code INTEGER, role INTEGER)")
        connection.execute("""INSERT INTO bundle_rows(seq,source_row_id,client_code,role)
            SELECT ROW_NUMBER() OVER (ORDER BY source_row_id)-1,source_row_id,client_code,?
            FROM selected_sources ORDER BY source_row_id""", (natural.UNSEEN,))
        actual_rows = connection.execute("SELECT COUNT(*) FROM bundle_rows").fetchone()[0]
        if actual_rows != n_rows:
            raise RuntimeError(f"retained row count mismatch: expected {n_rows}, selected {actual_rows}")
        connection.execute("CREATE INDEX bundle_rows_client_role_idx ON bundle_rows(client_code,role,seq)")
        for row in connection.execute("SELECT seq,client_code,label FROM bundle_rows JOIN reviews ON row_id=source_row_id ORDER BY seq"):
            seq, client_code, label = row
            labels[seq] = label
            codes[seq] = client_code

        seen_mask = np.zeros(len(keys), dtype=bool)
        seen_mask[seen] = True
        for client_code in range(len(keys)):
            if not seen_mask[client_code]:
                continue
            seqs = np.fromiter((row[0] for row in connection.execute(
                "SELECT seq FROM bundle_rows WHERE client_code=? ORDER BY seq", (client_code,))), dtype=np.int64)
            shuffled = roles_rng.permutation(seqs)
            n = len(seqs)
            n_test = max(1, int(round(data["role_fractions"][2] * n))) if n >= 2 else 0
            n_val = int(round(data["role_fractions"][1] * n)) if n - n_test >= 2 else 0
            n_train = n - n_val - n_test
            roles[shuffled[:n_train]] = TRAIN
            roles[shuffled[n_train:n_train + n_val]] = natural.VAL
            roles[shuffled[n_train + n_val:]] = natural.CLIENT_TEST
        connection.executemany("UPDATE bundle_rows SET role=? WHERE seq=?",
                               ((int(role), seq) for seq, role in enumerate(roles)))

        overlap = natural.assert_no_client_overlap(codes, roles)

        def train_texts_by_client():
            for client_code in seen:
                client_code = int(client_code)
                cursor = connection.execute("""SELECT r.text FROM bundle_rows b JOIN reviews r
                    ON r.row_id=b.source_row_id WHERE b.client_code=? AND b.role=? ORDER BY b.seq""",
                    (client_code, TRAIN))
                def text_batches(cursor=cursor):
                    while batch := cursor.fetchmany(5000):
                        yield [row[0] for row in batch]
                yield text_batches()

        vec = FederatedTfidf(cfg["features"]).fit_federated(train_texts_by_client())

        def transformed_chunks():
            cursor = connection.execute("""SELECT r.text FROM bundle_rows b JOIN reviews r
                ON r.row_id=b.source_row_id ORDER BY b.seq""")
            while batch := cursor.fetchmany(10_000):
                yield vec.transform([row[0] for row in batch])

        _write_csr_stream(out, "x_train", transformed_chunks(), (n_rows, len(vec.vocabulary_)))

    np.save(out / "y_train.npy", labels)
    np.save(out / "roles.npy", roles)
    np.save(out / "client_codes.npy", codes)
    with open(out / "client_keys.json", "w", encoding="utf-8") as f:
        json.dump([str(k) for k in keys], f)
    _save_features(out, vec)
    num_features = len(vec.vocabulary_)
    fit_documents = int(vec.n_docs_)
    vocabulary_hash = hashlib.sha256((out / "vocabulary.json").read_bytes()).hexdigest()
    return {
        "kind": "natural", "dataset": data["dataset"], "num_classes": amazon2023.NUM_CLASSES,
        "num_features": num_features, "fit_documents": fit_documents,
        "vocabulary_sha256": vocabulary_hash,
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


def load_bundle(path, include_test: bool = True) -> Bundle:
    """Load a bundle; cached per process so simulated clients share one copy."""
    path = Path(path)
    key = f"{path}|test={include_test}"
    if key not in _CACHE:
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        if (path / "x_train.data.npy").exists() or (path / "x_train.raw.json").exists():
            x_train = _load_csr_components(path, "x_train")
        else:
            x_train = sp.load_npz(path / "x_train.npz").tocsr()
        y_train, roles = np.load(path / "y_train.npy"), np.load(path / "roles.npy")
        if not include_test:
            x_test = sp.csr_matrix((0, x_train.shape[1]), dtype=x_train.dtype)
            y_test = np.empty(0, dtype=y_train.dtype)
            extra = dict(x_test=x_test, y_test=y_test)
            if meta.get("kind", "standard") == "natural":
                extra.update(client_codes=np.load(path / "client_codes.npy", mmap_mode="r"),
                             test_client_codes=np.empty(0, dtype=np.int64))
        elif meta.get("kind", "standard") == "natural":
            from src.data.natural import UNSEEN

            codes = np.load(path / "client_codes.npy")
            held = np.flatnonzero(roles == UNSEEN)
            extra = dict(x_test=x_train[held], y_test=y_train[held], client_codes=codes, test_client_codes=codes[held])
        else:
            extra = dict(x_test=sp.load_npz(path / "x_test.npz").tocsr(), y_test=np.load(path / "y_test.npy"))
        _CACHE[key] = Bundle(path=path, x_train=x_train, y_train=y_train, roles=roles,
                             num_classes=meta["num_classes"], meta=meta, **extra)
    return _CACHE[key]
