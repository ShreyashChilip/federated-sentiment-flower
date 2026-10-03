"""Amazon Reviews 2023 (McAuley Lab): raw per-category review files.

Source: Hugging Face dataset ``McAuley-Lab/Amazon-Reviews-2023``, files
``raw/review_categories/<Category>.jsonl``. Each line is one review with the
fields ``rating`` (1.0-5.0), ``title``, ``text``, ``asin``, ``parent_asin``,
``user_id``, ``timestamp``, ``helpful_vote``, ``verified_purchase``.

The task is the full 5-class rating task: label = rating - 1 in {0, ..., 4}.
The ratings are never collapsed to polarity.

Record-level cleaning (fixed data-quality rules, independent of any model):
  1. the rating must be one of 1, 2, 3, 4, 5;
  2. ``user_id`` and ``parent_asin`` must be present;
  3. the review body, after stripping white space, must be non-empty;
  4. exact duplicates of (user_id, parent_asin, text) are dropped, keeping the
     first occurrence in file order.
The model input is the title and the body joined by a newline.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HUB_ID = "McAuley-Lab/Amazon-Reviews-2023"
CLIENT_DEFINITIONS = ("user_id", "parent_asin")
NUM_CLASSES = 5
REVIEW_STORE_VERSION = 1


def resolve_file(cfg: dict, cache_dir: Path) -> Path:
    """Local path of the category file (downloaded and cached on first use)."""
    if cfg.get("local_path"):
        return Path(cfg["local_path"])
    if not cfg.get("category"):
        raise ValueError("data.category is not set; the Amazon category must be fixed in the config before any run")
    if not cfg.get("revision"):
        raise ValueError("data.revision is not set; pin the dataset revision so the download is reproducible")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        HUB_ID, f"raw/review_categories/{cfg['category']}.jsonl", repo_type="dataset",
        revision=cfg["revision"], cache_dir=str(cache_dir / "hf"),
    ))


def read_reviews(path: Path, max_records: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Parse and clean a category file. Returns the table and cleaning counts."""
    rows, seen = [], set()
    counts = {"lines": 0, "bad_json": 0, "bad_rating": 0, "missing_id": 0, "empty_text": 0, "duplicate": 0}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_records is not None and counts["lines"] >= max_records:
                break
            counts["lines"] += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                counts["bad_json"] += 1
                continue
            rating = r.get("rating")
            if rating not in (1, 2, 3, 4, 5, 1.0, 2.0, 3.0, 4.0, 5.0):
                counts["bad_rating"] += 1
                continue
            user, item = r.get("user_id"), r.get("parent_asin")
            if not user or not item:
                counts["missing_id"] += 1
                continue
            body = (r.get("text") or "").strip()
            if not body:
                counts["empty_text"] += 1
                continue
            key = hashlib.blake2b(f"{user}\x1f{item}\x1f{body}".encode("utf-8"), digest_size=12).digest()
            if key in seen:
                counts["duplicate"] += 1
                continue
            seen.add(key)
            title = (r.get("title") or "").strip()
            rows.append((user, item, int(rating) - 1, f"{title}\n{body}" if title else body, r.get("timestamp")))
    table = pd.DataFrame(rows, columns=["user_id", "parent_asin", "label", "text", "timestamp"])
    del rows, seen
    counts["kept"] = len(table)
    return table, counts


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_reviews(cfg: dict, cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    path = resolve_file(cfg, cache_dir)
    table, counts = read_reviews(path, cfg.get("max_records"))
    meta = {
        "hub_id": HUB_ID,
        "category": cfg.get("category"),
        "revision": cfg.get("revision"),
        "source_file": path.name,
        "source_sha256": file_sha256(path),
        "cleaning_counts": counts,
        "class_counts": np.bincount(table["label"].to_numpy(), minlength=NUM_CLASSES).tolist(),
    }
    return table, meta


def open_review_store(cfg: dict, cache_dir: Path) -> tuple[Path, dict]:
    """Create or reuse a disk-backed cleaned review store.

    The full category can contain millions of long reviews. Keeping Python
    tuples, a duplicate set, and a DataFrame alive together can exceed notebook
    memory before model preprocessing starts. SQLite keeps the same cleaned
    records and first-occurrence duplicate rule on disk instead.
    """
    source_path = resolve_file(cfg, cache_dir)
    source_hash = file_sha256(source_path)
    max_records = cfg.get("max_records")
    cache_dir = Path(cache_dir) / "review_store"
    cache_dir.mkdir(parents=True, exist_ok=True)
    limit_tag = str(max_records) if max_records is not None else "all"
    store_path = cache_dir / f"v{REVIEW_STORE_VERSION}_{source_hash[:16]}_{limit_tag}.sqlite"

    if store_path.exists():
        try:
            with sqlite3.connect(store_path) as connection:
                stored = dict(connection.execute("SELECT key, value FROM store_meta"))
            if (stored.get("complete") == "1" and stored.get("source_sha256") == source_hash
                    and stored.get("store_version") == str(REVIEW_STORE_VERSION)):
                return store_path, _review_store_metadata(store_path, cfg, source_path, source_hash)
        except sqlite3.Error:
            pass
        store_path.unlink(missing_ok=True)

    counts = {"lines": 0, "bad_json": 0, "bad_rating": 0, "missing_id": 0,
              "empty_text": 0, "duplicate": 0, "kept": 0}
    connection = sqlite3.connect(store_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute("""CREATE TABLE reviews (
            row_id INTEGER PRIMARY KEY, user_id TEXT NOT NULL,
            parent_asin TEXT NOT NULL, label INTEGER NOT NULL,
            text TEXT NOT NULL, duplicate_key BLOB NOT NULL UNIQUE
        )""")
        with open(source_path, "r", encoding="utf-8") as source:
            for line in source:
                if max_records is not None and counts["lines"] >= max_records:
                    break
                counts["lines"] += 1
                try:
                    review = json.loads(line)
                except json.JSONDecodeError:
                    counts["bad_json"] += 1
                    continue
                rating = review.get("rating")
                if rating not in (1, 2, 3, 4, 5, 1.0, 2.0, 3.0, 4.0, 5.0):
                    counts["bad_rating"] += 1
                    continue
                user, item = review.get("user_id"), review.get("parent_asin")
                if not user or not item:
                    counts["missing_id"] += 1
                    continue
                body = (review.get("text") or "").strip()
                if not body:
                    counts["empty_text"] += 1
                    continue
                duplicate_key = hashlib.blake2b(
                    f"{user}\x1f{item}\x1f{body}".encode("utf-8"), digest_size=12
                ).digest()
                title = (review.get("title") or "").strip()
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO reviews(user_id,parent_asin,label,text,duplicate_key) VALUES(?,?,?,?,?)",
                    (user, item, int(rating) - 1, f"{title}\n{body}" if title else body, duplicate_key),
                )
                if cursor.rowcount:
                    counts["kept"] += 1
                else:
                    counts["duplicate"] += 1
                if counts["lines"] % 5000 == 0:
                    connection.commit()
        connection.execute("CREATE INDEX reviews_user_label_idx ON reviews(user_id,label)")
        connection.execute("CREATE INDEX reviews_product_label_idx ON reviews(parent_asin,label)")
        connection.execute("CREATE TABLE store_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.executemany("INSERT INTO store_meta(key,value) VALUES(?,?)", [
            ("complete", "1"), ("source_sha256", source_hash),
            ("counts", json.dumps(counts)), ("max_records", limit_tag),
            ("store_version", str(REVIEW_STORE_VERSION)),
        ])
        connection.commit()
    except Exception:
        connection.close()
        store_path.unlink(missing_ok=True)
        raise
    connection.close()
    return store_path, _review_store_metadata(store_path, cfg, source_path, source_hash)


def _review_store_metadata(store_path: Path, cfg: dict, source_path: Path, source_hash: str) -> dict:
    with sqlite3.connect(store_path) as connection:
        stored = dict(connection.execute("SELECT key, value FROM store_meta"))
        class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for label, count in connection.execute("SELECT label, COUNT(*) FROM reviews GROUP BY label"):
            class_counts[int(label)] = int(count)
    return {
        "hub_id": HUB_ID, "category": cfg.get("category"), "revision": cfg.get("revision"),
        "source_file": source_path.name, "source_sha256": source_hash,
        "cleaning_counts": json.loads(stored["counts"]), "class_counts": class_counts.tolist(),
    }


def client_counts(store_path: Path, definition: str) -> tuple[list[str], np.ndarray]:
    """Return clients in the same sorted order as ``numpy.unique`` and class counts."""
    if definition not in CLIENT_DEFINITIONS:
        raise ValueError(f"client definition must be one of {CLIENT_DEFINITIONS}, got {definition!r}")
    with sqlite3.connect(store_path) as connection:
        num_clients = connection.execute(f"SELECT COUNT(DISTINCT {definition}) FROM reviews").fetchone()[0]
    counts = np.zeros((num_clients, NUM_CLASSES), dtype=np.int64)
    keys: list[str] = []
    with sqlite3.connect(store_path) as connection:
        rows = connection.execute(
            f"SELECT {definition}, label, COUNT(*) FROM reviews GROUP BY {definition}, label ORDER BY {definition}, label"
        )
        current = None
        client_index = -1
        for key, label, count in rows:
            key = str(key)
            if current is not None and key != current:
                keys.append(current)
                client_index += 1
            current = key
            if client_index < 0:
                client_index = 0
            counts[client_index, int(label)] = int(count)
        if current is not None:
            keys.append(current)
    return keys, counts
