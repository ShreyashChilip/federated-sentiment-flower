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
from pathlib import Path

import numpy as np
import pandas as pd

HUB_ID = "McAuley-Lab/Amazon-Reviews-2023"
CLIENT_DEFINITIONS = ("user_id", "parent_asin")
NUM_CLASSES = 5


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
