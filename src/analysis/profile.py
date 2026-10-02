"""Write the natural-client profile of one category and client definition."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.data import amazon2023, natural
from src.data.prepare import cache_root
from src.utils.config import PROJECT_ROOT, config_hash
from src.utils.env import collect_environment


def profile_and_save(cfg: dict, out_root=None) -> tuple[Path, dict]:
    """Profile raw clients and save ``profile.json`` and ``clients.csv``.

    The output directory is named after the category, the client definition
    and the hash of the source file, and is never overwritten.
    """
    data = cfg["data"]
    definition = data["client_definition"]
    if definition not in amazon2023.CLIENT_DEFINITIONS:
        raise ValueError(f"data.client_definition must be one of {amazon2023.CLIENT_DEFINITIONS}")
    table, source = amazon2023.load_reviews(data, cache_root(cfg))
    codes, keys = natural.encode_clients(table[definition].to_numpy())
    report = natural.profile_clients(codes, table["label"].to_numpy(), amazon2023.NUM_CLASSES, cfg["profile"]["thresholds"])
    per_client = report.pop("per_client_table")

    out_root = Path(out_root) if out_root else PROJECT_ROOT / cfg["experiment"]["results_dir"] / "exp0b_profile"
    out = out_root / f"{data.get('category') or 'local'}_{definition}_{source['source_sha256'][:8]}"
    if (out / "profile.json").exists():
        raise FileExistsError(f"{out} already holds a profile; results are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(PROJECT_ROOT)
    report = {
        "experiment_id": "exp0b_profile",
        "client_definition": definition,
        "source": source,
        "config_hash": config_hash(cfg),
        "official_environment": environment["platform_kind"] == "kaggle",
        "environment": environment,
        **report,
    }
    with open(out / "profile.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    with open(out / "clients.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["client_key", "num_reviews", *[f"count_{c + 1}_star" for c in range(amazon2023.NUM_CLASSES)]])
        for key, size, row in zip(keys, per_client["sizes"], per_client["label_counts"]):
            writer.writerow([key, int(size), *np.asarray(row).tolist()])
    return out, report
