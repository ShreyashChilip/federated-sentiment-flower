"""Pre-flight resource estimate for one benchmark run.

The estimate is deliberately conservative and is checked before a run is
launched: a run predicted to exceed the RAM or scratch-disk budget is not
started (unless forced), so an expensive configuration cannot silently OOM.

Memory model (resident set of the worker process):
  * interpreter + torch + scipy                         ~ 900 MB
  * labels, roles, partition indices, evaluator groups   ~ 40 bytes per row
  * model-sized arrays: global model, client model, start copy, client
    output, float64 accumulators, delta; + Adam-type moments (float64);
    + SCAFFOLD control variates and their deltas          k x P x 4 bytes
  * one evaluation chunk of the sparse matrix (+ copies) and its logits
  * the largest client's training rows (batches are views; one copy budgeted)
  * centralized: index array over all training rows
  * the memory-mapped feature matrix: evaluation reads rows spread over the
    whole file, so its pages become resident in the process (file-backed and
    reclaimable, but counted by RSS and by the container); counted in full.
    Kaggle Amazon pilot (Job 2): measured 2.9 GB peak, estimate ~3.4 GB.
Scratch disk: SCAFFOLD keeps one float32 model-sized row per client in a
file written with plain I/O (not mapped), so it is NOT in the process RSS;
the written rows sit in the kernel page cache (reclaimable). That amount is
reported as ``state_page_cache_gb`` and enters the worst-case container
figure, which is checked against physical memory as a warning only.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from src.benchmark.evaluation import CHUNK_ROWS
from src.data.prepare import bundle_dir

MODEL_COPIES = {"default": 12, "fedadam": 16, "fedyogi": 16, "fedadagrad": 16, "scaffold": 22}


def bundle_stats(cfg: dict) -> dict | None:
    path = bundle_dir(cfg)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = path / "x_train.raw.json"
    if raw.exists():
        r = json.loads(raw.read_text(encoding="utf-8"))
        rows, nnz = r["shape"][0], r["data_shape"][0]
    elif (path / "x_train.shape.json").exists():
        shape = json.loads((path / "x_train.shape.json").read_text(encoding="utf-8"))
        rows = shape[0]
        nnz = int(np.load(path / "x_train.indices.npy", mmap_mode="r").shape[0])
    else:  # legacy single-file bundle
        with np.load(path / "x_train.npz") as z:
            rows, nnz = int(z["shape"][0]), int(z["indices"].shape[0])
    return {"rows": int(rows), "nnz": int(nnz), "features": int(meta["num_features"]),
            "classes": int(meta["num_classes"]), "kind": meta.get("kind", "standard"),
            "matrix_bytes": int(nnz * 8 + rows * 8)}


def estimate(spec: dict, num_clients: int | None = None, max_client_rows: int | None = None) -> dict:
    cfg = spec["cfg"]
    stats = bundle_stats(cfg)
    if stats is None:
        return {"known": False, "reason": "feature bundle not built yet; it is built by the first run of the regime"}
    p = (stats["features"] + 1) * stats["classes"]
    model_bytes = p * 4
    copies = MODEL_COPIES.get(cfg["fl"].get("algorithm") if spec["kind"] == "federated" else "default", MODEL_COPIES["default"])
    avg_nnz = stats["nnz"] / max(stats["rows"], 1)
    rss = 900 * 2**20
    rss += stats["rows"] * 40
    rss += copies * model_bytes
    rss += CHUNK_ROWS * avg_nnz * 12 * 3 + CHUNK_ROWS * stats["classes"] * 8 * 4
    if max_client_rows:
        rss += max_client_rows * avg_nnz * 12
    if spec["kind"] == "centralized":
        rss += stats["rows"] * 16
    rss += stats["matrix_bytes"]
    scratch = 0
    if spec["kind"] == "federated" and cfg["fl"].get("algorithm") == "scaffold":
        scratch = (num_clients or 0) * model_bytes
    return {"known": True, "rss_gb": rss / 2**30, "scratch_disk_gb": scratch / 2**30,
            "mapped_matrix_gb": stats["matrix_bytes"] / 2**30, "state_page_cache_gb": scratch / 2**30,
            "worst_case_container_gb": (rss + scratch) / 2**30, "params": p, "bundle": stats}


def budget() -> dict:
    """RAM budget: FEDBENCH_RAM_GB, else 90% of physical memory."""
    try:
        import psutil

        total = psutil.virtual_memory().total / 2**30
    except Exception:
        total = 16.0
    ram = float(os.environ.get("FEDBENCH_RAM_GB") or 0.9 * total)
    scratch_dir = Path(os.environ.get("FEDBENCH_SCRATCH") or ".")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(scratch_dir).free / 2**30
    return {"ram_gb": ram, "physical_gb": total, "scratch_free_gb": disk, "scratch_dir": str(scratch_dir)}


def check(spec: dict, jobs: int = 1, **kw) -> tuple[bool, dict]:
    est = estimate(spec, **kw)
    b = budget()
    report = {"estimate": est, "budget": b, "jobs": jobs}
    if not est["known"]:
        return True, report
    problems = []
    if est["rss_gb"] * jobs > b["ram_gb"]:
        problems.append(f"estimated RSS {est['rss_gb']:.1f} GB x {jobs} job(s) exceeds the RAM budget {b['ram_gb']:.1f} GB")
    if est["scratch_disk_gb"] > 0.9 * b["scratch_free_gb"]:
        problems.append(f"SCAFFOLD client state needs up to {est['scratch_disk_gb']:.1f} GB; "
                        f"{b['scratch_free_gb']:.1f} GB free in {b['scratch_dir']}")
    report["problems"] = problems
    # Page cache is reclaimable, so exceeding physical memory here is a warning, not a refusal.
    if est["worst_case_container_gb"] * jobs > b["physical_gb"]:
        report["warnings"] = [f"RSS + SCAFFOLD state page cache {est['worst_case_container_gb']:.1f} GB x {jobs} "
                              f"exceeds physical memory {b['physical_gb']:.1f} GB (cache is reclaimable)"]
    return not problems, report
