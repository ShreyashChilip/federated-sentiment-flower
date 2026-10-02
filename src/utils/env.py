"""Environment and provenance metadata recorded with every run."""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import os
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = (
    "flwr", "torch", "transformers", "peft", "datasets", "scikit-learn", "numpy",
    "scipy", "pandas", "pyarrow", "ray", "matplotlib", "PyYAML", "psutil", "opacus",
)


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = _md.version(name)
        except _md.PackageNotFoundError:
            out[name] = None
    return out


def git_info(root: Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except Exception:
            return ""

    # Only trust git if the repository root is this project. The laptop has an
    # unrelated repository at the drive root which must not be reported.
    top = run("rev-parse", "--show-toplevel")
    if not top or Path(top).resolve() != Path(root).resolve():
        return {"commit": None, "dirty": None, "note": "project is not a git repository"}
    commit = run("rev-parse", "--verify", "--quiet", "HEAD") or None  # None until the first commit
    # "dirty" means a TRACKED file differs from the commit, i.e. the code or a
    # config that produced the run is not the committed one. New untracked
    # files (freshly created partitions, results) do not change the code and
    # are counted separately.
    untracked = [ln for ln in run("status", "--porcelain").splitlines() if ln.startswith("??")]
    return {"commit": commit, "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
            "untracked_paths": len(untracked)}


def platform_kind() -> str:
    # Kaggle kernels are Linux and always export KAGGLE_KERNEL_RUN_TYPE. A bare
    # path check is not enough: a "\kaggle\working" folder can exist on a
    # Windows drive and would mark laptop runs as official.
    if sys.platform.startswith("linux") and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


def hardware_info() -> dict:
    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
    except ImportError:
        pass
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return info


def collect_environment(root: Path) -> dict:
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "platform_kind": platform_kind(),
        "hardware": hardware_info(),
        "packages": package_versions(),
        "git": git_info(root),
    }
