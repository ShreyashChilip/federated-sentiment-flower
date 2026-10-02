"""Seeding utilities.

Every source of randomness in a run is derived from the experiment seed through
``derive_seed`` so that independent components (partitioning, client sampling,
batch order, dropout) never share a random stream.
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np

SEEDS = (42, 123, 456, 789, 2026)


def derive_seed(*parts) -> int:
    """Deterministically map any tuple of printable parts to a 32-bit seed."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (CPU and CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:  # torch-free utilities (partitioning) still work
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
