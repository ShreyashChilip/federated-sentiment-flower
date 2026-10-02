"""Process-level resource readings and the energy proxy."""
from __future__ import annotations

import os

import psutil


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def peak_vram_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**20
    except ImportError:
        pass
    return 0.0


def energy_proxy_joules(cpu_seconds: float, cpu_watts: float, gpu_seconds: float = 0.0, gpu_watts: float = 0.0) -> float:
    """Proxy only: active compute time multiplied by a declared power figure.

    This is not a measurement. The declared wattages are recorded in the run
    configuration and must be quoted wherever the proxy is reported.
    """
    return cpu_seconds * cpu_watts + gpu_seconds * gpu_watts
