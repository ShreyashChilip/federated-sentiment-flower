"""Model construction and weight (de)serialization helpers."""
from __future__ import annotations

import numpy as np
import torch

from src.models.lr import SparseLogisticRegression


def build_model(cfg: dict, input_dim: int, num_classes: int) -> torch.nn.Module:
    name = cfg["name"]
    if name == "lr":
        return SparseLogisticRegression(input_dim, num_classes)
    raise ValueError(f"model '{name}' is not implemented yet")


def trainable_names(model: torch.nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def get_weights(model: torch.nn.Module) -> list[np.ndarray]:
    """Trainable parameters only: this is exactly what is transmitted."""
    return [p.detach().cpu().numpy().copy() for p in model.parameters() if p.requires_grad]


def set_weights(model: torch.nn.Module, weights: list[np.ndarray]) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) != len(weights):
        raise ValueError(f"expected {len(params)} arrays, got {len(weights)}")
    with torch.no_grad():
        for p, w in zip(params, weights):
            p.copy_(torch.as_tensor(w, dtype=p.dtype, device=p.device))


def parameter_counts(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "transmitted_parameters": trainable,
        "transmitted_bytes_float32": trainable * 4,
    }
