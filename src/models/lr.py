"""Multinomial logistic regression over sparse TF-IDF inputs."""
from __future__ import annotations

import torch
from torch import nn


class SparseLogisticRegression(nn.Module):
    """Linear softmax classifier that accepts sparse or dense inputs.

    Weights start at zero: the objective is convex, so the starting point does
    not matter for the optimum and a zero start removes one source of
    seed-to-seed variance that has nothing to do with federated learning.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_classes, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse or x.layout == torch.sparse_csr:
            return torch.sparse.mm(x, self.weight.t()) + self.bias
        return x @ self.weight.t() + self.bias
