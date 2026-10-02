"""Server-side aggregation rules as pure functions over lists of NumPy arrays.

Keeping these independent of the FL framework makes them unit-testable and
lets the Flower server loop stay thin.
"""
from __future__ import annotations

import numpy as np

Weights = list  # list[np.ndarray]


def weighted_average(updates: list[Weights], num_examples: list[int]) -> Weights:
    """Example-weighted mean, accumulated in float64 in the given order."""
    total = float(sum(num_examples))
    out = [np.zeros_like(a, dtype=np.float64) for a in updates[0]]
    for weights, n in zip(updates, num_examples):
        for acc, w in zip(out, weights):
            acc += w.astype(np.float64) * (n / total)
    return [a.astype(np.float32) for a in out]


def simple_average(updates: list[Weights]) -> Weights:
    return weighted_average(updates, [1] * len(updates))


class ServerState:
    """Global model plus whatever the server-side rule needs to remember."""

    def __init__(self, weights: Weights, cfg: dict, num_clients: int):
        self.weights = [w.astype(np.float32) for w in weights]
        self.algorithm = cfg["algorithm"]
        self.server_lr = float(cfg.get("server_lr", 1.0))
        self.num_clients = num_clients
        self.round = 0
        if self.algorithm == "fedadam":
            self.beta1 = float(cfg.get("adam_beta1", 0.9))
            self.beta2 = float(cfg.get("adam_beta2", 0.99))
            self.tau = float(cfg.get("adam_tau", 1e-3))
            self.m = [np.zeros_like(w, dtype=np.float64) for w in self.weights]
            self.v = [np.zeros_like(w, dtype=np.float64) for w in self.weights]
        if self.algorithm == "scaffold":
            self.c_global = [np.zeros_like(w) for w in self.weights]

    def apply(self, client_weights: list[Weights], num_examples: list[int], c_deltas: list[Weights] | None = None) -> None:
        """Update the global model from one round of client results.

        * fedavg / fedprox: w <- w + server_lr * (weighted mean of w_i - w).
          With server_lr = 1 this is the usual weighted average.
        * fedadam (Reddi et al., 2021): Adam step on the weighted mean update.
        * scaffold (Karimireddy et al., 2020): unweighted mean of client updates
          scaled by server_lr, and c <- c + (|S| / N) * mean(c_i_new - c_i).

        If no client reported (all dropped out) the model is left unchanged.
        """
        self.round += 1
        if not client_weights:
            return
        if self.algorithm == "scaffold":
            mean = simple_average(client_weights)
        else:
            mean = weighted_average(client_weights, num_examples)
        delta = [m.astype(np.float64) - w.astype(np.float64) for m, w in zip(mean, self.weights)]

        if self.algorithm == "fedadam":
            new = []
            for i, (w, d) in enumerate(zip(self.weights, delta)):
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * d
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * d * d
                new.append((w + self.server_lr * self.m[i] / (np.sqrt(self.v[i]) + self.tau)).astype(np.float32))
            self.weights = new
        else:
            self.weights = [(w + self.server_lr * d).astype(np.float32) for w, d in zip(self.weights, delta)]

        if self.algorithm == "scaffold":
            mean_c = simple_average(c_deltas)
            scale = len(c_deltas) / self.num_clients
            self.c_global = [(c + scale * d).astype(np.float32) for c, d in zip(self.c_global, mean_c)]
