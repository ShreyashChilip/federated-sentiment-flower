"""Client partitioning: IID, Dirichlet label skew, quantity skew, natural clients.

All partitioners are pure functions of ``(labels, parameters, seed)`` and return
a list with one sorted index array per client. Every sample is assigned to
exactly one client.
"""
from __future__ import annotations

import numpy as np


def iid_partition(n: int, num_clients: int, seed: int) -> list[np.ndarray]:
    """Shuffle all samples and deal them into equally sized clients."""
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(part) for part in np.array_split(perm, num_clients)]


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_client_size: int = 10,
    max_tries: int = 1000,
) -> tuple[list[np.ndarray], int]:
    """Dirichlet label-skew partition (Hsu et al., 2019 recipe).

    For every class c a vector p_c ~ Dir(alpha * 1_N) is drawn and the samples
    of class c are split among the N clients in those proportions. Small alpha
    concentrates each class on few clients; large alpha approaches IID.

    If any client ends up with fewer than ``min_client_size`` samples the whole
    draw is repeated with the next state of the same random stream. The number
    of draws used is returned so that it is recorded with the partition.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    for attempt in range(1, max_tries + 1):
        buckets: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(labels == c))
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
            for client, part in enumerate(np.split(idx, cuts)):
                buckets[client].append(part)
        clients = [np.sort(np.concatenate(b)) for b in buckets]
        if min(len(c) for c in clients) >= min_client_size:
            return clients, attempt
    raise RuntimeError(
        f"no Dirichlet draw with every client >= {min_client_size} samples in {max_tries} tries "
        f"(N={num_clients}, alpha={alpha}); lower min_client_size or the client count"
    )


def quantity_skew_partition(
    n: int, num_clients: int, beta: float, seed: int, min_client_size: int = 10, max_tries: int = 1000
) -> tuple[list[np.ndarray], int]:
    """Unequal client sizes with IID labels: sizes ~ Dir(beta * 1_N)."""
    rng = np.random.default_rng(seed)
    for attempt in range(1, max_tries + 1):
        perm = rng.permutation(n)
        proportions = rng.dirichlet(np.full(num_clients, beta))
        cuts = (np.cumsum(proportions)[:-1] * n).astype(int)
        clients = [np.sort(part) for part in np.split(perm, cuts)]
        if min(len(c) for c in clients) >= min_client_size:
            return clients, attempt
    raise RuntimeError("no quantity-skew draw satisfied min_client_size")


def natural_partition(group_keys, min_client_size: int = 1) -> tuple[list[np.ndarray], list]:
    """One client per distinct metadata key (for example a reviewer id).

    Keys are taken from dataset metadata; no client identity is invented.
    Groups smaller than ``min_client_size`` are dropped and reported by the
    caller. Returns the index arrays and the matching keys, sorted by key.
    """
    keys = np.asarray(group_keys)
    order = np.argsort(keys, kind="stable")
    uniq, starts = np.unique(keys[order], return_index=True)
    groups = np.split(order, starts[1:])
    kept = [(k, np.sort(g)) for k, g in zip(uniq.tolist(), groups) if len(g) >= min_client_size]
    return [g for _, g in kept], [k for k, _ in kept]


def make_partition(labels: np.ndarray, cfg: dict, seed: int) -> tuple[list[np.ndarray], dict]:
    """Dispatch on ``cfg['scheme']``; returns clients and extra info to record."""
    scheme = cfg["scheme"]
    n, num_clients = len(labels), cfg["num_clients"]
    min_size = cfg.get("min_client_size", 10)
    if scheme == "iid":
        return iid_partition(n, num_clients, seed), {}
    if scheme == "dirichlet":
        clients, tries = dirichlet_partition(labels, num_clients, cfg["alpha"], seed, min_size)
        return clients, {"draws_used": tries}
    if scheme == "quantity_skew":
        clients, tries = quantity_skew_partition(n, num_clients, cfg["beta"], seed, min_size)
        return clients, {"draws_used": tries}
    raise ValueError(f"unknown partition scheme '{scheme}'")
