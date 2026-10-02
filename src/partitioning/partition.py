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
        f"(N={num_clients}, alpha={alpha}). For many clients and small alpha this rule cannot be met; "
        "see docs/PHASE_0_TO_PHASE_1_AUDIT.md, section 4, for the options"
    )


# Identifier stored with every partition made by dirichlet_client_partition.
# Any change to that function's sampling order must change this string.
DIRICHLET_CLIENT_ALGORITHM = "dirichlet_client/v1"


def dirichlet_client_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    """Client-wise Dirichlet label skew with equal client sizes.

    Every client k receives floor(n / N) samples (the first n mod N clients in
    a seeded random order receive one more) and a label distribution
    q_k ~ Dir(alpha * 1_C). Its class counts are drawn as
    Multinomial(size_k, q_k) and taken without replacement from the class
    pools. When a pool runs out, the shortfall is redrawn from the classes
    that still have stock, with q_k renormalized over them.

    No client can be empty or tiny, so no redraw rule is needed. Known
    artifact: clients served last are constrained by what is left in the
    pools and are therefore closer to the leftover class mix than to their
    own q_k. The realized class counts are saved with the partition.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    pools = [rng.permutation(np.flatnonzero(labels == c)) for c in classes]
    used = np.zeros(len(classes), dtype=np.int64)
    stock = np.array([len(p) for p in pools], dtype=np.int64)
    order = rng.permutation(num_clients)
    sizes = np.full(num_clients, len(labels) // num_clients, dtype=np.int64)
    sizes[order[: len(labels) % num_clients]] += 1
    clients: list = [None] * num_clients
    for k in order:
        q = rng.dirichlet(np.full(len(classes), alpha))
        take = np.zeros(len(classes), dtype=np.int64)
        need = int(sizes[k])
        while need > 0:
            available = stock - used - take
            w = np.where(available > 0, q, 0.0)
            w = w / w.sum() if w.sum() > 0 else (available > 0) / (available > 0).sum()
            draw = np.minimum(rng.multinomial(need, w), available)
            take += draw
            need -= int(draw.sum())
        clients[k] = np.sort(np.concatenate([pools[c][used[c]:used[c] + take[c]] for c in range(len(classes))]))
        used += take
    return clients


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


def make_partition(labels: np.ndarray, cfg: dict, seed: int, groups=None) -> tuple[list[np.ndarray], dict]:
    """Dispatch on ``cfg['scheme']``; returns clients and extra info to record.

    ``groups`` (natural scheme only) holds one integer client code per row, or
    -1 for rows that belong to no training client (held-out clients).
    """
    scheme = cfg["scheme"]
    if scheme == "natural":
        if groups is None:
            raise ValueError("the natural scheme needs the per-row client codes of the dataset")
        groups = np.asarray(groups)
        rows = np.flatnonzero(groups >= 0)
        parts, keys = natural_partition(groups[rows], min_client_size=1)
        return [rows[p] for p in parts], {"client_codes": [int(k) for k in keys]}
    n, num_clients = len(labels), cfg["num_clients"]
    min_size = cfg.get("min_client_size", 10)
    if scheme == "iid":
        return iid_partition(n, num_clients, seed), {}
    if scheme == "dirichlet":
        clients, tries = dirichlet_partition(labels, num_clients, cfg["alpha"], seed, min_size)
        return clients, {"draws_used": tries}
    if scheme == "dirichlet_client":
        return dirichlet_client_partition(labels, num_clients, cfg["alpha"], seed), {"algorithm": DIRICHLET_CLIENT_ALGORITHM}
    if scheme == "quantity_skew":
        clients, tries = quantity_skew_partition(n, num_clients, cfg["beta"], seed, min_size)
        return clients, {"draws_used": tries}
    raise ValueError(f"unknown partition scheme '{scheme}'")
