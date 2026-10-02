"""Natural clients: profiling, size filtering and the held-out client split.

A natural client is the set of all records that share one value of a real
metadata key (a reviewer id or a product id). No client identity is invented.

Order of operations, all independent of any model or metric:
  1. profile the raw clients (``profile_clients``);
  2. apply the client-size rule from the config (``filter_clients``);
  3. split the retained CLIENTS into seen and unseen (``split_clients``); every
     record of an unseen client is held out;
  4. inside each seen client, split its records into train / val / client-test
     (``assign_roles_within_clients``).
"""
from __future__ import annotations

import numpy as np

from src.data.roles import CLIENT_TEST, TRAIN, VAL

UNSEEN = 3  # role of every record that belongs to a held-out client
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
DEFAULT_THRESHOLDS = (1, 2, 5, 10, 20, 50, 100, 200)


def encode_clients(keys) -> tuple[np.ndarray, np.ndarray]:
    """Map metadata keys to integer codes (sorted key order). Returns codes, unique keys."""
    uniq, codes = np.unique(np.asarray(keys), return_inverse=True)
    return codes.astype(np.int64), uniq


def profile_clients(codes: np.ndarray, labels: np.ndarray, num_classes: int, thresholds=DEFAULT_THRESHOLDS) -> dict:
    """Describe the raw client population before any filtering or training."""
    codes, labels = np.asarray(codes), np.asarray(labels)
    n = len(codes)
    sizes = np.bincount(codes)
    k = len(sizes)
    # K x C table of label counts per client
    table = np.bincount(codes * num_classes + labels, minlength=k * num_classes).reshape(k, num_classes)
    share = table / sizes[:, None]
    majority = share.max(axis=1)
    mean_label = (table * np.arange(num_classes)).sum(axis=1) / sizes
    retention = []
    for t in thresholds:
        keep = sizes >= t
        retention.append({
            "min_reviews": int(t),
            "clients_retained": int(keep.sum()),
            "client_fraction_retained": float(keep.mean()),
            "records_retained": int(sizes[keep].sum()),
            "record_fraction_retained": float(sizes[keep].sum() / n),
        })

    def dist(v):
        return {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                **{f"p{p}": float(np.percentile(v, p)) for p in PERCENTILES}}

    return {
        "num_records": int(n),
        "num_raw_clients": int(k),
        "client_size": {**dist(sizes), "median": float(np.median(sizes))},
        "global_class_counts": np.bincount(labels, minlength=num_classes).tolist(),
        "global_class_distribution": (np.bincount(labels, minlength=num_classes) / n).tolist(),
        # how skewed individual clients are, over ALL raw clients
        "per_client_majority_class_share": dist(majority),
        "per_client_mean_label": dist(mean_label),
        "clients_with_single_class": int((majority == 1.0).sum()),
        "retention_by_min_reviews": retention,
        "per_client_table": {"sizes": sizes, "label_counts": table},
    }


def filter_clients(codes: np.ndarray, rule: dict) -> tuple[np.ndarray, dict]:
    """Apply the client-size rule. Returns a boolean mask over records and a report.

    Rule (all values come from the config and are fixed before training):
      * ``min_reviews``  keep clients with at least this many cleaned records;
      * ``max_reviews``  if set, larger clients keep a seeded random subset of
                         exactly this many records (limits very heavy clients);
      * ``max_clients``  if set, a seeded random subset of this many retained
                         clients is kept (compute limit).
    """
    if rule.get("min_reviews") is None:
        raise ValueError(
            "data.client_filter.min_reviews is not set. Run the client profiling first, then write the "
            "chosen threshold into the config (and CHANGELOG.md) before any training."
        )
    codes = np.asarray(codes)
    rng = np.random.default_rng(int(rule.get("filter_seed", 0)))
    sizes = np.bincount(codes)
    kept_clients = np.flatnonzero(sizes >= int(rule["min_reviews"]))
    if rule.get("max_clients") and len(kept_clients) > int(rule["max_clients"]):
        kept_clients = np.sort(rng.choice(kept_clients, size=int(rule["max_clients"]), replace=False))
    mask = np.isin(codes, kept_clients)
    capped = 0
    if rule.get("max_reviews"):
        cap = int(rule["max_reviews"])
        order = np.argsort(codes, kind="stable")
        starts = np.cumsum(sizes) - sizes
        for c in kept_clients[sizes[kept_clients] > cap]:
            rows = order[starts[c]:starts[c] + sizes[c]]
            mask[rng.permutation(rows)[cap:]] = False
            capped += 1
    report = {
        "rule": {k: rule.get(k) for k in ("min_reviews", "max_reviews", "max_clients", "filter_seed")},
        "raw_clients": int(len(sizes)),
        "raw_records": int(len(codes)),
        "clients_retained": int(len(kept_clients)),
        "records_retained": int(mask.sum()),
        "record_fraction_retained": float(mask.mean()),
        "clients_capped": capped,
    }
    return mask, report


def split_clients(client_ids: np.ndarray, unseen_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded split of client ids into (seen, unseen); both sorted and disjoint."""
    ids = np.unique(np.asarray(client_ids))
    if not 0.0 < unseen_fraction < 1.0:
        raise ValueError("unseen_fraction must be strictly between 0 and 1")
    n_unseen = int(round(unseen_fraction * len(ids)))
    if n_unseen == 0 or n_unseen == len(ids):
        raise ValueError("the unseen split would be empty or would contain every client")
    perm = np.random.default_rng(seed).permutation(ids)
    return np.sort(perm[n_unseen:]), np.sort(perm[:n_unseen])


def assign_roles_within_clients(codes: np.ndarray, seen: np.ndarray, fractions, seed: int) -> np.ndarray:
    """Roles per record: UNSEEN for held-out clients; a per-client split otherwise.

    Each seen client's records are shuffled and cut into train / val /
    client-test by ``fractions``. A client always keeps at least one training
    record, and at least one client-test record when it has two or more.
    """
    codes = np.asarray(codes)
    roles = np.full(len(codes), UNSEEN, dtype=np.int8)
    rng = np.random.default_rng(seed)
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    seen_set = set(np.asarray(seen).tolist())
    for rows in np.split(order, bounds):
        if codes[rows[0]] not in seen_set:
            continue
        rows = rng.permutation(rows)
        n = len(rows)
        n_test = max(1, int(round(fractions[2] * n))) if n >= 2 else 0
        n_val = int(round(fractions[1] * n)) if n - n_test >= 2 else 0
        n_train = n - n_val - n_test
        roles[rows[:n_train]] = TRAIN
        roles[rows[n_train:n_train + n_val]] = VAL
        roles[rows[n_train + n_val:]] = CLIENT_TEST
    return roles


def assert_no_client_overlap(codes: np.ndarray, roles: np.ndarray) -> dict:
    """Hard check: no client has records on both sides of the seen/unseen split."""
    codes, roles = np.asarray(codes), np.asarray(roles)
    unseen_clients = np.unique(codes[roles == UNSEEN])
    seen_clients = np.unique(codes[roles != UNSEEN])
    overlap = np.intersect1d(seen_clients, unseen_clients)
    if len(overlap):
        raise AssertionError(f"{len(overlap)} client(s) appear in both the seen and the unseen split")
    if not len(unseen_clients) or not len(seen_clients):
        raise AssertionError("seen or unseen client set is empty")
    return {"seen_clients": int(len(seen_clients)), "unseen_clients": int(len(unseen_clients)), "overlap": 0,
            "seen_records": int((roles != UNSEEN).sum()), "unseen_records": int((roles == UNSEEN).sum())}
