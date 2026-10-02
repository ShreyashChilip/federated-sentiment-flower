"""Fixed per-sample roles inside the official training split.

Every training sample is assigned exactly one role, independently of how the
samples are later partitioned among clients:

* ``TRAIN``       - used for local training and for fitting features,
* ``VAL``         - used for hyperparameter tuning and model selection,
* ``CLIENT_TEST`` - used for per-client (fairness) evaluation.

Because roles do not depend on the partition, the feature space and the
centralized baseline are identical for every alpha, client count and seed, and
each client holds roughly the same train/val/test proportions. The official
test split is never assigned a role and never enters this module.
"""
from __future__ import annotations

import numpy as np

TRAIN, VAL, CLIENT_TEST = 0, 1, 2
ROLE_NAMES = {TRAIN: "train", VAL: "val", CLIENT_TEST: "client_test"}


def assign_roles(labels: np.ndarray, fractions=(0.8, 0.1, 0.1), seed: int = 0) -> np.ndarray:
    """Return an int8 role per sample, stratified by label."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("role fractions must sum to 1")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    roles = np.empty(len(labels), dtype=np.int8)
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_train = int(round(fractions[0] * len(idx)))
        n_val = int(round(fractions[1] * len(idx)))
        roles[idx[:n_train]] = TRAIN
        roles[idx[n_train:n_train + n_val]] = VAL
        roles[idx[n_train + n_val:]] = CLIENT_TEST
    return roles
