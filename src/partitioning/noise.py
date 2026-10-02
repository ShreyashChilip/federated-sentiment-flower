"""Label noise applied to training-role samples only."""
from __future__ import annotations

import numpy as np


def flip_labels(labels: np.ndarray, eligible: np.ndarray, rate: float, num_classes: int, seed: int):
    """Replace ``rate`` of the eligible labels with a different class.

    Returns the noisy copy and the indices that were changed. Validation,
    client-test and global test labels are never passed as eligible.
    """
    labels = np.asarray(labels).copy()
    if rate <= 0:
        return labels, np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    eligible = np.asarray(eligible)
    flipped = np.sort(rng.choice(eligible, size=int(round(rate * len(eligible))), replace=False))
    shift = rng.integers(1, num_classes, size=len(flipped))
    labels[flipped] = (labels[flipped] + shift) % num_classes
    return labels, flipped
