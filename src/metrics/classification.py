"""Classification metrics computed from a confusion matrix.

Working from the confusion matrix lets per-client results be pooled exactly
(confusion matrices add) and keeps every reported number traceable to counts.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    flat = np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2)
    return flat.reshape(num_classes, num_classes)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    # Macro averages run over classes that occur in y_true. A client that holds
    # a single class is then scored on that class instead of being capped at
    # 1/num_classes by a class it has no examples of.
    present = support > 0
    k = max(int(present.sum()), 1)
    return {
        "n": int(total),
        "accuracy": float(tp.sum() / total) if total else float("nan"),
        "macro_f1": float(f1[present].sum() / k),
        "macro_precision": float(precision[present].sum() / k),
        "macro_recall": float(recall[present].sum() / k),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.astype(np.int64).tolist(),
    }
