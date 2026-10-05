"""Chunked, memory-bounded evaluation of a linear model on a feature bundle.

Groups of rows are evaluated straight from the memory-mapped CSR matrix in
chunks; no full copy of a split (in particular of the unseen-client split) is
ever materialized. Per-client results come from one confusion tensor per
group, built with a single ``bincount``.

Groups
  val          seen-client validation rows (pooled and per seen client)
  client_test  seen-client client-test rows (pooled and per seen client)
  unseen       natural bundles: every record of the held-out clients
               (pooled and per unseen client)
  test         standard bundles: the official test split (pooled only)

Macro-F1 for one client averages over the classes present in that client's
labels, exactly as ``src.metrics.classification.metrics_from_confusion``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from src.data.roles import CLIENT_TEST, VAL
from src.metrics.classification import metrics_from_confusion
from src.metrics.fairness import client_dispersion

CHUNK_ROWS = 32768


def linear_logits(x: sp.csr_matrix, weights) -> np.ndarray:
    w, b = weights
    return np.asarray(x @ np.asarray(w, dtype=np.float32).T, dtype=np.float32) + np.asarray(b, dtype=np.float32)


def per_client_scores(confusion: np.ndarray) -> dict:
    """Vectorized metrics_from_confusion over a (K, C, C) tensor."""
    cm = confusion.astype(np.float64)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    support = cm.sum(axis=2)
    predicted = cm.sum(axis=1)
    total = support.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        present = support > 0
        k = np.maximum(present.sum(axis=1), 1)
        macro_f1 = np.where(total > 0, (f1 * present).sum(axis=1) / k, np.nan)
        accuracy = np.where(total > 0, tp.sum(axis=1) / total, np.nan)
    return {"n": total.astype(np.int64), "accuracy": accuracy, "macro_f1": macro_f1,
            "classes_present": present.sum(axis=1).astype(np.int64)}


@dataclass
class Group:
    name: str
    rows: np.ndarray          # sorted row indices into the matrix
    owner: np.ndarray | None  # client index per row (None: pooled only)
    num_owners: int
    labels: np.ndarray
    source: str               # "train" (x_train) or "test" (x_test)


class Evaluator:
    def __init__(self, rt, test_matrix_loader=None):
        b = rt.bundle
        self.num_classes = b.num_classes
        self.x_train = b.x_train
        self._test_loader = test_matrix_loader
        self._x_test = None
        n = b.x_train.shape[0]
        owner = np.full(n, -1, dtype=np.int64)
        for cid, idx in enumerate(rt.clients):
            owner[np.asarray(idx)] = cid
        self.num_seen = len(rt.clients)
        roles = np.asarray(b.roles)
        y = np.asarray(b.y_train)
        self.groups: dict[str, Group] = {}
        for name, role in (("val", VAL), ("client_test", CLIENT_TEST)):
            rows = np.flatnonzero((roles == role) & (owner >= 0))
            self.groups[name] = Group(name, rows, owner[rows], self.num_seen, y[rows], "train")
        self.unseen_keys = None
        if b.kind == "natural":
            from src.data.natural import UNSEEN

            rows = np.flatnonzero(roles == UNSEEN)
            codes = np.asarray(b.client_codes)[rows]
            uniq, inv = np.unique(codes, return_inverse=True)
            self.unseen_keys = uniq
            self.groups["unseen"] = Group("unseen", rows, inv.astype(np.int64), len(uniq), y[rows], "train")
        else:
            self.groups["test"] = None  # loaded lazily: the test split is only read when asked for

    @property
    def held_out_group(self) -> str:
        return "unseen" if "unseen" in self.groups else "test"

    def _test_group(self) -> Group:
        if self.groups.get("test") is None:
            x_test, y_test = self._test_loader()
            self._x_test = x_test
            self.groups["test"] = Group("test", np.arange(x_test.shape[0]), None, 0, np.asarray(y_test), "test")
        return self.groups["test"]

    def predict(self, group: Group, weights) -> tuple[np.ndarray, float]:
        """Predicted class per row of the group, and the summed cross-entropy."""
        x = self.x_train if group.source == "train" else self._x_test
        preds = np.empty(len(group.rows), dtype=np.int64)
        loss = 0.0
        for lo in range(0, len(group.rows), CHUNK_ROWS):
            rows = group.rows[lo:lo + CHUNK_ROWS]
            logits = linear_logits(x[rows], weights).astype(np.float64)
            preds[lo:lo + len(rows)] = logits.argmax(axis=1)
            mx = logits.max(axis=1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(logits - mx).sum(axis=1))
            loss += float((lse - logits[np.arange(len(rows)), group.labels[lo:lo + len(rows)]]).sum())
        return preds, loss

    def evaluate(self, weights, names=("val",), per_client: bool = True) -> dict:
        out = {}
        for name in names:
            group = self._test_group() if name == "test" else self.groups[name]
            if not len(group.rows):
                out[name] = {"pooled": {"n": 0}, "clients": None}
                continue
            preds, loss = self.predict(group, weights)
            c = self.num_classes
            pooled = metrics_from_confusion(np.bincount(group.labels * c + preds, minlength=c * c).reshape(c, c))
            pooled["loss"] = loss / len(group.rows)
            entry = {"pooled": pooled, "clients": None}
            if per_client and group.owner is not None:
                cm = np.bincount(group.owner * c * c + group.labels * c + preds,
                                 minlength=group.num_owners * c * c).reshape(group.num_owners, c, c)
                scores = per_client_scores(cm)
                entry["clients"] = scores
                entry["dispersion"] = client_dispersion(scores["macro_f1"][scores["n"] > 0])
            out[name] = entry
        return out


def compact(result: dict, prefix: str) -> dict:
    """Flatten an evaluation entry into history-row fields."""
    row = {}
    pooled = result["pooled"]
    if pooled.get("n"):
        for key in ("loss", "accuracy", "macro_f1", "macro_precision", "macro_recall"):
            row[f"{prefix}_{key}"] = pooled[key]
    for key, value in (result.get("dispersion") or {}).items():
        if key != "num_clients":
            row[f"{prefix}_client_f1_{key}"] = value
    return row
