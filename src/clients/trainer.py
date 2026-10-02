"""Local training and evaluation shared by federated clients and baselines.

The same function trains a federated client, the centralized baseline and the
local-only baseline, so the optimizer, batch order rule and loss are identical
across everything that is compared.
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from src.metrics.classification import confusion_matrix, metrics_from_confusion
from src.models.registry import get_weights, set_weights


def to_tensor(x, device) -> torch.Tensor:
    if sp.issparse(x):
        coo = x.tocoo()
        idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
        # scipy guarantees in-range indices, so torch's invariant check is skipped explicitly
        t = torch.sparse_coo_tensor(idx, torch.from_numpy(coo.data), coo.shape, check_invariants=False)
        return t.coalesce().to(device)
    return torch.as_tensor(x).to(device)


def planned_steps(num_samples: int, batch_size: int, epochs: float) -> int:
    """Number of SGD steps for ``epochs`` (possibly fractional) local epochs."""
    if num_samples == 0:
        return 0
    return max(1, int(round(epochs * math.ceil(num_samples / batch_size))))


def local_train(
    model: torch.nn.Module,
    x,
    y: np.ndarray,
    *,
    epochs: float,
    batch_size: int,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    seed: int,
    device: str = "cpu",
    algorithm: str = "fedavg",
    mu: float = 0.0,
    c_global: list[np.ndarray] | None = None,
    c_local: list[np.ndarray] | None = None,
    logit_offset: np.ndarray | None = None,
) -> dict:
    """Run mini-batch SGD from the model's current weights.

    ``epochs`` may be fractional: a straggler that completes half of its local
    work is trained with ``epochs = 0.5 * E``.

    * ``fedprox``  adds (mu / 2) * ||w - w_global||^2 to the local objective
      (Li et al., 2020), implemented as the gradient term mu * (w - w_global).
    * ``scaffold`` corrects every step with (c_global - c_local) and returns the
      updated client control variate using option II of Karimireddy et al.
      (2020): c_new = c_local - c_global + (w_global - w_new) / (steps * lr).
    * ``logit_offset`` (one value per class) is added to the logits inside the
      training loss only. With the offset log(local class prior) this is the
      local label-prior correction (logit adjustment, Menon et al. 2021; the
      prior term of FedLC, Zhang et al. 2022, without its tuned margin): the
      client fits p(y | x) / prior_k(y), so its update no longer encodes its
      own label frequencies. Evaluation never uses the offset.
    """
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    start = [p.detach().clone() for p in params]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    correction = None
    if algorithm == "scaffold":
        correction = [
            torch.as_tensor(cg - cl, dtype=p.dtype, device=device) for cg, cl, p in zip(c_global, c_local, params)
        ]

    offset = None
    if logit_offset is not None:
        offset = torch.as_tensor(np.asarray(logit_offset), dtype=torch.float32, device=device)

    n = len(y)
    steps_total = planned_steps(n, batch_size, epochs)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    rng = np.random.default_rng(seed)

    wall0, cpu0 = time.perf_counter(), time.process_time()
    steps, loss_sum, seen = 0, 0.0, 0
    while steps < steps_total:
        order = rng.permutation(n)
        for lo in range(0, n, batch_size):
            if steps >= steps_total:
                break
            batch = np.sort(order[lo:lo + batch_size])
            xb = to_tensor(x[batch], device)
            yb = y_t[batch].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if offset is not None:
                logits = logits + offset
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            if algorithm == "fedprox" and mu > 0:
                for p, p0 in zip(params, start):
                    p.grad.add_(p.detach() - p0, alpha=mu)
            if correction is not None:
                for p, corr in zip(params, correction):
                    p.grad.add_(corr)
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach()) * len(batch)
            seen += len(batch)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0

    out = {
        "weights": get_weights(model),
        "num_examples": n,
        "steps": steps,
        "train_loss": loss_sum / max(seen, 1),
        "train_wall_s": wall,
        "train_cpu_s": cpu,
    }
    if algorithm == "scaffold":
        start_np = [s.cpu().numpy() for s in start]
        c_new = [
            cl - cg + (w0 - w1) / (max(steps, 1) * lr)
            for cl, cg, w0, w1 in zip(c_local, c_global, start_np, out["weights"])
        ]
        out["c_new"] = c_new
        out["c_delta"] = [cn - cl for cn, cl in zip(c_new, c_local)]
    return out


@torch.no_grad()
def evaluate(model: torch.nn.Module, x, y: np.ndarray, num_classes: int, device: str = "cpu", batch_size: int = 4096) -> dict:
    """Mean cross-entropy and confusion-matrix metrics on one data split."""
    n = len(y)
    if n == 0:
        return {"n": 0, "loss": float("nan"), "accuracy": float("nan"), "macro_f1": float("nan")}
    model.to(device).eval()
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    preds, loss_sum = [], 0.0
    for lo in range(0, n, batch_size):
        logits = model(to_tensor(x[lo:lo + batch_size], device))
        loss_sum += float(F.cross_entropy(logits, y_t[lo:lo + batch_size].to(device), reduction="sum"))
        preds.append(logits.argmax(dim=1).cpu().numpy())
    out = metrics_from_confusion(confusion_matrix(y, np.concatenate(preds), num_classes))
    out["loss"] = loss_sum / n
    return out


def evaluate_weights(model, weights, x, y, num_classes, device="cpu") -> dict:
    set_weights(model, weights)
    return evaluate(model, x, y, num_classes, device)


def label_prior_offset(labels: np.ndarray, num_classes: int, reference: np.ndarray | None = None, smoothing: float = 1.0) -> np.ndarray:
    """log of the add-``smoothing`` local class prior, optionally minus log of a reference prior.

    ``reference=None`` gives log prior_k(y): the model is pushed towards the
    class-balanced posterior. With the global class prior as reference the
    offset is log(prior_k(y) / prior(y)): only the client's deviation from the
    global label distribution is removed and the global prior stays in the model.
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64) + smoothing
    offset = np.log(counts / counts.sum())
    if reference is not None:
        offset = offset - np.log(np.asarray(reference, dtype=np.float64))
    return offset.astype(np.float32)
