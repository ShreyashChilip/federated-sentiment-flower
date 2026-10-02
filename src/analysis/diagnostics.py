"""Mechanism diagnostics for client-level structure in word/sentiment association.

Full definitions and their interpretation are in docs/CONFOUNDING_DIAGNOSTIC.md.
Nothing here trains or changes a model; these are descriptive statistics of
the TRAINING-role data and of already-trained model weights.

Notation, for one feature (vocabulary term) j:
    z_ij in {0, 1}   document i contains term j
    s_i              sentiment score of document i (label for 2 classes,
                     star index 0..4 for the 5-class task)
    k(i)             client of document i; n_k its number of documents
    zbar_kj, sbar_k  client means;  zbar_j, sbar  overall means

The pooled least-squares slope of s on z_j splits exactly into a within-client
and a between-client part:

    beta_pooled_j = lambda_j * beta_within_j + (1 - lambda_j) * beta_between_j

    beta_within_j  = sum_i (z_ij - zbar_k(i)j)(s_i - sbar_k(i)) / SSW_j
    beta_between_j = sum_k n_k (zbar_kj - zbar_j)(sbar_k - sbar) / SSB_j
    SSW_j = sum_i (z_ij - zbar_k(i)j)^2,  SSB_j = sum_k n_k (zbar_kj - zbar_j)^2
    lambda_j = SSW_j / (SSW_j + SSB_j)

beta_within is what a reader would call the term's sentiment association
*inside* a client; beta_between is the association carried by *which clients*
use the term. The client gap

    gap_j = beta_pooled_j - beta_within_j = (1 - lambda_j)(beta_between_j - beta_within_j)

is the part of the pooled association that exists only because of client
membership. A feature shows CLIENT STRUCTURE when its gap differs from what
is expected if term usage depended on the sentiment of the document but not on
its client (see ``permutation_null``). It is a CANDIDATE client-confounded feature
only if, in addition, the pooled association is larger in magnitude than the
within-client one or has the opposite sign (see ``flag_candidates``).
"Candidate" is a descriptive label; it is not a causal claim.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import stats as sps

from src.metrics.classification import confusion_matrix, metrics_from_confusion
from src.metrics.fairness import client_dispersion


def client_feature_counts(x: sp.csr_matrix, groups: np.ndarray, num_groups: int) -> sp.csr_matrix:
    """K x d sparse matrix: number of documents of client k that contain term j."""
    z = x.copy().tocsr()
    z.data = np.ones_like(z.data, dtype=np.float64)
    member = sp.csr_matrix((np.ones(len(groups)), (np.asarray(groups), np.arange(len(groups)))),
                           shape=(num_groups, len(groups)))
    return (member @ z).tocsr()


def association_decomposition(x: sp.csr_matrix, score: np.ndarray, groups: np.ndarray) -> dict:
    """Pooled, within-client and between-client association of every feature.

    ``groups`` must be integer client codes 0..K-1 with every client non-empty.
    All sums are exact; the K x d client table is kept sparse.
    """
    groups = np.asarray(groups)
    s = np.asarray(score, dtype=np.float64)
    n, d = x.shape
    sizes = np.bincount(groups).astype(np.float64)
    k = len(sizes)
    if (sizes == 0).any():
        raise ValueError("client codes must be contiguous with no empty client")
    counts = client_feature_counts(x, groups, k)             # C_kj
    z = x.copy().tocsr()
    z.data = np.ones_like(z.data, dtype=np.float64)

    df = np.asarray(counts.sum(axis=0)).ravel()               # sum_i z_ij
    s_bar_k = np.bincount(groups, weights=s) / sizes
    s_bar = s.mean()
    zs = np.asarray(z.T @ s).ravel()                          # sum_i z_ij s_i
    c_sbar = np.asarray(counts.T @ s_bar_k).ravel()           # sum_k C_kj sbar_k
    c2_over_n = np.asarray(counts.multiply(counts).T @ (1.0 / sizes)).ravel()  # sum_k C_kj^2 / n_k

    ss_total = df - df * df / n                               # z is binary, so sum z^2 = df
    ss_within = df - c2_over_n
    ss_between = c2_over_n - df * df / n
    cov_total = zs - df * s_bar
    cov_within = zs - c_sbar
    cov_between = c_sbar - df * s_bar

    def ratio(a, b):
        return np.divide(a, b, out=np.full(d, np.nan), where=b > 1e-12)

    beta_pooled, beta_within, beta_between = ratio(cov_total, ss_total), ratio(cov_within, ss_within), ratio(cov_between, ss_between)
    return {
        "document_frequency": df.astype(np.int64),
        "client_frequency": np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.int64),
        "beta_pooled": beta_pooled,
        "beta_within": beta_within,
        "beta_between": beta_between,
        "lambda_within": ratio(ss_within, ss_total),
        "gap": beta_pooled - beta_within,
        # label exposure of the frozen Experiment 0A protocol:
        # X_j = sum_k df_kj (sbar_k - sbar) / sum_k df_kj
        "exposure": ratio(cov_between, df),
        # variance of the term's usage rate across clients (size-weighted)
        "between_client_variance": ss_between / n,
    }


def permutation_null(x: sp.csr_matrix, score: np.ndarray, groups: np.ndarray, repeats: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of ``gap_j`` under the null hypothesis that
    term usage is independent of the client GIVEN the document's sentiment.

    Each draw keeps every document's client and score and shuffles the text
    (feature rows) among documents that have the same score. This preserves
    client sizes, the sentiment distribution of every client and the pooled
    association of every term with sentiment, and removes only client-specific
    usage. The gap is generally not zero under this null (clients with
    different mean sentiment restrict the range of the score inside a client),
    which is why the null mean is returned and subtracted.
    """
    rng = np.random.default_rng(seed)
    score = np.asarray(score)
    strata = [np.flatnonzero(score == v) for v in np.unique(score)]
    gaps = np.empty((repeats, x.shape[1]))
    for r in range(repeats):
        perm = np.arange(x.shape[0])
        for rows in strata:
            perm[rows] = rng.permutation(rows)
        gaps[r] = association_decomposition(x[perm], score, groups)["gap"]
    return np.nanmean(gaps, axis=0), np.nanstd(gaps, axis=0, ddof=1)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values); NaN inputs stay NaN."""
    p = np.asarray(p, dtype=np.float64)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if len(ok) == 0:
        return q
    order = ok[np.argsort(p[ok])]
    ranked = p[order] * len(ok) / np.arange(1, len(ok) + 1)
    q[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def flag_candidates(decomp: dict, null_mean: np.ndarray, null_sd: np.ndarray, repeats: int, rule: dict) -> dict:
    """Apply the pre-specified candidate rule (thresholds come from the config).

    The standardized gap (gap - null mean) / (null sd * sqrt(1 + 1/R)) is
    referred to a Student t distribution with R - 1 degrees of freedom, R being
    the number of null draws, which accounts for the null moments being
    estimated from few draws.
    """
    support = (decomp["document_frequency"] >= rule["min_document_frequency"]) & \
              (decomp["client_frequency"] >= rule["min_client_frequency"])
    scale = null_sd * np.sqrt(1.0 + 1.0 / repeats)
    z = np.divide(decomp["gap"] - null_mean, scale, out=np.full(len(null_sd), np.nan), where=scale > 0)
    z[~support] = np.nan
    p = 2 * sps.t.sf(np.abs(z), df=repeats - 1)
    q = benjamini_hochberg(p)
    structured = support & (q <= rule["fdr"])
    # Direction of the difference between pooled and within-client association:
    #   sign_reversed  pooled and within have opposite signs
    #   inflated       same sign, pooled is the larger in magnitude: client
    #                  membership adds association the term does not have
    #                  inside a client
    #   attenuated     same sign, pooled is the smaller: a real within-client
    #                  association is partly hidden when clients are pooled
    pooled, within = decomp["beta_pooled"], decomp["beta_within"]
    kind = np.where(np.sign(pooled) * np.sign(within) < 0, "sign_reversed",
                    np.where(np.abs(pooled) > np.abs(within), "inflated", "attenuated"))
    kind = np.where(structured, kind, "")
    # Only inflated or sign-reversed terms are candidates for being
    # client-confounded; attenuated terms are genuine signals, not spurious ones.
    return {"supported": support, "gap_z": z, "gap_q": q, "client_structure": structured, "gap_type": kind,
            "candidate": structured & (kind != "attenuated")}


def sentiment_direction(weight: np.ndarray) -> np.ndarray:
    """One signed sentiment coefficient per feature from a C x d weight matrix.

    2 classes: w_positive - w_negative (the usual logit coefficient).
    C classes: sum_c (c - mean(c)) * w_c, the contrast of the class weights
    with the centred star index, so a positive value means "pushes towards
    higher ratings". For C = 2 this equals (w_1 - w_0) / 2 up to the factor 2,
    which is undone so both cases are on the same scale.
    """
    w = np.asarray(weight, dtype=np.float64)
    c = w.shape[0]
    contrast = np.arange(c) - (c - 1) / 2.0
    out = contrast @ w
    return out * 2.0 if c == 2 else out


def bias_statistics(w_model: np.ndarray, w_reference: np.ndarray, exposure: np.ndarray, mask: np.ndarray) -> dict:
    """Relation between coefficient bias and client-level exposure.

    bias_j = w_model_j - w_reference_j. Reports Pearson and Spearman
    correlation of bias with exposure, and the partial Pearson correlation
    controlling for w_reference_j (which removes both the feature's own
    polarity and any overall scale difference between the two models).
    """
    m = np.asarray(mask) & ~np.isnan(exposure)
    b, x, c = (w_model - w_reference)[m], exposure[m], w_reference[m]
    out = {"num_features": int(m.sum()), "pearson": None, "spearman": None, "partial_pearson": None,
           "mean_abs_bias": float(np.abs(b).mean()) if m.any() else None,
           "scale_ratio": None}
    if m.sum() < 3 or np.std(b) == 0 or np.std(x) == 0:
        return out
    out["pearson"] = float(sps.pearsonr(b, x)[0])
    out["spearman"] = float(sps.spearmanr(b, x)[0])
    denom = float(c @ c)
    if denom > 0:
        out["scale_ratio"] = float((w_model[m] @ c) / denom)   # least-squares scale of model vs reference
        rb = b - np.polyval(np.polyfit(c, b, 1), c)
        rx = x - np.polyval(np.polyfit(c, x, 1), c)
        # A residual that is zero up to rounding (bias fully explained by the
        # reference weights) has no defined correlation.
        if np.std(rb) > 1e-9 * np.std(b) and np.std(rx) > 1e-9 * np.std(x):
            out["partial_pearson"] = float(sps.pearsonr(rb, rx)[0])
    return out


def per_group_metrics(predictions: np.ndarray, labels: np.ndarray, groups: np.ndarray, num_classes: int) -> list[dict]:
    """Accuracy and macro-F1 of one model for every client in ``groups``."""
    order = np.argsort(groups, kind="stable")
    bounds = np.flatnonzero(np.diff(groups[order])) + 1
    rows = []
    for idx in np.split(order, bounds):
        m = metrics_from_confusion(confusion_matrix(labels[idx], predictions[idx], num_classes))
        rows.append({"client": int(groups[idx[0]]), "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    return rows


def generalization_report(pred_seen, y_seen, g_seen, pred_unseen, y_unseen, g_unseen, num_classes: int) -> dict:
    """Seen-client vs unseen-client performance of one model.

    ``seen`` uses the client-test records of training clients; ``unseen`` uses
    all records of held-out clients. gap = seen macro-F1 - unseen macro-F1.
    """
    seen = metrics_from_confusion(confusion_matrix(y_seen, pred_seen, num_classes))
    unseen = metrics_from_confusion(confusion_matrix(y_unseen, pred_unseen, num_classes))
    seen_clients = per_group_metrics(pred_seen, y_seen, g_seen, num_classes)
    unseen_clients = per_group_metrics(pred_unseen, y_unseen, g_unseen, num_classes)
    return {
        "seen": {k: seen[k] for k in ("n", "accuracy", "macro_f1")},
        "unseen": {k: unseen[k] for k in ("n", "accuracy", "macro_f1")},
        "generalization_gap_macro_f1": seen["macro_f1"] - unseen["macro_f1"],
        "generalization_gap_accuracy": seen["accuracy"] - unseen["accuracy"],
        "seen_client_macro_f1": client_dispersion([r["macro_f1"] for r in seen_clients]),
        "unseen_client_macro_f1": client_dispersion([r["macro_f1"] for r in unseen_clients]),
        "per_client": {"seen": seen_clients, "unseen": unseen_clients},
    }


def write_feature_table(path: Path, terms: list[str], columns: dict) -> None:
    """CSV with one row per feature: index, vocabulary term, then ``columns``."""
    d = len(terms)
    for name, col in columns.items():
        if len(col) != d:
            raise ValueError(f"column '{name}' has {len(col)} entries for {d} features")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "term", *columns])
        for j in range(d):
            writer.writerow([j, terms[j], *[_cell(columns[name][j]) for name in columns]])


def _cell(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.8g}"
    return v
