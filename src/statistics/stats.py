"""Across-seed summaries and paired comparisons.

Each observation is the final metric of one seed. Two methods are compared on
the same seeds and the same saved partitions, so the comparison is paired.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats as sps


def summarize(values, confidence: float = 0.95) -> dict:
    """Mean, sample standard deviation and Student-t confidence interval."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    out = {"n": n, "mean": float(v.mean()) if n else None, "std": None, "ci95_low": None, "ci95_high": None}
    if n > 1:
        std = float(v.std(ddof=1))
        half = float(sps.t.ppf(0.5 + confidence / 2, n - 1) * std / np.sqrt(n))
        out.update(std=std, ci95_low=out["mean"] - half, ci95_high=out["mean"] + half)
    return out


def wilcoxon_exact(diffs) -> tuple[float | None, float | None]:
    """Two-sided exact Wilcoxon signed-rank p-value and rank-biserial correlation.

    Zero differences are discarded; tied magnitudes receive mid-ranks. The null
    distribution is enumerated over all sign assignments, which is exact for
    the handful of seeds used here. Returns ``(None, None)`` if every
    difference is zero.
    """
    d = np.asarray(diffs, dtype=np.float64)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return None, None
    if n > 20:
        res = sps.wilcoxon(d)
        ranks = sps.rankdata(np.abs(d))
        w_plus = ranks[d > 0].sum()
        return float(res.pvalue), float((2 * w_plus - ranks.sum()) / ranks.sum())
    ranks = sps.rankdata(np.abs(d))
    total = ranks.sum()
    w_plus = ranks[d > 0].sum()
    centre = total / 2
    observed = abs(w_plus - centre)
    count = sum(
        abs(np.dot(signs, ranks) - centre) >= observed - 1e-12
        for signs in itertools.product((0, 1), repeat=n)
    )
    return count / 2**n, float((2 * w_plus - total) / total)


def paired_comparison(a, b, confidence: float = 0.95) -> dict:
    """Compare method A with method B on paired observations (A minus B).

    Reports the mean difference with a t-interval, the paired t-test, the exact
    Wilcoxon signed-rank test, Cohen's d_z and the rank-biserial correlation.

    ``wilcoxon_min_p`` is the smallest two-sided p-value the Wilcoxon test can
    produce with this many pairs (2 / 2^n). With 5 seeds it is 0.0625, so a
    single 5-seed comparison cannot be significant at 0.05 under this test no
    matter how large the effect; the value is reported so that a
    non-significant result is not misread as evidence of no difference.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs equally long inputs")
    d = a - b
    n = len(d)
    s = summarize(d, confidence)
    out = {
        "n": n,
        "mean_diff": s["mean"],
        "diff_ci95_low": s["ci95_low"],
        "diff_ci95_high": s["ci95_high"],
        "cohens_dz": None,
        "t_p": None,
        "wilcoxon_p": None,
        "rank_biserial": None,
        "wilcoxon_min_p": 2 / 2**n if n else None,
    }
    if n > 1 and s["std"] and s["std"] > 0:
        out["cohens_dz"] = s["mean"] / s["std"]
        out["t_p"] = float(sps.ttest_rel(a, b).pvalue)
    out["wilcoxon_p"], out["rank_biserial"] = wilcoxon_exact(d)
    return out


def holm_correction(p_values) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()
