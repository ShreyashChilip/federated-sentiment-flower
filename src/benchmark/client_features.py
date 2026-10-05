"""Model-free descriptors of every client, used to characterize which clients fail.

Computed once per regime from the feature bundle (and, for natural bundles,
from the cleaned review store for token-level measures). Nothing here looks
at a model or a metric. Rows used:

* seen clients: their TRAIN rows (the data the client trains on);
* unseen clients: all of their records (they never train).

"Global" reference = all TRAIN rows of all seen clients (the federation's
training distribution), the same rows the vocabulary and IDF were fitted on.

Descriptors
  n                       number of rows
  class_<k>, classes_present, dominant_class_share, label_entropy (nats),
  label_entropy_norm      (entropy / log C), mean_label, label_tv_to_global
  vocab_size              distinct TF-IDF features used by the client
  feature_coverage        vocab_size / number of features (1 - sparsity)
  mean_nnz_per_doc        features per document (length proxy after the vocabulary cut)
  rare_feature_mass       share of the client's TF-IDF mass on features with global
                          train document frequency below the 10th percentile of df
  client_specific_share   share of the client's used features for which this client
                          holds >= 50% of the global training documents containing it
                          (seen clients only; unseen clients are not in the global df)
  centroid_cosine_dist    1 - cos(client mean TF-IDF vector, global mean vector)
  centroid_js_div         Jensen-Shannon divergence (base 2) of the two mean vectors,
                          each normalized to sum to one
  mean_tokens_per_doc     unigram tokens per document (raw text; natural bundles)
  oov_token_rate          share of the client's unigram tokens absent from the
                          vocabulary (raw text; natural bundles)
  unique_users_share      distinct reviewers / records (product clients; raw store)
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.data.roles import TRAIN

FIELDS_BASE = ["population", "client_index", "client_code", "n", "classes_present", "dominant_class_share",
               "label_entropy", "label_entropy_norm", "mean_label", "label_tv_to_global", "vocab_size",
               "feature_coverage", "mean_nnz_per_doc", "rare_feature_mass", "client_specific_share",
               "centroid_cosine_dist", "centroid_js_div"]
FIELDS_RAW = ["mean_tokens_per_doc", "oov_token_rate", "unique_users_share"]


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log2(p / m), 0.0).sum()
        b = np.where(q > 0, q * np.log2(q / m), 0.0).sum()
    return float(0.5 * (a + b))


def _label_stats(y: np.ndarray, c: int, global_dist: np.ndarray) -> dict:
    counts = np.bincount(y, minlength=c).astype(float)
    n = counts.sum()
    p = counts / n
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum())
    out = {f"class_{k}": int(counts[k]) for k in range(c)}
    out.update(classes_present=int((counts > 0).sum()), dominant_class_share=float(p.max()), label_entropy=ent,
               label_entropy_norm=ent / np.log(c), mean_label=float((p * np.arange(c)).sum()),
               label_tv_to_global=float(0.5 * np.abs(p - global_dist).sum()))
    return out


def compute(rt, store_path: Path | None = None, definition: str | None = None, chunk: int = 50_000) -> list[dict]:
    b = rt.bundle
    x = b.x_train
    c, d = b.num_classes, x.shape[1]
    roles = np.asarray(b.roles)
    y = np.asarray(b.y_train)
    seen_rows = [np.asarray(rt.client_rows(k, "train")) for k in range(rt.num_clients)]
    all_train = np.sort(np.concatenate(seen_rows))

    # global training mean vector and document frequency (train rows of seen clients)
    gsum = np.zeros(d, dtype=np.float64)
    gdf = np.zeros(d, dtype=np.int64)
    for lo in range(0, len(all_train), chunk):
        m = x[all_train[lo:lo + chunk]]
        gsum += np.asarray(m.sum(axis=0)).ravel()
        gdf += np.bincount(m.indices, minlength=d)
    gmean = gsum / len(all_train)
    gdist = gsum / gsum.sum()
    rare = gdf < np.percentile(gdf[gdf > 0], 10)
    global_label = np.bincount(y[all_train], minlength=c) / len(all_train)

    populations = [("seen", k, None, rows) for k, rows in enumerate(seen_rows)]
    if b.kind == "natural":
        from src.data.natural import UNSEEN

        held = np.flatnonzero(roles == UNSEEN)
        codes = np.asarray(b.client_codes)[held]
        order = np.argsort(codes, kind="stable")
        uniq, starts = np.unique(codes[order], return_index=True)
        for i, (code, group) in enumerate(zip(uniq, np.split(held[order], starts[1:]))):
            populations.append(("unseen", i, int(code), np.sort(group)))
        cc = np.asarray(b.client_codes)
        seen_codes = [int(cc[r[0]]) if len(r) else -1 for r in seen_rows]
        populations = [(p, k, seen_codes[k] if p == "seen" else code, rows) for p, k, code, rows in populations]

    out = []
    for pop, k, code, rows in populations:
        if not len(rows):
            continue
        m = x[rows]
        row = {"population": pop, "client_index": k, "client_code": code, "n": int(len(rows))}
        row.update(_label_stats(y[rows], c, global_label))
        csum = np.asarray(m.sum(axis=0)).ravel().astype(np.float64)
        cdf = np.bincount(m.indices, minlength=d)
        used = cdf > 0
        mass = csum.sum()
        row.update(
            vocab_size=int(used.sum()), feature_coverage=float(used.mean()),
            mean_nnz_per_doc=float(m.nnz / len(rows)),
            rare_feature_mass=float(csum[rare].sum() / mass) if mass > 0 else float("nan"),
        )
        if pop == "seen":
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(gdf > 0, cdf / np.maximum(gdf, 1), 0.0)
            row["client_specific_share"] = float((share[used] >= 0.5).mean()) if used.any() else float("nan")
        else:
            row["client_specific_share"] = float("nan")
        cmean = csum / len(rows)
        denom = np.linalg.norm(cmean) * np.linalg.norm(gmean)
        row["centroid_cosine_dist"] = float(1 - cmean @ gmean / denom) if denom > 0 else float("nan")
        row["centroid_js_div"] = _js(csum / mass, gdist) if mass > 0 else float("nan")
        out.append(row)

    if store_path is not None and b.kind == "natural":
        _add_raw_text_measures(out, b, store_path, definition)
    return out


def _add_raw_text_measures(rows: list[dict], b, store_path: Path, definition: str) -> None:
    """Token counts, out-of-vocabulary rate and reviewer diversity from the review store."""
    from sklearn.feature_extraction.text import CountVectorizer

    vocab = set(t for t in b.vocabulary() if " " not in t)
    analyzer = CountVectorizer(lowercase=True, ngram_range=(1, 1)).build_analyzer()
    keys = json.loads((b.path / "client_keys.json").read_text(encoding="utf-8"))
    by_code = {r["client_code"]: r for r in rows if r["client_code"] is not None}
    with sqlite3.connect(store_path) as con:
        for code, row in by_code.items():
            key = keys[code]
            texts, users = [], set()
            for text, user in con.execute(f"SELECT text, user_id FROM reviews WHERE {definition}=?", (key,)):
                texts.append(text)
                users.add(user)
            tokens = in_vocab = 0
            for t in texts:
                toks = analyzer(t)
                tokens += len(toks)
                in_vocab += sum(1 for w in toks if w in vocab)
            # Note: computed over all of the client's records (seen clients: all roles), a
            # descriptor of the client's language, not of any evaluation split.
            row["mean_tokens_per_doc"] = tokens / max(len(texts), 1)
            row["oov_token_rate"] = 1 - in_vocab / tokens if tokens else float("nan")
            row["unique_users_share"] = len(users) / max(len(texts), 1)


def write(rows: list[dict], out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = FIELDS_BASE + sorted({k for r in rows for k in r if k.startswith("class_")}) + \
        [f for f in FIELDS_RAW if any(f in r for r in rows)]
    path = out_dir / "client_features.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out_dir / "client_features_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return path
