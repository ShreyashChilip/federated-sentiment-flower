"""Globally consistent TF-IDF features for federated training.

Averaging logistic-regression coefficients is only meaningful if coordinate j
means the same term on every client. The feature space is therefore built once
from two additive statistics of the TRAIN-role documents:

* corpus term frequency  tf(t) = sum over clients of the client's count of t
* document frequency     df(t) = sum over clients of the client's number of
                                 documents containing t

The vocabulary is the ``max_features`` terms with the highest tf(t) among those
with df(t) >= ``min_df``; idf(t) = ln((1 + n) / (1 + df(t))) + 1. Both
statistics are sums of per-client counts, so the server can obtain them in one
preprocessing round (``fit_federated``) and the result is identical to fitting
on pooled data (``fit``). The equivalence is checked in the test suite.

Privacy note: per-client term counts are sensitive. Sending them in the clear
reveals each client's vocabulary; the paper states this and treats the
preprocessing round as requiring secure aggregation in deployment.

No validation, client-test or global-test document is used for fitting.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


def _counter(cfg: dict, vocabulary=None) -> CountVectorizer:
    return CountVectorizer(
        lowercase=True,
        ngram_range=tuple(cfg.get("ngram_range", (1, 2))),
        vocabulary=vocabulary,
        dtype=np.int32,
    )


def term_statistics(texts, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """What one client reports: terms, term counts, document counts, number of docs."""
    cv = _counter(cfg)
    counts = cv.fit_transform(texts)
    terms = cv.get_feature_names_out()
    tf = np.asarray(counts.sum(axis=0)).ravel().astype(np.int64)
    df = np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.int64)
    return terms, tf, df, counts.shape[0]


class FederatedTfidf:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None
        self.df_: np.ndarray | None = None
        self.n_docs_: int = 0

    def _finalize(self, terms: np.ndarray, tf: np.ndarray, df: np.ndarray, n_docs: int) -> "FederatedTfidf":
        keep = df >= self.cfg.get("min_df", 1)
        terms, tf, df = terms[keep], tf[keep], df[keep]
        # highest corpus frequency first; ties broken alphabetically
        top = np.lexsort((terms, -tf))[: self.cfg["max_features"]]
        top = top[np.argsort(terms[top])]
        self.vocabulary_ = {t: i for i, t in enumerate(terms[top].tolist())}
        self.df_ = df[top]
        self.idf_ = (np.log((1 + n_docs) / (1 + self.df_)) + 1.0).astype(np.float32)
        self.n_docs_ = n_docs
        return self

    def fit(self, train_texts) -> "FederatedTfidf":
        return self._finalize(*term_statistics(train_texts, self.cfg))

    def fit_federated(self, client_texts: list) -> "FederatedTfidf":
        """Build the same feature space from per-client statistics only."""
        tf, df, n = Counter(), Counter(), 0
        for texts in client_texts:
            if len(texts) == 0:
                continue
            c_terms, c_tf, c_df, c_n = term_statistics(texts, self.cfg)
            tf.update(dict(zip(c_terms.tolist(), c_tf.tolist())))
            df.update(dict(zip(c_terms.tolist(), c_df.tolist())))
            n += c_n
        terms = np.array(sorted(tf), dtype=object)
        return self._finalize(
            terms, np.array([tf[t] for t in terms], dtype=np.int64), np.array([df[t] for t in terms], dtype=np.int64), n
        )

    def transform(self, texts) -> sp.csr_matrix:
        counts = _counter(self.cfg, self.vocabulary_).transform(texts).astype(np.float32)
        if self.cfg.get("sublinear_tf", True):
            np.log(counts.data, out=counts.data)
            counts.data += 1.0
        counts = counts @ sp.diags(self.idf_)
        return normalize(counts, norm="l2", axis=1, copy=False).tocsr().astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        return sorted(self.vocabulary_, key=self.vocabulary_.get)
