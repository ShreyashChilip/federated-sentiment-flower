import numpy as np

from src.preprocessing.tfidf import FederatedTfidf

DOCS = [
    "the food was great and the service was great",
    "terrible food and rude service",
    "great place great staff",
    "the worst pizza ever",
    "service was slow but the pizza was great",
    "never coming back terrible",
    "great pizza",
    "rude staff and slow service",
]
CFG = {"max_features": 12, "ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True}


def test_federated_fit_equals_pooled_fit():
    pooled = FederatedTfidf(CFG).fit(DOCS)
    federated = FederatedTfidf(CFG).fit_federated([DOCS[:3], DOCS[3:4], [], DOCS[4:]])
    assert pooled.vocabulary_ == federated.vocabulary_
    assert np.array_equal(pooled.df_, federated.df_)
    assert np.allclose(pooled.idf_, federated.idf_)
    assert pooled.n_docs_ == federated.n_docs_ == len(DOCS)


def test_features_have_fixed_dimension_and_unit_norm():
    vec = FederatedTfidf(CFG).fit(DOCS)
    x = vec.transform(DOCS + ["completely unseen words here"])
    assert x.shape == (len(DOCS) + 1, len(vec.vocabulary_))
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
    # every row is unit length, or all zeros if none of its terms is in the vocabulary
    assert all(np.isclose(v, 1.0, atol=1e-5) or v == 0.0 for v in norms)
    assert np.isclose(norms[0], 1.0, atol=1e-5)
    assert norms[-1] == 0.0  # out-of-vocabulary document maps to the zero vector


def test_held_out_documents_do_not_change_the_feature_space():
    train = DOCS[:6]
    a = FederatedTfidf(CFG).fit(train)
    a.transform(DOCS[6:])  # transforming held-out text must not alter the fit
    b = FederatedTfidf(CFG).fit(train)
    assert a.vocabulary_ == b.vocabulary_ and np.array_equal(a.idf_, b.idf_)
    assert all(df >= CFG["min_df"] for df in a.df_)
