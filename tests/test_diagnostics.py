"""Mechanism diagnostics, checked against brute-force arithmetic on small synthetic data."""
import csv

import numpy as np
import scipy.sparse as sp

from src.analysis import diagnostics as dg
from src.clients.trainer import label_prior_offset, local_train
from src.models.registry import build_model, get_weights
from src.partitioning.partition import dirichlet_client_partition
from src.partitioning.store import partition_hash


def toy(seed=0, n_clients=30, per_client=40):
    """Feature 0: genuine (follows the rating inside a client).
    Feature 1: client habit (used by high-rating clients, unrelated inside a client).
    Feature 2: noise."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_clients), per_client)
    high = (np.arange(n_clients) % 2 == 0)[groups]
    score = np.where(high, 3.0, 1.0) + rng.integers(-1, 2, size=len(groups))
    client_mean = np.where(high, 3.0, 1.0)
    z0 = (score > client_mean) & (rng.random(len(groups)) < 0.9)
    z1 = high & (rng.random(len(groups)) < 0.7)
    z2 = rng.random(len(groups)) < 0.3
    x = sp.csr_matrix(np.column_stack([z0, z1, z2]).astype(np.float32) * 0.5)   # TF-IDF values, not 0/1
    return x, score, groups


def brute_force(z, s, g):
    zc, sc = z - z.mean(), s - s.mean()
    pooled = (zc @ sc) / (zc @ zc)
    zw = z - np.array([z[g == k].mean() for k in g])
    sw = s - np.array([s[g == k].mean() for k in g])
    within = (zw @ sw) / (zw @ zw)
    ks = np.unique(g)
    n_k = np.array([(g == k).sum() for k in ks])
    zb = np.array([z[g == k].mean() for k in ks]) - z.mean()
    sb = np.array([s[g == k].mean() for k in ks]) - s.mean()
    between = (n_k * zb) @ sb / ((n_k * zb) @ zb)
    return pooled, within, between, (zw @ zw) / (zc @ zc)


def test_decomposition_matches_brute_force_and_identity_holds():
    x, score, groups = toy()
    d = dg.association_decomposition(x, score, groups)
    dense = (x.toarray() > 0).astype(float)
    for j in range(3):
        pooled, within, between, lam = brute_force(dense[:, j], score, groups)
        assert np.isclose(d["beta_pooled"][j], pooled) and np.isclose(d["beta_within"][j], within)
        assert np.isclose(d["beta_between"][j], between) and np.isclose(d["lambda_within"][j], lam)
        assert d["document_frequency"][j] == dense[:, j].sum()
        assert d["client_frequency"][j] == len({g for g, v in zip(groups, dense[:, j]) if v})
    lam = d["lambda_within"]
    assert np.allclose(d["beta_pooled"], lam * d["beta_within"] + (1 - lam) * d["beta_between"])
    assert np.allclose(d["gap"], (1 - lam) * (d["beta_between"] - d["beta_within"]))


def test_exposure_matches_the_frozen_0a_definition():
    x, score, groups = toy(seed=3)
    d = dg.association_decomposition(x, score, groups)
    dense = (x.toarray() > 0).astype(float)
    r_k = np.array([score[groups == k].mean() for k in range(groups.max() + 1)])
    for j in range(3):
        df_k = np.array([dense[groups == k, j].sum() for k in range(groups.max() + 1)])
        assert np.isclose(d["exposure"][j], (df_k @ (r_k - score.mean())) / df_k.sum())


def test_client_habit_word_is_flagged_and_genuine_word_is_not():
    x, score, groups = toy(seed=1)
    d = dg.association_decomposition(x, score, groups)
    null_mean, null_sd = dg.permutation_null(x, score, groups, repeats=20, seed=0)
    flags = dg.flag_candidates(d, null_mean, null_sd, 20, {"min_document_frequency": 20, "min_client_frequency": 3, "fdr": 0.05})
    # the habit word looks strongly positive when pooled but carries nothing inside a client
    assert d["beta_pooled"][1] > 1.0 and abs(d["beta_within"][1]) < 0.2
    assert flags["candidate"][1]
    # the noise word has no client structure
    assert not flags["candidate"][2]
    # the genuine word keeps a clear within-client association; if pooling
    # changes it at all, it is attenuated, which is not a confounding candidate
    assert d["beta_within"][0] > 0.5
    assert not flags["candidate"][0] and flags["gap_type"][0] in ("", "attenuated")
    assert flags["gap_type"][1] == "inflated" and flags["client_structure"][1]
    # support thresholds remove rare features from consideration
    strict = dg.flag_candidates(d, null_mean, null_sd, 20, {"min_document_frequency": 10**6, "min_client_frequency": 3, "fdr": 0.05})
    assert not strict["candidate"].any() and np.isnan(strict["gap_z"]).all()


def test_null_is_calibrated_on_noise_features():
    """Many pure-noise terms in data whose clients differ in mean sentiment:
    the rule must flag about the nominal share or fewer, not most of them."""
    rng = np.random.default_rng(5)
    n_clients, per_client, d = 40, 50, 300
    groups = np.repeat(np.arange(n_clients), per_client)
    score = np.clip(np.repeat(rng.integers(0, 5, n_clients), per_client) + rng.integers(-1, 2, len(groups)), 0, 4).astype(float)
    x = sp.csr_matrix((rng.random((len(groups), d)) < 0.1).astype(np.float32))
    dec = dg.association_decomposition(x, score, groups)
    mean, sd = dg.permutation_null(x, score, groups, repeats=20, seed=1)
    flags = dg.flag_candidates(dec, mean, sd, 20, {"min_document_frequency": 20, "min_client_frequency": 3, "fdr": 0.05})
    assert flags["supported"].sum() == d
    assert flags["client_structure"].sum() <= 0.05 * d


def test_shuffled_clients_give_no_gap():
    x, score, groups = toy(seed=2)
    shuffled = np.random.default_rng(0).permutation(groups)
    d = dg.association_decomposition(x, score, shuffled)
    assert np.abs(d["gap"]).max() < 0.1


def test_benjamini_hochberg():
    q = dg.benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan, 0.005]))
    assert np.allclose(q[[0, 1, 2, 4]], [0.02, 0.04, 0.04, 0.02]) and np.isnan(q[3])


def test_sentiment_direction_two_and_five_classes():
    w2 = np.array([[1.0, -2.0], [3.0, 0.5]])
    assert np.allclose(dg.sentiment_direction(w2), [2.0, 2.5])                 # w_pos - w_neg
    w5 = np.zeros((5, 2))
    w5[4, 0], w5[0, 0] = 1.0, -1.0       # feature 0 pushes towards 5 stars
    w5[2, 1] = 3.0                       # feature 1 only favours the middle class
    out = dg.sentiment_direction(w5)
    assert out[0] == 4.0 and out[1] == 0.0


def test_bias_statistics_partial_correlation_ignores_overall_scale():
    rng = np.random.default_rng(0)
    w_ref = rng.normal(size=400)
    exposure = 0.5 * w_ref + rng.normal(size=400)          # exposure is related to polarity, as under label skew
    mask = np.ones(400, dtype=bool)
    shrunk = 0.6 * w_ref                                   # a model that is merely under-trained
    s = dg.bias_statistics(shrunk, w_ref, exposure, mask)
    assert abs(s["spearman"]) > 0.3                        # raw bias correlates with exposure through polarity alone
    assert s["partial_pearson"] is None                    # nothing is left once polarity is controlled for
    assert np.isclose(s["scale_ratio"], 0.6)
    biased = w_ref + 0.8 * exposure                        # a model whose coefficients really follow exposure
    assert dg.bias_statistics(biased, w_ref, exposure, mask)["partial_pearson"] > 0.9
    assert dg.bias_statistics(w_ref, w_ref, exposure, mask)["pearson"] is None   # no bias at all


def test_generalization_report():
    y_seen, g_seen = np.array([0, 0, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1])
    p_seen = np.array([0, 0, 1, 1, 1, 1])
    y_un, g_un = np.array([0, 1, 0, 1]), np.array([7, 7, 9, 9])
    p_un = np.array([0, 1, 1, 1])
    r = dg.generalization_report(p_seen, y_seen, g_seen, p_un, y_un, g_un, 2)
    assert r["seen"]["macro_f1"] == 1.0 and r["unseen"]["accuracy"] == 0.75
    assert np.isclose(r["generalization_gap_macro_f1"], 1.0 - r["unseen"]["macro_f1"])
    assert r["unseen_client_macro_f1"]["num_clients"] == 2 and r["unseen_client_macro_f1"]["best"] == 1.0
    assert r["unseen_client_macro_f1"]["worst"] < 1.0
    assert [c["client"] for c in r["per_client"]["unseen"]] == [7, 9]


def test_feature_table_maps_every_index_to_its_term(tmp_path):
    terms = ["awful", "excellent", "sig plus"]
    dg.write_feature_table(tmp_path / "f.csv", terms, {"w": np.array([-1.5, 2.0, np.nan]), "flag": np.array([True, False, True])})
    with open(tmp_path / "f.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [(int(r["feature_index"]), r["term"]) for r in rows] == list(enumerate(terms))
    assert rows[1]["w"] == "2" and rows[2]["w"] == "" and rows[0]["flag"] == "1"


def test_label_prior_offset_values():
    off = label_prior_offset(np.array([0, 0, 0, 1]), 3)
    assert np.allclose(np.exp(off), np.array([4, 2, 1]) / 7)                   # add-one smoothing, missing class kept finite
    rel = label_prior_offset(np.array([0, 0, 0, 1]), 3, reference=np.array([0.5, 0.25, 0.25]))
    assert np.allclose(np.exp(rel), (np.array([4, 2, 1]) / 7) / np.array([0.5, 0.25, 0.25]))


def test_logit_offset_changes_training_but_zero_offset_does_not():
    rng = np.random.default_rng(0)
    x = sp.csr_matrix(rng.normal(size=(64, 6)).astype(np.float32))
    y = np.array([0] * 60 + [1] * 4)                        # a heavily skewed client

    def run(offset):
        return local_train(build_model({"name": "lr"}, 6, 2), x, y, epochs=3, batch_size=64, lr=0.5, seed=0, logit_offset=offset)["weights"]

    plain, zero = run(None), run(np.zeros(2, dtype=np.float32))
    corrected = run(label_prior_offset(y, 2))
    assert all(np.array_equal(a, b) for a, b in zip(plain, zero))
    # without correction the bias learns the client's label frequency; with it, much less
    assert abs(plain[1][0] - plain[1][1]) > 3 * abs(corrected[1][0] - corrected[1][1])
    assert get_weights(build_model({"name": "lr"}, 6, 2))[1].tolist() == [0.0, 0.0]


def test_client_wise_dirichlet_partition():
    labels = np.repeat([0, 1], 5000)
    a = dirichlet_client_partition(labels, 100, 0.1, seed=1)
    assert np.array_equal(np.sort(np.concatenate(a)), np.arange(10000))        # exact cover
    assert {len(c) for c in a} == {100}                                         # equal sizes, no empty client
    assert partition_hash(a) == partition_hash(dirichlet_client_partition(labels, 100, 0.1, seed=1))
    assert partition_hash(a) != partition_hash(dirichlet_client_partition(labels, 100, 0.1, seed=2))

    def majority(alpha):
        parts = dirichlet_client_partition(labels, 100, alpha, seed=3)
        return np.mean([np.bincount(labels[c], minlength=2).max() / len(c) for c in parts])

    assert majority(0.1) > majority(1.0) > majority(100.0)
    uneven = dirichlet_client_partition(np.repeat([0, 1], [503, 500]), 10, 0.5, seed=0)
    assert sorted(len(c) for c in uneven) == [100] * 7 + [101] * 3
