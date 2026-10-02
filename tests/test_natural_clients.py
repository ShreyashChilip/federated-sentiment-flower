"""Natural-client data path. Uses a SYNTHETIC review file (tests/synthetic_reviews.py)."""
import json

import numpy as np
import pytest

from src.data import amazon2023, natural
from src.data.prepare import build_bundle, load_bundle
from src.data.roles import CLIENT_TEST, TRAIN, VAL
from src.utils.config import load_config
from tests.synthetic_reviews import write_reviews


@pytest.fixture(scope="module")
def review_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("amazon") / "Synthetic.jsonl"
    return path, write_reviews(path)


def make_cfg(review_file, tmp_path, definition="user", **overrides):
    path, _ = review_file
    base = {
        "data.local_path": str(path), "data.category": "Synthetic", "data.cache_dir": str(tmp_path / "cache"),
        "data.client_filter.min_reviews": 10, "features.max_features": 500, "features.min_df": 1,
        "partition.dir": str(tmp_path / "partitions"), "experiment.results_dir": str(tmp_path / "results"),
        "device": "cpu",
    }
    return load_config(f"exp0b_{definition}.yaml", overrides={**base, **overrides})


def test_cleaning_rules(review_file):
    path, truth = review_file
    table, counts = amazon2023.read_reviews(path)
    for key in ("bad_rating", "empty_text", "missing_id", "bad_json"):
        assert counts[key] == truth[key]
    assert counts["duplicate"] >= 1
    assert counts["kept"] + counts["duplicate"] == truth["valid"] + 1
    assert set(table["label"].unique()) <= {0, 1, 2, 3, 4}       # five classes, never collapsed to polarity
    assert len(set(table["label"].unique())) > 2
    assert (table["text"].str.strip() != "").all()
    assert table[["user_id", "parent_asin", "text"]].duplicated().sum() == 0


def test_category_and_revision_are_required(tmp_path):
    with pytest.raises(ValueError, match="category"):
        amazon2023.resolve_file({"category": None, "revision": "abc"}, tmp_path)
    with pytest.raises(ValueError, match="revision"):
        amazon2023.resolve_file({"category": "X", "revision": None}, tmp_path)


def test_profile_reports_sizes_labels_and_retention():
    codes = np.array([0] * 6 + [1] * 3 + [2] * 1)
    labels = np.array([4, 4, 4, 4, 3, 4, 0, 0, 1, 2])
    p = natural.profile_clients(codes, labels, 5, thresholds=(1, 3, 5))
    assert p["num_records"] == 10 and p["num_raw_clients"] == 3
    assert p["client_size"]["min"] == 1 and p["client_size"]["max"] == 6 and p["client_size"]["median"] == 3
    assert p["global_class_counts"] == [2, 1, 1, 1, 5]
    assert np.isclose(sum(p["global_class_distribution"]), 1.0)
    assert p["clients_with_single_class"] == 1
    by_t = {r["min_reviews"]: r for r in p["retention_by_min_reviews"]}
    assert by_t[1]["clients_retained"] == 3 and by_t[1]["record_fraction_retained"] == 1.0
    assert by_t[3]["clients_retained"] == 2 and by_t[3]["records_retained"] == 9
    assert by_t[5]["clients_retained"] == 1 and np.isclose(by_t[5]["record_fraction_retained"], 0.6)
    assert p["per_client_table"]["label_counts"].tolist()[1] == [2, 1, 0, 0, 0]


def test_filter_rule_must_be_set_and_is_applied():
    codes = np.repeat(np.arange(6), [2, 5, 8, 20, 30, 40])
    with pytest.raises(ValueError, match="min_reviews is not set"):
        natural.filter_clients(codes, {"min_reviews": None})
    mask, rep = natural.filter_clients(codes, {"min_reviews": 8})
    assert sorted(np.unique(codes[mask])) == [2, 3, 4, 5] and rep["clients_retained"] == 4
    assert np.isclose(rep["record_fraction_retained"], 98 / 105)
    mask, rep = natural.filter_clients(codes, {"min_reviews": 8, "max_reviews": 25, "filter_seed": 1})
    assert np.bincount(codes[mask], minlength=6).tolist() == [0, 0, 8, 20, 25, 25] and rep["clients_capped"] == 2
    again, _ = natural.filter_clients(codes, {"min_reviews": 8, "max_reviews": 25, "filter_seed": 1})
    assert np.array_equal(mask, again)                           # seeded
    mask, rep = natural.filter_clients(codes, {"min_reviews": 5, "max_clients": 2, "filter_seed": 3})
    assert len(np.unique(codes[mask])) == 2 == rep["clients_retained"]


def test_client_split_is_disjoint_and_roles_respect_it():
    codes = np.repeat(np.arange(20), 15)
    seen, unseen = natural.split_clients(np.arange(20), 0.2, seed=7)
    assert len(unseen) == 4 and len(seen) == 16 and not set(seen) & set(unseen)
    assert np.array_equal(seen, natural.split_clients(np.arange(20), 0.2, seed=7)[0])
    roles = natural.assign_roles_within_clients(codes, seen, (0.8, 0.1, 0.1), seed=1)
    report = natural.assert_no_client_overlap(codes, roles)
    assert report["overlap"] == 0 and report["unseen_clients"] == 4 and report["unseen_records"] == 60
    for c in unseen:                                             # every record of a held-out client is held out
        assert (roles[codes == c] == natural.UNSEEN).all()
    for c in seen:                                               # every seen client can train and be tested
        own = roles[codes == c]
        assert (own == TRAIN).sum() >= 11 and (own == VAL).sum() >= 1 and (own == CLIENT_TEST).sum() >= 1
        assert len(own) == 15 and (own != natural.UNSEEN).all()


def test_overlap_assertion_catches_a_leak():
    codes = np.repeat(np.arange(4), 5)
    roles = natural.assign_roles_within_clients(codes, np.array([0, 1, 2]), (0.8, 0.1, 0.1), seed=0)
    roles[np.flatnonzero(codes == 3)[0]] = TRAIN                 # leak one record of the held-out client
    with pytest.raises(AssertionError, match="both the seen and the unseen"):
        natural.assert_no_client_overlap(codes, roles)


@pytest.mark.parametrize("definition,column", [("user", "user_id"), ("product", "parent_asin")])
def test_natural_bundle_holds_out_whole_clients(review_file, tmp_path, definition, column):
    cfg = make_cfg(review_file, tmp_path, definition)
    b = load_bundle(build_bundle(cfg))
    assert b.kind == "natural" and b.num_classes == 5
    assert b.meta["client_definition"] == column                 # the two definitions are separate bundles
    assert b.meta["holdout"]["overlap"] == 0
    unseen_clients = set(b.client_codes[b.roles == natural.UNSEEN])
    seen_clients = set(b.client_codes[b.roles != natural.UNSEEN])
    assert unseen_clients and seen_clients and not unseen_clients & seen_clients
    assert b.x_test.shape[0] == (b.roles == natural.UNSEEN).sum() == len(b.test_client_codes)
    assert b.meta["client_filter"]["rule"]["min_reviews"] == 10
    assert np.bincount(b.client_codes).min() >= 10
    assert len(b.vocabulary()) == b.input_dim == b.meta["num_features"]


def test_user_and_product_bundles_do_not_share_a_directory(review_file, tmp_path):
    a = build_bundle(make_cfg(review_file, tmp_path, "user"))
    b = build_bundle(make_cfg(review_file, tmp_path, "product"))
    assert a != b


def test_vocabulary_is_fitted_on_training_rows_of_seen_clients_only(review_file, tmp_path):
    """Append a held-out-only marker word and check it cannot enter the feature space."""
    path, _ = review_file
    cfg = make_cfg(review_file, tmp_path)
    b = load_bundle(build_bundle(cfg))
    keys = json.loads((b.path / "client_keys.json").read_text())
    unseen_key = keys[int(b.client_codes[b.roles == natural.UNSEEN][0])]
    marked = tmp_path / "marked.jsonl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            out.append(line)
            continue
        if r.get("user_id") == unseen_key and r.get("text", "").strip():
            r["text"] += " heldoutmarker heldoutmarker"
        out.append(json.dumps(r))
    marked.write_text("\n".join(out) + "\n", encoding="utf-8")
    cfg2 = make_cfg((marked, None), tmp_path)
    b2 = load_bundle(build_bundle(cfg2))
    keys2 = json.loads((b2.path / "client_keys.json").read_text())
    assert b2.roles[b2.client_codes == keys2.index(unseen_key)].tolist().count(natural.UNSEEN) > 0   # same client is still held out
    assert "heldoutmarker" not in b2.vocabulary()
    assert "excellent" in b2.vocabulary()


def test_training_refuses_to_start_without_a_filter_rule(review_file, tmp_path):
    cfg = make_cfg(review_file, tmp_path, **{"data.client_filter.min_reviews": None})
    with pytest.raises(ValueError, match="min_reviews is not set"):
        build_bundle(cfg)
