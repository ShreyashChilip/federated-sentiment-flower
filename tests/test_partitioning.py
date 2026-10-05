import numpy as np
import pytest

from src.data.roles import CLIENT_TEST, TRAIN, VAL, assign_roles
from src.partitioning.noise import flip_labels
from src.partitioning.partition import dirichlet_partition, iid_partition, natural_partition, quantity_skew_partition
from src.partitioning.store import get_partition, partition_hash


@pytest.fixture
def labels():
    return np.random.default_rng(0).integers(0, 2, size=4000)


def _is_exact_cover(clients, n):
    merged = np.sort(np.concatenate(clients))
    return len(merged) == n and np.array_equal(merged, np.arange(n))


def test_every_sample_assigned_exactly_once(labels):
    n = len(labels)
    assert _is_exact_cover(iid_partition(n, 10, 1), n)
    assert _is_exact_cover(dirichlet_partition(labels, 10, 0.1, 1)[0], n)
    assert _is_exact_cover(quantity_skew_partition(n, 10, 0.5, 1)[0], n)


def test_same_seed_same_partition_different_seed_differs(labels):
    a = dirichlet_partition(labels, 10, 0.5, 7)[0]
    b = dirichlet_partition(labels, 10, 0.5, 7)[0]
    c = dirichlet_partition(labels, 10, 0.5, 8)[0]
    assert partition_hash(a) == partition_hash(b)
    assert partition_hash(a) != partition_hash(c)


def test_smaller_alpha_gives_more_label_skew(labels):
    def mean_majority_share(alpha):
        shares = []
        for seed in range(5):
            for idx in dirichlet_partition(labels, 10, alpha, seed)[0]:
                shares.append(np.bincount(labels[idx], minlength=2).max() / len(idx))
        return np.mean(shares)

    assert mean_majority_share(0.1) > mean_majority_share(1.0) > mean_majority_share(10.0)
    assert mean_majority_share(10.0) < 0.65


def test_min_client_size_is_enforced(labels):
    clients, _ = dirichlet_partition(labels, 10, 0.1, 3, min_client_size=10)
    assert min(len(c) for c in clients) >= 10


def test_natural_partition_groups_by_key():
    keys = ["u2", "u1", "u2", "u3", "u1", "u2"]
    clients, kept = natural_partition(keys, min_client_size=2)
    assert kept == ["u1", "u2"]
    assert [c.tolist() for c in clients] == [[1, 4], [0, 2, 5]]


def test_roles_are_stratified_disjoint_and_seeded(labels):
    roles = assign_roles(labels, (0.8, 0.1, 0.1), seed=5)
    assert np.array_equal(roles, assign_roles(labels, (0.8, 0.1, 0.1), seed=5))
    assert set(np.unique(roles)) == {TRAIN, VAL, CLIENT_TEST}
    for c in (0, 1):
        share = np.mean(roles[labels == c] == TRAIN)
        assert abs(share - 0.8) < 0.01


def test_label_noise_touches_only_eligible_rows(labels):
    eligible = np.arange(0, len(labels), 2)
    noisy, flipped = flip_labels(labels, eligible, 0.02, 2, seed=1)
    changed = np.flatnonzero(noisy != labels)
    assert np.array_equal(changed, flipped)
    assert len(flipped) == round(0.02 * len(eligible))
    assert np.isin(flipped, eligible).all()


def test_saved_partition_is_reloaded_not_regenerated(tmp_path, labels):
    cfg = {"scheme": "dirichlet", "num_clients": 5, "alpha": 0.5, "min_client_size": 10}
    roles = assign_roles(labels, (0.8, 0.1, 0.1), 0)
    with pytest.raises(FileNotFoundError):
        get_partition(tmp_path, "toy", labels, roles, 2, cfg, seed=42)
    clients, summary, created = get_partition(tmp_path, "toy", labels, roles, 2, cfg, seed=42, create=True)
    again, summary2, created2 = get_partition(tmp_path, "toy", labels, roles, 2, cfg, seed=42)
    assert created and not created2
    assert summary["sha256"] == summary2["sha256"] == partition_hash(again)
    assert sum(c["num_samples"] for c in summary["clients"]) == len(labels)


def test_tampered_partition_is_rejected(tmp_path, labels):
    cfg = {"scheme": "iid", "num_clients": 4}
    get_partition(tmp_path, "toy", labels, None, 2, cfg, seed=1, create=True)
    path = next(tmp_path.glob("*.npz"))
    data = dict(np.load(path))
    data["indices"] = data["indices"][::-1].copy()
    np.savez_compressed(path, **data)
    with pytest.raises(RuntimeError, match="hash"):
        get_partition(tmp_path, "toy", labels, None, 2, cfg, seed=1)


def test_loading_many_clients_keeps_one_copy_of_the_indices(tmp_path):
    """Regression (Kaggle OOM, 2026-10-05): every client must be a view of ONE
    index array; slicing ``npz[key]`` per client kept a full copy per client."""
    from src.partitioning.store import load_partition, partition_hash, save_partition

    perm = np.random.default_rng(0).permutation(50_000)
    clients = [np.sort(c) for c in np.array_split(perm, 5_000)]
    save_partition(tmp_path, "p", clients, {"sha256": partition_hash(clients)})
    loaded, _ = load_partition(tmp_path, "p")
    assert all(np.array_equal(a, b) for a, b in zip(clients, loaded))
    base = loaded[0].base
    assert base is not None and base.nbytes == 50_000 * 8
    assert all(c.base is base for c in loaded)
