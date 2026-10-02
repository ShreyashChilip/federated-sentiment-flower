import numpy as np
import scipy.sparse as sp
import torch

from src.clients.trainer import evaluate, local_train, planned_steps
from src.models.registry import build_model, get_weights, parameter_counts, set_weights
from src.strategies.aggregation import ServerState, weighted_average

D, C = 20, 2


def make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, D)).astype(np.float32)
    w = rng.normal(size=D)
    y = (x @ w > 0).astype(np.int64)
    return sp.csr_matrix(x), y


def fresh(weights=None):
    model = build_model({"name": "lr"}, D, C)
    if weights is not None:
        set_weights(model, weights)
    return model


def full_batch_gradient(weights, x, y):
    model = fresh(weights)
    xt = torch.tensor(x.toarray())
    loss = torch.nn.functional.cross_entropy(model(xt), torch.tensor(y))
    return [g.numpy() for g in torch.autograd.grad(loss, list(model.parameters()))]


def test_sparse_and_dense_forward_agree():
    x, _ = make_data(16)
    model = fresh([np.random.default_rng(1).normal(size=(C, D)).astype(np.float32), np.zeros(C, np.float32)])
    from src.clients.trainer import to_tensor

    assert torch.allclose(model(to_tensor(x, "cpu")), model(torch.tensor(x.toarray())), atol=1e-6)


def test_training_is_deterministic_for_a_seed():
    x, y = make_data()
    runs = [local_train(fresh(), x, y, epochs=2, batch_size=16, lr=0.1, seed=s)["weights"] for s in (3, 3, 4)]
    assert all(np.array_equal(a, b) for a, b in zip(runs[0], runs[1]))
    assert not np.array_equal(runs[0][0], runs[2][0])


def test_fractional_epochs_run_proportional_steps():
    assert planned_steps(100, 10, 1) == 10
    assert planned_steps(100, 10, 0.5) == 5
    assert planned_steps(5, 10, 0.5) == 1
    x, y = make_data(100)
    assert local_train(fresh(), x, y, epochs=0.5, batch_size=10, lr=0.1, seed=0)["steps"] == 5


def test_one_full_batch_fedavg_round_equals_one_centralized_step():
    """With one full-batch step per client, FedAvg is exactly gradient descent
    on the pooled data. This ties client training and aggregation together."""
    x, y = make_data(300)
    parts = [np.arange(0, 100), np.arange(100, 180), np.arange(180, 300)]
    lr = 0.5
    start = get_weights(fresh())
    locals_ = [local_train(fresh(start), x[p], y[p], epochs=1, batch_size=len(p), lr=lr, seed=0)["weights"] for p in parts]
    fed = weighted_average(locals_, [len(p) for p in parts])
    central = [w - lr * g for w, g in zip(start, full_batch_gradient(start, x, y))]
    for a, b in zip(fed, central):
        assert np.allclose(a, b, atol=1e-6)


def test_fedprox_gradient_contains_proximal_term():
    """One full-batch FedProx step from w != w_global... the proximal pull is
    zero at the start of a round, so compare after two steps instead."""
    x, y = make_data(64)
    lr, mu = 0.1, 1.0
    start = get_weights(fresh())
    out = local_train(fresh(start), x, y, epochs=2, batch_size=64, lr=lr, seed=0, algorithm="fedprox", mu=mu)
    w1 = [w - lr * g for w, g in zip(start, full_batch_gradient(start, x, y))]
    g1 = full_batch_gradient(w1, x, y)
    w2 = [w - lr * (g + mu * (w - w0)) for w, g, w0 in zip(w1, g1, start)]
    for a, b in zip(out["weights"], w2):
        assert np.allclose(a, b, atol=1e-6)
    plain = local_train(fresh(start), x, y, epochs=2, batch_size=64, lr=lr, seed=0)["weights"]
    dist = lambda ws: sum(float(((w - s) ** 2).sum()) for w, s in zip(ws, start))  # noqa: E731
    assert dist(out["weights"]) < dist(plain)  # FedProx stays closer to the global model


def test_fedprox_with_mu_zero_is_fedavg():
    x, y = make_data()
    a = local_train(fresh(), x, y, epochs=1, batch_size=16, lr=0.1, seed=1)["weights"]
    b = local_train(fresh(), x, y, epochs=1, batch_size=16, lr=0.1, seed=1, algorithm="fedprox", mu=0.0)["weights"]
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


def test_scaffold_control_variate_update():
    x, y = make_data(64)
    lr = 0.1
    start = get_weights(fresh())
    zeros = [np.zeros_like(w) for w in start]
    out = local_train(fresh(start), x, y, epochs=1, batch_size=64, lr=lr, seed=0,
                      algorithm="scaffold", c_global=zeros, c_local=zeros)
    grad = full_batch_gradient(start, x, y)
    # one step with zero control variates: c_new = (w0 - w1) / lr = the gradient
    for c_new, g in zip(out["c_new"], grad):
        assert np.allclose(c_new, g, atol=1e-5)
    # a non-zero correction shifts the step by -lr * (c_global - c_local)
    c_g = [np.full_like(w, 0.3) for w in start]
    shifted = local_train(fresh(start), x, y, epochs=1, batch_size=64, lr=lr, seed=0,
                          algorithm="scaffold", c_global=c_g, c_local=zeros)["weights"]
    for s, w in zip(shifted, out["weights"]):
        assert np.allclose(s, w - lr * 0.3, atol=1e-6)


def _state(algorithm, **kw):
    w = [np.zeros((2, 3), np.float32), np.zeros(2, np.float32)]
    return ServerState(w, {"algorithm": algorithm, **kw}, num_clients=4)


def test_server_fedavg_is_example_weighted_mean():
    st = _state("fedavg")
    a = [np.ones((2, 3), np.float32), np.ones(2, np.float32)]
    b = [np.full((2, 3), 4.0, np.float32), np.full(2, 4.0, np.float32)]
    st.apply([a, b], [1, 2])
    assert np.allclose(st.weights[0], 3.0) and np.allclose(st.weights[1], 3.0)


def test_server_matches_flower_builtin_weighted_aggregation():
    from flwr.app import ArrayRecord, MetricRecord, RecordDict
    from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

    rng = np.random.default_rng(0)
    updates = [[rng.normal(size=(2, 5)).astype(np.float32), rng.normal(size=2).astype(np.float32)] for _ in range(4)]
    sizes = [10, 250, 31, 7]
    records = [
        RecordDict({"arrays": ArrayRecord(numpy_ndarrays=u), "metrics": MetricRecord({"num-examples": n})})
        for u, n in zip(updates, sizes)
    ]
    reference = aggregate_arrayrecords(records, "num-examples").to_numpy_ndarrays()
    for mine, ref in zip(weighted_average(updates, sizes), reference):
        assert np.allclose(mine, ref, atol=1e-6)


def test_server_no_reports_leaves_model_unchanged():
    st = _state("fedavg")
    before = [w.copy() for w in st.weights]
    st.apply([], [])
    assert all(np.array_equal(a, b) for a, b in zip(before, st.weights)) and st.round == 1


def test_server_fedadam_first_step():
    st = _state("fedadam", server_lr=0.1, adam_beta1=0.9, adam_beta2=0.99, adam_tau=1e-3)
    d = 2.0
    st.apply([[np.full((2, 3), d, np.float32), np.full(2, d, np.float32)]], [5])
    m, v = 0.1 * d, 0.01 * d * d
    assert np.allclose(st.weights[0], 0.1 * m / (np.sqrt(v) + 1e-3), atol=1e-6)


def test_server_scaffold_updates_global_control_variate():
    st = _state("scaffold", server_lr=1.0)
    w = [np.ones((2, 3), np.float32), np.ones(2, np.float32)]
    dc = [np.full((2, 3), 2.0, np.float32), np.full(2, 2.0, np.float32)]
    st.apply([w, w], [1, 99], [dc, dc])
    assert np.allclose(st.weights[0], 1.0)          # unweighted mean of client models
    assert np.allclose(st.c_global[0], 2 / 4 * 2.0)  # |S|/N * mean(delta c)


def test_evaluate_and_parameter_counts():
    x, y = make_data(120)
    model = fresh()
    out = evaluate(model, x, y, C)
    assert out["n"] == 120 and abs(out["loss"] - np.log(2)) < 1e-6
    assert np.asarray(out["confusion_matrix"]).sum() == 120
    assert parameter_counts(model)["trainable_parameters"] == C * D + C
