"""Federated algorithms of the benchmark behind one interface.

Each algorithm is a client-side rule (what ``local_train`` does) plus a
server-side rule (how the round's client results become the next global
model). Update rules were checked against the original papers; the equations
and the verification notes are in ``docs/ALGORITHM_VERIFICATION.md``.

Aggregation is streaming: a round's client models are folded into float64
accumulators one at a time, in increasing client-id order, so a round with
thousands of participating clients never holds more than one client model in
memory. For FedAvg and SCAFFOLD the arithmetic (operation order and dtypes) is
identical to ``src.strategies.aggregation.ServerState``, so the benchmark
engine reproduces the Phase 0 Flower path bit for bit (tested).

Interface used by the engine, per round:

    algo.begin_round(x, plan)        plan = [(client_id, n_examples, planned_steps), ...]
    algo.client_kwargs(cid)          extra keyword arguments for local_train
    algo.add_client(cid, n, out)     fold one client's local_train output in
    x = algo.finish_round(x)         the next global model
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.benchmark.state_store import ClientStateStore

Weights = list  # list[np.ndarray], float32

ALGORITHMS = ("fedavg", "fedprox", "fednova", "scaffold", "fedadam", "fedadagrad", "fedyogi")


class Algorithm:
    name = "base"
    local_rule = "fedavg"          # algorithm argument passed to local_train
    downlink_model_copies = 1      # model-sized arrays sent server -> client
    uplink_model_copies = 1        # model-sized arrays sent client -> server
    hyperparameters: tuple = ()    # fl.* keys that only this algorithm uses

    def __init__(self, fl: dict, weights: Weights, num_clients: int, scratch_dir: Path | None = None):
        self.fl = fl
        self.server_lr = float(fl.get("server_lr", 1.0))
        self.num_clients = num_clients
        self.shapes = [w.shape for w in weights]
        self.round = 0
        self._acc = None
        self._total = 0.0
        self._count = 0
        self.last_update_norm = None

    # ---- per round -----------------------------------------------------------
    def begin_round(self, x: Weights, plan: list[tuple[int, int, int]]) -> None:
        self.round += 1
        self._x = x
        self._plan = {cid: (n, steps) for cid, n, steps in plan}
        self._total = float(sum(n for _, n, _ in plan))
        self._count = len(plan)
        self._acc = [np.zeros(s, dtype=np.float64) for s in self.shapes]

    def client_kwargs(self, cid: int) -> dict:
        return {}

    def add_client(self, cid: int, n: int, out: dict) -> None:
        # Same expression and order as src.strategies.aggregation.weighted_average.
        for acc, w in zip(self._acc, out["weights"]):
            acc += w.astype(np.float64) * (n / self._total)

    def finish_round(self, x: Weights) -> Weights:
        if not self._count:
            return x  # every selected client dropped out: model unchanged
        mean = [a.astype(np.float32) for a in self._acc]
        delta = [m.astype(np.float64) - w.astype(np.float64) for m, w in zip(mean, x)]
        new = self.server_step(x, delta)
        self._acc = None
        return new

    def server_step(self, x: Weights, delta: list[np.ndarray]) -> Weights:
        self.last_update_norm = float(np.sqrt(sum(float((d * d).sum()) for d in delta)))
        return [(w + self.server_lr * d).astype(np.float32) for w, d in zip(x, delta)]

    def summary(self) -> dict:
        return {"update_norm": self.last_update_norm}

    def close(self) -> None:
        pass


class FedAvg(Algorithm):
    name = "fedavg"


class FedProx(Algorithm):
    """Li et al. 2020: proximal term (mu/2)||w - x||^2 in the local objective."""
    name = "fedprox"
    local_rule = "fedprox"
    hyperparameters = ("mu",)

    def client_kwargs(self, cid: int) -> dict:
        return {"mu": float(self.fl["mu"])}


class FedNova(Algorithm):
    """Wang et al. 2020, eq. (6) with vanilla SGD: normalized averaging.

    x <- x + server_lr * tau_eff * sum_i p_i (y_i - x) / tau_i,
    p_i = n_i / n, tau_eff = sum_i p_i tau_i, tau_i = SGD steps actually taken.
    """
    name = "fednova"

    def begin_round(self, x, plan):
        super().begin_round(x, plan)
        self._tau_eff = 0.0
        self._x64 = [w.astype(np.float64) for w in x]

    def add_client(self, cid, n, out):
        tau = max(int(out["steps"]), 1)
        p = n / self._total
        self._tau_eff += p * tau
        for acc, y, x in zip(self._acc, out["weights"], self._x64):
            acc += (y.astype(np.float64) - x) * (p / tau)

    def finish_round(self, x):
        if not self._count:
            return x
        delta = [self._tau_eff * a for a in self._acc]
        self._acc, self._x64 = None, None
        return self.server_step(x, delta)

    def summary(self):
        return {**super().summary(), "tau_eff": getattr(self, "_tau_eff", None)}


class Scaffold(Algorithm):
    """Karimireddy et al. 2020, Algorithm 1, option II.

    Unweighted mean of client models; c <- c + (|S| / N) mean(c_i+ - c_i).
    Client control variates c_i persist between participations in a
    disk-backed store (dense, float32), so thousands of clients do not occupy RAM.
    """
    name = "scaffold"
    local_rule = "scaffold"
    downlink_model_copies = 2   # x and c
    uplink_model_copies = 2     # Delta y and Delta c

    def __init__(self, fl, weights, num_clients, scratch_dir=None):
        super().__init__(fl, weights, num_clients, scratch_dir)
        self.c_global = [np.zeros_like(w) for w in weights]
        if scratch_dir is None:
            raise ValueError("SCAFFOLD needs a scratch directory for the client control variates")
        self.store = ClientStateStore(Path(scratch_dir) / "scaffold_c_local.f32", num_clients, self.shapes)

    def begin_round(self, x, plan):
        # SCAFFOLD averages uniformly over the participating clients.
        super().begin_round(x, [(cid, 1, steps) for cid, _, steps in plan])
        self._c_acc = [np.zeros(s, dtype=np.float64) for s in self.shapes]

    def client_kwargs(self, cid):
        return {"c_global": self.c_global, "c_local": self.store.get(cid)}

    def add_client(self, cid, n, out):
        super().add_client(cid, 1, out)
        self.store.put(cid, out["c_new"])
        # Same arithmetic as simple_average(c_deltas) in ServerState.
        for acc, d in zip(self._c_acc, out["c_delta"]):
            acc += d.astype(np.float64) * (1 / self._total)

    def finish_round(self, x):
        count = self._count
        new = super().finish_round(x)
        if count:
            mean_c = [a.astype(np.float32) for a in self._c_acc]
            scale = count / self.num_clients
            self.c_global = [(c + scale * d).astype(np.float32) for c, d in zip(self.c_global, mean_c)]
        self._c_acc = None
        return new

    def summary(self):
        return {**super().summary(),
                "c_global_norm": float(np.sqrt(sum(float((c.astype(np.float64) ** 2).sum()) for c in self.c_global))),
                "clients_with_state": self.store.num_written}

    def close(self):
        self.store.close(delete=True)


class FedOpt(Algorithm):
    """Reddi et al. 2021, Algorithm 5 (example-weighted client updates).

    Delta = sum_i (n_i / n)(y_i - x); m = b1 m + (1 - b1) Delta; v by variant;
    x <- x + eta * m / (sqrt(v) + tau); v starts at tau^2; no bias correction.
    """
    hyperparameters = ("server_lr",)
    variant = ""
    default_betas = (0.9, 0.99)

    def __init__(self, fl, weights, num_clients, scratch_dir=None):
        super().__init__(fl, weights, num_clients, scratch_dir)
        b1, b2 = self.default_betas
        self.beta1 = float(fl.get(f"{self.name}_beta1", b1))
        self.beta2 = float(fl.get(f"{self.name}_beta2", b2))
        self.tau = float(fl.get("opt_tau", 1e-3))
        self.m = [np.zeros(s, dtype=np.float64) for s in self.shapes]
        self.v = [np.full(s, self.tau ** 2, dtype=np.float64) for s in self.shapes]

    def finish_round(self, x):
        if not self._count:
            return x
        delta = [a - w.astype(np.float64) for a, w in zip(self._acc, x)]
        self._acc = None
        self.last_update_norm = float(np.sqrt(sum(float((d * d).sum()) for d in delta)))
        new = []
        for i, (w, d) in enumerate(zip(x, delta)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * d
            d2 = d * d
            if self.variant == "adagrad":
                self.v[i] = self.v[i] + d2
            elif self.variant == "yogi":
                self.v[i] = self.v[i] - (1 - self.beta2) * d2 * np.sign(self.v[i] - d2)
            else:
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * d2
            new.append((w + self.server_lr * self.m[i] / (np.sqrt(self.v[i]) + self.tau)).astype(np.float32))
        return new


class FedAdam(FedOpt):
    name, variant = "fedadam", "adam"


class FedYogi(FedOpt):
    name, variant = "fedyogi", "yogi"


class FedAdagrad(FedOpt):
    name, variant = "fedadagrad", "adagrad"
    default_betas = (0.0, 0.0)   # Reddi et al. App. D: Adagrad without momentum


REGISTRY = {cls.name: cls for cls in (FedAvg, FedProx, FedNova, Scaffold, FedAdam, FedYogi, FedAdagrad)}


def make_algorithm(fl: dict, weights: Weights, num_clients: int, scratch_dir: Path | None = None) -> Algorithm:
    name = fl["algorithm"]
    if name not in REGISTRY:
        raise ValueError(f"unknown benchmark algorithm {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](fl, weights, num_clients, scratch_dir)


def register(cls) -> type:
    """Add an algorithm class (used by the proposed method and its ablations)."""
    REGISTRY[cls.name] = cls
    return cls
