"""EXPLORATORY / POST-SCREENING algorithms (protocol docs/EXPLORATORY_CCE_PROTOCOL.md).

Not part of the frozen fedbench_v1 benchmark. They are registered under new
names and are used only by configs/exploratory/cce_v1.yaml.

Notation (our sign convention): x = global model, y_i = client model after
local training, Delta_i = y_i - x, p_i = n_i / sum of n over reporting clients,
K_i = local SGD steps client i actually took, eta_l = client learning rate of
the round, lam = weight decay.

* FedExP (Jhunjhunwala et al., ICLR 2023, Algorithm 1), size-weighted:
      eta_g = max(1, sum_i p_i ||Delta_i||^2 / (2 (||Delta_bar||^2 + eps)))
      x <- x + eta_g Delta_bar,  Delta_bar = sum_i p_i Delta_i
  The paper averages uniformly and states the extension to size-weighted
  objectives; every algorithm of this benchmark uses p_i, so FedExP does too.
  As in the paper, the reported model is the average of the last two iterates.

* Coverage-normalized averaging (CAV; component averaging, Censor, Gordon &
  Gordon 2001, as used for partially held parameters in FL) and its
  magnitude-dispersion variant CCE-v2 share one decomposition:
      omega_i  = expm1(K_i log1p(-eta_l lam))     (weight-decay factor; exact for
                                                 zero-gradient coordinates under
                                                 plain SGD, momentum 0)
      Delta'_ij = Delta_ij - omega_i x_j          only on coordinates whose feature is
                                                 in client i's training rows
                                                 (exact support); 0 elsewhere
      Omega     = sum_i p_i omega_i               (weight decay at the FedAvg rate)
      M_j = sum_i p_i Delta'_ij,  A_j = sum_i p_i |Delta'_ij|,  S_j = sum_i p_i Delta'_ij^2
      c_j = sum_{i covers j} p_i                  (bias: always covered)
  CAV:     x_j <- x_j + Omega x_j + M_j / c_j          (c_j = 0: M_j = 0, no data step)
  CCE-v2:  x_j <- x_j + Omega x_j + eta_j M_j,  eta_j = S_j / A_j^2  (= r_j / c_j)
           with r_j = E[D'^2] / (E|D'|)^2 >= 1 over covering clients; A_j = 0 -> eta_j = 1.
  CCE-v2 is therefore CAV times r_j. |eta_j M_j| <= max_i |Delta'_ij| always.
  CCE-unit: the same pipeline with eta_j = 1, which equals FedAvg up to float
  rounding; it checks the weight-decay decomposition end to end.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.benchmark.algorithms import Algorithm, register

DIAGNOSTIC_ROUNDS = (1, 10, 25, 50)


# ---------------------------------------------------------------------------
@register
class FedExP(Algorithm):
    name = "fedexp"
    hyperparameters = ("fedexp_eps",)
    evaluates_average = True

    def __init__(self, fl, weights, num_clients, scratch_dir=None):
        super().__init__(fl, weights, num_clients, scratch_dir)
        self.eps = float(fl["fedexp_eps"])
        self._prev = None
        self.last_eta = None

    def begin_round(self, x, plan):
        super().begin_round(x, plan)
        self._x64 = [w.astype(np.float64) for w in x]
        self._sq = 0.0

    def add_client(self, cid, n, out):
        p = n / self._total
        sq = 0.0
        for acc, y, x in zip(self._acc, out["weights"], self._x64):
            d = y.astype(np.float64) - x
            acc += d * p                      # accumulate Delta_bar directly
            sq += float((d * d).sum())
        self._sq += p * sq

    def finish_round(self, x):
        self._prev = [w.copy() for w in x]
        if not self._count:
            return x
        delta = self._acc
        norm2 = float(sum(float((d * d).sum()) for d in delta))
        self.last_eta = max(1.0, self._sq / (2.0 * (norm2 + self.eps)))
        self.last_update_norm = float(np.sqrt(norm2))
        new = [(w.astype(np.float64) + self.last_eta * d).astype(np.float32) for w, d in zip(x, delta)]
        self._acc, self._x64 = None, None
        return new

    def evaluation_weights(self, x):
        """Average of the last two iterates (FedExP, Section 5)."""
        if self._prev is None:
            return x
        return [((a.astype(np.float64) + b.astype(np.float64)) / 2).astype(np.float32) for a, b in zip(x, self._prev)]

    def summary(self):
        return {"update_norm": self.last_update_norm, "fedexp_eta_g": self.last_eta}


# ---------------------------------------------------------------------------
class _CoverageBase(Algorithm):
    """Shared decomposition for CAV, CCE-v2 and CCE-unit (see module docstring)."""
    needs_support = True
    mode = "cav"

    def __init__(self, fl, weights, num_clients, scratch_dir=None):
        super().__init__(fl, weights, num_clients, scratch_dir)
        if float(fl.get("momentum", 0.0)) != 0.0:
            raise ValueError(f"{self.name} requires plain SGD (momentum 0): the weight-decay removal is exact only then")
        if len(self.shapes) != 2 or len(self.shapes[0]) != 2 or self.shapes[1] != (self.shapes[0][0],):
            raise ValueError(f"{self.name} expects [W (classes x features), b (classes)]")
        if float(fl.get("local_epochs", 1)) != 1.0 or float(fl.get("straggler_prob", 0.0)) > 0:
            raise ValueError(f"{self.name}: the support is the client's full training set, which needs one full local epoch")
        self.lam = float(fl.get("weight_decay", 0.0))
        self.num_features = self.shapes[0][1]
        self.uplink_extra_bytes = (self.num_features + 7) // 8   # feature-support bitmask per client
        self.diagnostics_dir: Path | None = None
        self._stats: dict = {}

    def round_lr(self) -> float:
        return float(self.fl["lr"]) * float(self.fl.get("lr_decay", 1.0)) ** (self.round - 1)

    def begin_round(self, x, plan):
        super().begin_round(x, plan)
        self._x64 = [w.astype(np.float64) for w in x]
        self._M = [np.zeros(s) for s in self.shapes]
        self._A = [np.zeros(s) for s in self.shapes]
        self._S = [np.zeros(s) for s in self.shapes]
        self._cov = np.zeros(self.num_features)          # weighted coverage per feature
        self._cov_delta = np.zeros(self.num_features)    # diagnostic: coverage read from the deltas alone
        self._omega = 0.0
        self._fedavg = [np.zeros(s) for s in self.shapes]  # plain weighted mean of Delta (for the unit check)

    def omega(self, steps: int) -> float:
        return float(np.expm1(steps * np.log1p(-self.round_lr() * self.lam)))

    def add_client(self, cid, n, out):
        if "support" not in out:
            raise KeyError(f"{self.name} needs the client's feature support (engine hook)")
        p = n / self._total
        steps = int(out["steps"])
        om = self.omega(steps)
        self._omega += p * om
        sup = np.unique(np.asarray(out["support"], dtype=np.int64))
        self._cov[sup] += p
        Wy, by = out["weights"]
        Wx, bx = self._x64
        # Data part only on the client's support columns (exactly 0 elsewhere): the update is
        # computed column-sparse; the result is identical to masking a full-width difference.
        dWp = (Wy[:, sup].astype(np.float64) - Wx[:, sup]) - om * Wx[:, sup]
        dbp = (by.astype(np.float64) - bx) - om * bx      # the bias is covered by every example
        for acc, d in ((self._M[0], dWp), (self._A[0], np.abs(dWp)), (self._S[0], dWp * dWp)):
            acc[:, sup] += p * d
        self._M[1] += p * dbp
        self._A[1] += p * np.abs(dbp)
        self._S[1] += p * dbp * dbp
        full_width = self.round in DIAGNOSTIC_ROUNDS or self.name == "cce_unit"
        if full_width:
            dW = Wy.astype(np.float64) - Wx
            if self.name == "cce_unit":
                self._fedavg[0] += p * dW
                self._fedavg[1] += p * (by.astype(np.float64) - bx)
            if self.round in DIAGNOSTIC_ROUNDS:
                # diagnostic only: would the deltas alone reveal the support? (relative residue bound)
                tol = 4 * max(steps, 1) * 2.0 ** -24 * np.abs(Wx).max(axis=0) + 1e-12
                self._cov_delta += p * (np.abs(dW - om * Wx).max(axis=0) > tol)

    def multipliers(self) -> list[np.ndarray]:
        raise NotImplementedError

    def finish_round(self, x):
        if not self._count:
            return x
        mult = self.multipliers()
        step = [self._omega * xx + m * M for xx, m, M in zip(self._x64, mult, self._M)]
        self.last_update_norm = float(np.sqrt(sum(float((s * s).sum()) for s in step)))
        self._diagnostics(mult, step)
        new = [(xx + s).astype(np.float32) for xx, s in zip(self._x64, step)]
        self._x64 = self._M = self._A = self._S = self._fedavg = None
        return new

    # ---- diagnostics -------------------------------------------------------
    def _ratio_r(self):
        """r_j = S_j c_j / A_j^2 on covered coordinates (magnitude dispersion, >= 1)."""
        rW = np.full(self.shapes[0], np.nan)
        cov = np.broadcast_to(self._cov, self.shapes[0])
        ok = (self._A[0] > 0) & (cov > 0)
        rW[ok] = self._S[0][ok] * cov[ok] / self._A[0][ok] ** 2
        rb = np.where(self._A[1] > 0, self._S[1] / np.maximum(self._A[1], 1e-300) ** 2, np.nan)
        return rW, rb

    def _diagnostics(self, mult, step):
        covered = self._cov > 0
        rW, rb = self._ratio_r()
        rv = rW[:, covered].ravel()
        rv = rv[np.isfinite(rv)]
        agree = np.isclose(self._cov_delta, self._cov, rtol=1e-9, atol=1e-12)
        self._stats = {
            "features_covered": int(covered.sum()),
            "coverage_median": float(np.median(self._cov[covered])) if covered.any() else None,
            "inv_coverage_p90": float(np.percentile(1 / self._cov[covered], 90)) if covered.any() else None,
            "r_median": float(np.median(rv)) if len(rv) else None,
            "r_p90": float(np.percentile(rv, 90)) if len(rv) else None,
            "r_share_gt2": float(np.mean(rv > 2)) if len(rv) else None,
            "r_bias_max": float(np.nanmax(rb)) if np.isfinite(rb).any() else None,
            "multiplier_median_covered": float(np.median(np.broadcast_to(mult[0], self.shapes[0])[:, covered])) if covered.any() else None,
            "delta_coverage_agreement": float(np.mean(agree)) if self.round in DIAGNOSTIC_ROUNDS else None,
            "omega_bar": self._omega,
        }
        if self.diagnostics_dir is not None and self.round in DIAGNOSTIC_ROUNDS:
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.diagnostics_dir / f"round{self.round:03d}.npz",
                                coverage=self._cov.astype(np.float32), coverage_from_deltas=self._cov_delta.astype(np.float32),
                                r_W=rW.astype(np.float32), r_b=rb.astype(np.float32),
                                multiplier_W=np.broadcast_to(mult[0], self.shapes[0]).astype(np.float32),
                                step_W=step[0].astype(np.float32), step_b=step[1].astype(np.float32))
            (self.diagnostics_dir / f"round{self.round:03d}.json").write_text(json.dumps(self._stats, indent=1), encoding="utf-8")

    def summary(self):
        return {"update_norm": self.last_update_norm, **self._stats}


@register
class CAV(_CoverageBase):
    """Coverage-normalized averaging: covered features take the covering clients' mean."""
    name = "cav"

    def multipliers(self):
        inv = np.where(self._cov > 0, 1.0 / np.maximum(self._cov, 1e-300), 0.0)
        return [np.broadcast_to(inv, self.shapes[0]), np.ones(self.shapes[1])]


@register
class CCEv2(_CoverageBase):
    """CCE-v2: eta_j = S_j / A_j^2 (= r_j / c_j); pure form, applied to every coordinate incl. the bias."""
    name = "cce"

    def multipliers(self):
        out = []
        for A, S in zip(self._A, self._S):
            eta = np.ones_like(A)
            ok = A > 0
            eta[ok] = S[ok] / A[ok] ** 2
            out.append(eta)
        return out


@register
class CCEUnit(_CoverageBase):
    """Same decomposition with eta_j = 1: must equal FedAvg up to float rounding."""
    name = "cce_unit"

    def multipliers(self):
        return [np.ones(s) for s in self.shapes]

    def finish_round(self, x):
        if self._count:
            # x + Omega x + M on every coordinate == x + weighted mean Delta (exact arithmetic)
            self._unit_gap = float(max(np.abs(self._omega * xx + M - f).max() for xx, M, f in zip(self._x64, self._M, self._fedavg)))
        return super().finish_round(x)

    def summary(self):
        return {**super().summary(), "max_abs_gap_to_fedavg_step": getattr(self, "_unit_gap", None)}
