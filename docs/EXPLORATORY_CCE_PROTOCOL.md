# EXPLORATORY / POST-SCREENING protocol: explore_cce_v1

Status: **pre-registration**, written 2026-10-09 before any exploratory run.
This study is **not part of the frozen fedbench_v1 benchmark**. It does not
change any official config, code path used by the official runs, or result.
It makes **no novelty claim and no accuracy promise**: its purpose is to test a
mechanism. All runs use seed 7 and seen-client **validation** data only; no
client-test, unseen-client or test data is evaluated (`eval.test: false`).

Branch `explore/cce-v1` (from `aa8c273`). Config `configs/exploratory/cce_v1.yaml`,
results under `results/exploratory/cce_v1/`, experiment name `explore_cce_v1_*`.

## 1. Motivation (from the completed screening; see results/fedbench_v1_screening)

On Amazon natural product clients, FedAvg/FedProx/FedNova/SCAFFOLD stay near the
majority class after 50 rounds (word weights 1-5% of centralized norm), while
FedAdam/FedAdagrad/FedYogi reach centralized quality. A median client holds 41
training reviews and covers 3.4% of the 50,000 features. Two explanations are
open: (a) a **global step-size** shortfall (FedAvg's server step is fixed at 1,
adaptive methods had 5 tuned server step sizes - a confound of the screening);
(b) **coverage dilution**: averaging a feature's update over all participating
clients, most of which never saw that feature.

## 2. Algorithms (exact definitions: `src/benchmark/exploratory_algorithms.py`)

Notation: x global model; y_i client model; Delta_i = y_i - x; p_i = n_i / sum n
(reporting clients); K_i local SGD steps taken; eta_l client lr; lam weight decay.

* **FedAvg + server lr** eta_g in {1, 3, 10, 30, 100}: x <- x + eta_g * sum_i p_i Delta_i.
  eta_g = 1 is the frozen FedAvg and serves as a **bit-identity sentinel**.
* **FedExP** (Jhunjhunwala et al., ICLR 2023, Alg. 1), size-weighted:
  eta_g = max(1, sum_i p_i ||Delta_i||^2 / (2(||Delta_bar||^2 + eps))), eps on the
  paper's grid {1e-3, 10^-2.5, 1e-2, 10^-1.5, 1e-1}; reported model = average of
  the last two iterates (as in the paper). The only deviation from the paper is
  the size weighting, which every algorithm of this benchmark uses.
* **Shared decomposition (CAV, CCE-v2, CCE-unit)**: omega_i = expm1(K_i log1p(-eta_l lam));
  Delta'_ij = Delta_ij - omega_i x_j on coordinates of features present in client
  i's training rows (exact support; bias always covered), 0 elsewhere;
  Omega = sum_i p_i omega_i; M_j = sum p_i Delta'_ij; A_j = sum p_i |Delta'_ij|;
  S_j = sum p_i Delta'_ij^2; c_j = sum of p_i over clients covering j.
  * **CAV** (coverage-normalized averaging; prior art: component averaging,
    Censor, Gordon & Gordon 2001; HeteroFL-style partial-parameter averaging):
    x_j <- x_j + Omega x_j + M_j / c_j.
  * **CCE-v2**: x_j <- x_j + Omega x_j + eta_j M_j, eta_j = S_j / A_j^2 (eta_j = 1 if A_j = 0),
    applied to every coordinate including the bias (pure form, no exemption).
  * **CCE-unit**: eta_j = 1 (equals FedAvg up to float rounding; decomposition check).
* **FedYogi** server lr 3.0: fairness check beyond the frozen grid edge (1.0).

### Mathematical status of CCE-v2 (established before any run)

Over the covering clients of coordinate j, with mu1 = E|Delta'|, mu2 = E[Delta'^2]:
A_j = c_j mu1, S_j = c_j mu2, so **eta_j = r_j / c_j with r_j = mu2 / mu1^2 >= 1**.

* eta_j equals inverse coverage **only if** all covering clients have equal |Delta'|
  (r_j = 1); sign conflicts of equal magnitude are not amplified.
* Otherwise CCE-v2 = **CAV x r_j**: an additional extrapolation by the dispersion of
  update magnitudes, including on fully covered coordinates and the bias, where it
  is pure magnitude amplification. The step is bounded: |eta_j M_j| <= max_i |Delta'_ij|.
* Coverage therefore justifies CAV; it does not justify r_j. The pilot tests
  whether r_j adds anything.
* Weight-decay removal is exact for zero-gradient coordinates under plain SGD with
  momentum 0 (asserted); on covered coordinates the error is second order
  (~ eta_l lam K |Delta|). Absent coordinates are defined by the exact support,
  never by a threshold: the float32 residue of analytic weight-decay removal was
  measured at up to 1.7e-5 |x| for K = 300 (never exactly 0), comparable to small
  genuine updates, so a delta-based coverage test is unreliable. Coverage read
  from the deltas is logged as a diagnostic only.
* Communication: CAV and CCE-v2 need a 50,000-bit support mask per client (+6.25 KB
  uplink, counted); under secure aggregation CCE-v2 would also need sum p|Delta'| and
  sum p Delta'^2 per coordinate.

## 3. Matrix (28 runs, seed 7, 50 rounds, frozen client lr 0.3 / wd 1e-6, E = 1, batch 32)

| Method | amazon_vg (C = 0.10) | yelp_dir01 (C = 0.25) |
|---|---|---|
| FedAvg eta_g in {1, 3, 10, 30, 100} | 5 | 5 |
| FedExP, 5 eps values | 5 | 5 |
| CAV | 1 | 1 |
| CCE-v2 | 1 | 1 |
| CCE-unit | 1 | 1 |
| FedYogi server lr 3.0 | 1 | 1 |

Reused read-only from the official tuning archive (seed 7, commit 3415496):

| Run | sha256 of final weights | Validation macro-F1 |
|---|---|---|
| amazon_vg FedAvg | d83ab0edceb35ff6aad68fb49659602ca171dae01c2fd3fb109b6a21bc88e149 | 0.175341 |
| yelp_dir01 FedAvg | cc0126c555f46f4d4ab1b2244e5110f0c5beab75c2bf4ad2f5a33bdfb0caf25e | 0.748031 |
| amazon_vg FedYogi lr 1.0 | ea275be812a5c84a7141d7f13a247b2aae9071729ea77cec9a84444c00d70fb4 | 0.575171 |
| yelp_dir01 FedYogi lr 0.1 | a07a2adc6e2d147ee1da22b0b0f037900bc2f9d1164312d21c65b60657695c36 | 0.940801 |

Estimated runtime (from the official seed-7 runs): amazon ~6 h, yelp_dir01 ~2 h,
total ~8 h (one Kaggle CPU session).

## 4. Metrics

Primary: final seen-client validation macro-F1 V (FedExP: two-iterate average;
last iterate also logged). Secondary: validation curve, rounds to 0.95 x
centralized validation macro-F1 (amazon 0.552, yelp_dir01 0.897; centralized
seed-42 reference), per-class F1, predicted-class shares, std of V over the last
10 rounds. Mechanism diagnostics (CAV/CCE, every round; full vectors at rounds
1, 10, 25, 50): coverage distribution, r_j distribution (median, P90, share > 2),
bias r, CCE multipliers, delta-coverage agreement; final ||W||, cosine to
centralized (seed-42 weights; same data, different batch order), coefficient
ratio |w| / |w_central| by quintile of |w_central| (FedAdam showed x4.3 in the
smallest quintile); FedExP eta_g per round.

## 5. Pre-registered decision rules (seed 7; amazon_vg primary)

Definitions: FedYogi* = max(official FedYogi, FedYogi lr 3.0);
C1* = max over FedAvg eta_g in {3, 10, 30, 100} and the five FedExP runs.

Integrity (stop and fix code before reading anything else):
* S1: FedAvg eta_g = 1 reproduces the official sha256 above in both regimes.
* S2: |V(CCE-unit) - V(FedAvg eta_g = 1)| <= 0.002 in both regimes.

Mechanism questions (amazon_vg):
* H1, "a global step suffices": C1* >= 0.55.
* H2, "coverage dilution": V(CAV) - C1* >= 0.03.
* H3, "magnitude dispersion r adds value": V(CCE) - V(CAV) >= +0.01;
  |V(CCE) - V(CAV)| < 0.01 means r adds nothing; <= -0.01 means r is harmful.
* Competitive: max(V(CAV), V(CCE)) >= FedYogi* - 0.005.
* Stable: not diverged and last-10-round std of V <= 0.02.

Specificity (yelp_dir01): coverage is high there, so CAV and CCE are predicted to
stay near FedAvg; a gain >= +0.10 over FedAvg (0.748) would contradict the
coverage explanation for that regime and requires reinterpretation, not a claim.

Outcome:
* **Abandon the method line** if H1 holds, or H2 fails, or max(V(CAV), V(CCE)) <
  FedYogi* - 0.02, or the candidate is unstable. Report the C1 result as the finding.
* **Proceed to a separately pre-registered 3-seed study** (all regimes, held-out
  evaluation) only if S1, S2 pass, H1 fails, H2 holds, and the candidate is
  competitive and stable. If H3 also holds, both CAV and CCE-v2 go forward;
  otherwise only CAV goes forward, framed as a mechanism study of prior art.
* Anything else is **inconclusive**; no further run without the author's decision.

Single-seed results are exploratory; no statistical claim is made from them.
