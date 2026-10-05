# Benchmark protocol fedbench_v1 (pre-registered)

Written 2026-10-05, before any benchmark run on any data. Changes after this
date are recorded in `CHANGELOG.md` with whether results had been seen.
Experiment 0A is complete (negative/diagnostic: the raw coefficient/exposure
association did not survive the predefined control). Nothing in this
benchmark presupposes the 0A hypothesis.

## 1. Regimes (frozen scientific settings inherited unchanged)

| Regime | Source config | Clients | Participation | Rounds / E / batch | lr, wd |
|---|---|---|---|---|---|
| `yelp_iid` | `exp0a_tuned.yaml`, cell `iid` | 100, IID | C = 0.25 | 50 / 1 / 32 | 0.3, 1e-6 (0A tuning, frozen) |
| `yelp_dir01` | `exp0a_tuned.yaml`, cell `label_skew` | 100, `dirichlet_client/v1`, alpha 0.1 | C = 0.25 | 50 / 1 / 32 | 0.3, 1e-6 |
| `amazon_vg` | `exp0b_frozen.yaml`, cell `natural` | Video_Games `parent_asin`, min 20 reviews; 80% seen / 20% unseen clients | C = 0.10 | 50 / 1 / 32 | tuned (stage `tune_base`, frozen 0B rule) |

Features: TF-IDF, 50,000 features, (1,2)-grams, min_df 5, sublinear tf, fitted
on seen-client TRAIN rows only. Model: multinomial logistic regression, zero
init, plain SGD. Centralized: 13 epochs (Yelp, 0A rule) and 10 epochs (Amazon,
`exp0b_frozen.yaml`). Local-only: 10 epochs (base config).

## 2. Execution engine (not a scientific change)

Sequential in-process simulation (`src/benchmark/engine.py`) on CPU. Same
sampling/dropout streams, local trainer and aggregation arithmetic as the
Phase 0 Flower path; FedAvg and SCAFFOLD final weights are bit-identical to
Flower on the test fixture. Reason: the Flower runtime keeps one node and one
message per natural client and cannot run tens of thousands of clients within
Kaggle memory. Communication = float32 payload of the exchanged arrays.

## 3. Algorithms and tuning

See `docs/ALGORITHM_VERIFICATION.md`. Client lr and wd are shared by all
algorithms. Algorithm-specific grids (seed 7, seen-client validation macro-F1
after round 50, ties to the smaller value): FedProx mu in {1e-3, 1e-2, 1e-1,
1}; FedAdam/FedAdagrad/FedYogi server lr in {0.01, 0.03, 0.1, 0.3, 1}, tau =
1e-3, betas from Reddi et al. A selection on a grid edge is recorded and not
silently extended. Held-out data (client-test, unseen clients, Yelp test) is
never evaluated in a tuning run; selection code refuses runs that contain it.

## 4. Stages

1. `pilot` - seed 0, 3 rounds: time, memory, artifacts only. No metric is used.
2. `tune_base` - Amazon lr x wd grid (frozen 0B rule).
3. `tune_algorithms` - algorithm grids, every regime.
4. `screening` (Phase A) - seeds 42, 123, 456 (the first three official seeds,
   fixed now), 3 regimes x 9 methods.
5. Phase B (focused) and Phase C (8-seed confirmation) are specified in a
   separate committed config **after** screening and **before** they run,
   following the rules in section 6.

## 5. Metrics and units of analysis

Pooled: accuracy, macro-F1, per-class precision/recall/F1 on seen-client
validation, seen-client client-test and the held-out set (Yelp official test;
Amazon unseen clients). Per client: macro-F1 over the classes present, on the
client-test split (seen) and on all records (unseen); distribution summaries
(mean, median, std, P10, P25, P75, P90, worst, best, IQR, spread) over clients
with at least 5 evaluated records (`MIN_EVAL_N`, fixed now; all clients are
still written to the per-client table). Seen-unseen gap = seen minus unseen,
pooled and mean-client. Convergence: validation curve every round; rounds and
bytes to a target = 0.95 x tuned FedAvg validation macro-F1 of the regime.
Efficiency: wall clock, time per round, bytes per round and total.

The replication unit is the seed. Client-level values of one run are
summarized within the run and never used as independent replicates.

## 6. Screening decision rules (fixed before any result)

Notation: for regime r, metric M, method a, seed s: M(r, a, s). "Shortfall"
of a = M(r, centralized, s) - M(r, a, s) (higher-is-better metrics).
Thresholds: Delta = 0.02 for pooled and mean-client metrics, 0.05 for tail
metrics (P10 and worst client).

A **candidate failure mode** is a (regime, metric) cell such that

1. *Material*: the best of the seven FL methods (by mean over screening
   seeds) has mean shortfall >= Delta;
2. *Consistent*: that best method's shortfall is positive in every screening
   seed (3/3);
3. *Heterogeneity-specific*: the best-method shortfall exceeds the shortfall
   of the matching control by >= Delta. Controls: `yelp_dir01` vs `yelp_iid`
   (same metric); Amazon unseen-client metric vs the same metric on seen
   clients;
4. *Unsolved*: no existing FL method closes it (no method's mean shortfall is
   below Delta / 2).

Additional failure patterns checked for every FL method (reported, not used
to select by themselves): average-up/tail-down vs FedAvg (pooled F1 higher
while P10 lower, same direction in 3/3 seeds); seen-only improvement (seen
mean-client F1 improves vs FedAvg but unseen does not); poor-solution
convergence (validation std over last 10 rounds < 0.005 while shortfall >=
Delta); divergence.

Candidates are ranked by the heterogeneity-specific excess (criterion 3).
The top candidate goes to Phase B only if criteria 1-4 hold; otherwise the
screening outcome is "no candidate" and the report says so.

**Client-level characterization** (Amazon; descriptors from
`src/benchmark/client_features.py`, computed without any model): per run,
Spearman rho between each descriptor and per-client macro-F1 (clients with
>= 5 records), separately for seen and unseen clients. A descriptor is
"associated with failure" if rho has the same sign in every seed and
|mean rho| >= 0.2. Because the tail of a model's own scores regresses to the
mean, tail membership for algorithm comparisons is defined model-free (by
descriptor quantiles or by the centralized model), never by the score of the
method being compared.

## 7. Phase B and C (rules)

Phase B tests whether the candidate persists across more than one setting,
with the screening seeds: participation C in {0.05, 0.10, 0.20}, local epochs
E in {1, 3}, client dropout p = 0.1, for FedAvg, the best existing method for
that metric and centralized. Phase C: all 8 official seeds in the main
setting, all 9 methods; paired Wilcoxon (exact), Holm within regime x metric,
mean difference with 95% t-interval, Cohen's d_z, rank-biserial.

## 8. Failed runs

A run counts only with a `COMPLETE` marker after artifact validation. A
failed or killed run is recorded with its return code, classification and
log, never substituted or interpolated, and the stage stops at a suspected OOM
until it is investigated.
