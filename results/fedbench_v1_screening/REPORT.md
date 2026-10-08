# fedbench_v1 screening: results analysis

Analysis of the completed, frozen 81-run screening stage. No experiment was
run, re-run or tuned for this report. Every number below is computed from the
raw per-run files (`final.json`, `history.json`, `clients.json`,
`unseen_clients.json`, `final_weights.npz`) of the two final Kaggle archives
(`fedbench_results_fedbench-job4-yelp.zip`, `fedbench_results_fedbench-job4-amazon (1).zip`),
merged with the tuning archive for the selections.

Labels used throughout: **[M]** directly measured; **[I]** interpretation;
**[H]** hypothesis that needs a further experiment.

## 0. Provenance and integrity

* 81/81 screening runs complete; experiment commit 3415496 for every run;
  every run official (Kaggle, clean tree). Seeds 42, 123, 456 only.
* Re-verified locally with `run_benchmark.py verify` (same code as on Kaggle):
  pilot 15, tune_base 9, tune_algorithms 66, screening 81, no problems; one
  bundle and vocabulary per regime; one partition per seed (Yelp) / per regime
  (Amazon).
* The two archives were merged member by member: 0 conflicting files (657
  shared files byte-identical).
* The tables here were built from the raw files independently of the analysis
  module and cross-checked against its `runs.csv`: 0 mismatches.
* The pre-registered screening rules, recomputed locally, give
  **no_candidate** for both regimes, matching Kaggle. The preliminary
  "candidate" seen after 18/27 Amazon runs is not used anywhere.

Files: `tables/` (per-run, per-regime summaries, paired contrasts, descriptor
correlations), `figures/` (Figures 1-6, made by `make_figures.py`),
`pipeline/` (the pre-registered pipeline outputs).

## 1-3. Per-regime results (mean ± SD over seeds 42/123/456)

Held-out = official Yelp test set; for Amazon, all records of the 5,423
**unseen** product clients. Per-client statistics use clients with >= 5
evaluated records (pre-registered). Rank = mean rank by held-out macro-F1
within each seed [min-max]. Communication = total float32 payload over 50 rounds.

### Yelp IID (100 clients, C = 0.25)

| Method | Held-out macro-F1 | Seen: mean client F1 | Seen: P10 client F1 | Seen: worst client F1 | Comm. (GB) | Rank |
|---|---|---|---|---|---|---|
| Centralized | 0.9478 ± 0.0000 | 0.9468 ± 0.0001 | 0.9343 ± 0.0016 | 0.9160 ± 0.0067 | 0 | 4.0 [4-4] |
| Local-only | – | 0.8452 ± 0.0042 | 0.8181 ± 0.0037 | 0.7470 ± 0.0577 | 0 | – |
| FedAvg | 0.9019 ± 0.0003 | 0.9040 ± 0.0002 | 0.8869 ± 0.0007 | 0.8659 ± 0.0037 | 1.00 | 5.7 [5-6] |
| FedProx | 0.9013 ± 0.0002 | 0.9037 ± 0.0002 | 0.8870 ± 0.0005 | 0.8659 ± 0.0037 | 1.00 | 8.0 [8-8] |
| FedNova | 0.9019 ± 0.0003 | 0.9040 ± 0.0002 | 0.8869 ± 0.0007 | 0.8659 ± 0.0037 | 1.00 | 5.3 [5-5] |
| SCAFFOLD | 0.9016 ± 0.0002 | 0.9040 ± 0.0001 | 0.8862 ± 0.0003 | 0.8665 ± 0.0037 | 2.00 | 7.0 [7-7] |
| FedAdam | 0.9537 ± 0.0005 | 0.9518 ± 0.0003 | 0.9405 ± 0.0007 | 0.9281 ± 0.0048 | 1.00 | 2.7 [2-3] |
| FedAdagrad | 0.9540 ± 0.0002 | 0.9530 ± 0.0003 | 0.9416 ± 0.0004 | 0.9259 ± 0.0012 | 1.00 | 2.3 [2-3] |
| **FedYogi** | **0.9547 ± 0.0002** | 0.9531 ± 0.0003 | 0.9413 ± 0.0007 | 0.9292 ± 0.0067 | 1.00 | **1.0 [1-1]** |

### Yelp Dirichlet alpha = 0.1 (100 equal-size clients, C = 0.25)

| Method | Held-out macro-F1 | Seen: mean client F1 | Seen: P10 client F1 | Seen: worst client F1 | Comm. (GB) | Rank |
|---|---|---|---|---|---|---|
| Centralized | 0.9478 ± 0.0000 | 0.8752 ± 0.0066 | 0.5693 ± 0.0282 | 0.5016 ± 0.0140 | 0 | 1.7 [1-2] |
| Local-only | – | 0.7880 ± 0.0198 | 0.4784 ± 0.0080 | 0.4312 ± 0.0044 | 0 | – |
| FedAvg | 0.6054 ± 0.2329 | 0.6253 ± 0.1653 | 0.3007 ± 0.2514 | 0.2292 ± 0.2030 | 1.00 | 7.3 [7-8] |
| FedProx | 0.6045 ± 0.2309 | 0.6245 ± 0.1639 | 0.3019 ± 0.2542 | 0.2290 ± 0.1998 | 1.00 | 7.3 [6-8] |
| FedNova | 0.6061 ± 0.2333 | 0.6257 ± 0.1656 | 0.3007 ± 0.2514 | 0.2292 ± 0.2030 | 1.00 | 6.3 [6-7] |
| SCAFFOLD | 0.8858 ± 0.0019 | 0.8245 ± 0.0063 | 0.5170 ± 0.0117 | 0.4718 ± 0.0094 | 2.00 | 5.0 [5-5] |
| FedAdam | 0.9428 ± 0.0030 | 0.8722 ± 0.0064 | 0.5730 ± 0.0246 | 0.4973 ± 0.0097 | 1.00 | 3.0 [3-3] |
| **FedAdagrad** | **0.9497 ± 0.0043** | 0.8803 ± 0.0043 | 0.6096 ± 0.0075 | 0.5045 ± 0.0093 | 1.00 | **1.3 [1-2]** |
| FedYogi | 0.9424 ± 0.0028 | 0.8720 ± 0.0061 | 0.5723 ± 0.0279 | 0.4973 ± 0.0097 | 1.00 | 4.0 [4-4] |

Seen-client P10/worst values are low for every method, centralized included,
because many clients hold almost a single class and macro-F1 is then scored on
a near-single-class client-test split. [I]

### Amazon Video Games, natural product clients (21,694 seen / 5,423 unseen, C = 0.10)

| Method | Unseen pooled macro-F1 | Unseen acc. | Seen pooled macro-F1 | Seen−unseen gap (pooled F1) | Unseen: mean client F1 | Unseen: P10 client F1 | Unseen: worst client F1 | Comm. (GB) | Rank |
|---|---|---|---|---|---|---|---|---|---|
| **Centralized** | **0.5795 ± 0.0031** | 0.7886 ± 0.0003 | 0.5835 ± 0.0035 | 0.0041 ± 0.0007 | 0.5477 ± 0.0023 | 0.3973 ± 0.0014 | 0.1785 ± 0.0045 | 0 | **1.0 [1-1]** |
| Local-only | – | – | 0.2221 ± 0.0004 | – | – | – | – | 0 | – |
| FedAvg | 0.1765 ± 0.0027 | 0.6192 ± 0.0009 | 0.1799 ± 0.0027 | 0.0034 ± 0.0000 | 0.1822 ± 0.0021 | 0.1146 ± 0.0011 | 0.0063 ± 0.0110 | 216.9 | 6.0 [6-6] |
| FedProx | 0.1751 ± 0.0024 | 0.6187 ± 0.0008 | 0.1785 ± 0.0023 | 0.0034 ± 0.0000 | 0.1811 ± 0.0019 | 0.1141 ± 0.0011 | 0.0063 ± 0.0110 | 216.9 | 7.0 [7-7] |
| FedNova | 0.1816 ± 0.0014 | 0.6209 ± 0.0005 | 0.1854 ± 0.0015 | 0.0038 ± 0.0001 | 0.1866 ± 0.0012 | 0.1167 ± 0.0007 | 0.0127 ± 0.0110 | 216.9 | 5.0 [5-5] |
| SCAFFOLD | 0.1517 ± 0.0000 | 0.6110 ± 0.0000 | 0.1531 ± 0.0000 | 0.0014 ± 0.0000 | 0.1625 ± 0.0000 | 0.1059 ± 0.0000 | 0.0000 ± 0.0000 | 433.8 | 8.0 [8-8] |
| FedAdam | 0.5671 ± 0.0042 | 0.7807 ± 0.0009 | 0.5726 ± 0.0041 | 0.0055 ± 0.0004 | 0.5350 ± 0.0029 | 0.3814 ± 0.0021 | 0.1597 ± 0.0160 | 216.9 | 3.0 [3-3] |
| FedAdagrad | 0.5639 ± 0.0067 | 0.7801 ± 0.0015 | 0.5673 ± 0.0068 | 0.0034 ± 0.0003 | 0.5311 ± 0.0060 | 0.3801 ± 0.0031 | 0.1564 ± 0.0216 | 216.9 | 4.0 [4-4] |
| FedYogi | 0.5746 ± 0.0038 | 0.7815 ± 0.0010 | 0.5793 ± 0.0044 | 0.0047 ± 0.0006 | 0.5425 ± 0.0044 | 0.3904 ± 0.0047 | 0.1597 ± 0.0160 | 216.9 | 2.0 [2-2] |

Majority-class-only macro-F1 on the unseen set is about 0.152 (61.1% of unseen
reviews are 5-star). [M]

## 4. Concise rankings (held-out macro-F1, mean over seeds)

| Rank | Yelp IID | Yelp Dir-0.1 | Amazon (unseen) |
|---|---|---|---|
| 1 | FedYogi 0.955 | FedAdagrad 0.950 | Centralized 0.580 |
| 2 | FedAdagrad 0.954 | Centralized 0.948 | FedYogi 0.575 |
| 3 | FedAdam 0.954 | FedAdam 0.943 | FedAdam 0.567 |
| 4 | Centralized 0.948 | FedYogi 0.942 | FedAdagrad 0.564 |
| 5 | FedNova / FedAvg 0.902 | SCAFFOLD 0.886 | FedNova 0.182 |
| 6 | SCAFFOLD 0.902 | FedNova 0.606 | FedAvg 0.177 |
| 7 | FedProx 0.901 | FedAvg 0.605 | FedProx 0.175 |
| 8 | – | FedProx 0.605 | SCAFFOLD 0.152 |

Seed-paired contrasts (`tables/paired_contrasts.csv`; t-interval with df = 2;
with 3 seeds the exact Wilcoxon p cannot go below 0.25, so sign consistency
and intervals are reported instead of significance) [M]:

| Contrast | Yelp IID | Yelp Dir-0.1 | Amazon |
|---|---|---|---|
| Best adaptive − FedAvg | +0.053 [+0.052, +0.054], 3/3 | +0.344 [−0.23, +0.91], 3/3 | +0.398 [+0.385, +0.412], 3/3 |
| FedYogi − centralized | +0.0069 [+0.0063, +0.0075], 3/3 | −0.0054 [−0.0125, +0.0017], 3/3 | −0.0049 [−0.0089, −0.0009], 3/3 |
| SCAFFOLD − FedAvg | −0.0003, 3/3 | +0.280 [−0.29, +0.85], 3/3 | −0.025 [−0.031, −0.018], 3/3 |
| FedNova − FedAvg | 0.0000 | +0.0008 | +0.0051 [+0.0019, +0.0083], 3/3 |
| FedProx − FedAvg | −0.0006, 3/3 | −0.0009, mixed | −0.0014, 3/3 |

The Dir-0.1 intervals are wide because FedAvg itself varies between seeds (section 5).

## 5. Convergence (Figure 2, Figure 4)

| Rounds to reach 0.95 x centralized validation F1 (post-hoc target; per seed) | Yelp IID | Yelp Dir-0.1 | Amazon |
|---|---|---|---|
| FedAvg / FedProx / FedNova | 41-42 | never | never |
| SCAFFOLD | 42 | never (plateau 0.886) | never (constant) |
| FedAdam | 3-4 | 13-20 | 13 |
| FedAdagrad | 5-7 | 4-6 | 20-24 |
| FedYogi | 5 | 13-20 | 13 |

* [M] The pre-registered target (0.95 x tuned FedAvg validation F1) is
  uninformative for Amazon: tuned FedAvg is near the majority class, so the
  target (0.167) is met by FedAvg itself around round 40. The post-hoc target
  above is labelled post-hoc and used for description only.
* [M] Yelp IID: adaptive methods reach the target about 10x faster in rounds,
  hence about 10x less communication to the same validation quality.
* [M] Yelp Dir-0.1: FedAvg, FedProx and FedNova oscillate from round to round
  (standard deviation of validation F1 over the last 10 rounds 0.153-0.156,
  vs 0.007 SCAFFOLD and 0.002-0.005 adaptive). The final model's predicted
  positive share is 19%, 64% and 1% in the three seeds (test set balanced).
  The large seed variance of FedAvg in this regime is therefore last-round
  instability, not a stable lower level.
* [M] FedAdagrad dips sharply in the first rounds on Yelp IID (0.50 at round 2)
  before converging; final performance is unaffected.

## 6. Client-level robustness

* [M] Amazon, seen vs unseen (pooled, the clean comparison): the seen−unseen
  macro-F1 gap is 0.001-0.006 for every method, **including centralized
  (0.0041)**. There is no meaningful participation gap for product clients,
  and the small gap that exists is not specific to federated training.
* [I] The per-client mean "gap" stored by the engine (seen client-test mean
  minus unseen mean, e.g. 0.18 for FedAvg) is not a valid gap measure: seen
  clients are scored on ~10% of their reviews (often 2-5), unseen clients on
  all of theirs (>= 20). A size-binned comparison
  (`tables/seen_unseen_size_bins.csv`) shows seen > unseen by 0.01-0.025 in every bin and method,
  centralized included, but bins on evaluated records compare large seen
  products with small unseen products, so it is not size-matched either.
* [M] Amazon unseen tail: FedYogi's P10 client F1 is 0.390 vs 0.397
  centralized (−0.007, 3/3 seeds); FedAdam and FedAdagrad −0.016/−0.017.
  FedAvg-family P10 is 0.106-0.117 (near the majority-class level).
* [M] Yelp Dir-0.1: adaptive methods match or exceed centralized on the seen
  P10 client (FedAdagrad 0.610 vs 0.569); SCAFFOLD 0.517; FedAvg family
  0.30 ± 0.25.
* [M] Client descriptors (pre-registered: same sign in 3/3 seeds and
  |mean rho| >= 0.2; Figure 6): for FedAvg and FedNova, per-client F1 is
  strongly associated with label composition (unseen: label entropy −0.75,
  dominant-class share +0.72, mean rating +0.71) and weakly with text length.
  For centralized, FedAdam, FedAdagrad and FedYogi, **no descriptor reaches the
  threshold** (largest |rho| = 0.15, lexical centroid distance). OOV rate and
  rare-feature mass: |rho| <= 0.09 for every well-optimized model.
* [I] The FedAvg/FedNova associations are what a near-majority-class model
  produces (it is right only on mostly-5-star clients); they describe the
  under-trained model, not client heterogeneity.

## 7. Consistently strong vs regime-specific

* [M] Consistently strong: FedYogi, FedAdagrad, FedAdam: top-4 in every regime,
  within 0.005-0.016 of centralized on Amazon and at or above centralized on
  Yelp. Ranking among the three is regime-specific and differences are small
  (<= 0.011).
* [M] Regime-specific: SCAFFOLD is strongly better than FedAvg under synthetic
  label skew (+0.28) and indistinguishable on IID, but worst on Amazon
  (constant majority-class predictor in all 3 seeds).
* [M] FedProx and FedNova never differ from FedAvg by more than 0.006 in any
  regime. FedNova equals FedAvg on Yelp (clients of equal size take equal
  local steps, the case in which FedNova reduces to FedAvg) and is +0.005
  better on Amazon (heterogeneous client sizes). [I]
* [M] Local-only is far below every reasonably trained federated or
  centralized model on its own clients (Amazon seen mean client F1 0.263
  vs 0.611 centralized).

## 8. The Amazon behaviour seen during tuning

The tuning observation is confirmed on held-out data in all three seeds.

* [M] FedAvg, FedProx, FedNova predict 5 stars for about 99% of unseen
  reviews; SCAFFOLD for 100% (Figure 3). Their unseen macro-F1 (0.152-0.182)
  is at or just above the majority-class level (0.152).
* [M] Each participating client takes on average 4.18 local SGD steps per round
  (2,169 clients per round).
* [M] Their output biases match centralized ones (seed 42: 5-star bias 1.64
  FedAvg vs 1.67 centralized), but the word-weight matrix is about 5% of the
  centralized norm (Frobenius norm, mean over seeds: FedAvg 8.7, FedNova 9.1,
  SCAFFOLD 1.6, centralized 166). Adaptive methods reach the centralized scale
  (FedAdam 203, FedYogi 200). The FedAvg-family models are **under-trained and
  bias-dominated**, not converged to a different solution.
* [I] This is the behaviour reported by Reddi et al. (ICLR 2021) for
  bag-of-words logistic regression on Stack Overflow (Recall@5: FedAdagrad
  66.8, FedAdam 65.2, FedYogi 66.5 vs FedAvgM 46.5, FedAvg 40.6), with their
  stated mechanism: words absent from a client give near-zero client updates,
  so rare-word coordinates move little under averaging, while adaptive
  accumulators let them take large steps. Our setting reproduces it with
  natural product clients and 5-class sentiment.
* [H] Two compatible explanations are not separated by this benchmark: (a) a
  step-budget effect (few local steps per small client, fixed server step
  1.0); (b) per-coordinate dilution of sparse-feature updates by averaging.
  Testing them needs the experiments in section 11.
* [H] SCAFFOLD's complete collapse on Amazon (word weights about 1% of
  centralized) is unexplained. Plausible factors are stale client control
  variates under 10% participation of 21,694 clients and its unweighted model
  averaging over clients of very different sizes; not tested.

## 9. Does the completed screening support a defensible finding?

Yes, but not a research gap. The pre-registered rules found no candidate
failure mode: no federated-specific shortfall that persists across seeds, is
larger under heterogeneity than in its control, and is left unsolved by every
existing method. Under the frozen protocol, the adaptive server optimizers
close the gap to centralized training in all three regimes. What the
screening does support is a set of benchmark findings (section 12).

Outcome classification (task brief, section 20): **C, no convincing gap
found.**

## 10. What is measured, what is interpreted, what is open

* Measured: all tables, convergence, the predicted-class shares, weight norms,
  local steps per round, pooled seen/unseen gaps, descriptor correlations.
* Interpreted: under-training of the FedAvg family on Amazon; the
  non-specificity of the participation gap; that descriptor associations for
  FedAvg/FedNova reflect the majority-class behaviour; that centralized here
  is limited by plain SGD for 13/10 epochs (adaptive federated methods beat it
  on Yelp IID).
* Open (needs experiments): step budget vs feature dilution; whether a tuned
  server learning rate without adaptivity closes the Amazon gap; SCAFFOLD's
  collapse; whether any of this holds with more seeds, other categories or
  non-linear models.

## 11. Possible post-screening experiments (exploratory; not run)

All of these would be **exploratory / post-screening**, kept outside the
frozen 81-run benchmark, with their own config and CHANGELOG entry.

1. **Server-step control (highest value).** FedAvg with a tuned server learning
   rate (server SGD, eta in {1, 3, 10, 30}; optionally server momentum,
   FedAvgM) on Amazon and Yelp Dir-0.1, validation-only tuning, same seeds.
   It separates "adaptivity" from "a larger server step", the main confound
   of the screening comparison (adaptive methods got 5 tuned server step sizes,
   FedAvg none). Cost: about 4 grid runs per regime plus 3 seeds x 2 regimes.
2. **Feature-dilution test (no new training).** With the Amazon bundle's
   per-feature document frequencies and the existing final weights: does
   |W_FedAvg| / |W_centralized| fall with the fraction of clients containing a
   feature, and do adaptive methods remove that dependence? Tests the
   Reddi et al. mechanism directly; needs the bundle (Kaggle).
3. **Budget sensitivity.** Amazon with local epochs E in {1, 3} (already listed
   in the protocol's Phase B) or more rounds for the FedAvg family, to see
   whether its shortfall is transient.

## 12. Strongest defensible findings

1. **Optimizer, not heterogeneity, dominates.** Under a fixed 50-round budget,
   the adaptive server optimizers (FedAdam/FedAdagrad/FedYogi) beat
   FedAvg/FedProx/FedNova/SCAFFOLD in all three regimes, by +0.05 (Yelp IID),
   +0.34 (Yelp Dir-0.1) and +0.39 macro-F1 (Amazon unseen clients), same
   sign in 3/3 seeds, and reach centralized quality (within −0.005 to +0.007).
2. **Natural product clients: the FedAvg family stays near the majority-class
   solution.** 99-100% 5-star predictions, word weights 1-5% of centralized,
   about 4 local steps per client per round. This reproduces the Reddi et al.
   sparse-text result with natural product clients and 5-class sentiment.
3. **No unseen-client (participation) gap specific to federated training.**
   Pooled seen−unseen gap 0.001-0.006 for every method, centralized included.
4. **Measured client heterogeneity does not predict which clients do badly**
   once the model is well optimized (no descriptor |rho| >= 0.2 for
   centralized or adaptive methods); lexical-shift descriptors in particular
   are uninformative (|rho| <= 0.15). Consistent with the 0A negative result.
5. **Synthetic label skew mainly causes instability.** Under Dirichlet 0.1,
   FedAvg-family final models depend on the seed (test F1 0.35-0.81), while
   SCAFFOLD and adaptive methods are stable (spread <= 0.01).

## 13. Limitations

1. **Three seeds and one budget.** Screening has 3 seeds (no test can reach
   p < 0.25) and a single round budget (50) with last-round evaluation; the
   FedAvg-family results, especially on Amazon and Dir-0.1, are
   budget- and evaluation-point-dependent.
2. **Unequal tuning and grid edges.** Adaptive methods had 5 tuned server step
   sizes, FedAvg/FedNova/SCAFFOLD none; 9 selections lie on a grid edge,
   including the Amazon client lr. "Centralized" is plain SGD for 13/10 epochs
   and is beaten by adaptive federated methods on Yelp IID, so it is not an
   upper bound.
3. **Scope.** One natural-client category (Video Games, product clients), one
   model family (TF-IDF + linear), and seen-client per-client metrics on
   small evaluation splits.

## 14. Proposed Results section structure

1. Experimental setup and provenance: regimes, frozen protocol, tuning,
   seeds, verification (section 0).
2. Overall comparison: Figure 1 and the three regime tables; concise ranking.
3. Optimization dynamics: Figure 2 (convergence, rounds and bytes to target);
   Figure 4 (instability under label skew).
4. Natural-client case study: Figure 3 (predicted-class shares, weight norms,
   local steps), the link to Reddi et al., and the participation-gap analysis.
5. Client-level robustness: unseen-client distribution (Figure 5), tail
   metrics, descriptor analysis (Figure 6).
6. Negative results: no candidate failure mode under the pre-registered
   rules; no FL-specific participation gap; no lexical-shift association.
7. Limitations and threats to validity.
