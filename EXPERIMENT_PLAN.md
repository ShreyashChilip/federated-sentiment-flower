# Experiment plan

Protocol version 1, frozen 2026-10-02 **before any official result exists**.
Any later change to this protocol is recorded in `CHANGELOG.md` with the date,
the reason and whether results had already been seen.

## 1. Rules

1. Official results come only from the Kaggle GPU environment. Laptop runs are
   for debugging and are stored under `results/_smoke*` (git-ignored).
2. The official test split is used for final reporting only. Hyperparameters,
   round budgets, targets and model selection use the validation role.
3. Reported final metric = metric of the **last round** on the test split. No
   best-round selection on test data.
4. Every comparison is paired: same seeds, same saved partitions, same features.
5. Negative and null results are reported with the same prominence as positive ones.
6. No number is typed by hand into a table, figure or manuscript section.

## 2. Fixed design choices

| Choice | Value | Rationale |
|---|---|---|
| Train/test | Official Yelp Polarity split (560,000 / 38,000) | Fixes the 598k inconsistency at the source |
| Roles inside official train | 80% train / 10% val / 10% client-test, stratified, fixed seed, independent of the partition | One feature space and one centralized baseline for every alpha, N and seed |
| Features (LR) | TF-IDF, unigrams + bigrams, fitted on train-role rows only; vocabulary and IDF are sums of per-client counts | Globally consistent coordinates; computable federatedly; no test leakage |
| Non-IID recipe | Per class, proportions ~ Dir(alpha * 1_N); redraw until every client has >= 10 samples | Standard recipe (Hsu et al.), stated exactly |
| Seeds | 42, 123, 456, 789, 2026 | A seed changes partition, client sampling, dropout, batch order |
| Local optimizer | Mini-batch SGD | Matches what the text of the paper says |
| Client sampling | Uniform without replacement, max(1, round(C*K)) per round, seeded per round | Explicit and reproducible |
| Dropout model | A selected client fails after receiving the model: downlink is charged, no update returns | Stated policy (reviewer 1.6) |
| Straggler model | A selected client is slow with probability q and completes a fraction of its local epochs; policy `partial` aggregates it, `drop` discards it | The setting FedProx was designed for |
| Per-client metric | Global model on each client's own client-test rows; macro-F1 over the classes that client holds | Fairness without touching the global test set |
| Communication | Sum of serialized Flower record bytes, per direction, over clients that actually exchanged a message; no compression; transport framing excluded | Reviewer 1.9 |
| Energy | Proxy = client CPU seconds x declared CPU watts (+ GPU seconds x declared GPU watts) | Labelled as a proxy everywhere |

## 3. Phases and gates

Each phase ends with `results/<experiment_id>/summary.md` and an entry in
`EXPERIMENT_LOG.md`. A phase starts only when the previous gate is met.

| Phase | Content | Gate to proceed | Status |
|---|---|---|---|
| 0 Infrastructure | Data, roles, features, partitions, LR, Flower client/server, FedAvg/FedProx/FedAdam/SCAFFOLD, logging, tests, smoke test | Tests pass; same seed gives identical weights | **Done on laptop** (see EXPERIMENT_LOG) |
| 0K Kaggle bring-up | Clean install on Kaggle, smoke test, record environment, pin dataset revision | Smoke test passes on Kaggle; `ENVIRONMENT.md` filled | Next |
| T Tuning | Validation-only search (section 4) | Search results saved; chosen values written to configs | Not started |
| 1 Baseline | Yelp, LR: centralized, local-only, FedAvg; N = 10, alpha = 0.1, 3 seeds | Centralized LR in the expected range for TF-IDF + LR on Yelp Polarity; FedAvg between local-only and centralized, or the deviation explained | Not started |
| 2 Core benchmark | Yelp, LR: FedAvg, FedProx, FedAdam, SCAFFOLD x alpha in {0.1, 0.5, 1.0, 10.0} + IID, 5 seeds | All runs complete; statistics table generated | Not started |
| 3 Sensitivity | One factor at a time around the Phase 2 default: C in {0.25, 0.5, 1.0}; E in {1, 3, 5}; N in {10, 50, 100}; mu in {0, 0.01, 0.1, 1.0}; gamma in {1.0, 0.99, 0.95}; dropout p in {0, 0.10}; stragglers; quantity skew; 2% label noise | - | Not started |
| 4 Models | LR vs TextCNN vs DistilBERT + LoRA; alpha in {0.1, 0.5, 10.0}; FedAvg, FedProx | TextCNN and LoRA implemented and tested | Not started |
| 5 Amazon 5-class | Repeat the core benchmark; macro-F1 primary | **Dataset decision (section 6)** | Blocked on decision |
| 6 Natural clients | Real client metadata; held-out clients | **Natural-client definition (section 6)** | Blocked on decision |
| 7 Novelty audit + diagnostics | Experiment 0A (section 5), 0B, literature audit | - | Audit seeded |
| 8 Method (only if justified) | Candidate method + ablation | Outcome C of 0A, or a natural-client effect in 0B, and no prior art | Conditional |
| 9 Privacy | DP (epsilon, delta) sweep; secure aggregation | - | Not started |
| 10 Final | Tables 1-16, Figures 1-18, statistics, manuscript | All required rows of REQUIREMENTS_MATRIX at `Run` | Not started |

## 4. Tuning protocol (same budget for every algorithm)

* Setting: Yelp, LR, N = 10, alpha = 0.5, C = 1.0, E = 1, seed 42 only (a
  tuning seed; final runs use all five seeds with the chosen values).
* Selection criterion: validation macro-F1 at the last round.
* Shared grid, 9 configurations per algorithm:
  * local lr in {0.03, 0.1, 0.3} for every algorithm, crossed with one
    algorithm-specific axis of 3 values:
  * FedAvg: lr decay gamma in {1.0, 0.99, 0.95}
  * FedProx: mu in {0.01, 0.1, 1.0} (mu = 0 is FedAvg and is reported as such)
  * FedAdam: server lr in {0.01, 0.03, 0.1}
  * SCAFFOLD: server lr in {0.5, 1.0, 1.5}
* Feature size (`max_features`) and round budget are fixed from the
  centralized validation curve before the federated grid is run.
* The target validation macro-F1 for convergence-to-target is fixed as 97% of
  the centralized validation macro-F1, before federated final runs.
* All tuning runs are saved under `results/tuning/`.

## 5. Experiment 0A protocol (controlled confounding diagnostic), frozen

* Data: Yelp Polarity, LR, N = 100, Dirichlet alpha = 0.1, and an IID control
  with N = 100. Seeds: all five.
* Conditions: (1) centralized LR, (2) FedAvg, (3) FedAvg with local
  label-prior logit correction (clients add log of their local class prior to
  the logits during training, as in logit adjustment / FedLC without margin
  tuning; prior smoothed by +1 per class).
* Quantities, per feature j with document frequency >= 50 in the train role:
  * coefficient bias  B_j = w_fed,j - w_central,j, where w is the
    positive-minus-negative class coefficient;
  * label exposure  X_j = sum_k df_kj * (r_k - r) / sum_k df_kj, where df_kj is
    the number of client k's train documents containing feature j, r_k is
    client k's positive rate and r the global positive rate.
  * To separate client-level exposure from the feature's own polarity, the
    primary statistic is the **partial** correlation of B_j with X_j
    controlling for w_central,j; plain Pearson and Spearman are reported too.
* The same statistics on the IID control give the null reference.
* Decision rule (fixed now): an effect is declared only if the Spearman
  correlation under alpha = 0.1 has the same sign in all five seeds and its
  mean absolute value exceeds the IID control's mean absolute value by at
  least 0.10. Outcomes A/B/C are then read off as defined in the project brief.
* Known caveat recorded before running: under Dirichlet label skew, documents
  are assigned to clients independently of their text given the label, so any
  exposure X_j is induced purely by the feature's association with the label.
  A null result here would not rule out lexical confounding with natural
  clients (Experiment 0B).

## 6. Decisions that need the authors

1. **Second dataset (Phase 5).** The classic "Amazon Review Full" 5-class
   benchmark (Zhang et al., 3.0M train / 650k test) has **no user or product
   ids**, so it cannot give natural clients. Amazon Reviews 2023 (McAuley Lab)
   has `user_id` and `parent_asin` and 5-star ratings but no official
   train/test split and needs a predefined category and sampling rule.
   Options: (a) classic benchmark for Phase 5 and Amazon 2023 for Phase 6;
   (b) Amazon 2023 for both, with a split defined here before any run.
2. **Natural client definition (Phase 6).** Reviewer (`user_id`) or product
   (`parent_asin`). These are scientifically different: user clients vary in
   writing style and rating habits; product clients vary in topic vocabulary.
3. **Target journal.** Decides citation style, length and how much of the
   benchmark goes to supplementary material.
4. **Seeds and the Wilcoxon test.** With 5 seeds the smallest two-sided
   Wilcoxon p-value is 0.0625, so no single 5-seed comparison can reach 0.05.
   Options: (a) 6+ seeds for the head-to-head comparisons (min p = 0.031 at 6,
   0.0078 at 8); (b) keep 5 seeds and pair across seed x alpha blocks; (c) keep
   5 seeds and report the paired t-test and effect sizes alongside Wilcoxon.

## 7. Compute staging

Costs are measured, not guessed: Phase 0K times one full-data LR round on
Kaggle, and each later phase's budget is computed from that measurement and
written here before the phase is launched. The runner skips finished runs, so
a Kaggle session that hits the time limit resumes where it stopped.

## 8. Deviations from the project brief

* `experiments/` currently holds `00_smoke_test.py`, `01_baseline.py` and
  `make_partitions.py`; later numbered scripts are added when their phase's
  gate is met.
* Flower's built-in strategies are not used directly. The server loop is
  written on Flower's Message API so that sampling, dropout and byte accounting
  are explicit; the weighted aggregation is unit-tested against Flower's own
  `aggregate_arrayrecords`.

## 9. Amendments to protocol version 1

Sections 1 to 8 above are kept as frozen on 2026-10-02. Changes are listed
here with their date, origin and whether any official result existed. Open
points are numbered as in `docs/PHASE_0_TO_PHASE_1_AUDIT.md`, section 8.

| # | Date | Amendment | Origin | Official results existed? | Status |
|---|---|---|---|---|---|
| A1 | 2026-10-02 | Experiment 0 (0A controlled diagnostic, 0B natural clients) is run before the tuning phase and before the Phase 1 baseline. Its hyperparameters are therefore not tuned and are marked PROVISIONAL in the configs | Author instruction | No | In force; values await approval (D3) |
| A2 | 2026-10-02 | Experiment 0 uses 8 seeds instead of 5 (decision 6.4, option a). The 0A decision rule reads "all official seeds" in place of "all five seeds" | Author instruction | No | **Seed list not yet supplied** (D1) |
| A3 | 2026-10-02 | Natural-client data is Amazon Reviews 2023 with two separate client definitions, `user_id` and `parent_asin`, analysed separately and never mixed (resolves decisions 6.1 and 6.2 for Phase 6) | Author instruction | No | In force; category pending (D6) |
| A4 | 2026-10-02 | 0A is stated to concern ordinary label heterogeneity only. It is not evidence for or against natural-client lexical confounding. This restates the caveat already in section 5 | Author instruction | No | In force |
| A5 | 2026-10-02 | Pilot runs (seed 0, a few rounds) are allowed before official runs for timing and plumbing. Pilot outputs may not change the protocol | Author instruction | No | In force |
| A6 | 2026-10-02 | Definition of "candidate client-confounded feature" and of the seen/unseen-client measurements fixed in `docs/CONFOUNDING_DIAGNOSTIC.md` | Preparation work, before any real data was loaded | No | In force |

Not amended, although the audit found problems (they need the authors' decision):

* The Dirichlet redraw rule of section 2 cannot be satisfied for N = 100,
  alpha = 0.1 (audit M1, decision D2).
* The 0A decision rule of section 5 uses the raw Spearman correlation, which
  responds to under-training (audit M4, decision D9).
* `weight_decay` is 0 (audit M5, decision D3).
