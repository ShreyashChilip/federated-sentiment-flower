# Phase 0 to Phase 1 audit

Date: 2026-10-02. Audited commit: `512364a` (Phase 0) plus the preparation
work committed with this document. Scope: the implementation compared with
`EXPERIMENT_PLAN.md` (protocol version 1), `REQUIREMENTS_MATRIX.md`,
`docs/EXISTING_MATERIALS_AUDIT.md`, all configs, experiment scripts,
partitioning, evaluation and statistics code.

No scientific result exists. Nothing in this document is a finding about
federated learning; the only numbers are properties of the partitioning code
and of the repository.

## 1. Implemented and tested

| Area | State | Evidence |
|---|---|---|
| Yelp Polarity, official split, fixed train/val/client-test roles | Done | `tests/test_partitioning.py`, smoke test |
| Federated TF-IDF (global vocabulary and IDF from train-role rows only) | Done | `tests/test_features.py` |
| IID, Dirichlet (per class), quantity-skew partitions; saved, hashed, reload-only | Done | `tests/test_partitioning.py` |
| Logistic regression; FedAvg, FedProx, FedAdam, SCAFFOLD on Flower 1.39 | Done | `tests/test_training.py`, `experiments/00_smoke_test.py` (31 checks) |
| Centralized and local-only baselines with the same trainer | Done | smoke test |
| Communication ledger, timing, memory, energy proxy, per-client metrics | Done | `tests/test_metrics_stats.py`, smoke test |
| Mean, std, 95% CI, paired t, exact Wilcoxon, Holm | Done | `tests/test_metrics_stats.py` |
| **New:** FedAvg + local label-prior logit correction (`fedavg_la`) | Done | `tests/test_diagnostics.py`, Experiment 0 smoke |
| **New:** equal-size client-wise Dirichlet partition (`dirichlet_client`), optional | Done | `tests/test_diagnostics.py` |
| **New:** Amazon Reviews 2023 reader, 5 classes, cleaning rules | Done; tested on a synthetic file and run once on a real 16,216-line category file (parser counts only) | `tests/test_natural_clients.py` |
| **New:** natural clients by `user_id` or `parent_asin`; profiling; size rule; held-out client split with overlap assertion | Done | `tests/test_natural_clients.py` |
| **New:** within/across-client association decomposition, null, candidate rule, coefficient bias, seen/unseen report, index-to-term table | Done | `tests/test_diagnostics.py` |
| **New:** results schema, immutable run directories, flat tables | Done | `tests/test_exp0_schema.py` |
| **New:** Experiment 0A and 0B drivers with pilot/full modes and official-run guard | Done | `experiments/00b_exp0_smoke.py` (45 checks) |
| Kaggle bring-up | Reported as passed by the authors | **Not in the repository**: `ENVIRONMENT.md` section 1 is still empty and no `kaggle_environment.json` is committed |

Test count: 73 unit tests (37 existing, 36 new).

Not implemented: TextCNN, DistilBERT + LoRA, FedNova, full FedLC margin,
FedDisco, Fisher/FIPA aggregation, FedGMA, DP, secure aggregation, figures,
tables, manuscript, and the candidate method (deliberately).

## 2. What Experiment 0A needs

| Item | State |
|---|---|
| Config `configs/exp0a.yaml`: Yelp, N = 100, IID control and Dirichlet alpha = 0.1, three conditions | Written |
| Script `experiments/exp0a_controlled_diagnostic.py` (pilot / full / analyze-only, resumable) | Written |
| Same feature space, preprocessing, optimizer settings, test set across conditions | Enforced by `consistency_checks`; analysis refuses to run if a check fails |
| Machine-readable outputs: `runs.csv`, `rounds.csv`, `clients.csv`, `summary.json`, per-seed `diagnostic.json`, `feature_confounding.csv`, `checks.json` | Written |
| The 8 official seeds | **Missing** (section 8, D1) |
| A feasible Dirichlet rule at N = 100, alpha = 0.1 | **Missing** (section 4, M1; section 8, D2) |
| Approved hyperparameters (`protocol.frozen: true`) | **Missing** (section 8, D3) |
| Pinned Yelp dataset revision | **Missing** (section 8, D5) |

## 3. What Experiment 0B needs

| Item | State |
|---|---|
| Configs `exp0b_user.yaml` and `exp0b_product.yaml` (separate bundle, partition, results) | Written |
| Profiling script `experiments/exp0b_profile_clients.py` (no training) | Written |
| Training script `experiments/exp0b_natural_clients.py` | Written |
| Held-out clients: whole clients, zero overlap asserted and recorded | Done |
| Vocabulary fitted on training rows of seen clients only | Done and tested |
| Amazon category | **Missing** (D6) |
| Client-size rule (`min_reviews`, optional caps) | **Missing by design**: chosen from the profile, before training (D7) |
| Seeds, hyperparameters, prior reference | **Missing** (D1, D3, D4) |
| Data licence check | **Missing** (D8) |

## 4. Implementation / protocol mismatches

**M1. The frozen Dirichlet redraw rule cannot be met for 0A.**
The plan says: per class, proportions ~ Dir(alpha), "redraw until every client
has >= 10 samples". Measured with the project's partitioner on 560,000
balanced labels, 200 independent draws each:

| N | alpha | Draws in which every client has >= 10 rows | Mean clients with < 10 rows | Mean empty clients |
|---|---|---|---|---|
| 100 | 0.1 | 0 of 200 | 21.7 | 11.6 |
| 50 | 0.1 | 0 of 200 | 9.5 | 4.9 |
| 10 | 0.1 | 39 of 200 (19.5%) | 1.4 | 0.7 |
| 100 | 0.5 | 175 of 200 (87.5%) | 0.1 | 0.0 |

So Experiment 0A as specified fails at partition creation. Two further
consequences: (a) the Phase 1 baseline (N = 10, alpha = 0.1) only accepts
about one draw in five, which means the rule itself selects less extreme
partitions than Dir(0.1) describes; (b) "100 clients" would in practice be
about 78 usable clients. The config keeps the frozen rule and therefore stops
with an error that points here. Options are in D2. This was not changed
silently.

**M2. Seed count.** The plan and `configs/base.yaml` list five seeds. The
current instruction is eight. No 8-seed list exists in any config, document or
commit, so none was invented. The 0A decision rule says "all five seeds"; the
code evaluates "all official seeds".

**M3. Order of phases.** The plan runs the tuning phase before any
experiment. Experiment 0 now comes first, so its hyperparameters are not
tuned. They are marked PROVISIONAL in the configs and the full run is blocked
until `protocol.frozen` is set (D3).

**M4. The frozen 0A decision rule is sensitive to under-training.** The rule
uses the raw Spearman correlation between coefficient bias and label exposure.
Under label skew, exposure is correlated with a term's polarity. A federated
model that is merely less converged than the centralized one (coefficients
uniformly smaller) then shows a large raw correlation with no client effect.
`tests/test_diagnostics.py::test_bias_statistics_partial_correlation_ignores_overall_scale`
builds this case: raw correlation above 0.3 in magnitude, partial correlation
controlling for the centralized coefficient undefined (nothing left). The plan
already names the partial correlation as the "primary statistic", but the
decision rule is written on the raw one. Both are computed and reported.
Changing the rule needs approval (D9).

**M5. No regularization.** `weight_decay` is 0 everywhere. Unregularized
logistic regression on high-dimensional text has no finite optimum when the
classes are separable, so coefficient magnitudes depend on how long each
model trained. This makes M4 worse and makes "bias relative to centralized"
depend on round and epoch budgets. An L2 term fixed in advance and identical
across conditions would give all three conditions the same well-defined
target (D3).

**M6. Attribution of the partition recipe.** The plan calls the per-class
recipe "Hsu et al.". To the best of my knowledge Hsu et al. (2019) draw a
label distribution per client with equal client sizes, which is what the new
`dirichlet_client` scheme does, and the per-class recipe is the one used in
later benchmarks. Recorded from memory; verify before the paper is written.

**M7. Label-prior correction on imbalanced data.** As frozen, `fedavg_la`
adds log(local prior). The trained model then estimates a class-balanced
posterior. On Yelp (balanced) this matches the centralized model's target. On
Amazon ratings (not balanced) it does not: the corrected federated model and
the uncorrected centralized model would aim at different targets, which would
show up as an accuracy difference unrelated to clients. The alternative
`fl.prior_reference: global` removes only each client's deviation from the
global prior. Implemented, not selected (D4).

**M8. Per-client metrics are final-round only.** The schema asks for
per-client metrics; they are recorded once per run (after the last round),
not per round. Per-round per-client evaluation would add one message per
client per round.

**M9. Flower built-in strategies are not used** (already listed as a
deviation in the plan); `run_simulation` is deprecated in Flower 1.39 and the
version is pinned.

**M10. Test metrics are logged every round** for convergence curves. They are
never used for selection, and the reported value is the last round.

**M11. Held-out clients are fixed by `holdout_seed`, not by the experiment
seed.** Across-seed spread in 0B therefore reflects client sampling and batch
order, not the choice of held-out clients (D10).

## 5. Reproducibility risks

| Risk | Status |
|---|---|
| Kaggle environment not recorded in the repository | Open. Commit `results/kaggle_environment.json` values into `ENVIRONMENT.md` |
| Yelp dataset revision not pinned (`data.revision: null`) | Open; the full run refuses to start without it |
| Amazon revision | Pinned to `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`; the source file's SHA-256 is recorded per bundle |
| Partitions created on Kaggle are not in git until downloaded and committed | Mitigated: hash recorded in every run and verified on load; the runbook includes the commit step |
| Kaggle image drift between sessions | Open; use "pin to original environment" and compare package versions in `runs.csv` |
| GPU nondeterminism | Same-seed identity was checked by the smoke test; it must be re-checked on the GPU actually used (the smoke test runs wherever `device: auto` resolves) |
| A run interrupted by the 12-hour limit | Handled: unfinished directories are renamed `seedK.incomplete-<time>` and the run restarts; finished runs are skipped |
| Results overwritten | Prevented: a directory with `final.json` cannot be written again |
| `--set` overrides on the command line | Each override changes the config hash and is stored in the run metadata; official runs should use committed configs only |
| Uncommitted code on Kaggle | The full run refuses if a tracked file differs from the commit; untracked files (new partitions, results) are counted but allowed |
| `data.max_records` takes the first N lines of a category file, not a random sample | Do not use it for official runs; choose a category that fits in memory instead |
| Memory: every Ray actor loads the full feature matrix | To be measured in the pilot |
| Amazon bundle built from a multi-gigabyte file | Category size must fit Kaggle RAM; see the runbook |

## 6. Files changed by the preparation work

Modified:
`src/clients/trainer.py`, `src/clients/flower_client.py`, `src/clients/baselines.py`,
`src/strategies/server.py`, `src/data/prepare.py`, `src/partitioning/partition.py`,
`src/partitioning/store.py`, `src/runner.py`, `src/utils/runtime.py`, `src/utils/env.py`,
`experiments/make_partitions.py`, `.gitignore`, `EXPERIMENT_PLAN.md`, `CHANGELOG.md`,
`REQUIREMENTS_MATRIX.md`, `README.md`, `REPRODUCIBILITY.md`, `EXPERIMENT_LOG.md`.

Added:
`src/data/amazon2023.py`, `src/data/natural.py`, `src/analysis/diagnostics.py`,
`src/analysis/exp0.py`, `src/analysis/exp0_driver.py`, `src/analysis/profile.py`,
`src/utils/schema.py`, `configs/exp0a.yaml`, `configs/amazon2023.yaml`,
`configs/exp0b_user.yaml`, `configs/exp0b_product.yaml`,
`experiments/exp0a_controlled_diagnostic.py`, `experiments/exp0b_profile_clients.py`,
`experiments/exp0b_natural_clients.py`, `experiments/00b_exp0_smoke.py`,
`tests/synthetic_reviews.py`, `tests/test_natural_clients.py`, `tests/test_diagnostics.py`,
`tests/test_exp0_schema.py`, `docs/PHASE_0_TO_PHASE_1_AUDIT.md`,
`docs/CONFOUNDING_DIAGNOSTIC.md`, `docs/KAGGLE_PHASE1_RUNBOOK.md`.

Moved: `NOVELTY_AUDIT.md` to `docs/NOVELTY_AUDIT.md` (rewritten).
Removed: `configs/natural_clients.yaml` (placeholder, replaced by `amazon2023.yaml`).

Files that still need editing by the authors before the first scientific run:
`configs/base.yaml` (`experiment.seeds`, `data.revision`), `configs/exp0a.yaml`
(`cells.label_skew`, PROVISIONAL values, `protocol.frozen`), `ENVIRONMENT.md`
(Kaggle section), `CHANGELOG.md` (one entry per decision).

## 7. Kaggle commands

The full sequence with explanations is in `docs/KAGGLE_PHASE1_RUNBOOK.md`.
In order:

```
python -m pytest tests -q
python experiments/00_smoke_test.py
python experiments/00b_exp0_smoke.py
python experiments/exp0a_controlled_diagnostic.py --mode pilot --create-partitions
python experiments/exp0a_tune.py                                        # validation-only tuning
python experiments/make_partitions.py --config exp0a_tuned.yaml --cells
python experiments/exp0a_controlled_diagnostic.py --config exp0a_tuned.yaml --mode full   # first scientific run
python experiments/exp0b_profile_clients.py --definition user --set data.category=<Category>
python experiments/exp0b_profile_clients.py --definition product --set data.category=<Category>
python experiments/exp0b_natural_clients.py --definition user --mode pilot --create-partitions
python experiments/exp0b_natural_clients.py --definition user --mode full --create-partitions
python experiments/exp0b_natural_clients.py --definition product --mode pilot --create-partitions
python experiments/exp0b_natural_clients.py --definition product --mode full --create-partitions
```

## 8. Decisions required before the first scientific run

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | The 8 official seeds | Supply the list | Keep the five existing seeds and add three, so earlier planning stays valid. The three values are yours to choose |
| D2 | Dirichlet rule for N = 100, alpha = 0.1 (M1) | (a) keep the per-class recipe, drop the minimum (`min_client_size: 0`); clients with no training rows never participate and the usable client count is reported. (b) switch 0A to `dirichlet_client`: 100 equal-size clients, each with its own Dir(0.1) label mix. (c) change N or alpha | (b). It gives exactly 100 clients, isolates label skew from size skew, and needs no rejection step. (a) mixes label skew with extreme size skew. Whichever is chosen applies to every later experiment that uses alpha = 0.1 |
| D3 | 0A hyperparameters: rounds, C, E, learning rate, weight decay, centralized epochs | Approve the PROVISIONAL values, or run the validation-only tuning phase first | Run the tuning phase for the learning rate and round budget on the validation role, and fix a small non-zero weight decay in advance (M5) |
| D4 | Label-prior correction reference (M7) | `none` (as frozen) or `global` | `none` for 0A (no difference on balanced data); `global` for 0B, or else apply the same correction to the centralized model |
| D5 | Pin the Yelp dataset revision | Record the Hugging Face commit seen on Kaggle | Required |
| D6 | Amazon category for 0B | Any of the 33 category files | A mid-sized category that fits Kaggle memory; fix it before looking at any profile, or state that it was chosen for size only |
| D7 | Client-size rule | `min_reviews`, optional `max_reviews` and `max_clients` | Choose from the profile's retention table, record in the config and CHANGELOG before training |
| D8 | Amazon licence and citation | Read the dataset card | Required before any derived number is published |
| D9 | 0A decision rule (M4) | Keep the raw Spearman rule, or declare an effect only if the partial correlation also satisfies it | Add the partial-correlation condition, as a dated protocol amendment made before any result |
| D10 | Held-out clients in 0B (M11) | One fixed held-out set, or a different set per seed | One fixed set for the first run; state it as a limitation |
| D11 | Is the pilot allowed to inform anything? | - | Timing and memory only. The pilot uses seed 0, which is not an official seed |

## 9. Decisions taken on 2026-10-02

| # | Decision | Where it is implemented |
|---|---|---|
| D1 | Seeds 42, 123, 456, 789, 1001, 2024, 31415, 271828 | `configs/base.yaml`; plan amendment A7 |
| D2 | `dirichlet_client`, 100 equal-size clients, alpha = 0.1; unequal sizes studied separately later | `configs/exp0a.yaml`; plan section 12 |
| D3 | Validation-only tuning of learning rate and weight decay before the official run | `experiments/exp0a_tune.py`; plan section 10 |
| D9 | Raw Spearman alone is insufficient; the partial-correlation criterion is required too | `decision_rule_0a`; plan section 11 |
| D5 | Yelp revision pinned | `configs/base.yaml`. The Kaggle environment report is **still missing** from the repository |

Mismatches M1 (infeasible redraw rule), M2 (seed count), M3 (untuned
hyperparameters), M4 (decision rule) and M5 (no regularization) are closed for
Experiment 0A by these decisions. M1 remains relevant for later experiments
that use the per-class recipe (Phase 1 baseline at N = 10, alpha = 0.1).

Still open: D4 (prior reference for 0B), D6 (Amazon category), D7 (client-size
rule), D8 (Amazon licence), D10 (held-out clients per seed), and the Kaggle
environment record.
