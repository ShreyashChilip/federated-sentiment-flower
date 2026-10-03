# Changelog

Protocol and code changes that can affect results. Each entry states whether
results had been seen when the change was made.

## 2026-10-03 (Experiment 0B memory handling; no official result existed)

* Natural-client preprocessing now releases the raw review rows and duplicate
  tracking set after constructing the cleaned table, drops unused columns
  before feature construction, fits TF-IDF from client-wise sufficient
  statistics, and streams transformed CSR chunks to disk. Runtime loading
  recognizes this streamed CSR format. This preserves the frozen cleaning,
  client filtering, train-role fitting and feature-selection rules; no
  experimental result had been seen. Regression coverage added for loading
  streamed bundles.
* Follow-up scale review found that parsing still materialized the entire
  category in Python before the streamed feature path. The 0B profile and
  bundle builders now ingest cleaned reviews and duplicate keys into a
  disk-backed SQLite store, aggregate client profiles from counts, and read
  client text in batches. A synthetic parity check confirms the existing
  seeded client caps, holdout/role splits and TF-IDF features are preserved.
  No official result existed.

## 2026-10-02

* Project created. Old code and numbers discarded (`docs/EXISTING_MATERIALS_AUDIT.md`).
* Protocol version 1 frozen in `EXPERIMENT_PLAN.md`, including the Experiment
  0A protocol and decision rule. No official result existed.
* Phase 0 implemented: data, roles, features, partitions, LR, Flower
  client/server (FedAvg, FedProx, FedAdam, SCAFFOLD), baselines, accounting,
  statistics, tests, smoke test.
* Smoke-test design changed after its first laptop run. The first version
  asserted "accuracy > 0.5" on a 3-client Dirichlet(0.5) split; the run gave
  exactly 0.50 because the draw produced nearly single-class clients and the
  averaged model predicted one class, while centralized training reached 0.79.
  The assertion now uses an IID split, and the Dirichlet run is kept as a
  reproducibility check with its accuracy recorded but not asserted. This is a
  change to a software check, not to the experimental protocol; the
  observation is logged in `EXPERIMENT_LOG.md`.
* Bug fixed before any official run: Kaggle detection used the existence of
  `/kaggle/working`, which exists on the laptop's D: drive, so a laptop
  summary was labelled as Kaggle. Detection now requires Linux and
  `KAGGLE_KERNEL_RUN_TYPE`; regression test added.
* Macro-F1 for per-client evaluation is averaged over the classes present in
  that client's labels (identical to standard macro-F1 on the global test
  set). Decided before any result.

## 2026-10-02 (preparation for Experiment 0; no official result existed)

Protocol:

* Amendments A1 to A6 recorded in `EXPERIMENT_PLAN.md` section 9. The frozen
  text of sections 1 to 8 was not edited.
* Three problems in the frozen protocol were found and **not** changed; they
  are listed for decision in `docs/PHASE_0_TO_PHASE_1_AUDIT.md` (M1, M4, M5).

Code that affects how results are produced:

* New condition `fedavg_la`: FedAvg with local label-prior logit correction
  (offset log of the add-one local class prior during training only).
  `fl.prior_reference: global` subtracts the log global prior; the default
  `none` is the frozen definition.
* New optional partition scheme `dirichlet_client` (equal-size clients, one
  Dirichlet label mix per client). The default scheme and its redraw rule are
  unchanged.
* Partition file names now include a non-default `min_client_size`.
* Natural-client bundles (Amazon Reviews 2023): cleaning rules, client-size
  filter, held-out client split, per-client roles.
* Run directories are immutable: a finished run cannot be overwritten; an
  unfinished one is renamed `*.incomplete-<UTC time>`.
* `run_metadata.json` gained `experiment_id`, `run_id`, `config_hash`,
  `official_environment`, `vocabulary_sha256` and, under `partition`, the
  type, alpha, partition seed, client count, client definition, filter rule
  and realized per-client class counts.
* The git "dirty" flag now means "a tracked file differs from the commit".
  Untracked files are counted separately. Reason: partitions created on
  Kaggle are new untracked files and must not block an official run, while
  modified code must.
* The centralized baseline now saves `final_weights.npz`.
* `client_count` for the simulation is read from the saved partition.

Diagnostic definition (made on a synthetic fixture, before any real data):

* First draft of the candidate rule flagged genuine sentiment words together
  with client-habit words. Added `gap_type` (inflated / sign_reversed /
  attenuated); only the first two are candidates.
* First null (shuffle documents across clients) flagged pure-noise words
  because it erased differences in client mean sentiment. Replaced by a
  shuffle of text among documents with the same sentiment score, with the
  null mean subtracted and a Student t reference. A calibration test on 300
  noise terms was added.
* The partial-correlation statistic now returns "undefined" when the bias is
  fully explained by the reference coefficients, instead of a correlation of
  rounding noise.

## 2026-10-02 (author decisions before any scientific 0A run; no official result existed)

* **Seeds (A7).** Official seeds are 42, 123, 456, 789, 1001, 2024, 31415,
  271828. Seed 2026 of the first list is no longer used. `configs/base.yaml`
  and `src/utils/seeding.py` updated.
* **Partition (A10).** The label-skew cell of 0A uses `dirichlet_client/v1`
  (100 equal-size clients, alpha = 0.1). Reason: the per-class recipe with a
  minimum client size cannot produce a partition at N = 100, alpha = 0.1, and
  dropping the minimum would mix extreme size skew into a label-skew
  diagnostic. The algorithm identifier is stored with every partition.
* **Tuning (A8, A12).** Validation-only tuning of learning rate and weight
  decay added (`experiments/exp0a_tune.py`, `src/analysis/tuning.py`). New
  config switch `eval.test: false` makes a run skip the test split entirely.
  The official run now also requires a tuning result produced on Kaggle and
  checks that the tuned values are the ones in use. Centralized epochs set to
  13 by the equal-passes rule. All formerly PROVISIONAL 0A values are now
  either fixed a priori or tuned.
* **Decision rule (A9).** The raw Spearman criterion alone no longer declares
  an effect; the partial-correlation criterion must hold as well, with equal
  signs. `summary.json` now reports both criteria, the outcome (A/B/C) and a
  `raw_only` flag.
* **Dataset pin (A11).** `data.revision` set for Yelp. The data fingerprint
  of the laptop subset is unchanged by the pin.
* **Experiment 0A tuning completed (A8, A12).** Official Kaggle validation-only
  tuning selected `fl.lr = 0.3` and `fl.weight_decay = 1e-6`, with both values
  at their predefined grid boundaries. The grid was not expanded. The frozen
  config records the clean Kaggle commit and tuning CSV hash; no official 0A
  result existed when this entry was made.
* Not done: the Kaggle environment report is still not in the repository, so
  `ENVIRONMENT.md` section 1 remains empty.
