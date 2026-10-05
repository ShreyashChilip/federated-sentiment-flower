# Changelog

Protocol and code changes that can affect results. Each entry states whether
results had been seen when the change was made.

## 2026-10-05 (memory bug: Amazon pilot OOM-killed; fix before any result)

* Job 1 (commit 10f519b): the Yelp pilots completed; `pilot/amazon_vg/fedavg/seed0`
  was killed with return code -9 after ~169 s. The runner stopped as designed.
  Nothing from the failed run is used; the failure stays recorded.
* Cause: `load_partition` indexed `npz["indices"]` inside the per-client loop.
  Every access decompresses the whole index array, and each client's slice
  keeps its own full copy alive, so memory grows as clients x rows (measured
  ~26.7 MB per client at 3.5M rows: 5.3 GB for 200 clients). Yelp (100
  clients) survived; tens of thousands of natural clients cannot. The array
  is now read once and every client is a view of it (flat ~27 MB for 20,000
  clients). Partition contents and hashes are unchanged; the hash check on
  load still runs. Regression test in `tests/test_partitioning.py`.
* Checked and ruled out: no dense materialization of TF-IDF; scipy keeps the
  memory-mapped data/indices as file-backed views (it copies only `indptr`,
  rows x 4 bytes); training/evaluation never load the held-out matrix; one
  model per process; no data duplication across processes (one worker at a
  time).
* Laptop memory check (`configs/smoke_bench/*many_clients.yaml`, SYNTHETIC):
  4,800 seen product clients, 420,814 reviews: FedAvg pilot completes,
  peak RSS 480 MB.
* `tools/kaggle_job.py` gained `--only` and `--retry-failed` pass-through.
* The same bug also affects `experiments/client_diagnostics.py` (it uses
  `load_runtime`); the Job 1 Amazon client diagnostics must be recomputed.

## 2026-10-05 (Kaggle job generator fix; no experiment had run)

* Job 1 stopped on Kaggle before any run with `NameError: name 'true' is not
  defined`: `tools/kaggle_job.py` wrote the JOB block of the Python kernel
  with `json.dumps`, which emits JSON literals. It now writes a Python literal
  (`pprint`), parses and compiles the generated script and checks that the
  embedded JOB equals the intended one before writing it. Diagnostic seeds
  are written as integers. Regression test: `tests/test_kaggle_job.py`. No
  experimental code or protocol changed.

## 2026-10-05 (screening threshold units clarified; no benchmark run had been made)

* `docs/BENCHMARK_PROTOCOL.md` section 6 and `src/benchmark/screening.py` both
  define Delta = 0.02 (mean/pooled) and 0.05 (tail). The unit was not
  stated. It is now written out: Delta is an absolute difference on the 0-1
  metric scale (0.02 = 2 percentage points), the same for every regime
  including 5-class Amazon, not a relative margin. This is what the code
  already did; no value or rule changed. A test pins it
  (`tests/test_benchmark_screening.py`). An earlier chat summary that called
  the protocol margin "2%" was a misstatement, not a protocol text.

## 2026-10-05 (FL strategy benchmark fedbench_v1, Phase 1; no benchmark result existed)

* Experiment 0A is treated as a completed negative/diagnostic result. Its
  official outputs are not in the repository; nothing below changes 0A code
  paths or configs.
* New package `src/benchmark/`: one algorithm interface (FedAvg, FedProx,
  FedNova, SCAFFOLD, FedAdam, FedAdagrad, FedYogi) with streaming
  aggregation; sequential CPU simulation engine; chunked evaluator with
  per-client seen/unseen metrics; disk-backed SCAFFOLD client state;
  resource estimator; staged resumable orchestrator with subprocess
  isolation, failure/OOM classification and manifests; analysis, screening
  rules and model-free client descriptors. Update rules verified against the
  original papers (`docs/ALGORITHM_VERIFICATION.md`). FedAvg and SCAFFOLD
  final weights are bit-identical to the Phase 0 Flower path (test).
* Benchmark FedAdam/FedYogi/FedAdagrad initialize v = tau^2 (Reddi et al.);
  the Phase 0 Flower FedAdam (v = 0) is unchanged and not used by the benchmark.
* Protocol and screening decision rules pre-registered in
  `docs/BENCHMARK_PROTOCOL.md`; matrix in `configs/benchmark.yaml`. Screening
  seeds fixed to 42, 123, 456 (first three official seeds).
* `prepare_run` no longer loads the held-out split (it only needs labels,
  roles and client codes); for natural bundles this avoids a full copy of the
  unseen-client matrix. No data or result changes.
* Metrics gained per-class precision/recall/support and P25/P75/P90 client
  percentiles (additional keys only).
* Engine settings added for the benchmark only: `device: cpu`,
  `eval.full_every: 10`. The frozen 0A/0B scientific settings are inherited
  unchanged.

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
