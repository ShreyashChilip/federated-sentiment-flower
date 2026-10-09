# Changelog

Protocol and code changes that can affect results. Each entry states whether
results had been seen when the change was made.

## 2026-10-09 (EXPLORATORY / POST-SCREENING, branch explore/cce-v1; no exploratory run yet)

* Pre-registered exploratory pilot `explore_cce_v1` (docs/EXPLORATORY_CCE_PROTOCOL.md):
  seed 7, validation only, 28 runs: FedAvg with server lr {1,3,10,30,100} (lr 1 =
  bit-identity sentinel), FedExP (published eps grid, two-iterate average), CAV
  (coverage-normalized averaging), CCE-v2, CCE-unit (decomposition check), FedYogi
  server lr 3.0. Written after the frozen screening results were known; it is not
  part of fedbench_v1 and makes no novelty or accuracy claim.
* New: `src/benchmark/exploratory_algorithms.py`, `src/benchmark/exploratory_decision.py`,
  `experiments/explore_cce_analyze.py`, `configs/exploratory/*`, tests.
* `src/benchmark/engine.py`: opt-in hooks only (evaluation weights, client feature
  support, diagnostics folder, extra uplink bytes). Frozen algorithms take the old
  path; FedAvg/SCAFFOLD Flower bit-equivalence tests pass unchanged.
* Kaggle wrapper/generator: `bench` option and a separate `results_exploratory`
  output folder (restored and archived separately from official results).
* Mathematical review before implementation: CCE-v2's S/A^2 equals r/c with
  r = E[D^2]/(E|D|)^2 >= 1, i.e. CCE-v2 = CAV x r (magnitude-dispersion
  extrapolation). Absent features are defined by exact support, because the float32
  residue of analytic weight-decay removal (measured up to 1.7e-5 |x|, never 0) makes
  delta-based coverage unreliable.

## 2026-10-08 (tuning complete; decision before screening: run screening as frozen; no screening result existed)

* Tuning complete on Kaggle, experiments at 3415496: tune_base 9/9,
  tune_algorithms 66/66, `verify` OK for both, all official. Selections
  (validation macro-F1, seed 7) as printed by the final session; the selection
  files were verified locally from that session's archive (EXPERIMENT_LOG, 2026-10-08):
  - yelp_iid: FedProx mu 0.001 (edge); FedAdam server lr 1.0 (edge);
    FedAdagrad 1.0 (edge); FedYogi 0.3.
  - yelp_dir01: FedProx mu 0.01; FedAdam 0.1; FedAdagrad 1.0 (edge); FedYogi 0.1.
  - amazon_vg: base lr 0.3, wd 1e-6 (both edge); FedProx mu 0.001 (edge);
    FedAdam, FedAdagrad, FedYogi server lr 1.0 (edge).
  Edge selections are recorded and frozen; no grid is extended (protocol section 3).
* Determinism across Kaggle sessions: the tune_base winner (lr 0.3, wd 1e-6)
  and the separately run tune_algorithms FedAvg reference have identical
  final weights (sha256 d83ab0edceb3...).
* Tuning-stage observation (validation data only), recorded before screening:
  on amazon_vg, FedAvg, FedProx, FedNova and SCAFFOLD stay at or near the
  majority-class solution within the frozen 50-round budget (they predict
  5 stars for ~99% of validation reviews; validation macro-F1 0.153-0.183 vs
  0.153 for majority-only; curves flat until ~round 30). FedAdam/FedAdagrad/
  FedYogi (server lr 1.0) reach 0.56-0.58. Likely cause: few local SGD steps
  per small natural client per round (~120 training reviews, batch 32),
  i.e. a step-budget effect rather than evidence about heterogeneity. Not a
  research-gap claim.
* Decision (author, 2026-10-08): run screening exactly as frozen (option A).
  No exploratory budget study is added before screening. Screening
  interpretation must state that the Amazon FedAvg-family results are
  budget-limited; the protocol's Phase B (local epochs E in {1, 3},
  participation) is where this is examined.

## 2026-10-07 (Job 3b first attempt stopped by the restore guard; nothing ran)

* The first Job 3b session (experiments 3415496, wrapper 5178cb3) stopped in
  restore, before any run: the Amazon partition summary differed between the
  Job 2 and Job 3a outputs. Cause: the partition indices are byte-identical
  (same .npz, sha256 2d803e8c...), but the summary records the seed of the run
  that created it (Job 2: 42; Job 3a: 7). A natural-client partition does not
  depend on the seed.
* Root cause of the re-creation: Kaggle mounts notebook outputs at
  /kaggle/input/notebooks/<owner>/<slug>/ (three levels); the Job 3a wrapper
  searched two levels, so Job 3a restored nothing and rebuilt both feature
  bundles and the Amazon partition (its first run took 70 min instead of ~31).
  The rebuilt bundles are identical to Job 2's on every recorded fingerprint
  (source sha256, vocabulary sha256, train/test fingerprints, role counts,
  fit documents) for Amazon and Yelp, so Job 3a results stand.
* Wrapper fix: two partition summaries that are both natural-scheme, have the
  same sha256 and differ only in `experiment_seed`/`partition_seed` are not a
  conflict; the first is kept, the other saved as `*.from-<origin>`. Any other
  difference still aborts. Offline replay with the real Job 2 and 3a archives
  in Kaggle's layout: restore succeeds, preflight finds 9 + 33 completed runs.
  Amazon tuning runs of 3b will record the Job 2 creating seed in
  `partition.partition_seed`; for a natural partition that field is unused.
* Experiment code unchanged since 3415496.

## 2026-10-07 (Kaggle wrapper: dataset inputs, preflight guard, pinned experiment commit; tuning in progress)

Job 3a (commit 3415496): tune_base 9/9 complete, Amazon selection lr = 0.3,
wd = 1e-6 (validation macro-F1 0.1753), both on the grid edge; recorded and
frozen as-is (protocol: the grid is not extended). tune_algorithms 33/66
complete (all of yelp_iid, 11 of yelp_dir01), no algorithm selection yet.
`verify` OK for both stages. Job 3a used the same bundles, vocabularies and
Amazon partition as Job 2 (checked from both archives).

Wrapper/generator only; experiment code (`src/`, `configs/`, `experiments/`,
requirements) unchanged since 3415496:

* Restore finds result folders and bundles at any depth under /kaggle/input
  (datasets mount deeper than notebook outputs) and extracts
  `fedbench_results_*.zip` files that a dataset kept zipped. Previously such
  inputs would not have been found and finished runs would have been redone.
* Preflight: a job can require a minimum number of restored completed runs per
  stage and named feature bundles; otherwise it stops before running anything.
* `tools/kaggle_job.py --code-commit`: the experiments run from an earlier
  commit while the wrapper is newer, allowed only if the experiment paths are
  identical between the two commits. Lets Job 3b/3c finish tune_algorithms on
  3415496, the commit of the Job 3a runs.
* Job 3a timing (for planning): Amazon FedAvg tuning runs ~31 min each in
  steady state (first run 70 min), Yelp runs ~9 min.

## 2026-10-06 (engineering fixes before Job 3; no tuning or screening result existed)

Approved before Job 3. Job 2 (commit 55f8104, Kaggle) completed the pilot
stage: 15/15 runs, `verify pilot` OK, Amazon diagnostics validated.

* SCAFFOLD client state (`src/benchmark/state_store.py`) is written and read
  with plain file I/O instead of a memory map. Job 2's Amazon SCAFFOLD pilot
  peaked at 8.75 GB RSS after 3 rounds: mapped rows that had been written
  stayed resident (measured 1:1 with the written state; ~21.7 GB projected for
  21,694 clients at 50 rounds, on a 31.35 GB machine). Now the process grows by
  ~5 MB for 2.8 GB of state. File layout unchanged (row cid, float32, at byte
  cid x width x 4). Bit-identical results: a SCAFFOLD run on the SYNTHETIC
  4,800-client population gives the same final-weights hash (c6b900b9...),
  clients with state and c_global norm as the memory-mapped store; the Flower
  equivalence test for SCAFFOLD still passes. Regression tests: file layout,
  and process growth while writing ~190 MB of state (fails on the old store).
* Resource estimator: the memory-mapped feature matrix is counted as resident
  (evaluation touches rows across the whole file). Amazon pilot: estimate
  ~3.4 GB vs 2.9 GB measured (was 1.1 GB). SCAFFOLD state is reported as
  reclaimable page cache and enters a worst-case figure that only warns.
* Planner refuses a tuning seed that is also an evaluation seed (each regime's
  official seeds and every evaluation stage's seeds). Seed 7 and the official
  seeds are unchanged.
* `run_benchmark.py verify` reports a stage that cannot be planned yet
  (missing selection) without a traceback and without failing.
* Unchanged, checked: `docs/BENCHMARK_PROTOCOL.md`, every config; all 171
  planned runs of pilot / tune_base / tune_algorithms / screening have
  identical keys and config hashes before and after this change.

## 2026-10-06 (provenance: diagnostics validation, verify command, stable pilot specs, safe restore; no scientific result existed)

* Planning bug found by the new `verify` command (local smoke results only):
  pilot and base-tuning specs fell back to placeholders only while no
  `tune_base` selection existed, so re-planning them after tuning changed
  their config hash and completed pilots no longer matched their spec. They
  now never read selections (`regime_config(untuned=True)`). Evaluation
  stages still require the selection. The 24 pilot/tune_base specs of
  `configs/benchmark.yaml` hash identically to commit a3b07c5, so the
  completed Kaggle pilot runs stay valid.
* `experiments/client_diagnostics.py`: the descriptor table is checked
  against the bundle (seen/unseen client counts and row totals, class counts,
  raw-text measures present) and `DIAG_COMPLETE` is written only if every
  check passes; an unvalidated table is recomputed, never reused. Laptop
  check (SYNTHETIC, 6,000 clients, 420,814 reviews): 18 s, peak RSS 398 MB.
* `run_benchmark.py verify`: re-validates every complete run and checks one
  bundle/vocabulary per regime, one partition per regime (natural) or per seed
  (synthetic), and that validated diagnostics use the same partition.
* Kaggle kernel restore merges several attached outputs without loss: status
  attempts are unioned by start time, differing logs are both kept, identical
  files skipped; a differing immutable artifact or bundle aborts the job
  before anything runs. The kernel runs `verify` at the end.
* No threshold, algorithm, dataset, partition or frozen config changed.

## 2026-10-06 (pilot analysis no longer labelled as screening evidence)

* `analyze_stage` applied the screening rules to every stage, so the Job 1
  pilot analysis reported `"screening_outcome": "candidate"`. Screening rules
  now run only for stages of kind `evaluate` (screening/confirmation); pilot
  and tuning stages report `not_applicable` with no cells or candidates.
  Thresholds, rules, algorithms, data and frozen configs unchanged. The Job 1
  pilot label is void. Tests: `tests/test_benchmark_screening.py`.

## 2026-10-06 (OOM fix validated on Kaggle and promoted; no scientific result existed)

* Kaggle CPU session (31.35 GB RAM), commit a3b07c5 (scratch branch
  `fedbench-oomfix`): `pilot/amazon_vg/fedavg/seed0` COMPLETE (attempt 2;
  attempt 1 at 10f519b stays recorded as killed, return code -9), 2231 s
  wall including the build of the Video_Games feature bundle (2.3 GB), no
  OOM/SIGKILL. COMPLETE is written only after artifact validation (seed,
  config hash, partition hash, round count, held-out metrics). Evidence as
  reported from the Kaggle log; the archive is not in the repository.
* `fedbench` fast-forwarded to a3b07c5 (no history rewritten). Partition
  algorithm, contents and hashes unchanged; frozen 0A/0B configs unchanged.
* Pilot metrics are engineering checks only and are not used for any decision.

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
