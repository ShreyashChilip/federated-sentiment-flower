# Reproducibility

## What is fixed, and where

| Source of variation | How it is controlled |
|---|---|
| Package versions | `requirements.txt`, `requirements-kaggle.txt`; actual versions stored per run |
| Dataset | Official Hugging Face split; revision pinned in `configs/base.yaml` (`data.revision`) once recorded on Kaggle; size, class counts and a fingerprint stored per run |
| Train/val/client-test roles | `src/data/roles.py`, seed `derive_seed("roles", dataset, role_seed)`; independent of the experiment seed |
| Features | Fitted on train-role rows only; cached in `data_cache/features/<dataset>_<hash>` keyed by the data and feature config |
| Partition | Created once by `experiments/make_partitions.py`, saved to `partitions/` as `.npz` (indices) + `.json` (client ids, sizes, class counts, parameters, seed, SHA-256). Runs only **load** partitions and verify the hash; a missing partition is an error |
| Model initialisation | `derive_seed("init", seed)`; LR starts at zero |
| Client sampling | `derive_seed("sample", seed, round)` |
| Dropout / stragglers | `derive_seed("dropout" or "straggler", seed, round)` |
| Batch order | `derive_seed("batch", seed, client_id, round)` |
| Label noise | `derive_seed("label_noise", seed)`; training-role labels only |
| Aggregation order | Client replies are sorted by client id before aggregation, so parallel execution order cannot change floating-point sums |
| Threads | One torch thread per simulated client |

Seeds used: 42, 123, 456, 789, 2026 (`src/utils/seeding.py`).

## What every run records (`run_metadata.json`)

Seed, full resolved configuration, dataset metadata, partition name and
SHA-256, label-noise count, timestamp (UTC), platform (Kaggle or local),
hardware, package versions, git commit and whether the tree was dirty.

Alongside it: `history.json` (per round), `clients.json` (per client),
`final.json` (final metrics, communication, resources, SHA-256 of the final
weights) and `final_weights.npz`.

## Verified so far

On the laptop, for FedAvg, FedProx, FedAdam and SCAFFOLD, two runs with the
same seed produce identical final-weight hashes, metrics and byte counts, and
a different seed produces a different model (`experiments/00_smoke_test.py`).
The same check must pass on Kaggle (Phase 0K) before any official run. GPU
determinism for the neural models is not yet verified.

Timing and memory readings are measurements and are not expected to repeat
exactly.

## Reproducing from a clean checkout

```
pip install -r requirements.txt            # laptop; on Kaggle: requirements-kaggle.txt
python -m pytest tests -q
python experiments/00_smoke_test.py
python experiments/make_partitions.py --config yelp.yaml --clients 10 --alphas 0.1 --no-iid
python experiments/01_baseline.py --config yelp.yaml
```

## Test-set hygiene

The official test split is read in two places only: feature **transformation**
(never fitting) in `src/data/prepare.py`, and evaluation. It has no role, is
never partitioned, and no hyperparameter or stopping decision reads it.
`tests/test_features.py` checks that held-out text cannot change the feature
space.

## Added with the Experiment 0 preparation

* **Immutable runs.** A run directory that contains `final.json` cannot be
  written again (`FileExistsError`). An unfinished directory is renamed
  `<name>.incomplete-<UTC time>` and kept. Each run has a random `run_id`.
* **Results schema.** `src/utils/schema.py` lists the fields every run must
  record and validates them; `collect_tables` writes `runs.csv`, `rounds.csv`,
  `clients.csv` and `schema_validation.json` per experiment.
* **Comparability checks.** Before any diagnostic is computed, the three
  conditions of a cell are checked for the same vocabulary hash, bundle,
  training split, test set, seed, optimizer settings and partition
  (`src/analysis/exp0.py`). A failed check stops the analysis.
* **Official-run guard.** `--mode full` starts only on Kaggle, from a commit
  with no modified tracked file, with `protocol.frozen: true`, the full seed
  list and a pinned dataset revision.
* **Natural clients.** The held-out client split is seeded by
  `data.holdout.holdout_seed`; the client-size rule is part of the config and
  of the bundle hash; zero overlap between training and held-out clients is
  asserted when the bundle is built and stored in its metadata.
* **Amazon source.** Repository revision pinned in `configs/amazon2023.yaml`;
  the SHA-256 of the category file is stored with every bundle and profile.
