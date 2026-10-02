# Kaggle runbook: Experiment 0 (Phase 1)

Each block is one notebook cell. Notebook settings: GPU on, Internet on,
"Pin to original environment". Replace `<COMMIT>` with the commit you intend
to run; every result records it.

Nothing here has been executed on Kaggle by the preparation work.

| Step | What it is | Scientific result? |
|---|---|---|
| 1 | Environment verification | No |
| 2 | Experiment 0A pilot | No (timing and plumbing; seed 0) |
| 3 | Experiment 0A tuning | No (validation data only; seed 7; test split never evaluated) |
| 4 | Experiment 0A full run, 8 seeds | **Yes. This is the first scientific run.** |
| 5 | Experiment 0B client profiling | Descriptive data statistics; no model |
| 6 | Experiment 0B training | Yes |

## What is already fixed, and what blocks step 4

Fixed and committed: the 8 seeds, the Yelp revision, the `dirichlet_client`
partition, the tuning protocol, the decision rule, all non-tuned
hyperparameters (`configs/exp0a.yaml`).

Step 4 refuses to start until:

1. the Kaggle environment report from step 1 is committed (`ENVIRONMENT.md`
   section 1);
2. step 3 has been run on Kaggle from a clean commit, and the file it writes
   has been committed as `configs/exp0a_tuned.yaml`;
3. the notebook runs from that commit with no modified tracked file.

## 0. Setup (every session)

```
!git clone https://github.com/ShreyashChilip/federated-sentiment-flower.git /kaggle/working/fl
%cd /kaggle/working/fl
!git checkout <COMMIT>
!pip install -q -r requirements-kaggle.txt
!git status --short        # must print nothing
```

To continue an earlier session, restore the saved `results/`, `partitions/`
and (optionally) `data_cache/` folders into `/kaggle/working/fl` before
running anything. Finished runs are skipped; nothing is overwritten.

## 1. Environment verification

```
!python -m pytest tests -q
!python experiments/00_smoke_test.py
!python experiments/00b_exp0_smoke.py
!python -c "import json; from pathlib import Path; from src.utils.env import collect_environment; e = collect_environment(Path('.')); Path('results').mkdir(exist_ok=True); Path('results/kaggle_environment.json').write_text(json.dumps(e, indent=1)); print(json.dumps(e, indent=1))"
```

Expected: all unit tests pass; both smoke tests end with `PASSED`; the
environment report shows `"platform_kind": "kaggle"`, a GPU, a git commit and
`"dirty": false`. Download `results/kaggle_environment.json`, copy its values
into `ENVIRONMENT.md` section 1 and commit both.

## 2. Experiment 0A pilot (not a scientific result)

```
!python experiments/exp0a_controlled_diagnostic.py --mode pilot --create-partitions
!cat results/exp0a_pilot/summary.md
```

Purpose: wall-clock per round, memory, disk, and a check that every file is
written at full data scale (560,000 training rows, 100 clients). It runs seed
0 for 3 rounds. From `results/exp0a_pilot/rounds.csv` take `round_wall_s` and
estimate the later steps:

```
tuning   = round_wall_s x 50 rounds x 9 grid points
full run = round_wall_s x 50 rounds x 2 conditions x 2 cells x 8 seeds  (+ 8 centralized runs of 13 epochs)
```

Write the estimate into `EXPERIMENT_PLAN.md` section 7. Do not use the pilot's
accuracy, F1 or correlations for any decision.

## 3. Experiment 0A tuning (validation only; not a scientific result)

```
!python experiments/exp0a_tune.py
!cat results/exp0a_tuning/selected.json
!zip -qr /kaggle/working/exp0a_tuning.zip results/exp0a_tuning partitions
```

It runs FedAvg on the label-skew cell with seed 7 for the 9 grid points
(learning rate x weight decay), selects by pooled validation macro-F1 at the
last round, and writes `tuning.csv`, `selected.json` and `exp0a_tuned.yaml`.
The test split is never evaluated.

Then, on the laptop:

1. copy `results/exp0a_tuning/exp0a_tuned.yaml` to `configs/exp0a_tuned.yaml`;
2. check that it says `frozen: true` and `official_environment: true`
   (otherwise the tuning pass was not made on Kaggle from a clean commit);
3. commit it together with `results/exp0a_tuning/tuning.csv`,
   `selected.json` and the tuning partition, add a `CHANGELOG.md` entry with
   the selected values and whether they lie on the grid edge, and push.

If a selected value is on the edge of the grid, do not extend the grid
quietly. Either accept the value or record a grid extension in `CHANGELOG.md`
before running it.

## 4. Experiment 0A full run (FIRST SCIENTIFIC RUN)

From the commit that contains `configs/exp0a_tuned.yaml`:

```
!python experiments/make_partitions.py --config exp0a_tuned.yaml --cells
!python experiments/exp0a_controlled_diagnostic.py --config exp0a_tuned.yaml --mode full
```

If one session cannot finish all seeds, run them in chunks; each call resumes:

```
!python experiments/exp0a_controlled_diagnostic.py --config exp0a_tuned.yaml --mode full --seeds 42 123
!python experiments/exp0a_controlled_diagnostic.py --config exp0a_tuned.yaml --mode full --seeds 456 789
!python experiments/exp0a_controlled_diagnostic.py --config exp0a_tuned.yaml --mode full --seeds 1001 2024
!python experiments/exp0a_controlled_diagnostic.py --config exp0a_tuned.yaml --mode full --seeds 31415 271828
!python experiments/exp0a_controlled_diagnostic.py --config exp0a_tuned.yaml --mode full --analyze-only
```

Save the outputs before the session ends:

```
!zip -qr /kaggle/working/exp0a_outputs.zip results/exp0a partitions
```

Then, on the laptop: unzip, commit `partitions/` and `results/exp0a/`
(`runs.csv`, `rounds.csv`, `clients.csv`, `summary.json`, `summary.md`,
`schema_validation.json`, the per-seed `diagnostics/` folders and each run's
JSON files), and add an `EXPERIMENT_LOG.md` entry.

Check before reading any number:

* `results/exp0a/schema_validation.json` reports 0 runs with problems;
* every `checks.json` has all checks passed;
* `runs.csv`: `official_environment` is True, `git_dirty` is False and
  `git_commit` equals `<COMMIT>` in every row.

The decision rule (EXPERIMENT_PLAN.md section 11) is evaluated by the code
and written to `summary.json` under `decision_rule_0a`. Reminder: 0A describes
ordinary label heterogeneity. It is not evidence about natural-client lexical
confounding.

## 5. Experiment 0B client profiling (no training)

Requires `data.category` (decision D6). Either commit it in
`configs/amazon2023.yaml` or pass it explicitly:

```
!python experiments/exp0b_profile_clients.py --definition user    --set data.category=<Category>
!python experiments/exp0b_profile_clients.py --definition product --set data.category=<Category>
!zip -qr /kaggle/working/exp0b_profile.zip results/exp0b_profile
```

Outputs per definition: `profile.json` (raw client count, size distribution
with percentiles, median, min, max, per-client label skew, global class
distribution, retention for each candidate threshold) and `clients.csv` (one
row per client with its size and star counts).

Category file sizes range from 9 MB to 31 GB. The reader loads the whole file,
so choose a category that fits in memory. Do not use `data.max_records` for an
official run: it keeps the first lines of the file, not a random sample.

After profiling, and before step 6, write into the configs and commit:

* `data.category`,
* `data.client_filter.min_reviews` (and `max_reviews`, `max_clients` if used),
  separately for `exp0b_user.yaml` and `exp0b_product.yaml` if they differ,
* `fl.prior_reference` (decision D4), the PROVISIONAL values, `protocol.frozen: true`,
* a `CHANGELOG.md` entry stating the rule and that no model result existed.

## 6. Experiment 0B training

Pilot first (seed 0, 3 rounds, not a result):

```
!python experiments/exp0b_natural_clients.py --definition user --mode pilot --create-partitions
!python experiments/exp0b_natural_clients.py --definition product --mode pilot --create-partitions
```

Full runs:

```
!python experiments/exp0b_natural_clients.py --definition user --mode full --create-partitions
!python experiments/exp0b_natural_clients.py --definition product --mode full --create-partitions
!zip -qr /kaggle/working/exp0b_outputs.zip results/exp0b_user results/exp0b_product partitions
```

`--create-partitions` is safe here: natural clients are determined by the
data and the committed filter rule, the partition is saved on first use, and
its hash is checked on every later load.

Check before reading any number: the same three points as in step 4, plus
`"overlap": 0` under `dataset.holdout` in any `run_metadata.json`.

## What each run directory contains

```
results/<experiment>/<cell>/<condition>/seed<k>/
    run_metadata.json   experiment id, run id, seed, config and its hash, dataset
                        fingerprints, vocabulary hash, partition (type, alpha, seed,
                        hash, client count, realized class counts), environment,
                        git commit, official flag
    history.json        per round: loss, validation and test metrics, bytes up/down,
                        clients selected/dropped/reported, timings
    clients.json        per client: validation and client-test metrics
    final.json          final metrics, communication totals, resources, weight hash
    final_weights.npz   the model
results/<experiment>/<cell>/diagnostics/seed<k>/
    checks.json  diagnostic.json  feature_confounding.csv
results/<experiment>/
    runs.csv  rounds.csv  clients.csv  summary.json  summary.md  schema_validation.json
```
