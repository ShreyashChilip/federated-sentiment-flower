# Kaggle runbook: Experiment 0 (Phase 1)

Each block is one notebook cell. Notebook settings: GPU on, Internet on,
"Pin to original environment". Replace `<COMMIT>` with the commit you intend
to run; every result records it.

Nothing here has been executed on Kaggle by the preparation work.

| Step | What it is | Scientific result? |
|---|---|---|
| 1 | Environment verification | No |
| 2 | Experiment 0A pilot | No (timing and plumbing; seed 0 is not an official seed) |
| 3 | Experiment 0A full run | **Yes. This is the first scientific run.** |
| 4 | Experiment 0B client profiling | Descriptive data statistics; no model |
| 5 | Experiment 0B training | Yes |

## Before step 3: blockers

Step 3 refuses to start until all of these are committed (the script prints
the unmet ones):

1. `configs/base.yaml`: `experiment.seeds` holds the 8 official seeds.
2. `configs/base.yaml`: `data.revision` holds the Yelp dataset commit.
3. `configs/exp0a.yaml`: the `label_skew` cell is feasible (decision D2 in
   `PHASE_0_TO_PHASE_1_AUDIT.md`); with the current frozen rule partition
   creation fails for N = 100, alpha = 0.1.
4. `configs/exp0a.yaml`: PROVISIONAL values approved and `protocol.frozen: true`.
5. `CHANGELOG.md`: one dated entry per decision.

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
!python -c "from huggingface_hub import HfApi; print('yelp revision:', HfApi().dataset_info('fancyzhx/yelp_polarity').sha)"
```

Expected: 73 tests pass; both smoke tests end with `PASSED`; the environment
report shows `"platform_kind": "kaggle"`, a GPU, a git commit and
`"dirty": false`. Copy the report into `ENVIRONMENT.md` section 1 and the
printed Yelp revision into `configs/base.yaml` (`data.revision`), then commit.

## 2. Experiment 0A pilot (not a scientific result)

```
!python experiments/exp0a_controlled_diagnostic.py --mode pilot --create-partitions
!cat results/exp0a_pilot/summary.md
```

Purpose: wall-clock per round, memory, disk, and a check that every file is
written at full data scale (560,000 training rows, 100 clients). It runs seed
0 for 3 rounds. From `results/exp0a_pilot/rounds.csv` take `round_wall_s` and
compute the cost of step 3:

```
cost = round_wall_s x rounds x 2 conditions x 2 cells x 8 seeds  (+ 8 centralized runs)
```

Write that estimate into `EXPERIMENT_PLAN.md` section 7. Do not use the
pilot's accuracy, F1 or correlations for any decision.

If the `label_skew` cell stops with "no Dirichlet draw with every client >=
10 samples", decision D2 has not been made yet. That is expected with the
frozen rule.

## 3. Experiment 0A full run (FIRST SCIENTIFIC RUN)

```
!python experiments/make_partitions.py --config exp0a.yaml --cells
!python experiments/exp0a_controlled_diagnostic.py --mode full
```

If one session cannot finish all seeds, run them in chunks; each call resumes:

```
!python experiments/exp0a_controlled_diagnostic.py --mode full --seeds <s1> <s2>
!python experiments/exp0a_controlled_diagnostic.py --mode full --seeds <s3> <s4>
...
!python experiments/exp0a_controlled_diagnostic.py --mode full --analyze-only
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

Reminder: 0A describes ordinary label heterogeneity. It is not evidence about
natural-client lexical confounding.

## 4. Experiment 0B client profiling (no training)

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

After profiling, and before step 5, write into the configs and commit:

* `data.category`,
* `data.client_filter.min_reviews` (and `max_reviews`, `max_clients` if used),
  separately for `exp0b_user.yaml` and `exp0b_product.yaml` if they differ,
* `fl.prior_reference` (decision D4), PROVISIONAL values, `protocol.frozen: true`,
* a `CHANGELOG.md` entry stating the rule and that no model result existed.

## 5. Experiment 0B training

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

Check before reading any number: the same three points as in step 3, plus
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
