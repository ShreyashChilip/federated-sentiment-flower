# EXPLORATORY confirmation on new seeds: confirm_v1

Status: **pre-registration**, written 2026-10-12 before any run of this study.
Not part of fedbench_v1; frozen configs, selections and results are unchanged.

## Question
Do the validation-selected round budgets of the manuscript (Table "frontier quality")
hold on seeds that played no part in tuning, selection or the earlier held-out checks?
Motivation: a leave-one-seed-out analysis of the existing runs showed the Yelp IID
selections to be stable but the Yelp Dirichlet-0.1 and Amazon selections to be seed-sensitive.

## Design (fixed before any result)
* Seeds 789 and 1001 (listed in the base config, never used so far). For Yelp the seed
  also changes the partition (new IID / Dirichlet-0.1 clients); the Amazon natural partition is fixed.
* Configuration identical to the official screening runs (verified field by field against the
  official seed-42 `run_metadata.json`); hyperparameters read from byte-identical copies of the frozen
  Job 3 selections (`configs/exploratory/confirm_v1_selections`, SHA256SUMS).
* Runs (50 rounds each; held-out metrics at rounds 10..50 give every budget, because nothing in
  training depends on the horizon): Yelp IID FedAdam + FedAdagrad; Yelp Dir-0.1 FedAdagrad;
  Amazon FedAdam + FedYogi. 10 runs.
* Configurations evaluated (from the three-seed selection; not re-selected):

| Regime | Target | Configuration | Reference |
|---|---|---|---|
| Amazon | <= 2 pp | FedAdam @ 30 | FedYogi @ 50 |
| Amazon | <= 1 pp | FedAdam @ 40 | FedYogi @ 50 |
| Yelp Dir-0.1 | <= 1 and <= 2 pp | FedAdagrad @ 20 | FedAdagrad @ 50 |
| Yelp IID | <= 2 pp | FedAdam @ 10 | FedAdagrad @ 50 |
| Yelp IID | <= 1 pp | FedAdam @ 20 | FedAdagrad @ 50 |

## Outcome rule (per row, on the two new seeds; delta = held-out macro-F1 change in pp vs. the same-seed reference)
* **Confirmed:** both new seeds have delta >= -target.
* **Partly confirmed:** mean delta >= -target, but one seed below -target.
* **Not confirmed:** mean delta < -target.
Five-seed descriptive summaries (three original + two new) are reported alongside, never used for selection.

## Manuscript consequences (decided in advance)
* Each row's outcome is reported as is, including negative outcomes.
* If the Yelp IID rows are confirmed, the paper may call the IID budgets reproducible on new seeds and
  partitions; otherwise that claim is removed.
* If Dirichlet-0.1 or Amazon rows are not confirmed, the paper states that reduced budgets are not
  reliable for those regimes under this selection rule.
* The support-only accounting conclusions do not depend on this study.
