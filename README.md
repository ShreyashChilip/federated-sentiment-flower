# Federated sentiment classification: a reproducible Flower benchmark

Rebuild of the study "Resource-Efficient Sentiment Classification through
Federated Learning: Performance Analysis of FedProx and FedAvg". The original
manuscript, reviewer comments and improvement plan are summarised in
`docs/EXISTING_MATERIALS_AUDIT.md`. No number from the original paper or its
code is reused.

**Current state: Phase 0 (infrastructure) is complete on the development
laptop. No official result exists yet.** Official results come only from
Kaggle.

## Where things are

| File | Purpose |
|---|---|
| `REQUIREMENTS_MATRIX.md` | Every reviewer requirement and its status |
| `EXPERIMENT_PLAN.md` | Frozen protocol, phases, gates, open decisions |
| `EXPERIMENT_LOG.md` | What was run, when, and what was observed |
| `NOVELTY_AUDIT.md` | Prior-art audit (seeded, not complete) |
| `ENVIRONMENT.md`, `REPRODUCIBILITY.md` | Environments and how runs are made repeatable |
| `CHANGELOG.md` | Every protocol or code change that could affect results |
| `configs/` | All settings; experiments take no hard-coded hyperparameters |
| `src/` | All research logic |
| `experiments/` | Thin scripts that call `src/` |
| `tests/` | Unit tests |
| `partitions/` | Saved client partitions (tracked in git) |
| `results/` | One directory per run with metadata, history and final metrics |

## What is implemented

* Yelp Polarity loading with the official split; fixed train/val/client-test roles.
* IID, Dirichlet label-skew, quantity-skew and natural-key partitioners; saved, hashed partitions.
* Federated TF-IDF features (globally consistent vocabulary and IDF, no test leakage).
* Logistic regression in PyTorch on sparse inputs.
* Flower (1.39, Message API) client and server: FedAvg, FedProx, FedAdam, SCAFFOLD;
  seeded client sampling, dropout, stragglers with partial work, LR decay.
* Centralized and local-only baselines sharing the same training code.
* Communication ledger (bytes per direction per participating client), timing,
  memory, energy proxy, per-client metrics and dispersion.
* Across-seed statistics: mean, std, 95% CI, paired t, exact Wilcoxon, effect sizes, Holm.
* Automatic `summary.md` per experiment.

Not implemented yet: TextCNN, DistilBERT + LoRA, Amazon 5-class, natural
clients, FedNova/FedLC/FedDisco/Fisher baselines, DP, secure aggregation,
figures and tables, manuscript. See `EXPERIMENT_PLAN.md`.

## Quick start

```
pip install -r requirements.txt
python -m pytest tests -q
python experiments/00_smoke_test.py
```

On Kaggle use `notebooks/kaggle_runner.py`.
