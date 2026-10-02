# Changelog

Protocol and code changes that can affect results. Each entry states whether
results had been seen when the change was made.

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
