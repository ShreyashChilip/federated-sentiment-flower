# Requirements matrix

One row per mandatory requirement. A row is **Done** only when the
implementation exists, the experiment has run **on Kaggle**, the output
table/figure has been generated from logs, and the manuscript section is
written from those outputs. Mentioning something in prose does not count.

Status values: `Not started` / `Code` (implemented and unit-tested, no official
results) / `Smoke` (also exercised end to end on the laptop at toy scale) /
`Run` (official Kaggle results exist) / `Done`.

Reviewer IDs refer to `docs/source_text/Revisions.docx.txt`. "PI" means the
item comes from `Planned Improvements.docx`.

| # | Requirement | Reviewer | Original problem | Required correction | Implementation | Experiment | Output | Manuscript section | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Rigorous English revision | 1.1 | Garbled sentences, inconsistent tense | Full rewrite from logged results | `manuscript/` | - | - | All | Not started |
| 2 | Structured Introduction | 1.2 | Reviewer structure only partly followed | Background / Challenges / Objectives / Contributions / Organization, <= 20 refs, 2024-2025 | `manuscript/` | - | - | 1 | Not started |
| 3 | Recent, analysed related work | 1.3 | Descriptive list, no method/result/limitation analysis | >= 15 refs 2023-2025, each with method, results, advantages, limitations; closing gap analysis | `docs/NOVELTY_AUDIT.md` then `manuscript/` | - | - | 2 | Not started (audit seeded) |
| 4 | Correct dataset totals | 1.4 | Table 1 sums to 560k, text says 418,600 | Official split; totals computed from data | `src/data/datasets.py`, `src/data/roles.py` | all | Table 1 | 4.1 | Smoke |
| 5 | Seeded partition script | 1.4 | Script partitioned dummy labels | Deterministic partitioner, saved indices and hash | `src/partitioning/` | all | `partitions/*`, Table 2 | 3, Reproducibility | Smoke |
| 6 | Explicit non-IID recipe | 1.4, 1.5 | Recipe stated, saved split did not match | Dirichlet label skew, exact algorithm documented | `src/partitioning/partition.py` | 02 | Fig 2, Fig 3 | 3 | Smoke |
| 7 | >= 3 heterogeneity levels | 1.5 | alpha = 1, 10 plots suppressed | alpha in {0.1, 0.5, 1.0, 10.0} + IID | `configs/base.yaml` | 02, 03 | Table 7, Fig 12 | 4 | Code (alpha grid not yet run) |
| 8 | Client participation fraction | 1.6, 1.11 | C confused with alpha | C in {0.25, 0.5, 1.0} as its own ablation | `src/strategies/server.py` | 03 (sensitivity) | Table 8, Fig 14 | 3, 4 | Smoke |
| 9 | Local epochs | 1.6, 1.11 | E fixed at 5 | E in {1, 3, 5} | `src/clients/trainer.py` | 03 | Table 8 | 3, 4 | Smoke |
| 10 | Batch size | 1.6 | Stated only | Config field, logged per run | `configs/*.yaml` | all | Table 4 | 3 | Smoke |
| 11 | Optimizer | 1.6 | Text said SGD, code used L-BFGS | SGD (+momentum) actually used | `src/clients/trainer.py` | all | Table 4 | 3 | Smoke |
| 12 | Local/global LR schedules | 1.6, 1.11 | gamma 0.98 vs 0.95 | Per-round local decay gamma and server LR in config | `src/strategies/server.py` | 03 | Table 4, Table 8 | 3 | Smoke |
| 13 | FedProx mu grid | 1.6, 1.11 | Post-hoc shrinkage, not FedProx | Proximal term in local objective; mu in {0, 0.01, 0.1, 1.0} tuned on validation | `src/clients/trainer.py` | 03 | Table 8 | 3, 4 | Smoke |
| 14 | Dropout/straggler policy | 1.6, 1.12 | Described, not measured | Seeded per-round dropout p; stragglers do partial local work | `src/strategies/server.py`, `src/clients/trainer.py` | 06 | Table 9 | 3, 4 | Smoke |
| 15 | Aggregation cadence | 1.6 | Stated | One aggregation per round, logged | `src/strategies/server.py` | all | Table 4 | 3 | Smoke |
| 16 | Random seeds | 1.6 | Three seeds, not all sources seeded | 42, 123, 456, 789, 2026; Python/NumPy/Torch/CUDA/sampling | `src/utils/seeding.py` | all | `run_metadata.json` | 3 | Smoke |
| 17 | Centralized TF-IDF + LR | 1.7 | Described, numbers untraceable | Same features, same optimizer | `src/clients/baselines.py` | 01 | Table 5 | 4 | Smoke |
| 18 | Local-only baseline | 1.7 | No numbers reported | Each client trains alone | `src/clients/baselines.py` | 01 | Table 5 | 4 | Smoke |
| 19 | Centralized transformer | 1.7 | Deferred | DistilBERT + LoRA centralized | `src/models/` | 04 | Table 5, Table 12 | 4 | Not started |
| 20 | Federated transformer | 1.7 | Deferred | DistilBERT + LoRA, FedAvg/FedProx | `src/models/` | 04 | Table 5, Table 12 | 4 | Not started |
| 21 | >= 3 seeds, preferably 5 | 1.8 | 3 seeds | 5 seeds for final runs | `configs/base.yaml` | all | all tables | 4 | Code |
| 22 | Mean +/- std | 1.8 | Std computed over rounds, not seeds | Across-seed std of final metric | `src/statistics/stats.py` | all | all tables | 4 | Code (unit-tested; needs multi-seed official runs) |
| 23 | 95% confidence intervals | 1.8 | Not done | t-interval across seeds | `src/statistics/stats.py` | all | all tables | 4 | Code (unit-tested; needs multi-seed official runs) |
| 24 | Paired statistical tests | 1.8 | Not done | Paired Wilcoxon across seeds (+ paired t, effect size, Holm) | `src/statistics/stats.py` | 03, 11 | Table 16 | 4 | Code (unit-tested; needs multi-seed official runs) |
| 25 | Rigorous communication accounting | 1.9 | Constant 38.15 MB regardless of participation | Serialized bytes per message, per direction, per participating client | `src/communication/accounting.py` | all | Table 11, Fig 8 | 4 | Smoke |
| 26 | Wall-clock time | 1.9 | 0.016 s totals | Per-client train time, per-round time, server aggregation time | `src/resources/monitor.py` | 05 | Table 12 | 4 | Smoke |
| 27 | Energy proxy | 1.9 | Claim contradicted own numbers | CPU time x declared TDP (+ GPU time x declared power), labelled as proxy | `src/resources/monitor.py` | 05 | Table 12, Fig 9 | 4, 5 | Smoke |
| 28 | System heterogeneity | PI 4 | Not simulated | Straggler fraction with partial local epochs | `src/clients/trainer.py` | 06 | Table 9 | 4 | Smoke |
| 29 | Client dropout | 1.12 | Brief mention | p in {0, 0.10} | `src/strategies/server.py` | 06 | Table 9 | 4 | Smoke |
| 30 | Unbalanced client sizes | 1.12 | Brief mention | Quantity-skew partition | `src/partitioning/partition.py` | 06 | Table 9 | 4 | Smoke |
| 31 | Mild label noise | 1.12 | Brief mention | 2% seeded flips on training labels only | `src/partitioning/noise.py` | 06 | Table 9 | 4 | Smoke |
| 32 | Per-client performance | 1.12 | Two sigma values, no source | Global model evaluated on every client's local test split | `src/metrics/classification.py`, server | all | Table 10, Fig 10 | 4 | Smoke |
| 33 | Worst-client performance | 1.12 | Missing | Min and 10th percentile client F1 | `src/metrics/fairness.py` | all | Table 10, Fig 11 | 4 | Smoke |
| 34 | Performance dispersion | 1.12 | Missing | Std, IQR, range across clients | `src/metrics/fairness.py` | all | Table 10 | 4 | Smoke |
| 35 | Harder dataset | 1.13 | Deferred | Amazon 5-class | `configs/amazon5.yaml`, `src/data/datasets.py` | 04, 05 | Table 6 | 4 | Not started (**decision needed**, see EXPERIMENT_PLAN 6) |
| 36 | Privacy discussion/mechanism | 1.14 | Limitation statement only | Threat model; DP with (epsilon, delta); secure aggregation if feasible | `src/privacy/` | 10 | Table 13, Fig 15 | 3, 5 | Not started |
| 37 | Convergence curves | 1.15 | Partially | Per-round loss/accuracy/macro-F1 | `src/analysis/` | all | Fig 4 | 4 | Code (per-round history logged; plotting not written) |
| 38 | Error bands | 1.15 | Declined | Mean +/- std and 95% CI bands across seeds | `src/analysis/` | all | Fig 5 | 4 | Not started |
| 39 | Convergence-to-target-F1 | 1.15 | Declined | Rounds and bytes to reach a predefined target | `src/metrics/convergence.py` | all | Fig 6, Table 11 | 4 | Code |
| 40 | Data-efficiency analysis | 1.15 | Declined | F1 vs fraction of training data | `experiments/` | 05 | Fig 7 | 4 | Not started |
| 41 | Separate Discussion section | 1.16 | Added, but claims unsupported | Rewrite from results incl. limitations | `manuscript/` | - | - | 5 | Not started |
| 42 | Correct figures/tables | 2.10, PI 8 | Caption/text contradictions; figures from RNG | Every figure and table generated from logs | `src/analysis/` | all | all | all | Not started |
| 43 | Clean references | 2.1, 2.6 | DOI mismatch (ref 7), swapped descriptions (21/22) | Verify each reference; target journal style | `manuscript/references` | - | - | References | Not started (**target journal needed**) |
| 44 | Reproducibility/code availability | 1.4, PI 1 | No code released | Pinned env, saved partitions, run metadata, public repo | `REPRODUCIBILITY.md`, `ENVIRONMENT.md` | 00 | - | Reproducibility statement | Smoke |
| 45 | Remove reviewer tags | PI 8 | `[1.2]`, `[2.5]` etc. in text | New manuscript written clean; automated check | `tests/` (to add with manuscript) | - | - | All | Not started |

## Additional reviewer points not in the 45

| Reviewer | Requirement | Plan |
|---|---|---|
| 1.10 | State deployment type, hardware, OS, network emulation, framework | `ENVIRONMENT.md` + automatic `run_metadata.json`; Flower simulation; no network emulation is claimed |
| 2.0 | Justify methods; compare multiple cases | Method records in `docs/NOVELTY_AUDIT.md`; staged benchmark |
| 2.2-2.4, 2.7 | Capitalization, abbreviations, abbreviation list | Manuscript pass, depends on target journal |
| 2.5 | Equations discussed and cited as "Equation n" | Manuscript pass |
| 2.8, 2.9 | Author biographies and contributions | Authors to supply |
| 2.11 | No copied images | All figures generated by `src/analysis` |

## Changes since Phase 0 (2026-10-02, Experiment 0 preparation)

| # | Change |
|---|---|
| 16, 21 | Experiment 0 is specified with 8 seeds (EXPERIMENT_PLAN amendment A2). The list has not been supplied; configs still hold the original five. Status unchanged |
| 35 | Harder dataset: Amazon Reviews 2023 reader (5 classes) implemented and tested on a synthetic file and one real file. Category not chosen. Status: Code |
| 32-34 | Per-client and worst-client metrics now also computed for held-out (unseen) clients in the natural-client path. Status: Smoke |
| 44 | Results schema with validation, immutable run directories and flat `runs.csv` / `rounds.csv` / `clients.csv`. Status: Smoke |
| - | New diagnostic experiments 0A and 0B prepared; see `docs/PHASE_0_TO_PHASE_1_AUDIT.md` for open decisions |
