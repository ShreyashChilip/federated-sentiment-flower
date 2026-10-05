# Experiment log

Newest entry last. Laptop entries are software checks, never paper results.

## 2026-10-02 - Phase 0 smoke test (laptop, NOT official)

* Script: `experiments/00_smoke_test.py`, config `configs/smoke.yaml`.
* Data: class-proportional 6,000-row subset of the Yelp Polarity train split
  (4,800 train / 600 val / 600 client-test roles) and 2,000 test rows; 5,000
  TF-IDF features; LR with 10,002 parameters.
* Setting: 3 clients, 2 rounds, C = 1, E = 1, batch 32, lr 0.5.
* Environment: Windows 11, Python 3.11.9, flwr 1.39.0, torch 2.14.1 (CPU).
* Outcome: 31 of 31 checks passed (including a stress run with C = 0.5, 30% dropout, 50% stragglers, LR decay, unequal client sizes and 2% label noise); 37 unit tests passed.

| Run (seed 42) | Test accuracy | Test macro-F1 | Uplink bytes | Downlink bytes |
|---|---|---|---|---|
| FedAvg, IID | 0.8095 | 0.8081 | 242,616 | 242,280 |
| FedProx (mu = 0.1), IID | 0.7355 | 0.7195 | 242,616 | 242,286 |
| FedAdam (server lr 0.1), IID | 0.8110 | 0.8109 | 242,616 | 242,286 |
| SCAFFOLD, IID | 0.8090 | 0.8075 | 484,212 | 483,888 |
| FedAvg, Dirichlet(0.5) | 0.5000 | 0.3333 | - | - |
| Centralized, 2 epochs | 0.7900 | 0.7850 | - | - |
| Local-only, Dirichlet(0.5), mean over clients | - | 0.3333 | - | - |

Verified:

* Same seed gives identical final-weight SHA-256, metrics and byte counts for
  all four algorithms; a different seed gives a different model.
* Byte counts match the model: 10,002 float32 parameters = 40,008 bytes per
  message plus a small config record; 3 clients x 2 rounds gives the totals above.
* SCAFFOLD transfers about twice the bytes (model plus control variate in both
  directions), and client control variates persist between rounds.

Observations (toy scale, 2 rounds, untuned; no conclusion may be drawn):

* On the Dirichlet(0.5) split the three clients were 94%, 99.7% and 98%
  single-class. FedAvg's global model predicted one class for every test
  review (confusion matrix [[0, 1000], [0, 1000]]), and each local-only model
  did the same. This is the label-skew failure the benchmark is meant to
  quantify, and it shows that with few clients a nominally "moderate" alpha
  can produce extreme skew: the realised per-client class counts, not alpha
  alone, must be reported (they are saved with every partition).
* FedProx with mu = 0.1 was behind FedAvg after 2 IID rounds. Expected, since
  the proximal term slows local progress and nothing is tuned; it is noted
  here so that it is not forgotten if the tuned results look different.

Next: Phase 0K (Kaggle bring-up), then the tuning phase.

## 2026-10-02 - Phase 1 script check (laptop, NOT official)

`experiments/01_baseline.py` was run on the smoke subset (5 clients, alpha 1.0,
5 rounds, seeds 42 and 123) only to exercise the runner, resume logic and
`summary.md` generation. Output under `results/_smoke_baseline/` (git-ignored).
This run exposed the Kaggle-detection bug recorded in `CHANGELOG.md`.

## 2026-10-02 - Experiment 0 preparation checks (laptop, NOT official)

* `python -m pytest tests -q`: all unit tests pass (37 from Phase 0 plus the
  new ones for natural clients, diagnostics, schema and guards).
* `experiments/00_smoke_test.py`: passes after the changes to trainer, client,
  server and run metadata.
* `experiments/00b_exp0_smoke.py`: 0A pilot on a 6,000-row Yelp subset with 5
  clients, and 0B pilot on a SYNTHETIC review file for both client
  definitions; all checks pass. Output under `results/_smoke_*` (git-ignored).
* Reader check on one real file (`Subscription_Boxes.jsonl`, 16,216 lines):
  parsed without error; 11 empty reviews and 208 duplicates dropped, 15,997
  kept; all five rating classes present. Only parser counts were looked at.
* FedAvg run with empty clients (8 clients, 5 of them with no data): the
  server selects only clients that have training rows; evaluation and
  metadata are written.
* Partitioner feasibility (a property of the code, not a result): with
  560,000 balanced labels, N = 100 and alpha = 0.1, 0 of 200 per-class
  Dirichlet draws give every client at least 10 rows.

No number from these checks may be reported.

## 2026-10-05/06 - Benchmark fedbench_v1: Kaggle job 1 and OOM check (engineering, NOT results)

* Job 1 (commit 10f519b, Kaggle CPU): Yelp IID and Dirichlet-0.1 pilots
  (seed 0, 3 rounds; FedAvg, SCAFFOLD, FedAdam, centralized, local-only)
  completed. `pilot/amazon_vg/fedavg/seed0` was killed (-9) while loading the
  natural-client partition: per-client slicing of `npz["indices"]` kept one
  full index copy per client (`CHANGELOG.md`). Stage stopped as designed;
  tuning never started. The Job 1 Amazon client diagnostics ran through the
  same loader and are treated as not produced.
* The Job 1 pilot analysis printed `"screening_outcome": "candidate"`: the
  screening rules were applied to pilot runs. That label has no meaning and
  is discarded; fixed so non-screening stages report `not_applicable`.
* OOM check (commit a3b07c5): `pilot/amazon_vg/fedavg/seed0` complete in
  2231 s, no OOM. Fix promoted to `fedbench`.
