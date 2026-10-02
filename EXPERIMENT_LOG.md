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
