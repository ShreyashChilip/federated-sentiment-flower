# extend_v1: heterogeneity sweep, robustness and DistilBERT (EXPLORATORY, pre-registered)

Written 2026-10-12, before any extend_v1 run. Nothing here changes fedbench_v1 (commit `3415496`): its configs,
selections and results stay frozen, and every extend_v1 result is stored under `results/exploratory/extend_v1/`
and reported as exploratory. Settings below were fixed before any result was seen; any later deviation will
be reported as such.

## Common rules
* Data: exactly the fedbench_v1 data. Yelp Polarity (all 560,000 official training reviews, roles 80/10/10 as in
  the benchmark, official 38,000-review test set held out); Amazon Reviews 2023 Video Games natural product clients
  (bundle `amazon2023_86405fdce84c`, unseen-product clients held out). No subsampling.
* Seeds: tuning seed 7 (validation only); evaluation seeds 42, 123, 456 (the fedbench_v1 screening seeds, so
  every run is paired with the official run of the same method and seed where one exists).
* Selection uses seen-client validation macro-F1 only. Held-out data (Yelp test, Amazon unseen clients) is
  read only for reporting; nothing is tuned or selected on it.
* Reporting: every completed run is reported, including failures and unfavourable results. Three seeds:
  mean, per-seed values and worst seed; no significance tests.
* Communication figures are model-based byte estimates (fedbench_v1 Eqs. 2-4 or the analogue stated below),
  not measured traffic. Wall times are simulation times, not deployment latency. No energy claims.

## Study A: label-heterogeneity sweep (TF-IDF + LR, CPU)
Configs `configs/exploratory/extend_v1/bench_alpha_{a005,a05,a1}.yaml`; cells in `yelp_cells.yaml`.
* Partitions: client-wise Dirichlet (`dirichlet_client`, equal-size clients, 100 clients) with alpha 0.05,
  0.5 and 1.0. Together with the official alpha 0.1 and IID regimes this gives five heterogeneity levels.
* Model and FL settings: the frozen Yelp settings (lr 0.3, wd 1e-6 from the 0A tuning; 50 rounds, C 0.25,
  E 1, batch 32).
* Methods: FedAvg, FedProx (mu in {0.001, 0.01, 0.1, 1}), FedAdagrad (server lr in {0.01, 0.03, 0.1, 0.3, 1}),
  tuned per alpha on validation with seed 7, ties as in fedbench_v1. 10 tuning + 9 evaluation runs per alpha.
* Outputs: held-out macro-F1 at rounds 10-50, client-test dispersion (mean, p10, worst), derived dense and
  support-only bytes, as in fedbench_v1.
* Analysis: held-out macro-F1 at round 50 versus alpha for each method (with the official alpha 0.1 and IID
  runs); FedProx/FedAdagrad minus FedAvg per alpha (paired by seed); support-only/dense byte ratio versus alpha.

## Study B: robustness (TF-IDF + LR, CPU)
Config `configs/exploratory/extend_v1/bench_robust.yaml`. No tuning: the frozen fedbench_v1 selections are reused
(copies with SHA-256 sums in `robust_selections/`; Dir-0.1 selections for B1/B2, IID selections for B3).
Methods FedAvg, FedProx, FedAdagrad; seeds 42, 123, 456; 9 runs per condition.
* B1 client dropout: official Dirichlet-0.1 regime; each selected client independently drops out of a round
  with probability 0.2 (`fl.dropout_prob`, seeded draws of the benchmark engine).
* B2 label noise: official Dirichlet-0.1 regime with 10% of TRAIN-role labels flipped uniformly to another
  class (`data.label_noise 0.1`, seeded per run); validation and held-out labels are clean.
* B3 quantity skew: IID labels, client sizes from Dirichlet(0.5) over clients (`quantity_skew`, 100 clients,
  minimum 10 reviews).
* Analysis: per method and seed, change in held-out macro-F1 at round 50 relative to the paired clean official
  run (B1, B2: yelp_dir01; B3: yelp_iid), plus client-test dispersion. B3 uses the IID selections, so its
  change mixes the effect of quantity skew with any loss from not re-tuning; this is stated with the result.

## Study C: DistilBERT, centralized and federated (GPU)
Config `configs/exploratory/extend_v1/bert.yaml`; script `experiments/distilbert_fl.py`; Kaggle launcher
`notebooks/kaggle_bert.py`; tests `tests/test_distilbert_fl.py`.
* Regimes: Yelp IID, Yelp Dirichlet-0.1, Amazon VG natural clients, resolved from the same configs and cells as
  fedbench_v1. Rounds (50), client fraction (0.25 Yelp, 0.10 Amazon), local epochs (1), batch size (32), client
  sampling, model-initialization and batch-order seeds are those of the benchmark.
* Data identity: Yelp roles and labels are checked against bundle `yelp_polarity_7355dff6b72f`; Amazon texts are
  read from the cleaned review store in the bundle's row order and their labels and client codes are checked
  against `amazon2023_86405fdce84c`. A mismatch aborts before any training.
  (Reconstruction checked locally on two synthetic Amazon bundles and on the official Yelp bundle.)
* Model: `distilbert-base-uncased` (resolved hub commit recorded per run), sequence-classification head,
  128-token truncation, fp16 autocast on GPU.
* Local optimizer, fixed a priori from standard fine-tuning practice and not tuned: AdamW, lr 5e-5, weight
  decay 0.01 (none on biases and LayerNorm), new optimizer state for every client in every round.
* Methods: FedAvg (sample-size-weighted averaging of all floating-point parameters); FedProx with mu tuned per
  regime over {0.001, 0.01, 0.1} on validation (seed 7; ties to the smaller mu; grid-edge selections flagged);
  centralized fine-tuning on the pooled TRAIN rows (3 epochs, batch 32) per dataset and seed.
* Evaluation: seen-client validation every 5 rounds and at round 50; held-out (Yelp official test, Amazon
  unseen-product clients) at rounds 10, 20, 30, 40, 50; per-client client-test macro-F1 dispersion at the end.
* Communication (derived): dense message = 4 bytes x all floating-point parameters. Support-only analogue:
  only the word-embedding rows of the client's token support are exchanged, all other parameters dense, plus
  the cheaper of int32 indices or a vocabulary bitmask. Exactness of support-only exchange is NOT claimed for
  this model (AdamW decoupled weight decay moves every row); the figure is an upper bound on what row sparsity
  could save, reported to contrast with the linear model.
* Runs: 9 tuning + 18 federated evaluation + 6 centralized runs = 33 runs.
* Tuning runs evaluate validation only; held-out data is never read in a tuning run.
* Compute: unknown until the pilot. The pilot is a timing and pipeline check, never reported as a result: per regime,
  on the full data, two FedProx rounds (mu 0.01, seed 42; FedProx steps cost at least as much as FedAvg steps) with
  a checkpoint, a genuine pause and a resume between them, then one validation, one held-out and one client-test
  pass. It records measured step time, per-client overhead, evaluation times, throughput, GPU and host memory and
  checkpoint size, and projects every planned run from these unit costs and the exact workload of that run
  (client sampling replayed). Assumptions and a range are written to `pilot.json`. If the projection exceeds the
  available quota, the author decides before any tuning run starts; the protocol is not shortened silently (no
  fewer rounds, clients, seeds or data).
* Analysis: held-out macro-F1 by round for FedAvg, FedProx and centralized per regime; federated minus
  centralized gap; FedProx minus FedAvg (paired by seed); comparison with the TF-IDF+LR runs on the same data
  and seeds (descriptive: model, optimizer and parameter count differ); derived bytes per round for both models.

## Integrity
* All configs, the script and tests are committed before any run; the commit is recorded in each run.
* fedbench_v1 artifacts are read-only inputs (selections copied with checksums; bundles used only for labels,
  roles and client codes).
* The confirm_v1 study is separate and unaffected.

## Amendments (all before any extend_v1 run)
* 2026-10-12 (after commit ab49d2b, before any launch): pilot redesigned to measure unit costs over two rounds with
  a real pause/resume and to project every planned run exactly; tuning runs no longer read held-out data; the
  support-size cache is stored in checkpoints. No study setting (data, regimes, model, optimizer, rounds, seeds,
  grids, evaluation points) changed.
* 2026-10-13 (after a pilot attempt that stopped at data preparation, before any training): the GPU launcher
  `notebooks/kaggle_bert.py` now installs every package the DistilBERT script imports (flwr pinned to the
  benchmark's 1.39.0, psutil, huggingface_hub) and checks the imports before any data work. Infrastructure only;
  no study setting changed. The failed attempt produced no result (0 runs) and used about one GPU-minute.
