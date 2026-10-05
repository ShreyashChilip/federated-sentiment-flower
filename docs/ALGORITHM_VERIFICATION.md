# Algorithm verification against original sources

Every federated algorithm in the benchmark (`src/benchmark/algorithms.py`) was
implemented from the original paper, read on 2026-10-05 from the arXiv PDF
(text extracted from the PDF, not from a secondary description). Notation:
x = global model, y_i = model of client i after local training,
Delta_i = y_i - x, n_i = training examples of client i, n = sum over the
reporting clients, tau_i = local SGD steps of client i, S = reporting clients,
N = all training clients.

| Algorithm | Source (read) | Client side | Server side | Implemented as |
|---|---|---|---|---|
| FedAvg | McMahan et al., AISTATS 2017 | SGD, E epochs | x <- sum_i (n_i / n) y_i | `weighted_average`, unchanged from Phase 0 |
| FedProx | Li et al., MLSys 2020, arXiv 1812.06127, Alg. 2 and Sec. 5 | minimize F_i(w) + (mu/2)||w - x||^2 with SGD | Alg. 2 samples devices with probability p_i and averages uniformly; the paper's experiments (Sec. 5) sample uniformly and average weighted by data size | uniform sampling, data-size-weighted average (= the experiments of the paper); proximal gradient mu (w - x) added at every step |
| FedNova | Wang et al., NeurIPS 2020, arXiv 2007.07481, eq. (4), (6), Sec. 5 | vanilla SGD: a_i = [1, ..., 1], ||a_i||_1 = tau_i | x <- x - tau_eff sum_i w_i eta d_i with eta d_i = (x - y_i) / tau_i, w_i = p_i, tau_eff = sum_i p_i tau_i (default). Under uniform sampling without replacement p_i is re-scaled over the sampled clients (paper, footnote to Sec. 3) | p_i = n_i / n over reporting clients; tau_i = SGD steps actually taken (stragglers take fewer); weight decay is part of the stochastic gradient, so a_i stays all-ones |
| SCAFFOLD | Karimireddy et al., ICML 2020, arXiv 1910.06378, Alg. 1 | y_i <- y_i - eta_l (g_i(y_i) - c_i + c); option II: c_i+ = c_i - c + (x - y_i) / (K eta_l) | (Delta x, Delta c) = (1/|S|) sum over S; x <- x + eta_g Delta x; c <- c + (|S| / N) Delta c | unweighted mean as in Alg. 1; option II; eta_g = 1; client control variates persist on disk between participations |
| FedAdagrad / FedYogi / FedAdam | Reddi et al., ICLR 2021, arXiv 2003.00295, Alg. 2 and Alg. 5 (the version used in all experiments) | SGD | Delta = sum_i (n_i / n) Delta_i (Alg. 5); m = b1 m + (1 - b1) Delta; Adagrad v = v + Delta^2; Yogi v = v - (1 - b2) Delta^2 sign(v - Delta^2); Adam v = b2 v + (1 - b2) Delta^2; x <- x + eta m / (sqrt(v) + tau). Initialization v_{-1} >= tau^2, m_{-1} = 0. No bias correction | v initialized to tau^2; b1 = b2 = 0 for FedAdagrad (App. D: "as typical versions of Adagrad do not use momentum"); b1 = 0.9, b2 = 0.99 for FedAdam and FedYogi; tau = 1e-3 (fixed in all experiments of the paper); server learning rate eta tuned on validation |

## Deviations from the Phase 0 implementation

* Phase 0 FedAdam initialized v at 0. The paper requires v_{-1} >= tau^2;
  the benchmark initializes v = tau^2. The Phase 0 Flower code is not
  changed (0A did not use FedAdam).

## Fairness rules shared by all algorithms

* Same partition, client sampling stream, dropout/straggler stream,
  per-client batch-order seed, zero initialization, local optimizer
  (plain SGD, momentum 0), client learning rate, weight decay, batch size,
  local epochs and number of rounds.
* Client learning rate and weight decay: tuned once with FedAvg (frozen
  protocol) and inherited by every algorithm.
* Algorithm-specific hyperparameters (FedProx mu; server learning rate of
  FedAdagrad/FedAdam/FedYogi) get a predefined grid, tuned with the tuning
  seed on seen-client validation macro-F1 only. FedNova and SCAFFOLD have no
  extra hyperparameter in their default form (tau_eff default; eta_g = 1).
