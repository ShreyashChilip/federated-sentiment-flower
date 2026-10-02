# Novelty audit

Status on 2026-10-02: **no novelty claim is supported.** This audit records
what prior work already covers and what has not yet been found. Absence from
this audit means "not found by the searches listed in section 6", never
"does not exist".

History: first version 2026-10-02 at the repository root (commit 512364a);
the second version added primary-source verification; this third version
(2026-10-02) adds a full reading of FIPA, EWWA-FL and FedCIFL (section 3a) and
corrects the FedCIFL entry, which had been based on the abstract only.

## 1. The candidate mechanism being audited

Feature-level correction for client-specific lexical bias in federated
sentiment classification, with two parts:

* **A.** within-client label-prior conditioning (clients train with their
  local class prior added to the logits);
* **B.** feature-level, information-aware aggregation (per-coordinate weights
  on the server instead of one weight per client).

The method is **not implemented**. It will only be considered if Experiment 0B
shows the phenomenon it targets (docs/CONFOUNDING_DIAGNOSTIC.md, section 8).

## 2. How each source was verified

| Tag | Meaning |
|---|---|
| PDF | Full text downloaded from the official or preprint source and read for the fields below (2026-10-02) |
| ABS | Publisher abstract page and code repository read; full text **not** read |
| SEARCH | Bibliographic details confirmed by web search only; content not read |

## 3. The eight known papers

| | FedLC | FedDisco | FedGMA | FedFisher |
|---|---|---|---|---|
| Reference | Zhang, Li, Li, Xu, Wu, Ding, Wu. "Federated Learning with Label Distribution Skew via Logits Calibration". ICML 2022 (PMLR 162) | Ye, Xu, Wang, Xu, Chen, Wang. "FedDisco: Federated Learning with Discrepancy-Aware Collaboration". ICML 2023 | Tenison, Sreeramadas, Mugunthan, Oyallon, Rish, Belilovsky. "Gradient Masked Averaging for Federated Learning". TMLR 2023 | Jhunjhunwala, Wang, Joshi. "FedFisher: Leveraging Fisher Information for One-Shot Federated Learning". AISTATS 2024 (PMLR 238) |
| Verified | PDF (arXiv 2209.00189) | PDF (arXiv 2305.19229) | PDF (arXiv 2201.11986) | PDF (arXiv 2403.12329) |
| Problem | Label distribution skew | Category (label) distribution heterogeneity | Heterogeneous clients; averaging lets dominant clients impose spurious directions | One-shot FL (single communication round) |
| Client structure | Synthetic: quantity-based and distribution-based label skew; also LEAF Synthetic and FEMNIST | Synthetic: NIID-1 (Dirichlet), NIID-2 (biased client groups) | Synthetic label, feature and quantity skew; FEMNIST as the real-world federated set | Synthetic Dirichlet partitions |
| Label/logit level | **Yes.** Pairwise label margin added to logits in the local loss, margin = tau * (n_y^(-1/4) - n_i^(-1/4)) | No | No | No |
| Feature/parameter level | No | No. One scalar weight per client | **Yes.** One mask value per coordinate | **Yes.** Per-parameter Fisher weights (diagonal or K-FAC) |
| Client-specific lexical/domain confounding | No | No | Motivated by invariant vs spurious mechanisms; not lexical | No |
| Aggregation | Unchanged (weighted average) | p_k proportional to ReLU(n_k - a * d_k + b), d_k = discrepancy between local and global label distribution | A_j = abs(mean over clients of sign(update_j)); mask = 1 if A_j >= tau else A_j; global update = mask * mean update | Fisher-weighted combination of locally trained models, solved once |
| Unseen clients | No | No | Out-of-distribution test environment (FedCMNIST); not held-out natural clients | No |
| Text / sentiment | No (SVHN, CIFAR-10/100, ImageNet subset) | Text yes (AG News topic classification, TextCNN); sentiment no | No (MNIST, FMNIST, CIFAR, TinyImageNet, FEMNIST) | No (FashionMNIST, SVHN, CIFAR-10/100, CINIC-10) |
| Overlap with the candidate | **Direct** for part A | Related precedent: label-aware aggregation, but per client | **Partial** for part B: coordinate-wise aggregation with an invariance motive | **Direct** for the generic idea of information-weighted per-parameter aggregation |
| Potentially distinct | Nothing in part A | Per-feature rather than per-client weighting | The weighting signal (sign agreement) is not label-prior-aware and not derived from a within/between-client association | Multi-round use and a confounding-specific weighting signal (but see FIPA in section 4) |

| | FedCIFL | FedDDL | FedNLP | FedKA |
|---|---|---|---|---|
| Reference | Guo, Yu, Cui, Yu, Li. "Federated Causally Invariant Feature Learning". AAAI 2025, 39(16), 16978-16986. doi:10.1609/aaai.v39i16.33866 | Qi, Zhou, Meng, Hu, Yu, Meng. "Federated Deconfounding and Debiasing Learning for Out-of-Distribution Generalization". IJCAI 2025 | Lin, He, Ze, Wang, Hua, Dupuy, Gupta, Soltanolkotabi, Ren, Avestimehr. "FedNLP: Benchmarking Federated Learning Methods for Natural Language Processing Tasks". Findings of NAACL 2022 | Sun, Chong, Ochiai. "Feature Distribution Matching for Federated Domain Generalization". ACML 2022 |
| Verified | PDF (AAAI proceedings, full paper) | PDF (IJCAI proceedings) | PDF (arXiv 2104.08815) | PDF (arXiv 2203.11635); venue by SEARCH |
| Problem | Federated feature selection that survives non-IID data and out-of-distribution clients | Attribute bias: local models learn non-causal associations | Benchmark of FL methods on NLP tasks | Generalizing a federated model to an unseen target domain |
| Client structure | Synthetic: 3 to 20 clients with different bias rates and unequal sizes. Real: Amazon Review (Books, DVDs, Electronics, Kitchen); three product domains are the training clients, the fourth is the unseen test domain | 10 clients on NICO-Animal / NICO-Vehicle, contexts unevenly distributed | Synthetic label-shift and quantity-shift Dirichlet partitions; natural factor only for MRQA (6 source datasets as 6 clients) | One client per source domain (3 client domains, 1 target) |
| Label/logit level | No | No | - | Pseudo-labels for the target domain by client voting |
| Feature/parameter level | **Feature level**: estimates a federated causal effect between each input feature and the label, treating the other features as confounders; sample reweighting | Representation level: counterfactual samples and causal prototypes | - | Representation level: feature distribution matching |
| Client-specific lexical/domain confounding | **Yes**: features whose relation to the label differs across clients/domains ("irrelevant features") are identified and removed; on Amazon Review these are lexical features | **Yes in concept**: the causal graph names the data source (client) and the image background as confounders; vision only | No | Domain shift between product categories |
| Aggregation | No model aggregation rule of its own. Clients send thresholded causal-effect vectors and candidate irrelevant-feature sets; the server fixes the set size by rank-weighted voting and removes the features with the smallest summed effect; repeated until nothing is removed. A standard FL model is then trained on the kept features | Model averaging plus prototype aggregation | FedAvg, FedProx, FedOpt | Global model aggregation plus fine-tuning on the unlabeled target |
| Unseen clients | **Yes** (out-of-distribution, non-participating clients) | Test on unseen context distributions | Not found in the text | **Yes** (target domain) |
| Text / sentiment | **Yes, sentiment**: Amazon Review cross-domain sentiment (about 1,000 positive and 1,000 negative reviews per domain, preprocessed binary features); downstream **logistic regression** and MLP | No (images) | Text yes (20News, OntoNotes, MRQA, Gigaword); **no sentiment dataset** | **Yes**: Amazon Review (Blitzer et al. 2007), 4 product domains, binary sentiment, BERT embeddings |
| Overlap with the candidate | **Direct for the problem statement** (client-unstable lexical features in federated sentiment, evaluated on unseen clients, linear model); **partial for the mechanism** (feature selection before training, not aggregation) | Related conceptual precedent ("client/source as confounder") | Benchmark precedent for the benchmark part of the paper | Related precedent for unseen-domain sentiment in FL |
| Potentially distinct | Client = individual user or product instead of product category; thousands of small clients instead of 3; 5-class ratings; soft per-coordinate treatment during aggregation instead of hard removal; a within/across-client association decomposition as the diagnostic. None of these has been shown to matter | Text; coordinate-level mechanism | Classical model, natural user/product clients, resource accounting, statistics | Client = user or product rather than product category; sparse lexical model; coordinate-level mechanism |

## 3a. Full reading of FIPA, EWWA-FL and FedCIFL (2026-10-02)

Read from the PDFs in full (method, experiments, limitations). The candidate
mechanism is **not implemented**; the comparison is against its description
in section 1.

| | FIPA | EWWA-FL | FedCIFL |
|---|---|---|---|
| Problem | Client drift under non-IID data; one scalar weight per client ignores which parameters a client's data actually constrains | One aggregation proportion per client model ignores that each parameter converges differently | Federated feature selection that holds up under non-IID clients and out-of-distribution (unseen) clients |
| What the client sends | Model update plus the top-r eigenpairs of its local Fisher / generalized Gauss-Newton matrix, computed at the broadcast model | Local gradients | Per-feature causal-effect estimates (weighted logistic regression after confounder-balancing sample reweighting in a supervised-autoencoder space) and its set of low-effect features |
| Server rule | theta <- theta + sum_m B_m * delta_m, with matrix weights B_m = (N_m / N) * pinv(H) * H_m and H = sum_m (N_m / N) * H_m. Low-rank, **not diagonal**; equals a centralized Gauss-Newton step when local solves are exact | Per client and per element: Adam-style moments of that client's gradient give b = alpha * m_hat / sqrt(v_hat + eps); proportions p = softmax over clients of b, element by element; global update = sum_c p_c * g_c | Voting on how many features to drop (ties broken by a rank of clients' autoencoder losses), then dropping the features with the smallest summed effect; iterated |
| Level of operation | Parameter space, with cross-parameter coupling | Individual parameter | Input feature (original feature space) |
| Signal used for weighting | Curvature / parameter identifiability from each client's data | Magnitude and stability of each client's gradient element | Stability of the feature-label relation after balancing confounders |
| Uses label priors or logit correction | No | No | No |
| Targets client-specific spurious features | No. It targets optimization misalignment | No | **Yes** |
| Client structure in experiments | Function fitting and PINNs with domain splits; image classification with 100 clients, 5% participation, Dirichlet label skew | 3 clients, IID and non-IID image splits | 3 to 20 synthetic clients; 3 Amazon product domains as clients |
| Unseen clients | No | No | **Yes** (held-out domain; cites the participation-gap framing) |
| Text / sentiment | No (regression, PDEs, CIFAR-10/100, Tiny-ImageNet) | No (MNIST, CIFAR-10/100, ILSVRC2012) | **Yes** (Amazon Review sentiment, binary) |
| Stated limitations | Extra computation and communication for eigenpairs; used as a short refinement after a long FedAvg/FedRCL warm-up; Fisher signals may leak more than updates | Not discussed | Binary features; binary classification; multi-label left to future work |
| Overlap with part A (label-prior conditioning) | None | None | None |
| Overlap with part B (feature-level, information-aware aggregation) | **Direct.** Fisher-informed parameter-wise aggregation in multi-round FL under heterogeneity is exactly the generic form of part B, and in a more general (non-diagonal) version | **Direct.** Element-wise aggregation proportions are exactly "per-coordinate weights on the server" | Partial: per-feature decisions made across clients, but as removal before training rather than as aggregation weights |
| Overlap with the motivating problem (client-specific lexical bias, unseen clients) | None | None | **Direct**, including the dataset family (Amazon sentiment) and a linear downstream model |

### What this reading rules out

The following may **not** be claimed as a contribution of this project:

1. per-parameter (parameter-wise) aggregation as such: FIPA, FedFisher, Fisher merging;
2. element-wise aggregation weights as such: EWWA-FL, and per-coordinate server scaling in FedAdam / FedYogi;
3. Fisher-weighted or information-weighted aggregation as such: FIPA, FedFisher, Fisher merging;
4. client-weighted aggregation by label statistics: FedDisco; by update alignment: FedAWA, FedAdp;
5. coordinate masking by client agreement to favour invariant directions: FedGMA;
6. local label-prior or logit correction: logit adjustment, FedLC;
7. the observation that some features have client-unstable relations with the label, that removing their influence helps unseen clients, and that this applies to federated sentiment on Amazon reviews with a logistic-regression model: FedCIFL;
8. "deconfounding" or "causal" federated learning as a framing: FedCIFL, FedDDL.

### What would have to be true for anything to remain

Only a narrow combination is not covered by what was read, and each part
needs evidence before it can be stated:

* the **signal**: weighting or correcting a coordinate by how far its pooled
  association with sentiment departs from its within-client association
  (docs/CONFOUNDING_DIAGNOSTIC.md), which is neither curvature (FIPA),
  gradient moments (EWWA-FL), sign agreement (FedGMA) nor a balanced causal
  effect (FedCIFL);
* the **client granularity**: thousands of individual users or products
  rather than a handful of product-category domains;
* the **setting**: sparse lexical models where every coordinate is a named
  term, with 5-class ratings.

A method built on this would have to be compared against FIPA (or a diagonal
Fisher variant if the full version is infeasible at 50,000 features), EWWA-FL,
FedGMA, FedAdam, FedLC and FedCIFL-style feature selection, and it would have
to beat them on the specific failure that Experiment 0B measures. If 0B does
not show that failure, there is nothing to build, and the contribution is the
benchmark, the diagnostic and the negative result.

## 4. Additional work found during this audit

| Work | Verified | Why it matters |
|---|---|---|
| Chang, He, Hao. "Fisher-Informed Parameterwise Aggregation for Federated Learning with Heterogeneous Data" (FIPA), arXiv 2601.13608, January 2026 | PDF (see 3a) | Replaces client-level scalar weights by parameter-specific Fisher weights in multi-round FL under heterogeneity. **Direct overlap with part B as a generic mechanism.** Must be read and used as a baseline |
| Hu, Ren, Hu, Deng, Xie. "An Element-Wise Weights Aggregation Method for Federated Learning" (EWWA-FL), arXiv 2404.15919, April 2024 | PDF (see 3a) | A different aggregation proportion for each parameter. Direct overlap with "per-coordinate aggregation" |
| Matena and Raffel, "Merging Models with Fisher-Weighted Averaging", NeurIPS 2022 | SEARCH | Per-parameter Fisher-weighted averaging of NLP models |
| Reddi et al., "Adaptive Federated Optimization", ICLR 2021 (FedAdagrad, FedAdam, FedYogi) | memory, venue to confirm | Server already rescales each coordinate; implemented here as FedAdam; must be a baseline for part B |
| Menon et al., "Long-tail learning via logit adjustment", ICLR 2021 | memory, venue to confirm | Origin of part A |
| Yuan, Morningstar, Ning, Singhal, "What Do We Mean by Generalization in Federated Learning?", ICLR 2022 | SEARCH | Defines the participation gap between participating and non-participating clients on natural-client datasets. The seen/unseen protocol of 0B follows this framing and must cite it |
| "Preliminary Steps Towards Federated Sentiment Classification" (KTEPS), arXiv 2107.11956 | SEARCH | Federated multi-domain sentiment classification. To read |
| "UserIdentifier: Implicit User Representations for Simple and Effective Personalized Sentiment Analysis", arXiv 2110.00135 | SEARCH | User-level sentiment on Yelp, IMDB and Sentiment140, including federated setups. Shows user-level structure in sentiment data is known. To read |
| FedAWA, CVPR 2025 | SEARCH | Adaptive aggregation weights from client update vectors (client-level) |
| FuseFL, NeurIPS 2024 | SEARCH | One-shot FL; isolated local models fit spurious correlations |
| Gholamiangonabadi and Grolinger, "Federated learning for sentiment analysis in the presence of non-IID data", IEEE Access 2024 | from the old reference list, DOI to verify | Closest prior study to the benchmark framing |

## 5. Classification

**Direct overlap (already published; cannot be claimed)**

* Part A, local label-prior / logit correction: logit adjustment and FedLC.
* Part B as a generic idea, per-parameter information-weighted aggregation:
  FedFisher, Fisher merging, FIPA (read in full), EWWA-FL (read in full);
  per-coordinate server scaling in FedAdam/FedYogi.
* The motivating problem itself, client-unstable lexical features in
  federated sentiment with unseen clients and a linear model: FedCIFL (read
  in full).
* "Deconfounding in federated learning" as a framing: FedDDL, FedCIFL.

**Partial overlap (same level of operation or same goal, different signal or data)**

* FedGMA: coordinate-wise aggregation motivated by invariant versus spurious
  mechanisms; the signal is sign agreement.
* FedCIFL, for the mechanism: it removes features before training; it does
  not weight coordinates during aggregation.

**Related conceptual precedent**

* FedDDL (client/source as a confounder, vision), FedKA (unseen-domain
  sentiment, representation alignment), FedDisco (label-aware aggregation per
  client), Yuan et al. (participation gap), UserIdentifier and KTEPS
  (user/domain structure in sentiment).

**Potentially distinct components (not found so far; not a novelty claim)**

1. A within-client versus across-client decomposition of term-sentiment
   association, measured on natural user-level and product-level clients, as
   a diagnostic for federated text models.
2. Using that decomposition, rather than Fisher information or sign
   agreement, as the per-feature signal for aggregation.
3. Doing so for sparse, interpretable lexical models where each coordinate is
   a named term, together with a seen/unseen-client evaluation.

Each of the three is only "not found by the searches below". Items 2 and 3
additionally depend on Experiment 0B showing that the phenomenon exists. If
0B is negative, the defensible contribution is the benchmark plus the
diagnostic and its negative result.

## 6. Search log

| Date | Query topic | Outcome |
|---|---|---|
| 2026-10-02 | FedLC; FedDisco; FedFisher and Fisher merging; FedGMA | Confirmed; PDFs read |
| 2026-10-02 | FL text classification, spurious lexical features, deconfounding | FedCIFL, FedDDL |
| 2026-10-02 | FedNLP | Confirmed; PDF read |
| 2026-10-02 | Coordinate-wise aggregation for sparse TF-IDF features | Only Byzantine-robust coordinate-wise methods |
| 2026-10-02 | Federated domain generalization, unseen clients, Amazon sentiment | FedKA |
| 2026-10-02 | Federated sentiment with natural user clients (Amazon/Yelp user ids) | KTEPS, UserIdentifier, LEAF Sentiment140 mentioned |
| 2026-10-02 | FL, spurious correlation, text shortcuts, client-specific lexical bias | FuseFL; centralized text-shortcut papers; no federated lexical-bias paper surfaced |
| 2026-10-02 | Participation gap | Yuan et al., ICLR 2022 |
| 2026-10-02 | Per-coordinate / element-wise aggregation weights | FIPA, EWWA-FL, FedAWA, FedLAW |
| 2026-10-02 | Full-text reading | FIPA (arXiv v1, 17 pages), EWWA-FL (arXiv v1, 9 pages), FedCIFL (AAAI, 9 pages; appendix not available in the proceedings PDF) |

## 7. Open items before any novelty statement

- [x] Read FedCIFL in full (done 2026-10-02; it does use Amazon Review sentiment).
- [x] Read FIPA and EWWA-FL in full (done 2026-10-02).
- [ ] Decide the per-parameter baselines: FIPA needs eigenpairs of a 100,002-parameter model per client per round; a diagonal-Fisher variant may be the feasible stand-in and must then be labelled as such.
- [ ] Check the FedCIFL appendix and the Wang et al. (2018) preprocessing it uses for the Amazon data (feature dimension and type), and its code, before designing a FedCIFL-style baseline.
- [ ] Read KTEPS and UserIdentifier; check their client construction and whether unseen users are evaluated.
- [ ] Search the ACL Anthology and Google Scholar (not yet done; only a general web index was used) for: federated sentiment with per-user clients; participation gap in text; FedRS; federated logit adjustment in NLP; word-level or embedding-row aggregation; fixed-effects or within-estimator ideas in FL; "Simpson's paradox" in FL.
- [ ] Check 2025-2026 arXiv listings for the same terms.
- [ ] Confirm venue and DOI of every row marked "memory" or "SEARCH" on the publisher page before citing.
- [ ] Write a short literature record for each baseline before implementing it (FedLC full margin, FedDisco, Fisher-weighted/FIPA, FedGMA).
- [ ] Re-audit against the phenomenon actually observed after Experiment 0B.

## 8. Sources

- FedLC: https://proceedings.mlr.press/v162/zhang22p.html , https://arxiv.org/abs/2209.00189
- FedDisco: https://arxiv.org/abs/2305.19229
- FedGMA: https://arxiv.org/abs/2201.11986
- FedFisher: https://proceedings.mlr.press/v238/jhunjhunwala24a.html , https://arxiv.org/abs/2403.12329
- FedCIFL: https://ojs.aaai.org/index.php/AAAI/article/view/33866 , https://ojs.aaai.org/index.php/AAAI/article/download/33866/36021 , https://github.com/Xianjie-Guo/FedCIFL
- FedDDL: https://www.ijcai.org/proceedings/2025/0677.pdf
- FedNLP: https://aclanthology.org/2022.findings-naacl.13/ , https://arxiv.org/abs/2104.08815
- FedKA: https://arxiv.org/abs/2203.11635
- FIPA: https://arxiv.org/abs/2601.13608
- EWWA-FL: https://arxiv.org/abs/2404.15919
- Participation gap: https://arxiv.org/abs/2110.14216
- KTEPS: https://arxiv.org/abs/2107.11956
- UserIdentifier: https://arxiv.org/abs/2110.00135
