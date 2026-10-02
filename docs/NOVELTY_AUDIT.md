# Novelty audit

Status on 2026-10-02: **no novelty claim is supported.** This audit records
what prior work already covers and what has not yet been found. Absence from
this audit means "not found by the searches listed in section 6", never
"does not exist".

History: first version 2026-10-02 at the repository root (commit 512364a);
this version replaces it and adds primary-source verification.

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
| Verified | ABS (AAAI page + README of github.com/Xianjie-Guo/FedCIFL) | PDF (IJCAI proceedings) | PDF (arXiv 2104.08815) | PDF (arXiv 2203.11635); venue by SEARCH |
| Problem | Federated feature selection that survives non-IID data and out-of-distribution clients | Attribute bias: local models learn non-causal associations | Benchmark of FL methods on NLP tasks | Generalizing a federated model to an unseen target domain |
| Client structure | Synthetic IID+OOD and non-IID+OOD settings (per the repository); "real-world datasets" in the abstract, unnamed | 10 clients on NICO-Animal / NICO-Vehicle, contexts unevenly distributed | Synthetic label-shift and quantity-shift Dirichlet partitions; natural factor only for MRQA (6 source datasets as 6 clients) | One client per source domain (3 client domains, 1 target) |
| Label/logit level | No | No | - | Pseudo-labels for the target domain by client voting |
| Feature/parameter level | **Feature level**: estimates a federated causal effect between each input feature and the label, treating the other features as confounders; sample reweighting | Representation level: counterfactual samples and causal prototypes | - | Representation level: feature distribution matching |
| Client-specific lexical/domain confounding | Spurious feature-label correlation from selection bias; not lexical in what was read | **Yes in concept**: the causal graph names the data source (client) and the image background as confounders; vision only | No | Domain shift between product categories |
| Aggregation | Not stated in what was read | Model averaging plus prototype aggregation | FedAvg, FedProx, FedOpt | Global model aggregation plus fine-tuning on the unlabeled target |
| Unseen clients | **Yes** (out-of-distribution, non-participating clients) | Test on unseen context distributions | Not found in the text | **Yes** (target domain) |
| Text / sentiment | Tabular binary features in the repository; downstream models include **logistic regression** | No (images) | Text yes (20News, OntoNotes, MRQA, Gigaword); **no sentiment dataset** | **Yes**: Amazon Review (Blitzer et al. 2007), 4 product domains, binary sentiment, BERT embeddings |
| Overlap with the candidate | **Partial, and the most serious**: feature-level, causal-invariance framing, OOD clients, linear downstream model | Related conceptual precedent ("client/source as confounder") | Benchmark precedent for the benchmark part of the paper | Related precedent for unseen-domain sentiment in FL |
| Potentially distinct | Sparse lexical features from natural user/product clients; correction applied in aggregation rather than as feature selection. **Cannot be asserted until the full paper is read** | Text; coordinate-level mechanism | Classical model, natural user/product clients, resource accounting, statistics | Client = user or product rather than product category; sparse lexical model; coordinate-level mechanism |

## 4. Additional work found during this audit

| Work | Verified | Why it matters |
|---|---|---|
| "Fisher-Informed Parameterwise Aggregation for Federated Learning with Heterogeneous Data" (FIPA), arXiv 2601.13608, 2026 | SEARCH | Replaces client-level scalar weights by parameter-specific Fisher weights in multi-round FL under heterogeneity. **Direct overlap with part B as a generic mechanism.** Must be read and used as a baseline |
| "An Element-Wise Weights Aggregation Method for Federated Learning" (EWWA-FL), arXiv 2404.15919, 2024 | SEARCH | A different aggregation proportion for each parameter. Direct overlap with "per-coordinate aggregation" |
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
  FedFisher, Fisher merging, FIPA, EWWA-FL; per-coordinate server scaling in
  FedAdam/FedYogi.
* "Deconfounding in federated learning" as a framing: FedDDL, FedCIFL.

**Partial overlap (same level of operation or same goal, different signal or data)**

* FedGMA: coordinate-wise aggregation motivated by invariant versus spurious
  mechanisms; the signal is sign agreement.
* FedCIFL: feature-level causal invariance with OOD clients and a linear
  downstream model; feature selection on tabular data.

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

## 7. Open items before any novelty statement

- [ ] Read FedCIFL in full (highest priority): datasets, what clients send, server rule, whether any text data is used.
- [ ] Read FIPA and EWWA-FL in full; decide which is the per-parameter baseline.
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
- FedCIFL: https://ojs.aaai.org/index.php/AAAI/article/view/33866 , https://github.com/Xianjie-Guo/FedCIFL
- FedDDL: https://www.ijcai.org/proceedings/2025/0677.pdf
- FedNLP: https://aclanthology.org/2022.findings-naacl.13/ , https://arxiv.org/abs/2104.08815
- FedKA: https://arxiv.org/abs/2203.11635
- FIPA: https://arxiv.org/abs/2601.13608
- EWWA-FL: https://arxiv.org/abs/2404.15919
- Participation gap: https://arxiv.org/abs/2110.14216
- KTEPS: https://arxiv.org/abs/2107.11956
- UserIdentifier: https://arxiv.org/abs/2110.00135
