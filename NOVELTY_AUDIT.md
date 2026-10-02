# Novelty audit

Status: **seeded, not complete.** No novelty claim is supported yet, and none
may be written in the manuscript until section 5 is closed.

Verification column: `searched` = bibliographic details confirmed by a web
search on 2026-10-02 (link given); `memory` = well-known work recorded from
memory whose venue/DOI must still be checked against the publisher page before
it is cited.

## 1. Candidate idea under audit

Feature-level correction for client-specific lexical confounding in federated
sentiment classification: (A) within-client label-prior conditioning plus
(B) feature-level, information-aware aggregation.

Neither component is new on its own (rows 1, 2, 4, 5, 6 below). A contribution
could only be the combination, aimed at an **empirically demonstrated**
failure mode with natural clients. Whether that failure mode exists is what
Experiments 0A and 0B test.

## 2. Closest prior work

| # | Work | Year, venue | Problem | Data / clients / model | What it does | What it does not do | Difference from the candidate | Verified |
|---|---|---|---|---|---|---|---|---|
| 1 | Zhang et al., "Federated Learning with Label Distribution Skew via Logits Calibration" (FedLC) | 2022, ICML (PMLR 162) | Label skew | CIFAR-10/100, ImageNet subset; synthetic skew; CNNs | Calibrates logits by local class frequency with a pairwise margin | No text; nothing feature-level; synthetic clients | Covers component A almost entirely | searched: https://proceedings.mlr.press/v162/zhang22p.html |
| 2 | Ye et al., "FedDisco: Federated Learning with Discrepancy-Aware Collaboration" | 2023, ICML | Label skew | Image benchmarks; synthetic skew | Client aggregation weight uses dataset size and local-vs-global label discrepancy | One scalar weight per client, not per feature; no text | Client-level, not coordinate-level | searched: https://arxiv.org/abs/2305.19229 |
| 3 | Tenison et al., "Gradient Masked Averaging for Federated Learning" (FedGMA) | 2023, TMLR | Heterogeneity, invariance | Image and real-world federated sets | Per-coordinate mask from sign agreement across client updates | Agreement is binary sign voting; no label-prior conditioning; no sparse-text analysis | **Closest to component B**: it is already a coordinate-wise, invariance-motivated aggregation | searched: https://arxiv.org/abs/2201.11986 |
| 4 | Jhunjhunwala et al., "FedFisher: Leveraging Fisher Information for One-Shot Federated Learning" | 2024, AISTATS (PMLR 238) | One-shot FL | Image benchmarks; neural nets | Fisher-weighted (diagonal, K-FAC) combination of client models | One-shot setting; not targeted at confounding | Information-weighted per-parameter aggregation exists | searched: https://proceedings.mlr.press/v238/jhunjhunwala24a.html |
| 5 | Matena and Raffel, "Merging Models with Fisher-Weighted Averaging" | 2022, NeurIPS | Model merging | NLP transformers | Per-parameter Fisher-weighted average | Not federated rounds | Same aggregation rule as a Fisher baseline | memory (surfaced in search; venue to confirm) |
| 6 | Menon et al., "Long-tail learning via logit adjustment" | 2021, ICLR | Class imbalance | Centralized | Adds log-prior to logits | Not federated | Origin of component A | memory |
| 7 | "Federated Causally Invariant Feature Learning" (FedCIFL) | 2025, AAAI | Spurious correlation, OOD | Tabular/image per abstract (to confirm) | Each client estimates feature-label causal effect treating other features as confounders; sample reweighting | Text and natural-client sentiment not confirmed | **High overlap in framing** ("features that are predictive but unstable across clients"); must be read in full | searched: https://ojs.aaai.org/index.php/AAAI/article/view/33866 |
| 8 | "Federated Deconfounding and Debiasing Learning for Out-of-Distribution Generalization" (FedDDL) | 2025, IJCAI | Confounding in FL | Vision | Structural causal graph, backdoor adjustment | Vision; no lexical features | Deconfounding in FL already claimed | searched: https://www.ijcai.org/proceedings/2025/0677.pdf |
| 9 | Lin et al., "FedNLP: Benchmarking Federated Learning Methods for Natural Language Processing Tasks" | 2022, Findings of NAACL | FL benchmark for NLP | Text classification and other tasks; synthetic non-IID partitions; transformers | Benchmarks FedAvg, FedProx, FedOpt | No resource/energy accounting; no natural-client sentiment; no classical models | Our benchmark must be positioned against it | searched: https://aclanthology.org/2022.findings-naacl.13/ |
| 10 | Sun et al., "Feature Distribution Matching for Federated Domain Generalization" (FedKA) | 2022, ACML | Unseen-domain generalization | Includes Amazon Review sentiment (domains as clients) | Aligns feature distributions | Domain = product category, not user; neural features | Prior federated-DG result on Amazon sentiment | searched: https://arxiv.org/abs/2203.11635 |
| 11 | Gholamiangonabadi and Grolinger, "Federated learning for sentiment analysis in the presence of non-IID data" | 2024, IEEE Access | Non-IID sentiment FL | Sentiment data; deep models | Sensitivity of deep models to non-IID | (to read) | Closest prior study to our benchmark framing; cited in the old manuscript as [18] | memory (from old reference list; DOI to verify) |
| 12 | Li et al., FedProx | 2020, MLSys | Statistical and systems heterogeneity | - | Proximal term, partial work | - | Baseline | memory |
| 13 | Karimireddy et al., SCAFFOLD | 2020, ICML | Client drift | - | Control variates | - | Baseline | memory |
| 14 | Reddi et al., "Adaptive Federated Optimization" (FedAdagrad/FedAdam/FedYogi) | 2021, ICLR | Server optimization | - | Adaptive server step, per-coordinate scaling | Not label- or exposure-aware | Per-coordinate server scaling already exists; must be a baseline for B | memory |
| 15 | Wang et al., FedNova | 2020, NeurIPS | Objective inconsistency | - | Normalized averaging | - | Baseline | memory |
| 16 | Hsu et al., "Measuring the Effects of Non-Identical Data Distribution" | 2019, arXiv | Dirichlet partitioning | - | The partition recipe used here | - | Method reference | memory |
| 17 | Yuan et al., "What Do We Mean by Generalization in Federated Learning?" | 2022, ICLR | Participation gap, unseen clients | Natural-client datasets | Separates out-of-sample and participation gaps | No lexical analysis | Defines the seen/unseen-client protocol to follow in Phase 6 | memory |

## 3. Observations so far

1. Component A is prior art (rows 1, 6). It can only be a baseline or an
   ingredient.
2. Component B has two strong precedents: sign-agreement masking (row 3) and
   Fisher weighting (rows 4, 5). FedAdam/FedYogi (row 14) already rescale
   coordinates. A new per-feature rule has to beat all three on the specific
   failure mode, not just FedAvg.
3. Causal/deconfounding framing in FL is already claimed (rows 7, 8). The word
   "deconfounding" should not be used unless the method has an explicit causal
   estimand; "client-specific lexical bias" is the safer description.
4. A targeted search for coordinate-wise aggregation driven by **feature
   document frequency / exposure for sparse text models** found no direct
   match. That is weak evidence of a gap, not proof: one search, general web
   index. It also may simply mean the setting (sparse linear text models in
   FL) is of limited interest, which the paper would have to argue against.
5. Independent of any method, an honest, fully logged resource and statistics
   benchmark of classical vs neural vs PEFT text models under FL, with natural
   clients, is not what FedNLP (row 9) provides. That supports Outcome A or B
   as a defensible fallback.

## 4. Search log

| Date | Query topic | Result |
|---|---|---|
| 2026-10-02 | FedLC | Confirmed ICML 2022 |
| 2026-10-02 | FedDisco | Confirmed ICML 2023 |
| 2026-10-02 | FedFisher, Fisher merging | Confirmed AISTATS 2024; Fisher merging surfaced |
| 2026-10-02 | FedGMA | Confirmed TMLR 2023 |
| 2026-10-02 | FL text, spurious lexical features, deconfounding | Found FedCIFL (AAAI 2025), FedDDL (IJCAI 2025) |
| 2026-10-02 | FedNLP | Confirmed Findings of NAACL 2022 |
| 2026-10-02 | Coordinate-wise aggregation for sparse TF-IDF features | No direct match (Byzantine-robust coordinate-wise methods only) |
| 2026-10-02 | Federated DG, unseen clients, Amazon sentiment | Found FedKA (ACML 2022) |

## 5. Still to do before any novelty statement

- [ ] Read FedCIFL, FedDDL and FedGMA in full; record datasets, client construction and exact method.
- [ ] Search ACL Anthology and Google Scholar for: federated sentiment with user-level clients; "participation gap" text; per-user Amazon/Yelp federated benchmarks; FedRS; FedLA / federated logit adjustment in NLP; word-level or embedding-row aggregation; feature-frequency-aware aggregation; federated sparse linear models.
- [ ] Check 2025-2026 arXiv listings for the same terms.
- [ ] Verify every `memory` row against the publisher page.
- [ ] Write a short literature record for each additional baseline before it is implemented (FedNova, FedLC, FedDisco, Fisher-weighted, FedGMA).
- [ ] Re-audit after Experiments 0A and 0B, against the phenomenon actually observed.
