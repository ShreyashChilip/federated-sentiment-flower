# Audit of existing project materials

Audited 2026-10-02. Sources live one directory above this project
(`D:\sem5\last-sem-dt-changes\`). Plain-text extractions of the three
authoritative documents are in `docs/source_text/`.

## 1. Documents

| File | What it is | Authoritative? |
|---|---|---|
| `Resource-Efficient Sentiment Classification through FL-IJATEE.docx` | Revised manuscript submitted to IJATEE (7,962 words, 15 images, reviewer tags still inline) | Yes (original manuscript) |
| `Revisions.docx` | Response-to-reviewers table: Reviewer 2 (2.0-2.11, editorial) and Reviewer 1 (1.1-1.16, technical) | Yes |
| `Planned Improvements.docx` | Self-assessment of shortcomings and the rebuild plan | Yes |
| `revision-plan.docx`, `1.2.docx`, `new-literature-review.docx` | Earlier drafting notes for the response and the introduction | Background only |
| `prevdoc.docx`, `doc2.docx`, `dpccccc.docx` | Earlier manuscript variants | Superseded |
| `latest_changes.docx`, `yooo changes from claude, absolute class.docx` | Drafted reviewer responses | **Not usable, see 4.3** |

## 2. The manuscript

Claims: FedProx outperforms FedAvg on Yelp Polarity with TF-IDF + logistic
regression over 10 simulated clients; FL reaches near-centralized accuracy with
lower CPU use and energy; the framework is "privacy-preserving".

Setup as written: custom (non-framework) FedAvg/FedProx, 10 clients simulated
sequentially on one Windows laptop, C = 0.5, E = 5, batch 32, SGD momentum 0.9,
seeds 42/123/456, Dirichlet alpha in {0.1, 0.5, 1.0, 10.0}.

## 3. Reviewer requirements

Reviewer 1 (technical) asked for: seeded partition script and corrected
totals (1.4); >= 3 heterogeneity levels (1.5); full FL hyperparameters (1.6);
centralized, local-only and transformer baselines (1.7); >= 3 seeds with
mean/std/95% CI and paired tests (1.8); bytes up/down, time, energy proxy
(1.9); deployment description (1.10); sensitivity to mu, C, E, gamma (1.11);
dropout, unbalanced sizes, label noise, per-client dispersion (1.12); a harder
dataset (1.13); threat model plus DP/secure aggregation or an explicit
limitation (1.14); per-round curves with error bands, convergence-to-target-F1,
data-efficiency plots (1.15); a separate Discussion (1.16); English (1.1);
structured Introduction (1.2); related work with gap analysis (1.3).

Reviewer 2 (editorial) asked for: method justification (2.0), reference
hygiene and journal style (2.1, 2.6), capitalization and abbreviations
(2.2-2.4, 2.7), equations referenced in text (2.5), figures/tables discussed
before they appear (2.10), no copied images (2.11), author material (2.8, 2.9).

Several were answered in prose only or declined: 1.7 (transformers deferred),
1.8 (no CI, no test), 1.13 (deferred), 1.14 (limitation statement only),
1.15 (declined). The full mapping is in `REQUIREMENTS_MATRIX.md`.

## 4. Problems found

### 4.1 Numbers in the manuscript that do not agree with each other

| Item | Manuscript says | Conflict |
|---|---|---|
| Training set size | 418,600 (70% of 598k) | Table 1 client counts sum to exactly 560,000, the official Yelp train split |
| Best F1 | 92.73% (Table 4), "approx. 92.7%" | Seed-averaged text reports 0.83 F1 at alpha = 0.5 and 0.67 at alpha = 0.1 |
| Global LR decay gamma | 0.98 (Section 3) | 0.95 (after Equation 8) |
| Communication | "38.15 MB per experiment" | "38.15 MB per uplink"; 3.8 GB total matches neither |
| Energy | "lower power consumption for FL" | Own figures: 83 kJ (FL) vs 70 kJ (centralized) |
| Sensitivity | "client fraction alpha in {0.1, 0.5, 1.0}" | alpha is the Dirichlet parameter, not participation |
| Table 5 class skew | 24-76% positive per client | Saved alpha = 0.1 split has clients at 0.002% and 99.98% positive |
| Round ranges in Tables 2-4 | "6-10" and "10-20" | Overlap |

### 4.2 Problems in the old code (`../code`, about 28,600 lines in 60+ scripts)

1. **FedProx is not FedProx.** `fed/app.py` runs a full scikit-learn L-BFGS
   `fit` each "epoch" and then shrinks coefficients toward the global model.
   That is neither the FedProx local objective nor mini-batch SGD as described.
2. **Reported variability is not across seeds.** `fed/summary_stats.csv`
   (source of "0.726 +/- 0.163") is computed over per-round rows, so the std
   mixes rounds and seeds and the CIs are too narrow.
3. **Communication is constant by construction**: 19.08 MB up and down for
   every run, independent of participation.
4. **Timing is implausible**: `total_time_sec` of about 0.016 s.
5. **The saved partition does not match the paper**: `partition_script.py`
   partitions `np.random.randint` dummy labels and mentions IMDB; the alpha = 0.1
   Yelp split contains a client with 5 samples.
6. **Several figure scripts draw from random number generators**, not from
   experiment logs (`fed/app.py` lines 1619-2432, `fed/advanced_visualizations.py`,
   `comprehensive_visualizations.py`), e.g.
   `fedprox_perf = np.random.normal(0.774, 0.014, 1000)`.
7. The Flower-based scripts are stubs written for flwr 1.5/1.6
   `start_simulation`, which the current release no longer documents.
8. No tests, no saved partitions, no per-run metadata, results for IMDB and
   Yelp mixed in the same folders.

### 4.3 Provenance of Tables 2-4 and of the drafted responses

No log, CSV or JSON in the old code tree reproduces the values in manuscript
Tables 2-4 (for example 93.12% accuracy). The file
`yooo changes from claude, absolute class.docx` opens with a request to write
realistic-sounding reviewer responses for code that had not been run, and the
text that follows contains specific figures and p-values (for example
"p = 0.043, paired Wilcoxon"). Those values have no experimental source.

**Consequence for this project:** no number, table or figure from the old
manuscript, the old code, or the drafted responses is carried forward. Every
value in the new paper is regenerated from logged runs of the new pipeline.

## 5. What is reused

Only the research question, the reviewer requirements and the improvement
plan. No code is reused.
