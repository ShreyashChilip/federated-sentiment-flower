# Client-level structure in word/sentiment association: diagnostic definitions

Status: definitions fixed on 2026-10-02, before any result. Implemented in
`src/analysis/diagnostics.py` and `src/analysis/exp0.py`; verified against
brute-force arithmetic in `tests/test_diagnostics.py`.

This document defines measurements. It reports no result and makes no claim
that confounding exists.

## 1. What the two experiments can and cannot show

| | Experiment 0A | Experiment 0B |
|---|---|---|
| Data | Yelp Polarity, 2 classes | Amazon Reviews 2023, 5 classes |
| Clients | Synthetic: IID control and Dirichlet label skew | Real: one client per `user_id`, or one per `parent_asin` (never mixed) |
| How text reaches a client | Independently of the text, given the label | Through who wrote it / what it is about |
| Can show | How ordinary label heterogeneity moves FedAvg coefficients, and whether a label-prior correction undoes it | Whether some terms are associated with sentiment across clients but not inside them, and whether models lean on such terms |
| Cannot show | Natural-client lexical confounding. Any client-level exposure in 0A is induced by the label alone | Causation. The quantities are observational |

## 2. Notation

For a document i: `z_ij = 1` if it contains vocabulary term j (TF-IDF value
greater than zero), else 0. `s_i` is its sentiment score: the label (0/1) for
two classes, the star index (0 to 4) for the five-class task. `k(i)` is its
client, `n_k` the client's number of training documents, `n` the total. Bars
denote means: `zbar_kj`, `sbar_k` within client k; `zbar_j`, `sbar` overall.

All quantities use **training-role documents of training (seen) clients
only**. Validation, client-test, held-out-client and global test documents
never enter them.

## 3. Association decomposition

The pooled association of term j with sentiment is the least-squares slope of
`s` on `z_j` (Equation 1). It splits exactly into a within-client part
(Equation 2) and a between-client part (Equation 3), weighted by the share of
the term's variance that lies within clients (Equation 4):

```
(1)  beta_pooled_j  = sum_i (z_ij - zbar_j)(s_i - sbar) / sum_i (z_ij - zbar_j)^2

(2)  beta_within_j  = sum_i (z_ij - zbar_k(i)j)(s_i - sbar_k(i)) / SSW_j
     SSW_j          = sum_i (z_ij - zbar_k(i)j)^2

(3)  beta_between_j = sum_k n_k (zbar_kj - zbar_j)(sbar_k - sbar) / SSB_j
     SSB_j          = sum_k n_k (zbar_kj - zbar_j)^2

(4)  lambda_j       = SSW_j / (SSW_j + SSB_j)

(5)  beta_pooled_j  = lambda_j * beta_within_j + (1 - lambda_j) * beta_between_j
```

Reading:

* **Within-client association** (Equation 2): among one client's own
  documents, do those containing the term have a different sentiment? Client
  mean sentiment and client usage rate are subtracted, so a client's general
  harshness or its habit of using the term cannot contribute.
* **Across-client association** (Equation 3): do clients that use the term
  more have a different average sentiment?
* Equation 5 is an identity, checked numerically in the tests.

Also recorded per term: document frequency, client frequency (number of
clients using it), and the between-client variance of its usage rate,
`SSB_j / n`.

## 4. Candidate client-confounded features

The **client gap** is the part of the pooled association that exists only
because of client membership (Equation 6):

```
(6)  gap_j = beta_pooled_j - beta_within_j = (1 - lambda_j)(beta_between_j - beta_within_j)
```

A non-zero gap alone is not enough, for two reasons. Rare terms and small
clients produce gaps by chance. And when clients differ in mean sentiment the
gap is generally not zero even for a term whose usage has nothing to do with
the client, because the sentiment score has a narrower range inside a client
than in the pool.

**Null hypothesis.** Term usage is independent of the client given the
document's own sentiment: `z_j independent of k | s`.

**Null distribution.** Each draw keeps every document's client and score and
shuffles the feature rows among documents with the same score. This preserves
client sizes, each client's sentiment distribution and each term's pooled
association with sentiment; it removes only client-specific usage. Over
`R = null_repeats` draws the gap has mean `m0_j` and standard deviation
`sd0_j`, and the standardized gap is (Equation 7)

```
(7)  t_j = (gap_j - m0_j) / (sd0_j * sqrt(1 + 1/R))
```

referred to a Student t distribution with R - 1 degrees of freedom (the null
moments are estimated from few draws). Two-sided p-values are adjusted by
Benjamini-Hochberg over the supported terms.

A term shows **client structure** when all three hold (thresholds are in the
config under `diagnostic`, fixed before any run):

1. document frequency >= `min_document_frequency`,
2. client frequency >= `min_client_frequency`,
3. BH q-value <= `fdr`.

Terms with client structure are then typed by how the pooled association
relates to the within-client one:

| `gap_type` | Condition | Reading |
|---|---|---|
| `sign_reversed` | pooled and within have opposite signs | Pooling reverses the direction the term has inside clients |
| `inflated` | same sign, abs(pooled) > abs(within) | Client membership adds association the term does not have inside a client |
| `attenuated` | same sign, abs(pooled) < abs(within) | A real within-client association is partly hidden when clients are pooled |

A term is a **candidate client-confounded feature** only if it has client
structure and its type is `inflated` or `sign_reversed`. Attenuated terms are
genuine sentiment signals and are reported, but they are not candidates.

"Candidate" is a descriptive label. It says the term's usage depends on the
client beyond what the document's sentiment explains, in a way that makes the
pooled association overstate or reverse the within-client one. It does not say
why. A high pooled association by itself never makes a term a candidate.

How this definition was arrived at, for the record: two earlier drafts were
tried on a synthetic file with a known construction (`tests/synthetic_reviews.py`),
before any real data was loaded. A first draft without gap types flagged the
genuine words together with the habit words. A first null that shuffled
documents across clients flagged pure-noise words, because it erased the
differences in client mean sentiment. The definition above flags the two
constructed habit words and no genuine or noise unigram, and a calibration
test on 300 pure-noise terms is part of the test suite. The synthetic file is
a software fixture; nothing about it is a finding.

Limits to state with any use of this label:

* q-values near the threshold are not reliable to the second decimal with 20
  null draws;
* the test detects dependence of usage on the client given sentiment; a term
  can show it for harmless reasons (topic, product type);
* with few documents per client, `beta_within` is noisy;
* for five classes the score treats stars as equally spaced, and the null
  conditions on the exact star value.

## 5. Model-side quantities

**Signed sentiment coefficient.** For a `C x d` logistic-regression weight
matrix W the coefficient of term j is `w_j = sum_c (c - (C-1)/2) * W_cj`,
multiplied by 2 when C = 2 so that it equals the usual `W_pos,j - W_neg,j`.
Positive means "pushes towards higher sentiment". Column j of W is term j of
`vocabulary.json` in the feature bundle; the feature table writes the index
and the term side by side and the test suite checks the order.

**Coefficient bias** (frozen in EXPERIMENT_PLAN.md section 5):
`B_j = w_fed,j - w_central,j`, computed for FedAvg and for FedAvg with the
label-prior correction, against the centralized model trained on the pooled
training rows with identical features, optimizer, batch size, learning rate
and weight decay.

**Client label exposure** (frozen): `X_j = sum_k df_kj (sbar_k - sbar) / sum_k df_kj`,
where `df_kj` is the number of client k's training documents containing term
j. It equals the numerator of Equation 3 divided by the term's document
frequency.

**Bias versus exposure.** Over supported terms: Pearson and Spearman
correlation of `B_j` with `X_j`, and the partial Pearson correlation
controlling for `w_central,j`.

Why the partial correlation matters: under label skew, exposure is itself
correlated with a term's polarity. If the federated model is simply less
converged than the centralized one (coefficients uniformly smaller), then
`B_j` is proportional to `-w_central,j` and correlates with exposure although
no client effect exists. `tests/test_diagnostics.py` constructs exactly this
case: the raw correlation is large and the partial correlation vanishes. The
frozen decision rule of 0A is stated on the raw Spearman correlation; see
PHASE_0_TO_PHASE_1_AUDIT.md section 4 for the consequence.

## 6. Performance quantities (0B)

For each of centralized, FedAvg and FedAvg + label-prior correction, on
identical rows:

* **seen-client performance**: client-test rows of training clients, pooled;
* **unseen-client performance**: every row of every held-out client, pooled;
* **generalization gap** = seen macro-F1 minus unseen macro-F1 (accuracy gap too);
* **client-level macro-F1 distribution** for seen and for unseen clients:
  mean, median, 10th percentile, worst, standard deviation, IQR, spread;
* macro-F1 of a single client is averaged over the classes that client holds.

Held-out clients never contribute a document to training, validation or
feature fitting. `natural.assert_no_client_overlap` raises if any client has
rows on both sides, and the count of overlapping clients (zero) is stored in
the bundle metadata.

## 7. Output files

Per cell and seed, under `results/<experiment>/<cell>/diagnostics/seed<k>/`:

* `checks.json`: comparability checks; analysis refuses to run if one fails;
* `feature_confounding.csv`: one row per vocabulary term with
  `feature_index, term, document_frequency, client_frequency,
  global_association, within_client_association, across_client_association,
  lambda_within, between_client_variance, client_exposure, client_gap,
  gap_null_mean, gap_null_sd, gap_z, gap_q, supported, client_structure,
  gap_type, candidate_confounded, w_centralized,
  w_fedavg, w_fedavg_la, coefficient_bias_fedavg, coefficient_bias_fedavg_la`;
* `diagnostic.json`: counts, bias-versus-exposure statistics (all supported
  terms, and candidates only), mean absolute weight on candidate terms per
  model, and the performance report.

Per experiment: `summary.json`, `summary.md`, `runs.csv`, `rounds.csv`,
`clients.csv`, `schema_validation.json`.

## 8. What would count as support for a client-specific lexical effect

Written down now so that it is not chosen after the fact. All of these would
have to hold in 0B, for a given client definition, across seeds:

1. candidate terms exist in a number clearly above what the false-discovery
   level (`fdr` times the number of supported terms) would produce by chance;
2. models put weight on candidate terms in the direction of the pooled
   association rather than the within-client one;
3. unseen-client performance is lower than seen-client performance by more
   than the seen clients' own train-to-test gap;
4. the label-prior correction does not remove points 2 and 3.

If point 4 fails, a feature-level method is not needed. If points 1 to 3
fail, the phenomenon is absent for that client definition and is reported as
absent.
