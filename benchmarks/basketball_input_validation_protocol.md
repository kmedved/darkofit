# Basketball input-validation and sklearn-compliance protocol

## Decision and scope

This phase closes item 8 of `BEST_OF_BOTH_PLAN.md`: harden DarkoFit's public
input boundary without changing any model policy, estimator default, fitted
state, prediction, or serialized model. Basketball remains the primary and
fatal development gate because it is fast, representative of the intended
sports workload, and already has ordinary-fold, held-team, and cold-player
views. A failure on any of those views rejects the candidate. Broader datasets
are confirmation-only after a basketball pass; they are not part of this
iteration. No TabArena or CTR23 task is opened.

The baseline is clean, pushed DarkoFit `main` at
`e2f4535b79bfc1c4d10397758e2dba2994e55b51`. The comparison source is the
clean synced ChimeraBoost 0.15.0 checkout at
`851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`. Substantial literal adaptation
from its validation helpers requires Apache-2.0 attribution in `NOTICE`.

The candidate may change only input coercion, validation, feature metadata,
sklearn tags, focused tests, and the source-bound basketball runner. It may
not add a constructor parameter or automatically reorder columns.

## Reproduced baseline gaps

Direct probes against the baseline established that DarkoFit currently:

- silently discards a NumPy masked-array mask;
- silently discards the imaginary component of complex numeric features;
- accepts positive or negative infinity during fit and prediction;
- neither warns when fit-time feature names disappear at prediction nor when
  names appear only at prediction;
- silently deduplicates repeated categorical indices;
- rejects categorical column names rather than resolving them;
- reports only NumPy's raw conversion error for an unmarked nonnumeric
  DataFrame column; and
- extracts names only from `.columns`, missing PyArrow's `.column_names`
  convention.

Pandas nullable numeric fit/eval/predict and sparse-input rejection already
work and must remain covered.

With true estimator defaults, scikit-learn 1.7's `check_estimator` currently
reports eight failures for each wrapper. The shared actionable failures are
complex/non-finite input handling, empty-data and reshape messages, the
column-vector warning wording, the missing-`y` message, and the all-zero
weight message. The classifier additionally needs a one-class/one-sample
message. The remaining failure is the existing documented algorithmic
deviation: sample weighting is not bit-exactly equivalent to literal row
removal/repetition.

## Candidate behavior

The shared wrapper/core boundary must:

1. Reject masked arrays at fit, explicit validation, prediction, and staged
   prediction with guidance to use `filled(np.nan)`.
2. Reject complex numeric features before any lossy cast.
3. Reject infinity in numeric feature columns at fit and prediction while
   continuing to accept NaN as a missing value.
4. Honor `sklearn.config_context(assume_finite=True)` only as a
   prediction-time escape hatch for the O(n) infinity scan.
5. Capture flat scalar feature names from pandas/Polars `.columns` and
   PyArrow `.column_names`; reject renamed or reordered names and emit the
   sklearn-consistent warning when names exist on only one side.
6. Resolve string or mixed string/integer `cat_features` against named input
   at fit time, reject unknown names and name use on unnamed input, and reject
   duplicate resolved indices. The fit-only API remains fit-only.
7. Preserve nullable missing values when coercing pandas, Polars, PyArrow, and
   compatible frame-like inputs. Optional integrations may use duck typing;
   DarkoFit must not import optional packages unconditionally.
8. Name unmarked nonnumeric DataFrame-like columns in the conversion error.
9. Publish sklearn tags declaring NaN support and sparse rejection.
10. Use sklearn-compatible errors/warnings for empty inputs, 1-D inputs,
    missing targets, column-vector targets, all-zero weights, and one-class
    classification.

NaN remains supported, sparse matrices remain unsupported, column order is
never repaired automatically, and sample-weight numerics do not change.

## Correctness and compatibility gates

Focused tests must cover regression, binary and multiclass classification,
direct core boosters, explicit eval sets, staged predictions, and
distributional prediction. They must include:

- masked, complex, infinite, NaN, sparse, empty, 1-D, and malformed target
  inputs;
- named categorical resolution and duplicate/unknown-name errors;
- named/unnamed, reordered, and renamed fit/eval/predict combinations;
- pandas nullable numeric and categorical columns;
- mandatory frame-like PyArrow/Polars behavior using dependency-free test
  doubles, plus real integration tests when either package is installed;
- `assume_finite=True` prediction behavior; and
- archive round trips retaining `n_features_in_` and `feature_names_in_`.

Both public wrappers must pass scikit-learn 1.7 `check_estimator` under their
true defaults, with only
`check_sample_weight_equivalence_on_dense_data` registered as an expected
failure. That exception must remain explicitly documented as weighting the
loss rather than reproducing integer row replication bit-for-bit.

The complete project suite must pass. Any changed valid-input prediction,
fitted metadata field, feature importance, or archive byte rejects the
candidate.

## Frozen basketball gate

The unchanged creator fold 0, complete held-team view, and corrected 585-row
cold-player view are used at 18 threads with explicit `darkofit.warmup()`
outside timing. Six reciprocal fresh-process blocks compare:

- `validated`: ordinary prediction with the new default validation; and
- `assume_finite`: the same calls inside
  `sklearn.config_context(assume_finite=True)`.

Every worker receives a unique empty Numba cache and records import, warmup,
fit, prediction, validation-only, fitted metadata, warnings, predictions,
and archive bytes. All observations are retained.

The immutable baseline references are:

- fold prediction:
  `6200db22da190d8c0787d7794c1fb8d859af737ab7e3026716e23aa1be95125f`;
- held-team prediction:
  `1693ff2070b05bb705810aba0d9b27b5a0a01dc6f4ee51939a3ee30af3698cdf`;
- cold-player prediction:
  `b9dc899fcabc5a3a7892da41d839bac70f7d50da9553e2e57770501f71694c82`;
- model archive: 382,557 bytes and
  `50a7e6f0a6f8500a55a6ba088ad25137335ed4354a4b4e908ea17f023c91ec71`;
- resolved learning rate `0.052312`, 1,000 CatBoost-mode trees, and
  `iteration_limit`; and
- warmup artifact SHA-256
  `6548a88749d2c141b4ab6fc887f95059e9768636b40ab45bfaa9617c17d5ee47`,
  whose warmed median first fit was `1.606730271` seconds.

## Promotion gates

The input-validation layer ships only if:

1. Every worker reproduces all three prediction hashes and arrays exactly.
2. Every worker reproduces the archive byte count and SHA-256 exactly.
3. Timing-free fitted metadata and behavior fingerprints are identical across
   arms and repeats.
4. Both wrappers pass `check_estimator` with only the preregistered
   sample-weight-equivalence expected failure.
5. Every focused compatibility behavior above passes.
6. The `validated` median first fit is at most `1.10x` the frozen warmed
   baseline (`1.7674032981` seconds).
7. The `validated` median prediction time is at most `1.10x` the
   `assume_finite` arm.
8. Each arm's fit IQR/median is at most `0.25`, the paired prediction-ratio
   IQR/median is at most `0.25`, and each prediction IQR/median is at most
   `0.50`.
9. No unexpected warning, worker failure, non-finite output, cache reuse, or
   missing cold-player row occurs.

Failure closes this implementation attempt without weakening thresholds,
discarding blocks, or rerunning the formal campaign. Passing authorizes only
the validation/compliance layer; it does not authorize a model default or
quality-policy change, nor does it establish broad tabular-data superiority.
