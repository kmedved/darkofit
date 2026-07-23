# Group-centered categorical crosses v1 development contract

_Frozen before candidate implementation and before any candidate quality
outcome._

Contract identity:
`group-centered-categorical-crosses-v1-development-20260722`.

## Authority and claim

Revision R1 in `BEAT_CHIMERABOOST_PLAN.md` funds categorical crosses as the
next one-at-a-time quality mechanism. The source intake is
`chimeraboost_v0210_changelog_triage_20260722.md`, binding ChimeraBoost release
`v0.21.0` at `26fed8a` and donor commit `e0d401b` under Apache-2.0.

The candidate claim is narrow: on eligible mixed numeric/categorical scalar-
RMSE CatBoost-mode data, a deterministic validation race can safely select a
bounded block of target-free group-centered numeric columns. It is not a
pairwise categorical-combination revival, a classification claim, or a
general cross-feature API.

## Exact private candidate

The private candidate changes the candidate branch's default fit only. It
adds no public constructor parameter and remains unmergeable under this
contract.

Eligibility requires all of:

1. one non-ensemble `DarkoRegressor` with scalar `loss="RMSE"`;
2. explicit `tree_mode="catboost"` resolution;
3. at least 2,000 selection-training rows after any holdout;
4. at least one numeric and one declared categorical input feature; and
5. no preset, automatic tree-mode race, automatic learning-rate probe,
   callbacks, global linear residual, distributional/interval calibration,
   refit, or ordered boosting.

Ineligible fits execute the exact control path and record a stable reason.

For an eligible fit:

1. create a deterministic 15% validation holdout when the caller supplied no
   `eval_set`; keep declared groups disjoint and subset sample weights;
2. fit a control audition with early stopping on the selection train/validation
   rows;
3. rank original input features by that audition's split-gain importance;
4. take at most the top four numeric and top three categorical features and
   form every numeric×categorical pair (at most 12);
5. for pair `(x, c)`, fit `mean_fit(x | c)` on selection-fit rows only,
   weight-aware and ignoring non-finite `x`; transform to
   `x - mean_fit(x | c)`, using the fit-time global weighted mean for unseen
   categories;
6. fit the augmented audition under the same seed, rows, validation data,
   weights, and boosting policy;
7. select the augmented lane only when its validation RMSE is strictly lower;
   ties select control; and
8. fit the selected lane from scratch under the caller's original full-fit
   semantics. The final group means use only the rows used by that final fit.

The fitted record includes eligibility/reason, seed, split provenance and
hashes, pair list, group/global-mean provenance, both validation RMSE values,
selection decision, selection costs, final lane, and final preprocessing
state. Declines must be prediction/state exact to control. Selected models
reconstruct crosses from original inputs at predict and staged-predict time.

## Invariants before quality evidence

Before M6, tests must establish:

1. ineligible numeric, classification, small, ensemble, and unsupported-mode
   fits are exact control behavior with stable reasons;
2. selection rows and validation rows are disjoint, group-disjoint when
   groups exist, deterministic, and weight-correct;
3. group means are target-free, weight-aware, exclude zero-weight influence,
   treat missing categories consistently, and use the recorded global fallback
   for unseen categories;
4. pair generation is deterministic, importance-ranked, bounded to 12, and
   contains only declared numeric×categorical pairs;
5. a selected automatic final fit is prediction and normalized-state exact to
   a forced fit with the recorded pairs; a decline is exact to control;
6. feature names, NumPy/object and pandas categorical inputs, missing values,
   empty prediction batches, ambient Numba thread restoration, staged
   predictions, clone/get-params, and repeated fits remain valid;
7. safe-NPZ round trips preserve predictions, pair/mean payloads, fitted
   metadata, corruption rejection, and deterministic resave;
8. feature importance folds each centered column into its numeric parent;
   exact TreeSHAP fails loudly while centered columns are active because that
   attribution cannot honestly be assigned to only one original feature; and
9. no classifier, ensemble, distributional, interval, explicit ordinal,
   linear-leaf, or linear-residual behavior changes.

Mechanism synthetics must include a category-specific numeric-baseline problem
where centering has headroom, an unseen-category case, a weighted zero-row
case, and a no-category exact no-op.

## Frozen evidence sequence and stop

After invariants and relevant M5 correctness sentinels pass, execute exactly
one M6 v3 inspection using mechanism id
`group_centered_categorical_crosses_v1`, inspection index 1, and the immutable
60-pair grid/rule in `m6_quality_successor_v3_contract.md`.

- `advance` under all three M6 v3 gates yields only
  `eligible_for_mechanism_specific_spent_attribution`.
- `kill` closes this exact candidate.
- A harness or execution failure spends inspection 1 and closes that execution
  identity; any repaired execution needs a new explicit identity and owner
  authority.

No outcome-specific threshold, pair cap, selection rule, or eligibility scope
may change after inspection. Costs and engagement are fully reported but are
not M6 ranking gates. The 60 cells are dependent spent development evidence,
not independent datasets.

Every material execution requires clean committed published control,
candidate, and harness sources; exact source hashes; fresh workers; exclusive
machine; create-only outputs; and a 12-field `TESTING_LOG.md` entry. No fresh,
TabArena, release ladder, public API/default, merge, release, or lockbox access
is authorized.
