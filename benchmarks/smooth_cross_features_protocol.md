# Smooth cross-feature development screen

## Question and evidence tier

Can a target-free external implementation of ChimeraBoost's numeric
diff/product feature policy:

1. reproduce the native implementation exactly;
2. improve a full-budget constant/linear selector on the spent smooth panel;
   and
3. maintain an acceptable selector regression profile?

This is development evidence on already-spent coordinates. It cannot promote a
default or make a fresh competitive claim. There are no win-count gates.
Exactness is a correctness requirement; quality and cost are reported as
measurements used to scope the later T5 candidate.

## Data boundary

- Tasks: grid stability `361251`, kin8nm `361258`, and space_ga `361623`.
- Coordinates: OpenML repeat 0, folds 3–9, sample 0 (21 total).
- All tasks are in the spent CTR23 confirmation partition.
- Every CTR23 lockbox task is explicitly denied.
- The official outer test rows are never used for model or feature selection.
- Each outer training fold gets a deterministic inner validation split:
  `ShuffleSplit(test_size=0.20, random_state=4)`.

## Frozen base model

Both libraries use seed 4, six threads, CatBoost-style depth-6 trees, L2 1,
128 bins, learning rate 0.1, 2,000 rounds, minimum child hessian 1, early
stopping with the shared explicit validation rows, and full-budget selection.

The base selector fits constant and linear leaves independently and keeps the
lower best validation RMSE (ties retain constant leaves).

## Frozen cross policy

1. Rank original numeric features by the selected base model's split-gain
   importance, descending with original column order as the deterministic tie
   break.
2. Take at most the top six numeric features.
3. For every unordered pair, append the difference and product columns, in
   pair-major `diff`, `prod` order (at most 30 columns).
4. Refit the selected base leaf lane from scratch on the augmented matrix.
5. Keep the crossed model only when its best validation RMSE is strictly lower
   than the base model's. Ties retain the cheaper base.

All transforms use features only. Missing values propagate naturally through
the arithmetic and remain missing.

## Arms and checks

For every coordinate the runner records:

- `darko_base`: the full-budget constant/linear validation winner;
- `darko_cross_policy`: the base-versus-cross validation winner; and
- `chimera_full_product`: ChimeraBoost with `linear_leaves=None`,
  `cross_features=None`, and `selection_rounds=None`.

The runner fails unless DarkoFit's external selected policy and ChimeraBoost's
native full product agree exactly on:

- selected constant/linear lane;
- selected cross-feature decision and ordered pair list;
- fitted borders and normalized tree fingerprint;
- complete validation history;
- prediction bytes, retained tree count, best validation RMSE, and test RMSE.

The analyzer reports equal-dataset geometric-mean quality ratios, leave-one-out
ratios, worst dataset and split ratios, selection counts, summed fit seconds,
and RSS. Timings are diagnostic and not eligible for an engineering claim.

