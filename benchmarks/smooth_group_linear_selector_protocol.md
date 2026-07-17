# Smooth group-safe linear-leaf selector development gate

## Question

Does the frozen 3% validation-margin selector retain most of fixed linear
leaves' smooth-data benefit while declining at least one weak selection?

This is development evidence only. It follows the passed basketball
group-safety gate and uses only already-spent CTR23 confirmation coordinates.
It cannot authorize a public selector, a default, fresh-confirmation claims,
or lockbox use.

## Frozen data boundary

- Tasks: grid stability `361251`, kin8nm `361258`, and space_ga `361623`.
- Coordinates: OpenML repeat 0, folds 3 through 9, sample 0: 21 total.
- The task, dataset, feature/target, and split-index hashes must be recorded.
- All tasks must remain numeric, complete members of the CTR23-v3
  confirmation partition.
- Every task in the CTR23 lockbox is explicitly denied.
- Three task workers run concurrently with six threads each. Configuration
  waves remain sequential. Every worker warms fold 3 once outside timing.

## Frozen arms

| Arm | Policy |
|---|---|
| `darko_default` | Current `DarkoRegressor(random_state=4)` |
| `smooth_margin_selector` | Frozen selector below |
| `darko_linear_current` | Current defaults plus `linear_leaves=True` |
| `chimera_product` | Unmodified clean ChimeraBoost 0.15.0 product defaults |

The local ChimeraBoost source must remain clean at tag `v0.15.0`, commit
`851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`.

## Frozen selector

Within each outer training fold:

1. make one deterministic 20% target-stratified validation split using random
   state 4 and DarkoFit's `weighted_stratified` regression policy;
2. fit constant- and linear-leaf variants to the same selection rows, with
   the same explicit validation rows, current defaults, early stopping,
   best-model retention, and no refit;
3. compute `(constant RMSE - linear RMSE) / constant RMSE`;
4. select linear leaves only when the improvement is at least `0.03`; and
5. fit the selected leaf type from scratch on the complete outer training
   fold with current full-data defaults.

The 3% threshold and 20% fraction were frozen by the preceding mechanism
probe and basketball gate. They are not tuned in this campaign.

## Development gates

The selector advances to a fresh-confirmation design only if all hold:

1. its equal-task geometric-mean RMSE ratio versus default is at most `0.98`;
2. it wins at least two of three datasets and at least 14 of 21 splits;
3. no dataset-level geometric-mean ratio versus default exceeds `1.00`;
4. it selects linear leaves on at least 14 but fewer than all 21 coordinates;
5. it retains at least 90% of fixed-linear's equal-task improvement over
   default; and
6. no selector dataset is more than 1% worse than fixed linear.

ChimeraBoost product results are reported on the same coordinates but are not
a development promotion gate. Timing and peak RSS are descriptive because
the selector intentionally performs two extra selection fits and no public
runtime policy exists.

Passing means only: build the fresh confirmation registry/power design and
test this exact frozen selector there. Failure closes this selector shape.
The CTR23 lockbox remains sealed either way.
