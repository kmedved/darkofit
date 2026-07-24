# Accuracy-v2 component ablation

This normal development benchmark asks whether guarded numeric crosses add
value to the existing explicit A10 accuracy profile, and separately measures
the 10,000-round horizon inside the same integrated path.

The grid uses only the four regression datasets in the already-spent M6-v3
medium slice, seeds 0--2, and ordinary plus stress-weighted views. No holdout,
TabArena task, or sports ship-check is consulted.

Four arms share the A10 configuration (`tree_mode="auto"`, learning rate 0.1,
L2 3, 128 bins, one target-stat permutation, automatic linear leaves):

1. `a1`: 1,000-round ceiling;
2. `a1_cross`: A1 plus the historical top-six numeric diff/product audition;
3. `a10`: the current `preset="accuracy"` fallback; and
4. `a10_cross`: A10 plus the same audition.

Each cross audition uses only the inner validation split. It engages when the
crossed validation RMSE is at most 95% of the uncrossed RMSE, then refits the
selected best-prefix configuration on all outer-training rows. A decline
reuses the uncrossed final prediction exactly. Outer-test rows are used only
for the reported development comparison.

Accuracy-v2 is selected only if A10+crosses engages at least once, strictly
improves equal-cell quality over A10, and preserves M6-v3's dataset-harm and
leave-one-dataset-out bounds. Otherwise the existing A10 preset remains the
profile and no API is added. Fit and prediction times are telemetry.
