# Basketball robust-head screen

## Question

Can an existing robust DarkoFit regression head improve the ChimeraBoost
creator's noisy basketball target without weakening ordinary-fold robustness,
the overlap-exposed held-team score, or genuinely cold-player generalization?

This is a basketball-first development screen. It does not change defaults and
does not consume CTR23 development, confirmation, or lockbox coordinates.

## Frozen boundary

- Source before protocol implementation: DarkoFit `ab86269`.
- Data, creator preprocessing, ten unshuffled folds, random state 4, and the
  held-team/seen-player/cold-player views are those fingerprinted by
  `basketball_harness.py`.
- The three arms use product defaults except for the declared loss and the
  tree mode required by distributional heads:
  - control: `DarkoRegressor(loss="RMSE", random_state=4)`;
  - Student-t location: `DarkoRegressor(loss="StudentT",
    tree_mode="lightgbm", random_state=4)`;
  - MAE: `DarkoRegressor(loss="MAE", random_state=4)`.
- No sample weights, external evaluation set, refit, ensemble, categorical
  transform, or per-arm tuning is used.
- Each worker performs one complete first-fold warmup outside timing, then
  evaluates all ten folds sequentially. Imports and data loading remain
  outside the measured interval.

## Fatal quality screen

Each candidate is paired to RMSE on the same folds. It advances only if all
conditions hold:

1. mean R² gain is at least `+0.002`;
2. every leave-one-fold-out mean delta is nonnegative;
3. overlap-exposed held-team R² does not regress; and
4. cold-player R² does not regress.

The first fresh-worker block evaluates all three arms. Candidates that fail
stop immediately and receive no stability claim.

## Timing and stability

Survivors receive two further fresh-worker blocks. Arm order reverses relative
to the control in the middle block. Stability is gated only on the candidate /
control ratio paired within each block:

- wall, summed fit, and summed prediction ratios each require
  `IQR(ratio) / median(ratio) <= 0.10`.

Per-arm millisecond-scale dispersion is descriptive only and cannot reject a
candidate. No absolute runtime ceiling is imposed in this zero-new-code screen;
runtime remains a product-cost input if a head advances.

## Decision

A passing head advances to the preregistered multi-season sports suite. This
screen alone cannot promote a default. If neither head passes, robust loss is
closed as an explanation for the creator-benchmark gap in its present form.
