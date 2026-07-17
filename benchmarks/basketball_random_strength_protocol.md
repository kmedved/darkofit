# Basketball random-strength screen

## Question

Does split-score noise provide enough broad quality value on the primary noisy
sports workload to justify keeping `random_strength` as a public 1.0 parameter?

This is the one cheap screen required by Track Z1 of
`BEYOND_PARITY_PLAN.md`. It is a retirement screen, not a default-promotion
campaign, and it consumes no CTR23 development, confirmation, or lockbox
coordinates.

## Frozen boundary

- The DarkoFit source, protocol, and runner are committed before the formal
  run; a dirty tree is ineligible.
- Data, preprocessing, ten unshuffled creator folds, random state 4, and the
  overlap-exposed/seen-player/cold-player guardrails are the fingerprinted
  basketball boundary in `basketball_harness.py`.
- Arms:
  - control: product defaults (`random_strength=0.0`);
  - candidate 1: `random_strength=0.5`;
  - candidate 2: `random_strength=1.0`.
- Every other model parameter remains at the product default. There is no
  sample weighting, validation set, early stopping, refit, ensemble, or
  per-fold tuning.
- Each arm runs once in a fresh worker. Runtime is recorded only as operating
  cost; it is not a decision gate.

The two nonzero values cover a moderate and the midpoint setting of the
existing `[0, 2]` tuning range without turning this cleanup screen into a
hyperparameter search.

## Fatal quality gate

Each candidate is paired with the control on identical folds and advances only
if all conditions hold:

1. mean R² gain is at least `+0.002`;
2. every leave-one-fold-out mean delta is nonnegative;
3. overlap-exposed held-team R² does not regress; and
4. cold-player R² does not regress.

The seen-player subset is reported diagnostically. Exact prediction and fitted
metadata fingerprints are retained.

## Decision

- If either candidate passes every gate, keep `random_strength` public and move
  only the best passing value to a fresh sports confirmation protocol.
- If neither passes, deprecate `random_strength` in 0.10 and remove it in 1.0.
  The implementation remains unchanged during the deprecation cycle.
- This screen cannot change any default.
