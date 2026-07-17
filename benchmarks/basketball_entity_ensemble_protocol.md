# Basketball entity-aware ensemble screen

## Question

Can a five-member player-identity bootstrap ensemble improve the unchanged
creator-fold basketball score while preserving held-team and cold-player
quality?

This is the materially different S2 mechanism. It is not a rescue run for the
closed row-bootstrap OOB-5 candidate and cannot authorize a default or public
API change.

## Frozen mechanism

- Source before implementation: clean, published `main` at `b7a5b2c`.
- Control: `DarkoRegressor(random_state=4)`.
- Candidate: five deterministic members. For each external training set:
  1. sample its unique exact source `Player` identities with replacement,
     drawing as many identities as the unique-player count;
  2. include every row belonging to each sampled identity, repeating the
     complete player block when an identity is drawn more than once;
  3. use every row belonging to an unselected identity as that member's
     group-disjoint OOB evaluation set;
  4. early-stop without refit on that exact OOB set; and
  5. average the five predictions arithmetically.
- `Player` is grouping metadata only and never becomes a model feature.
- The five members share one numeric preprocessor and binned matrix fitted
  once on the external training features. This dataset has no categorical
  model columns, so the shared transform is unsupervised; member training
  indexes rows from that common binned matrix and OOB evaluation uses the same
  fitted transform.
- The runner records player draws, selected/OOB identities, row indices,
  shared-preprocessor state, member metadata, predictions, and guardrail
  hashes.

The player-bootstrap and shared-binning design received one development probe.
The promotion thresholds below were already declared in
`BEYOND_PARITY_PLAN.md`; this run is a source-bound fatal-screen audit, not a
blind estimate.

## Fatal quality gate

The first reciprocal block runs control then candidate on the unchanged ten
unshuffled creator folds and the overlap-exposed team holdout with its
seen-player and 585-row cold-player subsets.

The candidate survives only if all are true:

1. mean ten-fold R² gain is at least `+0.004`;
2. every leave-one-fold-out mean delta is nonnegative;
3. overlap-exposed held-team R² does not regress; and
4. cold-player R² is strictly positive versus control.

Failure stops the campaign. No timing-stability or cost claim is then
available, and the entity-aware ensemble is closed as shaped.

## Timing for a survivor only

A survivor receives two more reciprocal fresh-worker blocks:

```text
control, entity_ensemble5
entity_ensemble5, control
control, entity_ensemble5
```

Wall, summed-fit, and summed-prediction candidate/control ratios each require
`IQR / median <= 0.10`. Candidate median wall time must be at most `3.0x`
control, a tighter budget than the closed row-bootstrap implementation because
preprocessing is shared. Peak RSS must be at most `3.0x`.

Passing would advance only to S4's fresh sports confirmation. A public
`n_ensembles` API remains forbidden before that confirmation.

No CTR23 development, confirmation, or lockbox coordinate is used.
