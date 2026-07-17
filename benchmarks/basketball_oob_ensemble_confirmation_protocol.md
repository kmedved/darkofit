# Basketball OOB ensemble stable confirmation protocol

## Decision boundary

The original five-member OOB-ensemble campaign passed every frozen basketball
quality gate but could not authorize API work because both arms slowed together
in its third and final timing block. This successor asks only whether a clean,
position-balanced timing campaign reproduces the already-frozen behavior and
is stable enough to advance the mechanism to a separate opt-in API phase.

This confirmation does not authorize a default change. Basketball remains the
complete and fatal gate: the unchanged creator folds, overlap-exposed team
holdout, and corrected 585-row cold-player subset are the only data used. No
TabArena or CTR23 coordinate is consulted.

The run binds:

- DarkoFit base commit `10209a8e9f9a8ed8d16aa2bf991c4fe78c255252`;
- the exact current `darkofit/` package manifest after the TreeSHAP phase;
- the original OOB runner, shared basketball harness, creator runner, player
  guardrails, and original immutable artifact by SHA-256;
- a clean `main` equal to `origin/main`; and
- this protocol hash plus the confirmation runner hash recorded in the output.

The OOB mechanism remains the independent DarkoFit prototype in
`run_basketball_oob_ensemble.py`: five deterministic training-sized
bootstraps, each member early-stopped on its exact out-of-bag complement, no
refit, and arithmetic-mean prediction. No ChimeraBoost source is copied.

## Frozen behavior

Every fresh worker must reproduce the original artifact's exact ten fold
prediction hashes and full held-team prediction hash for its arm. R² values,
the cold-player and seen-player subset hashes, deterministic bootstrap/OOB
plans, fitted member count, validation source, early-stopping reason, and model
metadata invariants are revalidated. Within this campaign, each arm's complete
timing-free behavior fingerprint must also be identical across all six runs.

The quality result must therefore reproduce the original decision:

- mean fold R² delta is nonnegative;
- at least six of ten folds improve;
- every leave-one-fold-out mean delta is nonnegative;
- overlap-exposed team-holdout R² does not regress; and
- cold-player R² does not regress.

Any prediction-hash drift fails the confirmation even if the new scores happen
to pass.

## Position-balanced timing

Run six fresh-process reciprocal blocks at 18 threads:

```text
default, oob_ensemble5
oob_ensemble5, default
default, oob_ensemble5
oob_ensemble5, default
default, oob_ensemble5
oob_ensemble5, default
```

Each worker performs one complete first-fold fit and prediction outside its
timer, then measures the unchanged ten folds. No block is discarded and no
threshold is adjusted after observation.

The original max/min gate was underpowered with only three observations and
made a single shared slowdown determinative. This successor preregisters
robust spread measures over six observations:

- each arm's steady-wall IQR/median is at most `0.20`;
- each arm's summed-prediction IQR/median is at most `0.20`;
- the six within-block candidate/default wall ratios have IQR/median at most
  `0.15`;
- candidate median wall time is at most `4.0x` default; and
- candidate median summed prediction time is at most `6.0x` default.

These retain the original feature-specific complexity budgets. Passing does
not claim parity with a single model; it establishes that a deliberately
five-model opt-in accuracy feature has stable, bounded cost.

## Decision

A pass advances only to an isolated public-API implementation phase. That
phase must independently prove sklearn cloning, regression/classification,
weights, categoricals, groups, thread/process budgeting, serialization,
calibration, and unsupported distributional behavior before release, followed
by the unchanged basketball/cold-player gate again on the actual API.

A failure closes the OOB-ensemble attempt without rerunning or weakening this
protocol. It cannot be rescued by averaging the two historical campaigns or
dropping an inconvenient block.
