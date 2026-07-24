# Automatic linear-selector v3 CTR23 ship-check

This is the fixed general holdout check required by `SHIP_RULES.md` before
the automatic linear-leaf selector can become a public default.

CTR23 has already been opened by the automatic-depth ship-check, so it is
**observed release-validation**, not pristine evidence. It remains the
repository's fixed general ship-check and must not be used to retune this
candidate.

## Comparison

The runner uses the nine CTR23 lockbox tasks and official folds 0–2 from the
committed CTR23 snapshot. Each pair uses one clean source commit:

- control: `linear_leaves=False`;
- candidate: `linear_leaves="auto"` with the fixed 2-SE selector.

All other model parameters, input rows, order, seeds, and current automatic
policies are identical. Each fit runs in a fresh worker. The runner verifies
the official split fingerprints, safe-NPZ round-trip, persisted selector
metadata, ambient Numba thread restoration, and clean source identity.

## Readout

Quality is reported as candidate/control RMSE by official fold, then as an
equal-fold geometric mean within each task and an equal-task geometric mean
across the nine tasks. The report also includes the task bootstrap upper
ratio, leave-one-task-out sensitivity, worst task, task wins/ties/losses,
selector engagement counts, fit time, prediction time, and peak RSS.

The candidate is eligible for the next holdout step only when:

1. all integrity checks pass;
2. the equal-task RMSE ratio is at most `1.0`; and
3. no task's equal-fold RMSE ratio exceeds `1.0`.

This deliberately tests the selector's safety claim, not just its pooled
average. If CTR23 rejects the default, the newest untouched sports season is
not consulted because it cannot reverse a conjunctive ship rule. If CTR23
passes, the sports-season ship-check remains required before exposure.

Benchmark bugs are fixed and rerun as normal software. Outputs are
create-only so each material run remains auditable.
