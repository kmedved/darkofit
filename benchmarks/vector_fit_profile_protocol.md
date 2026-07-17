# Classification and distributional fit-path profile

## Purpose

Measure before optimizing the classification, multiclass, and distributional
fit paths that were not covered by the scalar-RMSE matched-engine campaign.
This profile selects opportunities for E1; it cannot authorize a kernel change
or performance claim.

## Frozen workload

- Clean committed DarkoFit source, exactly 18 threads.
- Deterministic 50,000-row, 24-feature numeric matrix.
- Forty boosting rounds, learning rate 0.1, 128 bins, L2 1, full rows and
  columns, ordered boosting off, no validation, no early stopping, training
  loss evaluation off, and timing telemetry on.
- Six paths:
  1. scalar RMSE CatBoost-mode control;
  2. binary Logloss CatBoost mode;
  3. four-class CatBoost per-class trees;
  4. four-class LightGBM shared-vector trees;
  5. Gaussian LightGBM distributional trees; and
  6. Student-t LightGBM distributional trees.
- CatBoost paths use depth 6. LightGBM paths use 64 leaves and
  `min_child_samples=20`.

Each path runs in three fresh workers. A same-shape 5,000-row, three-round fit
warms its exact path outside timing before the formal fit.

## Output and decision

The artifact records total fit time, seconds per round, fitted tree count,
resolved lane, prediction shape/hash, and DarkoFit's phase telemetry:

```text
preprocess / grad_hess / tree_build / train_update /
validation_predict / loss_eval
```

For each path, medians are computed across workers. Phase shares are derived
from the sum of named phases rather than total wall time so unattributed Python
overhead cannot distort their ranking. The selected opportunity is the largest
stable, behavior-relevant phase on a non-control path. A phase is only a lead
for subsequent profiling; E1 still requires an exactness oracle and a separate
performance protocol.

No external comparator, default change, CTR23 coordinate, or lockbox task is
in scope.
