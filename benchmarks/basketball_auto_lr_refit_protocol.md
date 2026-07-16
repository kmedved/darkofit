# Basketball auto-LR early-stop/refit protocol

## Purpose and immutable parent

This diagnostic asks whether DarkoFit's current automatic learning rate can be
combined with internal early stopping and exact full-data refit to reduce
training time without weakening basketball accuracy. It does not change the
creator benchmark, production defaults, or any lockbox data.

The data URL, byte and SHA-256 pins, `MP > 500` filter, feature list, target,
seed 4, and unshuffled 10-fold split are inherited unchanged from
`run_basketball_creator_benchmark.py`. The creator's alphabetical team holdout
also remains unchanged, but is correctly labeled **player-overlap-exposed**.
The supplemental cold-player score uses only holdout rows whose exact source
`Player` string is absent from the creator training rows. `Player` is not a
model feature. The source has no season or date column, so this protocol makes
no temporal-generalization claim.

## Frozen arms

| Arm | DarkoRegressor parameters beyond `random_state=4` and telemetry |
|---|---|
| `default` | `{}` |
| `auto_lr_early_stopping_refit` | `early_stopping=True`, `early_stopping_rounds=None`, `validation_fraction=0.1`, `use_best_model=True`, `refit=True`, `refit_strategy="exact"` |

The candidate leaves `learning_rate=None` and `iterations=1000`, exactly as the
current default. Automatic patience therefore resolves from the fitted
learning rate. Every external fold creates its internal validation split from
that fold's training rows only, selects a best prefix, and then exactly refits
that prefix on the full external-fold training data.

## Quality and guardrail evidence

For both arms, persist all external-fold predictions and hashes, per-fold R²,
mean R², the unchanged overlap-exposed team-holdout score, the cold-player and
seen-player subset scores, full holdout predictions and hashes, resolved
learning rates, selected and final tree counts, selection and final stop
reasons, phase timings, and resolved thread count.

The candidate may advance only if all quality gates pass:

1. mean 10-fold R² is not below the default;
2. at least 6 of 10 folds improve;
3. the mean R² delta remains nonnegative after omitting any one fold;
4. overlap-exposed team-holdout R² is not below the default; and
5. cold-player R² is not below the default.

This is a diagnostic advance decision only. Passing does not authorize a
production-default change.

## Clean timing

Run three fresh-process reciprocal blocks:

1. `default`, candidate;
2. candidate, `default`;
3. `default`, candidate.

Each process receives the full machine thread allocation, warms one complete
first-fold fit and prediction outside timing, then times all 10 folds
sequentially. Prediction hashes and fitted structural metadata must match
exactly across the three repetitions for each arm. Timing is admissible only
if each arm's maximum/minimum steady-time ratio is at most 1.20. A material
speed gain requires the candidate median steady time to be at least 20% below
the default median.

## Conditional kernel profile

Profile tree-building kernels at the basketball shape only if every quality
gate passes, timing is stable, and the material-speed gate passes. Capture
golden prediction hashes before profiling and verify them afterward. If any
quality or speed gate fails, record that profiling was intentionally skipped;
do not optimize a rejected policy.
