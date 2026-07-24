# Feature recipes

These are opt-in Tier-E recipes, not automatic recommendations. Each section
states both the use case and the strongest observed counterevidence. Validate
on a split that matches deployment—especially entity- or time-disjoint splits
for sports data.

## Automatic group-centered categorical crosses

```python
model = DarkoRegressor(
    categorical_crosses=True,
    random_state=4,
)
model.fit(X_train, y_train, cat_features=categorical_columns)
```

This selector tests whether subtracting within-category numeric means creates
useful generic interaction features, using separate control and augmented
auditions. It never forces the crosses: inspect
`group_centered_categorical_crosses_["selected"]` and `["reason"]` after fit.
The two auditions make eligible fits materially more expensive than the
default path.

On the spent Diamonds development coordinates, the automatic selector reduced
RMSE to `0.7245×` the control. A smaller healthcare coordinate was below the
selector's row floor; forcing the same mechanism there measured `1.0081×`, so
small-data support is deliberately deferred rather than inferred. On a
player-disjoint basketball guardrail with mixed numeric and categorical
features, the selector engaged in all 11 fits and measured `0.9960×` fold
RMSE, `0.9969×` held-team RMSE, and `0.9940×` cold-player RMSE. These are
development and guardrail measurements, not a universal quality promise.

The public v1 opt-in is limited to single-model scalar-RMSE
`tree_mode="catboost"` regressors. Ensembles, classification, distributional
or interval modes, automatic tree-mode selection, presets, callbacks, refit,
ordered or ordinal categorical paths, automatic learning-rate probes, and
linear residual/leaf modes must be evaluated separately and therefore fail
loudly when combined with this opt-in. Leaving `categorical_crosses=False`
keeps the prior engine exactly.

## Ensemble v3

```python
from darkofit import DarkoRegressor

model = DarkoRegressor(
    ensemble_mode="v3",
    n_ensembles=8,
    random_state=4,
)
model.fit(X_train, y_train)
```

V3 is the fixed eight-member, 80%-without-replacement OOB recipe. Use
`ensemble_bootstrap="groups"` and pass `groups=` when complete entities must
stay together. The default member policy uses learning rate `0.15` and column
sampling `0.85`; dedicated `ensemble_member_*` parameters can override either
member value without changing the base estimator's settings.

On the frozen 13-case development panel, v3 beat the corresponding single
model in all 13 cases and reduced pooled error to `0.9655×`. This is an
explicit quality/cost trade: median fit time was `5.03×` the single, prediction
time was `6.25×`, and serialized size was `6.18×`; peak process RSS was
`1.09×` in the measured harness. Those figures characterize the tested panel
and machine rather than promising the same ratios elsewhere. Keep the default
single model when latency or storage matters more than the measured quality
gain.

## Noisy or heavy-tailed targets

```python
from darkofit import DarkoRegressor

student = DarkoRegressor(loss="StudentT", tree_mode="lightgbm")
median = DarkoRegressor(loss="MAE")
```

Use a robust head when the deployment objective values tail resistance or a
distributional Student-t model, not as a generic RMSE upgrade. On the frozen
basketball screen, Student-t and MAE improved cold-player R² by `0.00535` and
`0.00070`, respectively, but reduced mean creator-fold R² by `0.00798` and
`0.00846`. Both therefore remain closed as sports defaults.

## Split-score randomization

```python
model = DarkoRegressor(random_strength=0.5)
```

`random_strength` adds deterministic, seed-controlled noise to split scores.
It can reduce brittle split selection on noisy targets. The initial basketball
screen improved mean R² by `0.00212` and cold-player R² by `0.00730`.
Fresh multi-season confirmation was effectively null (`+0.000036` equal-cell
R²), had a negative leave-one-cell-out result, and cost `1.835×` fit time.
Keep it opt-in and validate it on the actual deployment panel.

## Smooth numeric relationships

```python
model = DarkoRegressor()  # linear_leaves="auto"
rollback = DarkoRegressor(linear_leaves=False)
forced = DarkoRegressor(linear_leaves=True)
```

Local linear leaves fit a ridge-regularized slope inside each oblivious-tree
leaf. They are most plausible when nearby rows follow smooth numeric
relationships. The automatic selector uses a deterministic, group-safe
holdout and a two-standard-error paired-gain guard. It fits the chosen lane
from scratch and records the split, scores, uncertainty, reason, and final
resolution in `automatic_linear_selector_`.

The current evidence is deliberately asymmetric: automatic selection improved
the three Protein development coordinates by about 4.5–5.4%, improved three
smooth CTR23 tasks with no loss on the other six, and stayed bit-exact to
constant leaves on all three small 2020 basketball targets. On CTR23, the
audition cost `2.20x` fit time and `1.28x` prediction time in the recorded
run. Use `False` when that selection cost matters more than the possible
quality gain; use explicit `True` only when the deployment data already
justifies forcing local-linear leaves.

## Declared ordinal categories

```python
orders = {
    "size": ["small", "medium", "large"],
    "grade": ["C", "B", "A"],
}

model = DarkoRegressor()
model.fit(X_train, y_train, ordinal_features=orders)
```

Declare order only when it comes from domain semantics. Unknown non-missing
values fail closed. The representation is target-free and does not add
columns.

If the input uses ordered pandas categoricals, the regressor can audition the
declared order automatically:

```python
model = DarkoRegressor()
model.fit(X_train, y_train, ordinal_features="select")
print(model.automatic_ordinal_selector_["selected"])
```

For a mapping such as `orders`, request the same audition with
`ordinal_selection=True`. The selector never treats bare integer category
codes as semantic order. It currently supports scalar-RMSE CatBoost fits and
records exact fallback or a loud incompatibility instead of silently changing
the request. `ordinal_features=None` is the rollback.

This can be powerful but is dataset-specific. On the two historical
declared-order development domains, the selector engaged in all six
seeded comparisons and reduced equal-dataset outer-test RMSE by about `18.5%`
without losing any coordinate. Those datasets were already used to develop
the mechanism; this is not evidence for arbitrary categoricals or new
datasets. Never infer a universal order merely because category labels look
sortable.

## Accuracy profile

```python
model = DarkoRegressor(preset="accuracy", random_state=4)
model.fit(X_train, y_train)
```

The accuracy preset applies the frozen A10 configuration: 10,000-round cap,
early stopping, LR `0.1`, L2 `3`, 128 bins, one target-stat permutation, and
validation selection among CatBoost, LightGBM, and hybrid tree modes. On the
spent 13-dataset development panel it measured `2.44%` lower RMSE than
ChimeraBoost 0.14.1 and `3.64%` lower than the historical DarkoFit default,
but Diamonds supplied `87.6%` of the ChimeraBoost advantage. It also used the
slower non-oblivious modes often enough to raise inference cost materially.

Treat it as an opt-in, validation-backed profile. Inspect
`model.model_.auto_params_["preset"]` and
`model.tree_mode_selection_` after fitting.
