# Feature recipes

These are opt-in Tier-E recipes, not automatic recommendations. Each section
states both the use case and the strongest observed counterevidence. Validate
on a split that matches deployment—especially entity- or time-disjoint splits
for sports data.

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
model = DarkoRegressor(
    linear_leaves=True,
    tree_mode="catboost",
    loss="RMSE",
)
```

Local linear leaves fit a ridge-regularized slope inside each oblivious-tree
leaf. They are most plausible when nearby rows follow smooth numeric
relationships. On three spent smooth development datasets, fixed linear
leaves reduced equal-task RMSE by `7.97%`; a validation selector retained
most of the gain. On a fresh 14-lineage panel, however, the selector's
aggregate improvement was only `1.07%`, with the best lineage carrying most
of it. It remains a narrow opt-in, not a default.

Do not use this recipe as a noisy-sports default. A random-split selector
regressed creator-fold, held-team, and cold-player basketball quality. Group-
aware selection safely declined every basketball fold but is not yet a public
automatic policy.

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

This can be powerful but is dataset-specific. In categorical development,
declared order reduced Diamonds RMSE by about `25%`, while an automatically
identified ordinal policy regressed the FPS benchmark by about `32%` and its
worst split by `144%`. Never infer a universal order merely because category
labels look sortable.

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
