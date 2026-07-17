# Fresh smooth/process selector confirmation

## Question

Does the exact 3% validation-margin local-linear selector generalize to the
outcome-unseen registry v2 panel, preserve categorical and noisy-tabular
guardrails, and beat ChimeraBoost 0.15 product defaults?

This is the sole confirmation run on registry v2. No task, fold, arm, or
threshold may be changed after execution begins.

## Frozen evidence boundary

- Registry v1 file SHA-256:
  `37799ed0b788af3c1d69c8f0f7cf37a656fde998ce2d54b5c4a2196c369df4c3`.
- Registry v2 file SHA-256:
  `0d878d690e32f6781a170fa3e5c232eef13d20d51d25b352c96a20ddc87e3970`.
- Coordinates: the exact 20 tasks × repeat 0 × folds 0–2 × sample 0 in
  registry v1: 60.
- Primary: 14 `smooth_process` lineages. Guardrails: 3 `categorical` and 3
  `noisy_tabular`.
- Related tasks are not loaded or counted.
- Clean ChimeraBoost 0.15.0 must be commit `851ab7f`.
- Three task workers run concurrently, six threads each. Each task/config
  worker warms fold 0 once outside timing, then runs all three frozen folds.
- Job failure, imputation, source drift, task/data identity drift, or
  incomplete output fails closed. No task may be dropped.

## Frozen arms

1. `darko_default`: current `DarkoRegressor(random_state=4)`.
2. `smooth_margin_selector`: the exact preceding policy—one deterministic 20%
   weighted-target-stratified internal split; constant and local-linear
   selection fits on identical rows with explicit validation, early stopping,
   best-model retention and no refit; select linear only at relative validation
   RMSE improvement `>=0.03`; fit the selected leaf type from scratch on the
   complete outer training fold.
3. `darko_linear_fixed`: current defaults plus `linear_leaves=True`.
4. `chimera_product`: unmodified ChimeraBoost 0.15.0 product defaults.
5. `catboost_product`: CatBoost regressor defaults with random seed 4,
   six threads, quiet output and file writing disabled.

All arms receive the same raw train/test indices and declared categorical
columns. CatBoost alone receives a semantics-preserving categorical transport:
categorical values become strings and missing categorical values use one
fixed sentinel, because CatBoost's public API rejects numeric NaN categorical
tokens. No target or numeric feature is transformed.

## Frozen gates

The selector advances to a lockbox power freeze only if every gate passes.

Primary 14-lineage gates:

1. selector/default equal-lineage geometric-mean RMSE ratio `<=0.98`;
2. selector wins at least 9 of 14 lineages;
3. no primary lineage selector/default ratio exceeds `1.02`; and
4. selector/ChimeraBoost product equal-lineage ratio `<=1.00`.

Guardrail gates, separately for categorical and noisy-tabular:

1. selector/default equal-lineage ratio `<=1.00`; and
2. no lineage ratio exceeds `1.02`.

CatBoost, fixed-linear, split records, selection frequency, wall/fit/predict
time and RSS are report-only. Timing is not a claim because config waves are
not reciprocal; basketball already established the selector's frozen cost
boundary.

## Interpretation

Passing authorizes only a new target-blind power calculation from the observed
fresh lineage effects. The CTR23 lockbox stays sealed unless that separately
frozen calculation reaches 80%. Failure closes this selector shape for
promotion; development may continue on a materially different mechanism
without using lockbox data.
