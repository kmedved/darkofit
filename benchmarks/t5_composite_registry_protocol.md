# T5 composite confirmation registry and power freeze

## Purpose

Freeze a genuinely outcome-unseen, contamination-screened confirmation panel
for one exact DarkoFit composite policy before any model is fitted on a
candidate task. This registry authorizes only T5 execution. It does not change
defaults, open any earlier lockbox, or turn development measurements into
confirmation evidence.

Registry construction may use task metadata, feature schemas, source
descriptions, official split indices, and opaque semantic fingerprints. It may
read target bytes only to produce opaque hashes used by the contamination
screen. It must not compute target statistics, fit a model, score a prediction,
or inspect a result involving a candidate task.

## Panel

The declaration names 25 supervised-regression tasks from 25 source-reviewed
lineages:

- 9 `smooth_numeric`;
- 9 `mixed_categorical`; and
- 7 `applied_noisy`.

The strata describe coverage; they are not votes or separate quality gates.
Repeat 0, folds 0–2, sample 0 are frozen, producing 75 outer coordinates.
Every dataset receives equal weight after geometric reduction across its three
coordinates.

The panel deliberately rejects uploads whose nominal target was merely exposed
through a regression task. Survival-time outcomes remain ordinary scalar
regression targets for this campaign; no survival-specific claim is made.

## Contamination boundary

The builder binds:

1. DarkoFit immediately before the registry files, commit `da5e2d3`;
2. clean ChimeraBoost 0.15.0, commit `851ab7f`;
3. ChimeraBoost's OpenML, Grinsztajn, PMLB, high-cardinality, and TabArena
   benchmark catalogs;
4. the frozen CTR23-v3 source/confirmation/lockbox records;
5. both earlier fresh-selector registries and the native-ordinal C2 registry;
6. exact task IDs, OpenML dataset IDs, normalized names, conservative
   six-character name containment, repository references at the bound heads,
   opaque semantic fingerprints, and CTR23's near-lineage alarms; and
7. pairwise near-lineage alarms among the 25 nominees.

Any source or candidate collision, ambiguous canonicalization, name/target/ID
drift, non-regression task, split drift, duplicate lineage, or pairwise
near-match fails the freeze closed. The one ordinal declaration (`Riga:
condition`) is checked against the complete feature domain without consulting
the target.

### Target-blind registry amendment

The committed v1 builder stopped before writing an artifact because OpenML
task `168887` (`CD4`) names `Future_CD4` as its task target while the dataset
metadata has no default target. The shared semantic-fingerprint builder
correctly rejects that ambiguity. No model was fitted, no target statistic was
computed, and no candidate outcome was scored. The nominee was replaced
target-blind with task `363204` (`dataFTR`, target `time`), preserving the
25-lineage and 9/9/7 stratum design. The failed attempt is recorded in
`t5_composite_registry_invalid_attempt.md`.

## Exact composite nominee

The evaluated unit is a selection-guarded automatic policy, not a bare
hyperparameter arm.

1. `n_outer_train < 2,000`: return the current product default exactly.
2. Otherwise reserve a shared, deterministic 20% inner validation split with
   `ShuffleSplit(random_state=4)`.
3. Fit a current-default audition on the inner training rows.
4. Fit the challenger with 10,000 rounds, learning rate 0.1, L2 3, 128 bins,
   `ts_permutations=1`, automatic tree mode, and 100-round tree-mode
   auditions followed by the selected lane's full-budget fit.
5. If CatBoost tree mode wins, separately race constant and per-leaf-linear
   leaves at full budget. Other tree modes remain constant-leaf.
6. Apply only complete ordinal maps frozen in this registry.
7. From the selected uncrossed challenger, rank numeric features by split gain,
   take at most six, append every pair's difference and product, and refit the
   selected tree/leaf lane. Crosses engage only when validation RMSE improves
   by at least 5% (`cross <= 0.95 * uncrossed`).
8. The whole challenger engages only when its validation RMSE improves by at
   least 0.5% over the current-default audition
   (`challenger <= 0.995 * control`). Otherwise the final model is an exact
   current-default full-data fit.
9. Refit the selected configuration from scratch on all outer-training rows
   using the selected resolved learning rate and exact selected best-prefix
   round count. No outer-test row participates in any decision.

The runner must record every audition, selected mode/leaf/cross/ordinal lane,
best iteration, resolved learning rate, stop reason, final-refit parameters,
prediction hash, fit/predict seconds, and peak RSS. Exact declines must match a
separate current-default full-data fit byte for byte.

ChimeraBoost 0.15.0 and the installed CatBoost product default run on the same
outer coordinates. They are competitive comparators, not ingredients in the
DarkoFit promotion decision.

## Tier-D decision rule

All gates are conjunctive and use candidate/control outer-test RMSE:

1. equal-dataset geometric-mean ratio `<= 0.995`;
2. one-sided hierarchical 95% bootstrap upper bound `<= 1.002`;
3. least-favorable leave-one-dataset-out ratio `<= 0.998`;
4. worst dataset ratio `<= 1.005` (selection-guarded route);
5. equal-dataset total fit-seconds ratio `<= 6.0`, worst-dataset fit ratio
   `<= 12.0`, equal-dataset prediction ratio `<= 1.5`, and equal-dataset peak
   RSS ratio `<= 2.5`; and
6. complete execution, finite outputs, source/hash integrity, no imputation,
   and no protocol deviations.

The hierarchical bootstrap resamples datasets, then the three coordinates
within each selected dataset, with seed `20260717` and 100,000 replicates.
Win counts are descriptive only.

## Design-time power

The plausible-effect pool is deliberately conservative and outcome-spent:

- the 13-dataset A10-versus-product-default development table, transformed by
  the same 0.5% validation guard (`test ratio` when the validation ratio is
  `<= 0.995`, otherwise `1.0`);
- the single most favorable A10 lineage (Diamonds) removed before simulation,
  so power does not depend on the historical outlier; and
- the three dataset effects from the spent 5%-margin smooth-cross nominee,
  each included once.

This yields 15 lineage profiles. With seed `20260717`, draw 25 profiles with
replacement in each of 200,000 simulations. A simulation applies the four
quality gates above; its design-time one-sided upper bound is the normal
bootstrap approximation on log ratios. Execution is authorized only when at
least 80% of simulations pass. This is a design calculation conditional on
spent development effects, not evidence that T5 works.

## No-rerun rule

Once any T5 outcome is scored, the registry, candidate, folds, margins,
comparators, aggregation, bootstrap, and gates are immutable. Failure closes
this candidate. Repairs may resume only from raw records when they do not
change fitted models or predictions; any model-affecting change requires a new
candidate and new outcome-unseen panel.
