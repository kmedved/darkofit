# Native ordinal C2 development and confirmation protocol

Status: **frozen before registry construction, runner implementation, or any
C2 target outcome inspection**.

Date frozen: 2026-07-17.

## Question and evidence boundary

Track C2 asks whether the C1 native ordinal-at-binning mechanism preserves the
large target-free representation benefit previously observed for semantically
ordered categorical values, without the prediction-width penalty that closed
the earlier column-adding safe-ordinal implementation.

The basketball C1 screen authorized development only. This protocol therefore
has two forward-only tiers:

1. a spent development panel used to establish mechanism engagement, quality,
   exact no-engagement behavior, and operating cost;
2. a target-unseen confirmation panel that may be scored only if every
   development and power gate passes.

Neither tier touches the CTR23 lockbox. A passing result supports the explicit
`ordinal_features` capability only. It does not authorize guessing order for
nominal categories or changing the global default.

## Frozen declarations and source boundary

`benchmarks/native_ordinal_c2_declarations.json` is binding. It was authored
against clean DarkoFit `main` at
`a74299e67307f44675c4f2b73d581a633885387b` and clean ChimeraBoost at
`851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`, before any C2 model fit.

The registry builder must be committed and pushed on DarkoFit `main` before it
runs. It may inspect feature values, feature metadata, official OpenML split
indices, and opaque target digests. It must not print, summarize, rank, or use
target values. Registry selection and ordered-category declarations are
target-blind.

Every task uses official OpenML repeat `0`, folds `0`, `1`, and `2`, sample
`0`. The builder must bind task ID, dataset ID/name, default target name,
feature-only fingerprint, opaque target digest, exact feature domains, and
official split hashes. Any drift fails closed.

## Development panel

The following tasks are already spent and are development-only:

| Task | Dataset | Role |
|---:|---|---|
| 361236 | auction_verification | exact nominal no-engagement |
| 361252 | video_transcoding | exact nominal no-engagement |
| 361268 | fps_benchmark | engaged: ordered versions/settings |
| 361622 | cars | exact numeric no-engagement |
| 363631 | diamonds | engaged: cut/color/clarity |
| 363372 | bookprice_prediction | exact nominal no-engagement |
| 363375 | ae_price_prediction | engaged: monetary `mrp` |
| 363471 | munich-rent-index-1999 | engaged: location quality |

Only declarations with a non-empty `ordinal_features` map enter the positive
quality estimand. Empty-map tasks are exact no-engagement controls; their
candidate and control predictions, scores, logical model state after removing
only the empty ordinal declaration, and preprocessing state must be bitwise
identical.

FPS orders only explicitly versioned or intensity-like fields. CPU/GPU names,
architectures, memory types, and game names remain nominal. Diamonds uses the
published orders already frozen in the prior ordinal campaign. Apparel `mrp`
uses numeric currency order. Munich location is source-defined as average,
good, and top.

## Target-unseen confirmation registry

Five independent lineages are declared before development outcomes:

| Task | Dataset | Ordered predictor | Semantic basis |
|---:|---|---|---|
| 363221 | nwtco | disease stage I–IV | survival/cchs package documentation |
| 363217 | flchain | FLC group 1–10 | survival package analysis grouping |
| 363227 | tumor | Charlson score | pammtools documentation |
| 363226 | patient | ICU admission year | pammtools documentation |
| 363201 | rdata | diagnosis year | relsurv documentation |

The builder must exclude a confirmation task on any exact OpenML ID/name hit
in the TabArena/ChimeraBoost exposure catalog, structured repository exposure
at the prefreeze revisions, or exact/target-blind near-lineage match to CTR23,
the spent I3 registry, or a C2 development task. Canonicalization ambiguity
also excludes. The current ChimeraBoost catalog exposes names and OpenML
dataset IDs rather than feature sketches, so it supports exact exposure checks
but not a separate renamed-copy near-lineage calculation; this limitation is
recorded rather than overclaimed. All five declarations must survive; there is
no substitution after scoring.

The confirmation registry is frozen before development execution. Confirmation
targets remain outcome-unseen until the development analyzer emits an explicit
authorization.

## Frozen arms

Both arms use the same official OpenML train/test indices and:

```python
DarkoRegressor(random_state=4)
```

- `control`: product-default fit with unchanged native categorical handling.
- `candidate`: the identical fit with only the task's exact declared
  `ordinal_features` mapping.

There is no tuning, early-stopping eval set, target-derived representation,
per-task model policy, manual learning rate, or retry after result inspection.
The explicit map must remove only its declared columns from nominal target
statistics and put one numeric code column per declaration through the existing
numeric binner. Added-column count and added target-stat-block count are zero.
Unknown non-missing values fail closed; missing values use the numeric missing
bin.

## Execution and timing

Each coordinate runs as a reciprocal pair of isolated fresh workers. Alternate
order by coordinate index. Every worker uses a unique empty Numba cache,
disables import warmup, performs one explicit untimed `darkofit.warmup()`, and
then fits and predicts one arm. Runtime, package versions, source state,
environment, cache state, warnings, fitted metadata, ordinal telemetry,
preprocessor state, prediction bytes, RMSE, fit/predict time, and peak RSS are
recorded.

Quality uses paired outer-test RMSE. A deterministic 20% tail of each official
training partition, selected only from row indices and seed
`20260717 + task_id + fold`, is a diagnostic validation view; the remaining
80% is the fit view. The official OpenML test partition is never used for
training, model selection, or early stopping. Both arms share identical
fit/validation/test row hashes.

Timing is interpreted through same-coordinate candidate/control ratios. Raw
per-arm dispersion is diagnostic. No failed coordinate, warning, deadline hit,
or outlier may be dropped or rerun into the same evidence namespace.

## Development gates

All gates are conjunctive:

1. complete 48-worker grid, clean/pushed frozen source, exact runtime/data/split
   bindings, unique warmed caches, finite metrics, and zero failures or
   unexpected warnings;
2. exact ordinal telemetry and preprocessing engagement on all four engaged
   tasks: exactly the declared columns and categories, zero added columns, zero
   added target-stat blocks, and no target use;
3. bitwise-exact predictions and normalized logical state on every
   no-engagement coordinate;
4. across the four engaged tasks, equal-task geometric-mean candidate/control
   test RMSE ratio at most `0.980`, at least three task wins, worst task ratio
   at most `1.020`, and worst split ratio at most `1.050`;
5. each engaged task's validation ratio at most `1.020`;
6. across all 24 paired coordinates, median fit ratio at most `1.150`, median
   public-predict ratio at most `1.100`, and median peak-RSS ratio at most
   `1.100`;
7. fit and prediction paired-ratio IQR/median at most `0.150`.

Failure closes this native implementation shape or sends it back to
development. Thresholds cannot be weakened after inspection.

## Preregistered confirmation power

The registry builder simulates the five-lineage confirmation design from the
33 spent, fresh-coordinate `safe_ordinal / fixed_base_native` test-RMSE ratios
in
`benchmarks/tabarena_regression_ordinal_confirmation_paired_splits.csv`.
It performs 200,000 deterministic cluster-bootstrap simulations with seed
`20260717`: for each simulated lineage, select Airfoil or Diamonds with equal
probability, sample three splits within that source lineage, and retain only
25% of the observed log effect (75% shrinkage toward no effect). Each
simulation evaluates the point gates and the exact one-sided 95% task-bootstrap
upper gate. For five tasks, the builder enumerates all `5^5` bootstrap samples
through their 126 multinomial count vectors rather than approximating that
interval with another random draw.

A simulation passes the frozen confirmation point gates below. The registry is
adequately powered only when at least 80% of simulations pass. This calculation
uses spent outcomes only and runs before C2 targets are scored.

## Confirmation gates

Confirmation may run exactly once only after a development pass and a registry
power probability of at least `0.80`. All gates are conjunctive:

1. complete 30-worker grid with the same integrity, telemetry, and operating
   contracts as development;
2. equal-lineage geometric-mean candidate/control test RMSE ratio at most
   `0.995`;
3. a one-sided 95% task-cluster bootstrap upper bound, using 100,000 draws and
   seed `20260718`, strictly below `1.000`;
4. at least three of five lineage wins;
5. worst lineage ratio at most `1.020` and worst split ratio at most `1.050`;
6. every lineage validation ratio at most `1.020`;
7. median fit, prediction, and peak-RSS ratios at most `1.150`, `1.100`, and
   `1.100`, respectively, with fit/prediction paired-ratio IQR/median at most
   `0.150`.

A confirmation failure closes promotion. It cannot be repaired by deleting a
lineage, changing an order, tuning a model, or opening the lockbox.

## Decisions

- Development failure: `close_native_ordinal_c2_development`.
- Development pass but inadequate power:
  `close_native_ordinal_c2_confirmation_underpowered`.
- Development and power pass:
  `authorize_native_ordinal_c2_confirmation_once`.
- Confirmation pass: retain and document explicit native ordinal handling,
  then run the deferred mode-mix attribution; no automatic nominal-category
  ordering follows.
- Confirmation failure: retain only as experimental/explicit if warranted by
  the development evidence; do not promote or touch the lockbox.
