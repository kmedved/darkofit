# TabArena regression accuracy-shootout protocol

_Status: source-frozen development protocol. No result from this campaign may
be presented as independent confirmation. The 13 tasks and all selected outer
coordinates have already informed policy development._

## Objective and decision boundary

This campaign answers one narrow development question: can an accuracy-oriented
DarkoFit profile reach the current ChimeraBoost 0.14.1 default on the spent
13-dataset TabArena regression panel?

The campaign also isolates how much of the candidate movement comes from the
fixed parameter/horizon base and how much comes from validation-selected tree
mode. It does not change package defaults, validate a generic representation
policy, or establish out-of-sample superiority.

Development parity is defined before execution as an equal-dataset geometric
mean

```text
RMSE(DarkoFit accuracy candidate) / RMSE(ChimeraBoost 0.14.1) <= 1.000
```

on the 39 fixed outer coordinates below. It is necessary but not sufficient:
the native candidate must also have a worst dataset-level `A10/P` RMSE ratio no
higher than 1.02 and a worst leave-one-dataset-out `A10/M` geometric-mean ratio
no higher than 1.01. These harm checks prevent one influential task (especially
Diamonds) from carrying the aggregate. The report must include dataset and
split win counts, the worst dataset and split, and every leave-one-dataset-out
estimate. These rules only steer the subsequent profile freeze. Confidence
intervals and registered-leaderboard comparisons are descriptive on this spent
panel.

## Frozen aggregation hierarchy

All quality gates use paired **outer test** RMSE. For arm `X`, comparator `Y`,
dataset `d`, and outer coordinate `s` in `r0f0`, `r1f1`, and `r2f2`, define the
paired split ratio

```text
r[d,s; X/Y] = test_RMSE[d,s; X] / test_RMSE[d,s; Y]
```

from the two results on that exact task/repeat/fold coordinate. Ratios are
formed before any reduction; neither RMSE values nor squared errors are first
averaged across splits. Every numerator and denominator must be finite and
strictly positive, and the complete paired grid is required. A missing or
invalid pair invalidates the campaign rather than being dropped.

Within each dataset, reduce the three paired split ratios in log space:

```text
R[d; X/Y] = exp(mean_s(log(r[d,s; X/Y])))
```

The equal-dataset campaign estimate then gives each of the 13 datasets one
vote, irrespective of row count or test-set size:

```text
G[X/Y] = exp((1 / 13) * sum_d(log(R[d; X/Y])))
```

This same paired-split, within-dataset, then equal-dataset hierarchy applies to
every reported primary contrast. The three development gates are exactly:

```text
parity:        G[A10/M] <= 1.000
dataset harm: max_d R[d; A10/P] <= 1.020
LODO harm:    max_k exp((1 / 12) * sum_(d != k)(log(R[d; A10/M]))) <= 1.010
```

The report emits all 13 dataset ratios and all 13 leave-one-dataset-out (LODO)
ratios; “worst” means the numerical maximum. Descriptive split and dataset
wins/losses/ties use ratios below/above/exactly equal to one, and the worst
split is likewise the maximum paired split ratio. Validation RMSE is reported
with the identical hierarchy and explicit `validation` labels, but it does not
replace outer test RMSE in any gate above. No row-weighted, pooled-error,
ratio-of-means, median, or selectively complete alternative may be substituted.

## Fixed tasks and coordinates

The primary lane uses `r0f0`, `r1f1`, and `r2f2` for the exact 13 tasks from
the attested same-machine campaign:

| Order | Dataset | OpenML task |
| ---: | --- | ---: |
| 1 | `airfoil_self_noise` | 363612 |
| 2 | `Another-Dataset-on-used-Fiat-500` | 363615 |
| 3 | `concrete_compressive_strength` | 363625 |
| 4 | `diamonds` | 363631 |
| 5 | `Food_Delivery_Time` | 363672 |
| 6 | `healthcare_insurance_expenses` | 363675 |
| 7 | `houses` | 363678 |
| 8 | `miami_housing` | 363686 |
| 9 | `physiochemical_protein` | 363693 |
| 10 | `QSAR-TID-11` | 363697 |
| 11 | `QSAR_fish_toxicity` | 363698 |
| 12 | `superconductivity` | 363705 |
| 13 | `wine_quality` | 363708 |

Every new outer job uses the same eight AutoGluon child folds, one bag set,
model seed policy, validation split, sample weights, 18-CPU allocation, and
single 3,600-second soft **outer-job** budget as the source campaign. The eight
children do not receive 3,600 seconds apiece.

## Frozen arms

The primary causal chain uses native representation throughout:

| Code | Role | Execution | Manual DarkoFit configuration |
| --- | --- | --- | --- |
| `P` | Product default | Reuse | `{}` |
| `B10` | Parameter/horizon attribution | New | `iterations=10000`, `tree_mode="catboost"`, `l2_leaf_reg=3`, `max_bins=128`, `learning_rate=0.1`, `ts_permutations=1` |
| `A10` | Accuracy candidate | New | identical to `B10`, except `tree_mode="auto"` |
| `M` | Live ChimeraBoost comparator | Reuse | exact v0.14.1 default |
| `C` | CatBoost reference | Reuse | exact 1.2.10 default |

All unspecified parameters retain the DarkoFit 0.9.0 default. The campaign
nevertheless freezes the implicit behavior that affects selection:

- automatic tree-mode candidate order is exactly `catboost`, `lightgbm`, then
  `hybrid`;
- selection minimizes each candidate's original validation RMSE and an exact
  tie selects the earlier candidate in that order;
- `early_stopping=True`, `early_stopping_rounds=None`, and the fixed 0.1
  learning rate resolve to 50 rounds of patience; `use_best_model=True` retains
  the best validation prefix;
- `ordered_boosting="auto"` resolves off for scalar regression;
- `linear_residual=False`, `auto_learning_rate_probe=False`,
  `target_ordered_cat_codes="off"`, and `ts_permutations=1`.

Children fit sequentially. At each child start, AutoGluon passes only the
unspent remainder of the 3,600-second outer allocation. The child's monotonic
callback derives one soft deadline from that remainder, including the
source-frozen safety margin. The `A10` wrapper charges all three internal
candidates to that one child deadline; neither a later candidate nor a later
bag child resets the clock to 3,600 seconds. A tree already in progress may
finish after the soft deadline, as in the source campaign, but the deadline
state and elapsed time must be recorded.

All three `A10` candidates must fit and emit finite validation scores for every
child. Any skipped candidate, deadline hit, `time_limit` stop, non-finite score,
or change in candidate count/order invalidates the complete campaign rather
than silently changing the selector.

The primary contrasts are `A10/M`, `A10/P`, `A10/B10`, and `B10/P`. The first
is the parity decision; the remaining contrasts are attribution only.

## Representation diagnostics

Representation is deliberately excluded from the primary candidate. Two
non-poolable diagnostics may run on Airfoil and Diamonds after their
benchmark-only transforms exist:

1. `A10 + resolver`: only target-free automatic rules are allowed. Numeric
   strings may be restored only when every non-missing training value parses
   losslessly and validation/test values satisfy the frozen training schema;
   pandas categoricals are ordinal only when `ordered=True`. Otherwise the
   feature remains on the native categorical path.
2. `A10 + resolver + declared maps`: the same resolver plus an explicit,
   source-declared order supplied before fit. Unknown values fail closed. The
   mapping may not depend on labels' lexical order, frequency, target values,
   validation rows, test rows, dataset name checks inside the resolver, or a
   fitted model.

The resolver-only lane should be active for Airfoil-class numeric strings and
a no-op for unordered Diamonds labels. The declared-map lane may additionally
restore the published Diamonds orders. Both lanes report marginal movement
relative to native `A10`; neither can establish core engine parity or enter the
primary aggregate. Automatic resolver behavior cannot ship from this spent
diagnostic alone. Declared maps may remain an explicit adapter option, but are
never an automatic core-model policy.

## Reused evidence contract

Only the already-attested quality observations for `P`, `M`, and `C` may be
reused. The source is commit `a1ff4b74510b5e314bb41c27b40544910741543d`,
whose DarkoFit package subtree is
`52278b0326419a45a72bdfd3afcfc13019087838`. The exact ChimeraBoost comparator
is tag commit `9c9ea6e704a9fe2bfe6d6c284b22de73914be048`; CatBoost is wheel version
1.2.10.

The runner/analyzer must revalidate these committed inputs before joining a new
candidate result:

| Artifact | SHA-256 |
| --- | --- |
| `tabarena_regression_same_machine_primary_paired_splits.csv` | `3e7bbe21e0ffe40771f2065dc252dbd4314550f8ab350f2fbed9641401b341b1` |
| `tabarena_regression_same_machine_summary.json` | `ca23618bdc3d9e0ab38557e7738c66e95827945ad34e3eb63005f253c92ccf01` |
| `tabarena_regression_same_machine_completion_attestation.json` | `213f462aa06103e97864ecd786b75e8fd8e11743c77f556262fa39bdb3e1b7d9` |
| `tabarena_regression_same_machine_run_manifest.json` | `2869acaaa4bcc8319d9ba03744a4a9ca8602ed349553a031c3d84ab537de72ee` |

Reuse is allowed only when a committed cross-attestation matches the prior
Python and dependency lock, hardware identity, CPU allocation, task schemas,
outer coordinates, child-fold contract, product package subtree, base adapter
bytes, and every source artifact hash above. Any mismatch requires rerunning
all affected arms; it may not be waived. Any base-adapter byte change,
including telemetry or resolver edits, requires rerunning `P` across all 39
coordinates. Fixed numeric and categorical fixture checks are diagnostic only;
they cannot authorize reuse or waive that rerun.

Reused training, inference, and memory values are historical same-machine
context, not paired timing controls for the newly executed arms. Only `A10`
versus `B10`, run in a balanced alternating order after identical warmup, may
support a causal operational contrast. Comparator timing must be rerun if a
freeze decision depends on it.

## New execution grid and warmup

The mandatory new primary grid is 13 datasets x 3 coordinates x 2 arms = 78
outer jobs and 624 selected child models. Internal auto-mode candidates are
additional fits and must be counted separately in fitted metadata.

`A10` and `B10` alternate first/second position within every dataset and repeat
block. The exact ordered grid is emitted and hashed before execution. Warmup is
outside measured jobs, uses 18 threads, covers numeric and categorical data,
and exercises all three automatic tree modes plus prediction for each selected
tree family. Diagnostic resolver warmup is separate and cannot populate a
measured result cache.

Every result file is runner-owned, uniquely keyed by protocol digest, arm,
task, repeat, and fold. Resume may consume only byte- and provenance-validated
results from this exact campaign. There is no imputation or substitution.

## Required child and prediction telemetry

For every child, preserve the existing requested/attempted/completed/retained
rounds, resolved learning rate, selected lane and tree mode, stop reason,
deadline state, and complete automatic-mode candidate metadata. Add or derive:

- retained tree count;
- total retained node and leaf counts;
- maximum and mean retained depth;
- categorical feature count and generated raw-code/target-stat feature counts;
- repeated batch-prediction wall time split into external representation,
  wrapper preprocessing, binning, and tree traversal;
- the number of timed rows and repetitions and whether the first call was
  excluded as warmup.

Phase timing uses `time.perf_counter()` and records raw repetitions after a
discarded first call. The full end-to-end inference time remains authoritative;
phase values are diagnostic and must reconcile to it within explicitly
reported uninstrumented overhead. Current committed telemetry does not expose
reliable node counts or prediction-phase decomposition, so the runner may not
claim those fields until a dedicated profiler and tests land. If a phase
boundary cannot be measured without changing predictions, mark it unavailable
rather than estimating it. Missing diagnostic phase timing does not invalidate
the quality estimand; fabricated or inferred timing does.

## Development decision tree: at most two iterations

Iteration 1 is the frozen `A10` shootout.

- If all three development gates pass, the profile may freeze unchanged.
- If any gate fails, one and only one iteration 2 may compare two otherwise
  identical `A10` candidates: `linear_residual=False` and
  `linear_residual=True` with `linear_residual_alpha=1.0`,
  `linear_residual_features="auto"`, `linear_residual_fit_intercept=True`, and
  `linear_residual_standardize=True`. Each tree-mode candidate is fit in both
  lanes; selection minimizes original-target validation RMSE, and an exact tie
  favors `linear_residual=False` then the earlier frozen tree-mode candidate.
  All six candidates must complete within the same child deadline derived from
  the remaining 3,600-second outer allocation; candidate or lane transitions
  do not reset it. Selection stays wholly child-training/validation local;
  outer test values cannot select the lane.
- Inference has no veto and remains report-only throughout this development
  chain. Reused product-default timing is historical and cannot trigger a
  freeze decision. No mode restriction, cost-aware selector, or other
  prediction-changing cost policy may be designed from iteration-1 telemetry
  or consume iteration 2.
- If measured inference is operationally unacceptable, the accuracy profile
  may be disclosed as an explicit accuracy preset or abandoned. Any attempt to
  optimize its cost starts a separately preregistered campaign on new
  development data with a fresh, balanced product-default timing control; it
  cannot ship from this spent panel.

No unmeasured policy may be frozen. After iteration 2, or after an unchanged
iteration-1 pass, the complete profile freezes regardless of outcome. The
freeze includes parameters, horizon, mode candidates and selection rule,
linear policy, representation policy, resource policy, and source bytes.

## External confirmation and lockbox gates

The CTR23 registry and its confirmation/lockbox assignment must be committed
before any model is run on an eligible CTR23 task. Its contamination scope is
the union of known DarkoFit and ChimeraBoost-comparator development exposure,
because a task used to design either side is not neutral evidence for their
comparison. The full profile and exact outer-coordinate subset must be frozen
in a later execution commit before the confirmation panel is opened.

The proposed one-shot confirmation gate is provisional until the CTR23
coordinate inventory is frozen. At that freeze, it must define the bootstrap
hierarchy, tie tolerance, and whether the final claim rests on lockbox alone or
on a predeclared sequential combination. The current design target is:

- primary `D/M` equal-dataset test-RMSE ratio: one-sided hierarchical 95%
  upper bound below 1.000; a point estimate at or below 0.995 is desired but
  not required;
- guardrail versus product-default DarkoFit: a simultaneous max-regret upper
  bound no higher than 1.02; dataset point estimates above 1.01 are flagged;
- completeness: all prescribed jobs and child metadata, zero failures,
  imputations, cache substitutions, or unresolved DarkoFit stop reasons;
- resources: every outer job has one 3,600-second soft budget, each sequential
  child's callback is bounded by only the unspent outer allocation, and no
  child or internal candidate receives an independent 3,600-second reset;
  inference is reported with a 3x product-default soft owner flag;
- ties use a source-frozen numerical tolerance defined before execution;
  exact prediction equality and source-frozen structural no-ops are reported
  separately. Win counts are descriptive, never a substitute for the primary
  interval.

The hierarchical bootstrap resamples eligible datasets and then the actual
task-defined repeat/fold hierarchy within each sampled dataset; shared folds
are not treated as independent observations. Dataset-level guardrails use the
frozen within-task resampling unit and are reported with their limited effective
sample size. The lockbox runs once only after a complete confirmation pass,
under identical gates and with zero post-confirmation changes. The lockbox
runner must refuse to enumerate a job without a passing confirmation
attestation bound to the exact profile, registry, partition, source, runner,
analyzer, dependencies, and coordinate hashes.

## Explicit exclusions

- `ts_permutations=4` is excluded.
- No dataset-specific hyperparameter or tree-mode rule is allowed.
- Registered ChimeraBoost results are a secondary scoreboard column only.
- Resolver diagnostics cannot rescue a failed native parity contrast.
- No CTR23 model, prediction, validation score, or test score may be inspected
  while constructing the contamination registry or panel assignment.
