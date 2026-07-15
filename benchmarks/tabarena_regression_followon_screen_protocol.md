# TabArena scalar-regression isolated follow-on screen

Status: **source-frozen before execution**. This is an exploratory mechanism
screen, not a default-policy confirmation. It follows the attested cap-horizon
campaign, which formally retained the 1,000-round control. The decision is
copied into this source; the runner does not dynamically read or reinterpret
the prior campaign.

## Causal background and isolated arms

Every arm uses the same child-fold validation data, fold-wise seeds, one
eight-fold bag set, sequential local fitting, 3,600-second outer budget,
monotonic wall-clock callback, warmup, and base policy:

```text
iterations=1000
tree_mode="catboost"
l2_leaf_reg=3
max_bins=128
learning_rate=0.1
ts_permutations=1
linear_residual=False
early_stopping=True
use_best_model=True
```

The 39 native controls are shared analytically; no control is rerun for each
candidate. Each candidate changes exactly one declared lever:

| Arm | Only change | Dataset scope | Jobs |
| --- | --- | ---: | ---: |
| baseline | none | all 13 | 39 |
| auto | `tree_mode="auto"` | all 13 | 39 |
| linear | `linear_residual=True` | all 13 | 39 |
| ts4 | `ts_permutations=4` | 5 child-visible categorical datasets | 15 |
| ordinal | source-frozen domain representation | Airfoil, Diamonds | 6 |
| onehot | target-free low-cardinality representation | Fiat, Diamonds, Food, Healthcare, Miami, Wine | 18 |
| **Total** | | | **156 outer jobs / 1,248 child fits** |

The three screen coordinates are `r0f0`, `r1f1`, and `r2f2` (registered split
IDs 0, 4, and 8). Outer test RMSE is explicitly used for research selection;
these coordinates are spent and cannot later serve as confirmation data.

The TS4 scope is Airfoil, Used Fiat 500, Diamonds, Food Delivery, and Healthcare
Expenses. At the child-model boundary AutoGluon's upstream feature generator
has already converted Miami's and Wine's binary categoricals to numeric 0/1,
so `ts_permutations` is a structural no-op there. They are excluded from the
formal TS4 estimand and covered by identity/schema tests instead.
Native child metadata v2 binds both sides of AutoGluon's child-specific feature
alignment: the ordered external feature names/count/digest and the ordered
schema actually passed to DarkoFit. AutoGluon may remove a feature that is
constant inside one child training fold. Such a removal is accepted only when
the adapter records the exact dropped name, observes exactly one distinct value
with missing included, and the fitted schema is the order-preserving external
schema minus those audited constants. The runner also requires this entire
pre-representation audit to match the paired native control for all 744 native
candidate child fits.

The surviving categorical schema remains exact for all 13 datasets. The five
TS4 datasets must expose, respectively,
`attack-angle`, `model`, `cut/color/clarity`, the three Food categorical
columns, and Healthcare `region`; every TS4 child therefore proves the lever is
structurally active. Miami, Wine, and the other numeric panel datasets must
expose no native categorical columns. Healthcare `sex` and `smoker` are numeric
at the native boundary but remain declared semantic positions for its separate
one-hot mechanism test.

The auto arm is a one-parameter product-policy test, not a pure topology
ablation. LightGBM and hybrid candidates also select their mode-specific
categorical preprocessing (K-fold target statistics plus raw category codes),
so any result must be interpreted as the complete `tree_mode="auto"` policy.
Likewise, the linear arm enables DarkoFit's normal additive ridge-residual
lane when usable numeric features exist; it does not force linear tree leaves.
The analyzer therefore reports the selected mode and selected-lane counts.

## Representation safety boundary

The ordinal arm never invents ranks from category labels, first appearance,
lexical order, target means, validation rows, or test rows. It recognizes only
three exact input schemas and applies these source-frozen meanings:

- Airfoil: restore `attack-angle` to its physical numeric value. AutoGluon
  lexically compacts the source string labels before the child boundary, so a
  source-frozen 27-entry code-to-angle tuple is used (for example code 13 is
  3.0 degrees); compact codes are never treated as physical values directly.
- Diamonds: `cut` is `Fair < Good < Very Good < Premium < Ideal`; `color` is
  `J < I < H < G < F < E < D`; `clarity` is
  `I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF`. AutoGluon's pinned
  category-memory generator compacts the source labels alphabetically before
  the child boundary, so the adapter freezes the audited compact-code-to-rank
  maps rather than treating compact codes as ordinal values.
- Miami identity/schema check: `avno60plus` already arrives numeric 0/1 at the
  child-model boundary. Its declared mapping remains tested but is excluded
  from the formal ordinal quality estimand.

An unexpected column layout, categorical set, nonnumeric Airfoil value, or
undeclared Diamonds/Miami value fails the child fit. Missing ordinal values
remain numeric NaN.

These are the exact child-model schemas after AutoGluon's upstream feature
generator (which places surviving categorical columns after numeric columns).

The one-hot arm learns category identities from the child training rows only
and never reads the target. A categorical feature is one-hot encoded only when
that child observes at most eight nonmissing categories. Missing values get a
dedicated indicator and unseen validation/test categories are all-zero. A
categorical feature above the threshold remains categorical and continues
through DarkoFit's native target-statistic path. In particular, Food Delivery's
high-cardinality `Delivery_person_ID` stays native while `Type_of_order` and
`Type_of_vehicle` are one-hot. Dense output is capped at 256 features.

Per-child metadata records native external/fitted schema digests and audited
constant drops, or the source-frozen ordinal domain/training-fold one-hot
schema digest, encoded positions and category counts, remaining native
target-stat positions, validation transform count, and unseen-category count.
Completion fails unless every record proves the declared boundary.

## Ordering, resources, and deadlines

Jobs run as coordinate groups so the control is fit once. For each candidate
arm independently, its scoped occurrences alternate candidate-before-control
and candidate-after-control. Counts differ by at most one:

| Arm | Candidate before | Candidate after |
| --- | ---: | ---: |
| auto | 20 | 19 |
| linear | 20 | 19 |
| ts4 | 8 | 7 |
| ordinal | 3 | 3 |
| onehot | 9 | 9 |

Candidate order on each side also reverses across coordinate groups. The
runner reconstructs this schedule and rejects missing, duplicate, out-of-scope,
or misconfigured jobs. It resolves AutoGluon's child CPU allocation once,
pins it on every experiment, and uses that exact thread count for the numeric
and categorical warmup outside `run_jobs`. The attested warmup contains nine
exact stages: numeric and categorical CatBoost, LightGBM, and hybrid paths;
numeric and categorical linear-residual paths; and categorical CatBoost with
four target-stat permutations. Its stage, requested/resolved mode, selected
lane, input-kind, and permutation counts must match the frozen manifest before
completion or analysis can succeed. Representation transforms have separate
mechanism preflight tests and are not folded into model-kernel timing warmup.
The warmup config always requests the pinned child CPU count. On its fixed
2,048-row training sample, production's leaf-wise thread policy intentionally
resolves LightGBM/hybrid stages to `min(requested, 2)` threads; CatBoost and
linear-residual stages resolve to the full request. The attestation validates
both the requested and mode-specific resolved counts.

The auto arm shares one wall-clock callback across CatBoost, LightGBM, and
hybrid candidates. Every candidate must have `fit_status="fitted"`, and every
candidate-level and selected-model deadline flag must be false. A skipped or
deadline-hit auto candidate invalidates the result rather than masquerading as
a fair mode comparison. Any fitted child with a `time_limit` stop invalidates
campaign completion.

Each auto child's selected-mode refit parameters are validated coherently.
For LightGBM/hybrid children the wrapper intentionally retains
`depth=-1, num_leaves=None`; the bound core resolves that default to its
31-leaf capacity. AutoGluon's bag-level `child_hyperparameters_fit` is only a
fieldwise compression across mixed modes and is not interpreted as a coherent
tree policy; its exact fields/common flags and aggregated iteration count are
validated separately.

## Provenance, resume, and attestation

The hardened cap-campaign boundaries are reused:

- Full execution requires clean, committed DarkoFit and editable TabArena
  checkouts. The manifest binds exact source blobs, Git trees, interpreter,
  packages, behavior-affecting environment variables, host identity, CPU and
  memory resources, protocol digest, resolved child CPUs, and ordering counts.
- A zero-start output directory is the default. `--resume` requires the exact
  manifest. Before mutation, result and state paths must be real regular files;
  symbolic links and unexpected result paths fail fast. Resume is allowed only
  for a trusted, runner-owned cache directory because runner-side cache
  validation unpickles those cached results. The standalone analyzer remains
  non-executable-data-only: it never unpickles and uses the attested JSON plus
  raw-result byte hashes.
- Resume validation parses and checks every cached result. If any arm for a
  coordinate is missing or invalid, every existing job in that coordinate's
  shared-control group is archived and the group reruns. This avoids comparing
  a stale control with a new candidate. Warmup and resume histories are
  append-only and attested.
- The runner validates its own pickles, writes a normalized non-executable JSON
  payload, and seals every result digest and byte size in a completion
  attestation. The normalized analysis payload uses schema version 2; version 1
  payloads are intentionally incompatible because version 2 binds each child's
  fitted policy metadata and complete soft-deadline audit tuple. The version is
  also part of the frozen protocol digest, so old and new campaigns cannot
  share an artifact identity. Normalized child rows retain the ordered external
  feature schema so the standalone analyzer recomputes every native schema
  digest and audited drop relation itself. The analyzer hashes but never
  unpickles result files.
  Its five outputs use fixed campaign-root filenames so resume can archive every
  stale analysis artifact before any rerun.
  It revalidates the exact grid, row schemas, fitted metadata, representation
  safety, source, dependencies, runtime, hardware, and all attested bytes
  before computation, immediately before decision publication, and after the
  atomic summary/report write.

Dry-run and execution examples:

```bash
/Users/kmedved/.venvs/tabarena-darko312/bin/python \
  benchmarks/run_tabarena_regression_followon_screen.py \
  --output-dir .cache/tabarena-regression-followon-screen-20260713 \
  --time-limit 3600 \
  --dry-run

/Users/kmedved/.venvs/tabarena-darko312/bin/python \
  benchmarks/run_tabarena_regression_followon_screen.py \
  --output-dir .cache/tabarena-regression-followon-screen-20260713 \
  --time-limit 3600
```

## Exploratory survivor gate

Every candidate is evaluated only on its declared scope using paired
`log(candidate/control)` ratios. Splits are averaged within dataset and datasets
receive equal weight. The report includes validation, time, inference, memory,
per-dataset estimates, worst splits, selected mode/lane, stop reasons, rounds,
and a 10,000-draw seeded hierarchical bootstrap that resamples datasets and
the three screen coordinates within dataset.

The auto arm uses each child's validation fold to choose CatBoost, LightGBM,
or hybrid, so its validation metric is supportive but not an independent
estimate of mode-policy quality. The paired outer test metric is the
exploratory screen decision metric for that arm.

An arm survives to an independently frozen holdout experiment only if all of
these pass:

1. its exact declared grid and all eight child metadata records are complete;
2. equal-dataset test-RMSE ratio is at most 0.995;
3. equal-dataset validation-RMSE ratio is at most 1.002;
4. no scoped dataset's point test-RMSE ratio exceeds 1.005;
5. a strict majority of the arm's applicable dataset point estimates improve;
6. there are zero fitted or auto-candidate wall-clock stops;
7. equal-dataset training-time ratio is at most 4.0, inference-time ratio at
   most 1.25, and peak-memory ratio at most 2.0.

Bootstrap bounds are reported but are not a small-screen hard gate. Split-level
regret is diagnostic rather than a gate. A survivor is not a new default and
survivors must not be bundled: freeze each one separately, then validate on
mechanism-level holdout repeats or, preferably, genuinely unseen datasets. The
other 27 panel coordinates are unused for these mechanism arms but were
already inspected during cap-horizon selection, so they are not globally
untouched confirmation data. Only after a DarkoFit policy is frozen should
ChimeraBoost 0.14.1 and CatBoost be rerun on the same machine.
