# TabArena safe-ordinal mechanism confirmation protocol

Status: **source-frozen before execution**. This campaign confirms only the
safe, source-declared ordinal representation that survived the isolated
follow-on screen. It does not reopen the tree-mode, target-statistic,
one-hot, linear-residual, cap-horizon, or per-dataset tuning questions.

## Claim and evidence boundary

The exploratory screen used `r0f0`, `r1f1`, and `r2f2` for Airfoil and
Diamonds. This confirmation uses every remaining registered coordinate for
those two datasets:

| Dataset | Task ID | Registered coordinates | Excluded screen coordinates | Confirmation coordinates |
| --- | ---: | ---: | ---: | ---: |
| Airfoil self noise | 363612 | 30 (`r0f0` through `r9f2`) | 3 | **27** |
| Diamonds | 363631 | 9 (`r0f0` through `r2f2`) | 3 | **6** |
| **Total** | | **39** | **6** | **33** |

These 33 coordinates are unused by the ordinal mechanism. They are not globally
unseen: the preceding cap-horizon campaign already inspected them. This is
therefore a weaker mechanism-level replication, not independent external
confirmation. In addition, both transforms encode dataset-specific
semantics. A successful campaign may support these exact Airfoil and Diamonds
representations; it cannot justify treating arbitrary categorical columns as
ordinal, learning order from data, or claiming replication on unseen datasets.

The six excluded coordinates remain excluded from every arm. Their screen
scores are never pooled into this analysis.

## Frozen arms and three contrasts

Every coordinate runs three adjacent jobs. Each job uses one eight-fold bag
set, sequential local fitting, the same outer split, the same child-fold
validation data, fold-varying seeds, the same pinned CPU allocation, a
3,600-second outer budget, the monotonic wall-clock callback, and the attested
process-local warmup.

| Code | Arm | Representation | Model policy |
| --- | --- | --- | --- |
| `P` | `product_default_native` | Native categorical path | Empty manual model config; use the source-frozen product adapter defaults |
| `B` | `fixed_base_native` | Native categorical path | Fixed screen control below |
| `O` | `fixed_base_safe_ordinal` | Source-declared, target-free ordinal/numeric path | Same fixed policy as `B` |

The fixed `B` and `O` policy is:

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

`O` and `B` differ only at the representation boundary. `P` deliberately has
an empty manual model config. Its initialized child parameters must be the
source-frozen adapter defaults (`iterations=1000`, `early_stopping=True`,
`tree_mode="catboost"`, and `diagnostic_warnings="never"`) plus the exact
fold seed. In particular, `P` retains automatic learning-rate resolution and
the product defaults for settings that `B` fixes. The runner and analyzer must
not silently copy the fixed-base settings into `P`.

The three predeclared paired contrasts have distinct roles:

1. **Primary causal contrast, `O / B`:** isolates the safe-ordinal
   representation under one fixed model policy.
2. **Deployment contrast, `O / P`:** asks whether that frozen ordinal policy
   is preferable to the current product default as actually deployed.
3. **Attribution contrast, `B / P`:** describes how much of the deployment
   result comes from the fixed-base model policy rather than representation.
   It is report-only and can neither advance a policy nor reselect one.

There are **33 coordinate groups, 99 outer jobs, 792 child fits, 33 primary
pairs, 33 deployment pairs, and 33 attribution pairs**. No control is shared
across different outer coordinates.

## Exact coordinate and execution order

Coordinates are ordered first by the task order above, then by repeat and
fold. `r0f0`, `r1f1`, and `r2f2` are removed. Within successive coordinate
groups the job order follows this exact six-permutation cycle:

```text
P B O
B O P
O P B
O B P
P O B
B P O
```

The cycle restarts at `P B O` for each dataset. Airfoil's 27 groups contain
four complete cycles followed by `P B O`, `B O P`, and `O P B`, so each arm
appears nine times in each position. Diamonds' six groups contain one complete
cycle, so each appears twice in each position. Across the campaign, every arm
therefore appears in the first, second, and third position exactly 11 times.
The runner reconstructs the complete ordered grid and rejects a missing,
duplicate, extra, out-of-scope, or misconfigured job. It records a digest of
that ordered grid in the manifest so resume cannot change the schedule.

## Representation and child-metadata boundary

The source-frozen transforms are exactly those audited by the exploratory
screen:

- Airfoil restores `attack-angle` to its physical numeric value through the
  frozen 27-entry compact-code-to-angle tuple. Compact codes are never used as
  numeric angles directly.
- Diamonds maps `cut` as
  `Fair < Good < Very Good < Premium < Ideal`, `color` as
  `J < I < H < G < F < E < D`, and `clarity` as
  `I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF`. The audited compact
  category-code-to-rank maps are frozen in source.

The transform may not inspect targets, validation rows, test rows, label
lexical order, frequency, target means, or first appearance. Missing ordinal
values remain numeric NaN. An unexpected feature order, categorical set,
compact category domain, or value fails the child fit rather than falling back
to an inferred representation.

Every one of the 792 children must carry complete fitted metadata, including
best iteration, attempted/completed/retained rounds, requested and resolved
learning rate, requested and selected tree mode, selected lane, exact refit
parameters, early-stopping state, stop reason, wall-clock state, CPU allocation,
and representation audit. `B` and `O` must request and resolve learning rate
`0.1` exactly. `P` must have no manual learning-rate override; its finite,
positive resolved learning rate must agree with its trained exact-refit
learning rate.

For native `P` and `B` children, metadata schema v2 binds the ordered external
feature schema to the ordered schema actually passed to DarkoFit. A feature
may disappear only through the audited AutoGluon child-fold constant-drop
rule: the exact name must be recorded with one distinct value (missing
included), and the fitted schema must equal the external schema minus those
columns while preserving order. Airfoil must expose exactly `attack-angle` as
native categorical; Diamonds must expose exactly `cut`, `color`, and `clarity`.

For `O`, metadata must prove the exact source-frozen domain, target-free
fitting, one transform fit on child-training rows, the expected evaluation
transforms, input/output counts, category-schema digest, encoded positions, and
zero undeclared or unknown validation values. The adapter requires the exact
source-frozen full child feature order; a missing, extra, or reordered column
fails closed. The normalized payload retains that ordered child schema under
the completion attestation. All eight child records must be complete and
unique for every outer job.

Only fully fitted, valid, inferable outer and child models are admissible. Any
model failure, imputation, deadline hit, `time_limit` stop, nonfinite metric,
incomplete metadata, or representation mismatch invalidates campaign
completion.

## Warmup, provenance, resume, and safe analysis

The hardened cap and follow-on campaign boundaries remain in force:

- A full run requires clean, committed DarkoFit and editable TabArena
  checkouts. The imported modules must resolve to those exact checkouts.
- Before fitting, `run_manifest.json` binds the protocol and ordered-grid
  digests, exact source blobs and Git trees, interpreter and dependency
  versions, behavior-affecting environment variables, sanitized remotes,
  host identity, CPU and memory resources, output root, time limit, and the
  resolved child CPU allocation.
- The attested process-local warmup uses that same pinned child CPU count and
  records the expected numeric/categorical routes, selected modes and lanes,
  requested/resolved learning rates, target encodings, permutation counts,
  thread policies, and prediction fingerprints before measured jobs run.
- The default is a missing or empty output directory. Reuse requires explicit
  `--resume` and an exact manifest match. Before any read or mutation, every
  cached result, state file, and analysis artifact must be a real regular file
  beneath the campaign root; symbolic links and unexpected paths fail closed.
- Resume is coordinate-group atomic. If any of `P`, `B`, or `O` for a
  coordinate is missing, unreadable, incomplete, or invalid, all existing
  results for that coordinate are archived and all three arms rerun. Valid
  triads remain cached. Warmup and resume histories are append-only and bound
  into the completion attestation.
- The runner alone may validate its trusted pickles. It emits a normalized,
  non-executable JSON payload and a completion attestation that seals the byte
  digest and size of all 99 raw results and every state artifact. The
  standalone analyzer never unpickles campaign input. It independently
  verifies source, runtime, hardware, manifest, attestation, exact grid,
  schemas, representations, fitted metadata, raw-result hashes, and protected
  output paths before computation, immediately before publication, and after
  atomic publication of the complete output set.

Example preflight and execution:

```bash
/Users/kmedved/.venvs/tabarena-darko312/bin/python \
  benchmarks/run_tabarena_regression_ordinal_confirmation.py \
  --output-dir .cache/tabarena-regression-ordinal-confirmation-20260718 \
  --time-limit 3600 \
  --dry-run

/Users/kmedved/.venvs/tabarena-darko312/bin/python \
  benchmarks/run_tabarena_regression_ordinal_confirmation.py \
  --output-dir .cache/tabarena-regression-ordinal-confirmation-20260718 \
  --time-limit 3600
```

Use the same command plus `--resume` only for an interrupted run whose exact
manifest is intact.

## Frozen estimands and uncertainty

For a contrast `N / D`, ratios below one favor the numerator. Metrics are
paired at the outer coordinate and transformed to `log(N / D)`. Fold estimates
are averaged within repeat; repeat estimates are averaged within dataset; the
two dataset estimates then receive equal weight. This hierarchy is mandatory:
Airfoil's 27 coordinates must not outweigh Diamonds' six. Validation RMSE,
training time, inference time, and peak RSS use the same equal-dataset
aggregation. Arithmetic ratios of pooled raw values are not decision inputs.

Uncertainty uses **10,000** draws with seed **20260718**. The two datasets are
fixed in every draw. Within each dataset, repeat blocks are sampled with
replacement, and folds are sampled with replacement inside each selected
repeat block; each draw is then averaged by repeat, by dataset, and equally
across the two fixed datasets. The one-sided upper bound is the 0.95 quantile;
the two-sided 95% interval is also reported. Resampling the two datasets is not
allowed.

The exact one-sided sign test uses all **13** predeclared dataset-repeat point
estimates: ten Airfoil repeat blocks and three Diamonds repeat blocks. A strict
ratio below one is a win; a tie is conservatively a non-win. Its fixed-null
tail is `sum(comb(13, k), k=wins..13) / 2**13`. Therefore `p < 0.05` requires
at least ten wins. The worst coordinate and all split win/loss/tie counts are
reported as diagnostics only; there is no split-level hard gate.

## Exact advancement gates

The safe-ordinal policy advances from this campaign only if **both** the
primary `O / B` gate and the deployment `O / P` gate pass in full. A threshold
is inclusive unless explicitly stated as strict.

### Primary causal gate: `O / B`

1. The exact grid, all 264 paired child comparisons (528 arm-side child
   records), fitted metadata, representation audits, provenance, and finite
   non-imputed metrics validate.
2. Equal-dataset test-RMSE ratio is at most **0.995** and the fixed-dataset
   hierarchical-bootstrap one-sided 95% upper ratio is strictly below **1.0**.
3. **Both** dataset test-RMSE point ratios are at most **0.995**.
4. The exact one-sided sign-test p-value over 13 repeat blocks is strictly
   below **0.05**.
5. Equal-dataset validation-RMSE ratio is at most **1.002**, and neither
   dataset's validation-RMSE ratio exceeds **1.005**.
6. There are zero failed, imputed, deadline-hit, or `time_limit`-stopped jobs
   or children.
7. Equal-dataset training-time, inference-time, and peak-RSS ratios are at
   most **1.50**, **1.25**, and **1.25**, respectively.

### Deployment gate: `O / P`

1. The exact grid, all 264 paired child comparisons (528 arm-side child
   records), representation audits, provenance, and finite non-imputed metrics
   validate.
2. Equal-dataset test-RMSE ratio is at most **0.995** and the fixed-dataset
   hierarchical-bootstrap one-sided 95% upper ratio is strictly below **1.0**.
3. No dataset test-RMSE point ratio exceeds **1.005**.
4. Equal-dataset validation-RMSE ratio is at most **1.002**.
5. The exact one-sided sign-test p-value over 13 repeat blocks is strictly
   below **0.05**.
6. There are zero failed, imputed, deadline-hit, or `time_limit`-stopped jobs
   or children.
7. Equal-dataset training-time, inference-time, and peak-RSS ratios are at
   most **1.50**, **1.25**, and **1.25**, respectively.

### Attribution contrast: `B / P`

`B / P` receives the same point estimates, uncertainty summaries, per-dataset
and per-repeat tables, resource metrics, child metadata summaries, and worst
coordinate diagnostics. It has **no pass/fail gate** and cannot advance,
reject, rescue, tune, or reselect either policy. Its sole purpose is to
attribute the deployment contrast after the two frozen decisions are evaluated.

A pass confirms only the exact source-declared Airfoil and Diamonds mechanism
under this frozen policy. Product integration must remain explicit and
fail-closed, and genuinely unseen datasets with independently declared ordinal
semantics remain the stronger next validation step. Only after the DarkoFit
policy decision is frozen should ChimeraBoost 0.14.1 and CatBoost be rerun on
the same machine.
