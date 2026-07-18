# T7b CatBoost-gap attribution protocol

## Status and scope

T7b is a prospective, development-only mechanism-attribution campaign. It
uses exactly the eight already-spent C2 development tasks, repeat 0, folds
0–2, and the deterministic C2 inner split. It does not open a confirmation
task, CTR23 lockbox coordinate, T5 lineage, or any new dataset. No T7b result
can change a default or serve as confirmation evidence.

The binding coordinate declaration is
`t7b_catboost_gap_attribution_coordinates.json`. The binding source freeze is
`t7b_catboost_gap_attribution_freeze.json`. The campaign must remain
unexecuted until those files, this protocol, the runner, analyzer, and tests
all agree byte-for-byte with the hashes in the freeze.

## Runtime and shared fit policy

The campaign runs only in the frozen `tabarena-darko312` environment:

- CPython 3.12.13;
- NumPy 2.4.6;
- pandas 2.3.3;
- SciPy 1.16.3;
- scikit-learn 1.7.2;
- Numba 0.66.0;
- OpenML 0.15.1; and
- CatBoost 1.2.10.

The parent and every worker compare this complete environment contract before
loading a task or fitting a model. A missing package or any version mismatch
fails closed. CatBoost runs on CPU with six threads. The seeds are 4, 17, and
29. Every arm uses:

- `loss_function="RMSE"`;
- 1,000 iterations (CatBoost's T7 product default);
- the exact C2 fit, validation, and outer-test rows;
- the exact categorical feature declaration and CatBoost frame conversion
  used by T7;
- no evaluation set, early stopping, or refit;
- `allow_writing_files=False`; and
- the coordinate's frozen T7 resolved default learning rate, passed
  explicitly and unchanged to every arm and seed.

There are 24 task/fold coordinates, 72 coordinate/seed executions, eight arms,
and 576 fits. Arm order is a deterministic rotation over the 72 executions.
Each execution runs in an isolated subprocess and is durably spooled before
aggregation. Warmup precedes timed work.

## Frozen arms

| Arm | Override beyond the shared policy |
|---|---|
| `baseline` | none |
| `random_strength_0` | `random_strength=0` |
| `bootstrap_no` | `bootstrap_type="No"` |
| `no_split_noise_or_row_sampling` | both preceding overrides |
| `l2_leaf_reg_1` | `l2_leaf_reg=1` |
| `one_hot_max_size_0` | `one_hot_max_size=0` |
| `one_hot_max_size_255` | `one_hot_max_size=255` |
| `leaf10_any_improvement` | 10 leaf-estimation iterations with `AnyImprovement` backtracking |

No additional arm, post-hoc combination, per-dataset exception, changed seed,
or changed learning rate is authorized.

## Integrity controls

The worker and parent enforce:

1. The source freeze names a 40-character DarkoFit model-source commit
   `H_model`. Every stored source hash is independently reproduced from the
   corresponding `git show H_model:path` blob. The live files must reproduce
   the same hashes at execution time.
2. Execution occurs from a clean, pushed `main` commit descended from
   `H_model`. The complete name-only diff from `H_model` to the execution
   commit must consist of exactly
   `benchmarks/t7b_catboost_gap_attribution_freeze.json`. This requires a
   model-source commit followed by a dedicated freeze-only commit and prevents
   the freeze from silently blessing post-freeze model changes.
3. The exact Python/dependency contract above is stored in the freeze, copied
   into the raw protocol binding and runtime record, and checked before data
   access in the parent and workers.
4. C2 registry, immutable C2 raw anchor, frozen T7 raw artifact, C2 loader,
   CatBoost frame helper, T7 runner, protocol, runner, analyzer, tests, and
   coordinate declaration must match the source freeze.
   `tests/conftest.py` is deliberately excluded from the model-affecting
   source map because it only assigns pytest markers; it remains covered by
   the clean two-commit execution boundary.
5. The task identity, full source-data fingerprint, categorical declaration,
   official outer split, deterministic inner split, dimensions, and
   coordinate order must match the C2/T7 boundary.
6. For seed 4, every baseline validation and test prediction must byte-match
   the frozen T7 default prediction hashes.
7. Cars (task 361622) is the numeric negative control. For every fold and
   seed, both one-hot arms must byte-match the baseline on validation and
   test. Any difference fails the worker closed. CatBoost may omit the
   irrelevant resolved one-hot value on this numeric task; the exact requested
   policy is still persisted and checked.
8. Every arm must resolve to CatBoost CPU, seed, 1,000 iterations, and the
   coordinate's frozen learning rate. The complete requested constructor/fit
   policy and the resolved loss, evaluation, thread, tree, sampling,
   regularization, categorical, leaf-estimation, learning-rate, iteration, and
   seed parameters are persisted and checked. Apart from declared overrides
   and CatBoost's bootstrap-coupled sampling fields, every arm must resolve to
   the baseline policy.
9. The parent must recheck the exact clean-pushed-main source state, source
   freeze, and coordinate declaration after every worker completes and before
   publishing the raw artifact.

After execution, the analyzer verifies the frozen source hashes from the
historical `H_model` blobs rather than requiring the current checkout to
remain byte-identical forever. This preserves auditability after later
repository work while the execution path still requires live-file equality.

## Estimands

For each non-baseline arm, report arm/baseline RMSE ratios for validation and
test:

- per task, pooling the three folds and three seeds geometrically;
- equal-dataset geometric mean;
- each seed block;
- wins, losses, and ties at task level;
- worst task; and
- least-favorable leave-one-task-out ratio.

Also report the combined `no_split_noise_or_row_sampling` arm against each
component. Because this campaign does not power an interaction estimand, this
is an **incremental-vs-components** comparison, not a complementarity claim.
Report the direct paired ratios and additive log departure. Timing and memory
are descriptive only and must include machine/runtime metadata.

## Predeclared interpretation

The two directions answer different questions and must not be conflated.

An ablation is an explanatory **contributor** only when disabling or reducing
the mechanism makes CatBoost worse and all of the following hold:

- equal-dataset test and validation arm/baseline ratios are greater than 1;
- the multiplicity-adjusted hierarchical-bootstrap lower bound for test is
  greater than 1;
- the least-favorable leave-one-task-out test ratio is greater than 1; and
- all three seed-block test ratios are greater than 1.

For every arm, report the signed fraction of the historical CatBoost/DarkoFit
gap **erased by the CatBoost perturbation** on the log-RMSE scale:

`log(arm / CatBoost baseline) / log(DarkoFit / frozen T7 CatBoost default)`.

A positive value means the ablation numerically erases that fraction of
CatBoost's historical advantage by making CatBoost worse; values above 1 are
reported uncapped. A negative value means the arm improves CatBoost. This is
not a causal fraction explained: that claim would require a matched DarkoFit
mechanism bridge. The binding historical bridge uses seed 4 because the frozen
DarkoFit/CatBoost denominator used seed 4. The all-three-seed estimator is
reported separately as sensitivity evidence.

Separately, an arm is a **promising_config** only when it improves CatBoost
and all of the following hold:

- equal-dataset test ratio is at most 0.995;
- equal-dataset validation ratio is at most 1.005;
- the multiplicity-adjusted hierarchical-bootstrap upper bound for test is
  below 1;
- worst-task test ratio is at most 1.02;
- least-favorable leave-one-task-out test ratio is at most 1.00; and
- no seed-block test ratio exceeds 1.005.

An arm can receive at most one direction label; otherwise it is
`not_attributed`. These labels describe CatBoost mechanisms on spent
development data. Even a promising configuration only authorizes a separately
implemented, outcome-unseen DarkoFit experiment.

The combined noise/sampling arm receives only a descriptive
`incremental_vs_both_components` status. In the explanatory direction, both
direct combined/component ratios must be at least 1.005; in the promising
direction, both must be at most 0.995. Missing that bar means “not incremental
versus both,” not “redundant.” No inferential interaction claim is authorized.

The hierarchical bootstrap uses 100,000 deterministic draws with seed 7017.
It resamples the eight tasks and the three folds within each selected task.
The three seeds are crossed, fixed repeat blocks: every sampled fold averages
all three seeds rather than resampling them independently. There are seven
arms and two outcome-selected directions, so the 14 directional claims use
Bonferroni familywise alpha 0.05. The binding lower and upper quantiles are
0.0035714285714285718 and 0.9964285714285714.

The summary binds both the exact raw-file SHA-256 and the distinct canonical
payload SHA-256. The raw payload cannot contain its own file hash.
