# Minimal unseen CTR23 regression confirmation protocol

_Status: source-frozen confirmation protocol. No eligible CTR23 model score,
prediction, validation metric, or test metric may be opened until this protocol,
the runner, the analyzer, and their protocol tests are committed together in a
clean tree._

## 2026-07-16 forward-only harness amendment

The completed schema-version-1 campaign, its historical verifier, and its
reviewed publication remain authoritative at commit
`c3e47d5697826793c097214bb60ce68fd713c443`. The source-freeze status above
describes that original campaign and must not be read as a claim that the
post-outcome changes below were preregistered.

Schema version 2 is forward-only harness hardening for any future campaign. It
retains swap-in telemetry end to end, constrains the analyzer to explicit JSON
and source-registry allowlists, and fails recovery closed unless shutdown and
the post-teardown sample are proved. It is intentionally not a replay verifier
for the published schema-version-1 namespace. These amendments do not rerun,
reanalyze, requalify, or change the CTR23 evidence, and they do not alter the
terminal A10 decision: confirmation was not established and no default or
preset promotion is authorized.

## Objective and stopping boundary

This campaign asks one narrow question: does the already-frozen DarkoFit `A10`
accuracy profile confirm against the live ChimeraBoost 0.14.1 engine on a small,
audited, previously unused CTR23 panel?

The campaign is deliberately the minimum defensible confirmation rather than a
new development cycle. It runs three fixed official outer folds on all nine
tasks assigned to the confirmation half of the committed contamination
partition. A single fold per task would be only a directional smoke because the
smallest test folds contain 65 and 81 rows. Three folds provide within-task
replication while retaining the complete target-blind task allocation.

The run ends after the frozen analysis regardless of its result. It may not add
folds, tasks, arms, transformations, or tuning; change a threshold; open the
lockbox; or change a package default. A passing result supports only a minimal
unseen parity statement for `A10` versus ChimeraBoost. It does not establish
superiority, CatBoost parity, a production default, or a generally safe preset.

## Audited registry boundary

The campaign consumes the version-3 artifacts created under
[`ctr23_contamination_registry_protocol.md`](ctr23_contamination_registry_protocol.md).
The following semantic identities are frozen:

| Artifact identity | SHA-256 |
| --- | --- |
| Suite snapshot | `95bb2bb5d9c65ea21cb7642151bedb831ed67712bae28166a0bddc64670f0364` |
| Contamination registry | `9bda6f8b94b71575fa8275ed724ab80976c93555d898fbec8f474fcc78c6639d` |
| Confirmation/lockbox partition | `24e060ed3626fed23967294138d5768c3d9e7241f4ed06cf9b8180d512e81ee8` |
| Registry bundle | `21980c6ddaf3f5b70e866fbcc6c59c04a98b666687234147a7cedcc0b8271516` |
| Manual evidence catalog | `66529d85f9f1caea2d04784ae6666704cb6c3b5e56e06460066a482c5358ce75` |
| Frozen declarations | `bd20852afdacdbd55d20fd4adfe7331c760f651061a912fa8424f5d77675dcc9` |

OpenML suite 353 has 35 regression tasks. The audited union of known DarkoFit
and historical ChimeraBoost exposure excludes 16 tasks and fails one ambiguous
Puma-family task closed. The remaining 18 tasks were allocated target-blind to
nine confirmation and nine lockbox tasks. The allocator did not receive target
values or target statistics. This is independence from the audited development
histories, not proof that no external author has ever encountered a public
dataset.

The runner must validate the complete semantic and byte identities of the suite
snapshot, registry, partition, declarations, and evidence catalog before
building experiments, loading any CTR23 task or target, fitting, or creating
any result. It must also prove that every selected task is `eligible`, has no
ambiguity alarm or exclusion reason, is in `confirmation_task_ids`, and is
absent from `lockbox_task_ids`. A mismatch aborts before creating a result.

## Frozen tasks and official coordinates

Task order is the order below. Every task uses only OpenML sample 0, repeat 0,
and folds 0, 1, and 2, written `r0f0`, `r0f1`, and `r0f2`. No other repeat or
fold is permitted.

| Order | OpenML task | Dataset | Rows | Predictors | Categorical | Missing predictors | Official regime |
| ---: | ---: | --- | ---: | ---: | ---: | :---: | --- |
| 1 | 361236 | `auction_verification` | 2,043 | 7 | 2 | no | 1 x 10-fold CV |
| 2 | 361251 | `grid_stability` | 10,000 | 12 | 0 | no | 1 x 10-fold CV |
| 3 | 361252 | `video_transcoding` | 68,784 | 18 | 2 | no | 1 x 10-fold CV |
| 4 | 361258 | `kin8nm` | 8,192 | 8 | 0 | no | 1 x 10-fold CV |
| 5 | 361268 | `fps_benchmark` | 24,624 | 43 | 14 | yes | 1 x 10-fold CV |
| 6 | 361269 | `health_insurance` | 22,272 | 11 | 7 | no | 1 x 10-fold CV |
| 7 | 361619 | `student_performance_por` | 649 | 30 | 17 | no | 10 x 10-fold CV |
| 8 | 361622 | `cars` | 804 | 17 | 0 | no | 10 x 10-fold CV |
| 9 | 361623 | `space_ga` | 3,107 | 6 | 0 | no | 1 x 10-fold CV |

The selected subset is materialized in
[`ctr23_minimal_confirmation_coordinates.json`](ctr23_minimal_confirmation_coordinates.json).
The authoritative train/test indices remain those in
[`ctr23_suite_snapshot.json`](ctr23_suite_snapshot.json), not a regenerated
split and not the stale suite prose about holdouts. The runner must require the
coordinate file and suite snapshot to agree exactly. For each task/fold it must
match `train_size`, `test_size`, `train_index_sha256`, and
`test_index_sha256` before fitting.

The canonical coordinate manifest contains 27 dictionaries in task-table order
and then fold order. Each dictionary has exactly these keys:

```text
dataset, fold, openml_task_id, repeat, sample,
test_index_sha256, test_size, train_index_sha256, train_size
```

Serialize the list with UTF-8 JSON, sorted dictionary keys, separators `(',',
':')`, and no trailing newline. Its SHA-256 must be:

```text
6cef3b771c20440c9dad6b737797f50650d84217ee99cf8fc6fcfcbd85829c0b
```

The exact metadata path matters. TabArena's lossy
`task_metadata_collection_from_openml()` fallback fabricates three folds and
policy-derived repeats for uncached tasks and is forbidden. The runner must use
the official OpenML task objects and exact committed split metadata, then bind
the resulting `TaskMetadataCollection` to the coordinate manifest above.

## Frozen arms and job grid

All arms use native input representation. No numeric-string resolver, ordinal
mapping, declared category map, target-ordered category-code experiment, or
other representation diagnostic is allowed.

| Code | Arm | Coordinates | Manual configuration |
| --- | --- | ---: | --- |
| `A10` | Frozen DarkoFit accuracy profile | all 27 | `iterations=10000`, `tree_mode="auto"`, `l2_leaf_reg=3`, `max_bins=128`, `learning_rate=0.1`, `ts_permutations=1` |
| `M` | Live ChimeraBoost 0.14.1 default | all 27 | `{}` through the official TabArena adapter |
| `D` | DarkoFit 0.9.0 product default | all 27 | `{}` |
| `C` | CatBoost 1.2.10 default | only the nine `r0f0` coordinates | `{}` through AutoGluon/TabArena |

`A10` also freezes the behavior behind its explicit parameters:

- automatic tree-mode candidate order is `catboost`, `lightgbm`, `hybrid`;
- every candidate must fit and emit a finite child-validation RMSE;
- selection minimizes the original child-validation RMSE and an exact tie
  selects the earlier candidate;
- early stopping uses 50 rounds of patience and `use_best_model=True`;
- scalar-regression `ordered_boosting="auto"` resolves off;
- `linear_residual=False`, `auto_learning_rate_probe=False`, and
  `target_ordered_cat_codes="off"`;
- native categorical handling and `ts_permutations=1` remain unchanged.

The official empty-config comparator defaults must be validated before running.
In particular, `M` resolves to 10,000 estimators, learning rate 0.1, depth 6,
`l2_leaf_reg=1`, 128 bins, four categorical permutations, ordered boosting off,
50-round patience, and validation selection between its supported leaf lanes.
`C` resolves to CatBoost 1.2.10's official CPU adapter with 10,000 iterations,
learning rate 0.05, RMSE evaluation, the AutoGluon adaptive stopping callback,
and best-model retention when an eval set exists. `D` remains the unmodified
DarkoFit product default, including its automatic learning-rate policy.

The CTR23-only comparator adapter observes the official callback bindings
without changing model parameters or callback return values. For every `M` and
`C` child it writes an exact `ctr23_time_callback_audit` sidecar with only
`schema_version`, `kind`, `engine`, `time_limit_seconds`,
`time_callback_instrumented`, `time_callback_instance_count`,
`time_callback_call_count`, and `time_callback_hit`. The received child budget
must be finite and in `(0, 3600]`. `M` must expose one callback instance, or two
when both leaf-lane candidates were fitted, with at least one invocation per
instance. `C` must expose exactly one instance and at least one invocation.
Every audited callback return must prove `time_callback_hit=false`; `A10` and
`D` have no comparator sidecar. CatBoost's separate memory callback is
structurally ineligible on this panel under its five-million-cell threshold.

The exact grid is therefore:

```text
27 A10 + 27 M + 27 D + 9 C = 90 outer jobs
90 outer jobs x 8 AutoGluon bag children = 720 selected children
27 A10 jobs x 8 children x 3 candidates = 648 A10 candidate fits
```

Every outer job uses the same eight-child, one-bag-set construction, seed
policy, child-validation construction, sample-weight handling, and 3,600-second
soft outer-job budget. A child receives only the unspent remainder of its outer
budget. A deadline hit, time-limit stop, missing auto candidate, non-finite
metric, changed child count, or changed selection contract invalidates the
attempt; it is never imputed or dropped.

## Primary and guardrail estimands

Only paired **outer test RMSE** enters a gate. Validation RMSE is descriptive.
Every required RMSE must be finite and strictly positive, and all required arms
must share the exact task/repeat/fold coordinate.

For task `d`, selected fold `j`, and arms `X` and `Y`, define the paired log
ratio

```text
l[d,j; X/Y] = log(test_RMSE[d,j; X] / test_RMSE[d,j; Y]).
```

The task point ratio gives each of its three folds equal weight:

```text
R[d; X/Y] = exp(mean_j(l[d,j; X/Y])).
```

The equal-task point estimate gives each of the nine tasks one vote:

```text
G[X/Y] = exp((1 / 9) * sum_d(log(R[d; X/Y]))).
```

No row weighting, pooled squared error, ratio of arithmetic means, median,
selective completeness rule, or unpaired alternative may replace this
hierarchy.

### Primary `A10/M` confirmation gate

Use exactly 10,000 paired hierarchical-bootstrap draws with
`numpy.random.Generator(numpy.random.PCG64(20260719))`. In each draw:

1. sample nine task positions with replacement from the nine frozen tasks;
2. independently for each sampled task occurrence, sample three of that task's
   three fold positions with replacement;
3. average the three sampled paired log ratios within each task occurrence;
4. average the nine task-occurrence means and exponentiate.

The one-sided 95% upper bound is
`numpy.quantile(draw_ratios, 0.95, method="higher")`. The primary gate passes
only when this bound is **strictly less than 1.000**. `G[A10/M] <= 0.995` is a
desired, report-only target and is not a second gate. There is no fallback to a
point-estimate-only decision.

### Simultaneous `A10/D` max-regret guardrail

Use a fresh generator
`numpy.random.Generator(numpy.random.PCG64(20260720))` and exactly 10,000
draws. Keep all nine tasks present once in every draw. Within each task,
resample its three paired `A10/D` fold log ratios three times with replacement,
average, and exponentiate. The draw statistic is the maximum of the nine task
ratios.

The simultaneous max-regret upper bound is
`numpy.quantile(draw_maxima, 0.95, method="higher")`. The guardrail passes only
when that bound is **less than or equal to 1.020**. Report every task point ratio
and flag any `R[d; A10/D] > 1.010`; a point flag is diagnostic and is not a
replacement gate.

Both the primary gate and max-regret guardrail must pass. Before either is
computed, the integrity gate must establish the exact 90-job/720-child grid,
finite positive metrics, exact arm behavior, and zero failures, imputations,
deadline hits, time-limit stops, worker failures, or recovery mixing.

### CatBoost is descriptive only

`C` runs only `r0f0`, producing one paired observation per task. It cannot
enter the primary or guardrail gate, break a tie, change the pass/fail label, or
authorize another run. Report equal-task geometric point ratios for `A10/C`,
`M/C`, and `D/C` on those nine shared coordinates.

For a descriptive task-bootstrap interval, use exactly 10,000 paired-log-ratio
draws with `numpy.random.Generator(numpy.random.PCG64(20260721))`, sampling nine
tasks with replacement. Use `numpy.quantile(..., method="higher")` for both the
2.5th and 97.5th percentiles. Label these intervals descriptive, low-power, and
single-fold; they are not evidence for CatBoost parity.

## Quality-only two-worker execution

Production uses exactly two persistent processes created with multiprocessing
`spawn`. Each worker has a private scratch/current-working directory and each
selected child retains the frozen 18-CPU allocation. The parent is the only
writer of the manifest, ordered schedule, journal, safe payload, invalid-attempt
marker, and completion attestation. It releases the next static two-job wave
only after validating both reports from the prior wave. The 90 jobs form 45
waves; the complete order, worker slot, partner, task, coordinate, arm, and
schedule digest are written in the zero-start manifest before any CTR23 or
production model execution and before wave zero. The synthetic-only preflight
precedes that manifest. Production wave count: 45 waves.

The explicit execution policy is `quality_only_swap_in`:

- swap-in is allowed but measured and retained as operational audit evidence;
- schema-version-2 operational evidence starts a monotonic host-counter
  lifecycle before worker creation, retains setup checkpoints, cross-binds
  every measured dispatch to its lifecycle sample range, and ends only after
  confirmed worker shutdown;
- a failed worker session also takes a bounded post-teardown counter snapshot
  and persists it in the schema-version-2 invalid-attempt marker without
  replacing the original workload error; the marker distinguishes confirmed
  shutdown from a failed teardown or failed telemetry capture;
- every production wave retains its lifecycle sample range and its independently
  recomputable swap-in and swap-out deltas; the safe payload and completion
  attestation duplicate the same bounded `swap_audit` summary;
- any positive swap-out delta over the complete preflight worker lifecycle,
  any measured production dispatch, or the complete production worker
  lifecycle invalidates the attempt;
- combined resident memory must remain below 80% of physical memory;
- no timing, throughput, inference-speed, training-speed, or memory-performance
  comparison is admissible from this run;
- raw time, memory, load, and swap fields may be retained only for integrity and
  failure diagnosis.

The preflight is synthetic-only and non-reusable. Synthetic numeric and
categorical warmup, including all three `A10` automatic tree modes and
prediction, occurs serially in both persistent workers; the same synthetic
behavior projection is then checked under a simultaneous two-worker probe.
The preflight records `ctr23_fit_count=0`, must not load a CTR23 target, and may
not populate a production result cache. Its disposable scratch namespace is
removed before wave zero. OpenML task data and official split definitions may
be cache-primed without fitting, but no CTR23 fit, metric, or prediction may be
computed or opened outside the frozen 90-job grid.

The journal must record process/worker identity, dispatch and barrier state,
arm/task/coordinate identity, completion status, swap boundaries, peak RSS,
deadline state, result path, and result hash. Any parent interruption
invalidates that namespace: there is no in-place resume and no result may be
reused. Only an eligible concurrent production failure with the canonical
runner-written marker, a captured post-teardown lifecycle sample, and confirmed
worker shutdown may authorize the fresh full sequential recovery below. A
worker-stop failure or unavailable final telemetry is explicitly
`not_recoverable`.

### Fresh sequential recovery

Any two-worker failure, crash, OOM, forbidden swap-out, RSS breach, deadline or
time-limit hit, malformed report, behavior mismatch, or incomplete wave marks
the entire concurrent attempt invalid. No result from that namespace may enter
analysis.

The sole recovery is a complete sequential rerun of all 90 jobs in a fresh
namespace. It must be explicitly authorized by the runner-written invalid
attempt marker, bind the invalid manifest/marker hashes, prove that the failed
workers were shut down and that the post-teardown swap sample was retained,
preserve every model, task, coordinate, resource, dependency, threshold, and
analysis setting, and start with an empty result cache. Concurrent and
sequential results may never be mixed. Sequential recovery remains
`quality_only_swap_in`, requires zero swap-out and all other integrity gates,
and is disclosed in every result row.
Any campaign namespace inside the DarkoFit repository, including both the
invalid concurrent source and fresh sequential destination, must be Git-ignored
before source provenance is collected; namespaces outside the repository remain
allowed. This keeps generated artifacts outside the authenticated clean tree.

## Safe analysis boundary and attestations

The runner alone may deserialize trusted TabArena result caches. After checking
the complete grid and child metadata, it writes a strict JSON safe-analysis
payload. The payload may contain only:

- protocol, source, registry, manifest, schedule, policy, and execution hashes;
- the 90 unique job identities with finite test/validation RMSE and bounded
  fitted metadata;
- the 720 unique selected-child identities and fitted audit fields, including
  resolved learning rate, selected/requested mode or lane, best iteration/tree
  count, candidate count/order for `A10`, and stop reason;
- completeness, deadline, worker, recovery, swap, and RSS integrity summaries.

It must not contain model objects, pickles, predictions, targets, feature rows,
or arbitrary repr strings. JSON serialization uses `allow_nan=False`; mappings
must have string keys and values must be recursively JSON-safe. The runner
hashes the payload and every raw result artifact, then writes a completion
attestation binding those hashes to the zero-start manifest and journal.

The independent analyzer may read source-bound protocol/provenance files. Its
registry reader is limited, before any path construction or filesystem call,
to exactly four hard-coded repository-relative JSON paths:
`benchmarks/ctr23_suite_snapshot.json`,
`benchmarks/ctr23_contamination_registry.json`,
`benchmarks/ctr23_partition.json`, and
`benchmarks/ctr23_manual_evidence_catalog.json`. Registry keys supplied by a
document never confer filesystem authority.

From the completed campaign root the analyzer may read exactly these seven
strict JSON files:
`run_manifest.json`, `completion_attestation.json`,
`analysis_payload.json`, `wave_schedule.json`, `preflight_report.json`,
`concurrency_history.json`, and `warmup_history.json`. A sequential recovery
may additionally read only the exact runner-attested `invalid_attempt.json` and
source `run_manifest.json` named by its recovery record, and may perform only
an existence check for source `completion_attestation.json` to reject recovery
from an already completed campaign.

The completion attestation is the runner-authored root statement and has no
external signature. It hash-binds the manifest and hash-and-size-binds the five
other campaign JSON artifacts; the manifest in turn binds source, protocol,
registry, schedule, runtime, and policy identities. The analyzer independently
checks that internal chain without overstating it as third-party authentication.

The runner is the sole authority that may enumerate, stat, open, hash, decode,
or deserialize a raw result. Analyzer raw-result access is
`forbidden_no_stat_enumerate_open_hash_or_decode`: it must not inspect a
`results.pkl`, model cache, prediction, target, dataset, or experiments tree.
It validates the exact 90-entry runner-attested result metadata manifest and
its safe-payload digest, but does not independently rehash current raw bytes.
Its campaign-root reader must reject every filename outside the seven-name JSON
allowlist before filesystem access. It must revalidate all permitted hashes and
integrity gates before calculating a ratio. Its
machine-readable summary and tabular exports repeat `execution_mode`,
`swap_policy`, `timing_admissible=false`, and the performance-evidence
disposition, plus the retained swap-in audit counts, so detached files cannot
be mistaken for speed evidence.

Analysis must be deterministic: a second analyzer invocation over unchanged
inputs must reproduce every decision artifact byte for byte.

## Source and environment provenance

The frozen model behavior is the code shipped through DarkoFit commit
`92290df75e102fe0064851555f1ad2e6802703f3`. Benchmark-only runner/protocol
commits may follow it, but the `darkofit/` Git subtree must remain exactly
`52278b0326419a45a72bdfd3afcfc13019087838`. Any package-code change requires a
new protocol revision before an outcome is opened.

Comparator and harness identities are:

- ChimeraBoost tag `v0.14.1`, commit
  `9c9ea6e704a9fe2bfe6d6c284b22de73914be048`, tree
  `62ad7bd44e6d37dddee53d6a2230d45d3b8040c5`, clean detached checkout;
- TabArena commit `4cd1d2526874962daae048a6f2dcf34aa272f3fa`,
  tree `a293df372a613c7358ba5fcd746f58d580cde7d6`, clean checkout;
- CatBoost wheel version 1.2.10, with every installed distribution file hashed;
- `tabarena_adapter.py` SHA-256
  `13fefe3937ca9a4020412023324aa724e073330377fdd6ecd476fd29fc63740a`;
- `tabarena_comparator_adapters.py` SHA-256
  `87d56048271b5cfb66d689015e76ead935f593a752fd1479a05ec5374df01c2a`;
- `tabarena_ctr23_adapters.py` SHA-256
  `9ab74de39cdcffaa5f92031519eeb68c4087c58b09aa4c1401293324d925cab3`.

The required runtime is Python 3.12.13 on the attested 18-core arm64 host with
128 GiB physical memory. The dependency lock includes AutoGluon common/core/
features/tabular 1.5.1b20260712, DarkoFit 0.9.0, CatBoost 1.2.10, Graphviz 0.21,
llvmlite 0.48.0, Numba 0.66.0, NumPy 2.4.6, OpenML 0.15.1, pandas 2.3.3,
psutil 7.1.3, PyArrow 24.0.0, scikit-learn 1.7.2, SciPy 1.16.3, TabArena
0.0.1, and liac-arff 2.5.0. The zero-start manifest binds the exact executable,
hardware identity, environment variables affecting native/JIT threading, Git
heads/trees/status/remotes, imported module paths, distribution files, runner,
analyzer, protocol, registry artifacts, adapters, ordered grid, and output
namespace before wave zero.

## Mandatory terminal state

After one valid concurrent attempt or its one authorized fresh sequential
recovery, the analyzer publishes the frozen decision and the campaign stops.
Specifically prohibited after seeing any safe-payload metric are:

- running the remaining seven official folds or any additional repeat;
- changing `A10`, adding a representation or linear lane, or tuning any arm;
- changing the bootstrap, thresholds, aggregation, task weights, or exclusions;
- opening or running the nine lockbox tasks;
- promoting an accuracy preset or changing the DarkoFit default;
- rerunning only favorable, failed, noisy, or borderline tasks.

Any future experiment is a separately authorized research project on a newly
declared evidence boundary. This protocol itself authorizes no such follow-on.
