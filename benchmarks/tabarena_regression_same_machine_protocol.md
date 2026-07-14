# Same-machine TabArena comparator protocol

Status: **source-frozen before execution**. This campaign characterizes the
current out-of-box regression behavior of DarkoFit, ChimeraBoost 0.14.1, and
CatBoost 1.2.10 on one machine. It is not another policy-selection screen.

The preceding safe-ordinal confirmation ended `do_not_advance`: its quality
gates passed, but its frozen DarkoFit ordinal/native inference-time ratio was
`1.2652`, above the predeclared `1.25` limit. Nothing in this campaign may
reinterpret that outcome, advance the rejected candidate, or select a new
default. In particular, the Airfoil/Diamonds ordinal work below is a separate
all-engine representation diagnostic and is never pooled with the primary
panel.

## Claims and frozen lanes

The campaign has two disjoint lanes.

### Primary: out-of-box product defaults

The primary lane runs three adjacent jobs on each of `r0f0`, `r1f1`, and
`r2f2` for all 13 registered regression tasks:

| Order | Dataset | Task ID |
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

| Code | Arm | Required implementation | Manual model config |
| --- | --- | --- | --- |
| `D` | `darkofit_product_default` | Current DarkoFit TabArena adapter | `{}` |
| `M` | `chimeraboost_0_14_1_default` | Official TabArena ChimeraBoost adapter, exact 0.14.1 source | `{}` |
| `C` | `catboost_1_2_10_default` | Official AutoGluon/TabArena CatBoost adapter, exact 1.2.10 wheel | `{}` |

This is **39 coordinate groups, 117 outer jobs, and 936 child fits**. The
three engines retain their product defaults; the benchmark does not force
hyperparameter parity. The report must therefore disclose at least these
material differences:

| Setting | DarkoFit | ChimeraBoost 0.14.1 | CatBoost 1.2.10 |
| --- | --- | --- | --- |
| Iteration cap | 1,000 | 10,000 | 10,000 |
| Learning rate | automatic (`None`) | resolves to 0.1 with early stopping | 0.05 |
| Tree depth | automatic | resolves to 6 | library default |
| L2 | automatic | 1 | library default |
| Bins | 254 | 128 | library default |
| Categorical permutations | 1 | 4 | native CatBoost path |
| Ordered boosting | scalar-regression `auto` resolves off | off | library/adapter default |
| Linear residual/leaves | off | `linear_leaves=None`, validation selection | native CatBoost |
| Early stopping | DarkoFit adapter default | enabled, patience resolves to 50 | AutoGluon adaptive |

The benchmark adapters may add strict JSON telemetry, but native arms must
inherit the official fit and preprocessing paths without changing their
defaults. ChimeraBoost's possible constant-versus-linear validation selection
is part of its product behavior, including its additional work under the same
outer deadline.

### Diagnostic: identical safe-ordinal representation

Airfoil and Diamonds alone receive a second, separately named lane. It runs
the same three engines, on the same three outer coordinates, with the exact
same source-frozen, target-free ordinal transform applied before each engine:

| Code | Arm |
| --- | --- |
| `D` | `darkofit_product_default_safe_ordinal` |
| `M` | `chimeraboost_0_14_1_default_safe_ordinal` |
| `C` | `catboost_1_2_10_default_safe_ordinal` |

This is **6 coordinate groups, 18 outer jobs, and 144 child fits**. Native
counterparts already exist in the primary lane; a fourth arm is forbidden.
Each diagnostic arm must be identical to its native counterpart except for the
common representation boundary. Cross-engine ordinal comparisons and
within-engine ordinal/native uplift are reportable, but they are diagnostic
only. They may not be pooled with the 13-dataset primary estimand, used to
rescue the rejected DarkoFit ordinal policy, or used to select a default.

The transform is the already-audited source declaration:

- Airfoil restores `attack-angle` compact codes to the frozen 27 physical
  angle values.
- Diamonds maps `cut`, `color`, and `clarity` to their frozen declared ranks.

The transform is fitted exactly once on each child-training fold. It may not
inspect targets, validation/test rows, label lexical order, frequency, target
means, or row order. Missing values remain numeric NaN. Unknown schemas,
feature order, compact category domains, or values fail closed. All encoded
columns are numeric, and no engine may retain a native categorical/target-stat
route for them.

Across both lanes the frozen total is **45 coordinate groups, 135 outer jobs,
and 1,080 child fits**.

## Exact execution order and resources

Each coordinate's three jobs are adjacent. Starting from the first primary
coordinate and continuing without a per-dataset restart, successive groups
use this exact six-permutation cycle:

```text
D M C
M C D
C D M
C M D
D C M
M D C
```

The primary lane repeats the cycle continuously over all 39 groups. Each
engine appears 13 times in each position. The diagnostic lane starts a new
lane-local cycle and uses exactly its six rows once; each engine appears twice
in each position. Because TabArena internally buckets a dispatch by task, the
runner dispatches the complete primary lane first and the complete diagnostic
lane second; a single mixed-lane dispatch is forbidden. The runner records a
SHA-256 digest of the complete ordered grid. Missing, duplicate, extra,
reordered, out-of-lane, or misconfigured jobs invalidate execution.

Every outer job uses:

```text
num_bag_folds=8
num_bag_sets=1
model_random_seed=0
vary_seed_across_folds=True
fold_fitting_strategy="sequential_local"
num_cpus=18
num_gpus=0
ag.max_time_limit=3600
raise_on_model_failure=True
calibrate=False
```

Outer jobs also run sequentially (`debug_mode=True`); there is no hidden outer
parallelism. Every child must resolve to exactly 18 CPUs and zero GPUs. The
3,600-second limit is per outer job. Failed, imputed, invalid, uninferable,
deadline-hit, `time_limit`-stopped, incomplete, or nonfinite jobs/children are
reported and invalidate a complete comparison; they are never silently
removed or imputed.

## Warmup and fitted telemetry

Warmup occurs before `run_jobs` and outside all measured records. One attested
process-local warmup covers each engine's numeric and categorical regression
route plus small and large prediction batches, using the same 18-thread
policy. The ChimeraBoost route uses at least 1,000 training rows and an
explicit eval set so constant/linear selection is exercised. CatBoost is
imported and fit through its official wrapper. `CHIMERABOOST_WARMUP` must be
unset, exactly empty, or whitespace-trimmed `0`—the only values that the
pinned 0.14.1 implementation actually treats as disabled—so import-time work
cannot contaminate timing.

Warmup history is append-only, carries deterministic prediction fingerprints,
and is sealed into completion attestation. Warmup failure prevents measured
execution.

Every child carries two strict JSON objects with common schemas:

- `comparator_fit`: engine, requested/attempted/retained/best iterations,
  resolved parameters and learning rate, selected mode/lane where applicable,
  CPU/GPU allocation, stop reason, deadline state, and engine-specific audit
  fields such as ChimeraBoost linear selection or CatBoost tree count.
- `benchmark_representation`: native or safe-ordinal identity, ordered feature
  schema binding, fitted schema, categorical inputs, fold-local constant drops,
  and—for ordinal—frozen domain/digest, encoded positions, one training fit,
  evaluation-transform counts, target-free status, and zero unknown values.

The unmodified ChimeraBoost and AutoGluon CatBoost adapters do not persist
which callback requested every non-cap termination. Their telemetry therefore
uses a null stop reason when early stopping, time, memory, and (for ChimeraBoost)
no-legal-split cannot be distinguished without changing the official fit path.
These unresolved competitor stops are counted and prominently disclosed; they
are never silently relabeled. A known time/deadline stop still invalidates the
campaign, while unresolved stops qualify the descriptive comparison rather
than being presented as proof that no competitor callback truncated a child.
In particular, a timed ChimeraBoost automatic constant/linear selection is
unresolved because the official wrapper persists only the winning candidate,
not the shared callback outcome of both candidate fits.

The six adapters must expose the same common field sets. Each native/ordinal
pair must resolve the same official engine defaults, and the three ordinal
adapters must produce the same transformed matrix and representation audit for
the same child-fold input. Native adapters may drop only attested
child-training-fold constants while preserving feature order.

## Exact source and runtime provenance

Execution requires a clean, committed DarkoFit checkout and a clean editable
TabArena checkout. Imported modules must resolve to those exact repositories.
The ChimeraBoost source is the clean exact `v0.14.1` tag commit:

```text
9c9ea6e704a9fe2bfe6d6c284b22de73914be048
```

The current post-tag documentation-only `main` commit is not labeled as the
exact tag. CatBoost is exactly 1.2.10; every file in its installed wheel
manifest is hashed, including Python sources, the native extension, metadata,
and notebook assets. Every installed `autogluon.common`, `autogluon.core`,
`autogluon.features`, and `autogluon.tabular` file is likewise hashed because
their resource, preprocessing, callback, utility, and default-parameter
modules can alter the fit. Each installed byte must match the SHA-256 and size
declared by its wheel `RECORD` (the `RECORD` file itself is the sole permitted
unhashed entry), and the imported critical module paths must resolve into
those attested distributions. TabArena's ChimeraBoost wrapper is also hashed.
The manifest additionally binds:

- DarkoFit, TabArena, and ChimeraBoost Git HEAD/tree/status and sanitized
  remotes;
- every relevant benchmark/package source blob and imported module path;
- Python executable/version, dependency versions, environment variables,
  platform, host identity, CPU, and memory;
- exact protocol, ordered-grid, output-root, runtime, and warmup digests.

An import from a stale editable checkout, an installed distribution whose
metadata disagrees with its source, a dirty source tree, an unbound file, or a
runtime mismatch fails before measured work.

## Fresh cache, atomic resume, and safe analysis

The default is a nonexistent or empty campaign directory. Registered TabArena
rows and historical timings are never reused. `--resume` is permitted only for
the exact attested source/runtime/protocol. Every runner-managed path that is
read, archived, hashed, or replaced must be a real regular file beneath the
campaign root; symbolic-link components, unexpected `results.pkl` paths, and
unexpected managed output targets fail closed. Unrelated entries are outside
the campaign artifact set and are never opened or traversed by the runner or
analyzer.

Resume is lane-and-coordinate-triad atomic. If any `D`, `M`, or `C` result in
a triad is missing, corrupt, incomplete, inconsistent, or invalid, all three
members of that triad are archived and rerun. A native primary triad and an
ordinal diagnostic triad at the same dataset/coordinate are distinct groups.
Only complete validated triads remain cached.

The runner alone may open its trusted TabArena pickle files. It emits a
normalized, finite, non-executable JSON payload and an attestation that binds
every raw result by relative path, byte count, and SHA-256 digest. The
standalone analyzer treats raw results as opaque bytes. Its own source may not
directly import or call `pickle`, call a runner raw-result decoder, or invoke
TabArena deserialization; attested raw-result bytes are used only for path,
size, and SHA-256 verification. Transitive dependencies may import the
standard-library `pickle` module for unrelated internals (NumPy does so during
a normal import), which is not the security boundary. The analyzer verifies
the manifest, runtime, sources, import identities, ordering, schemas, raw byte
hashes, and one-to-one result binding before computation, immediately before
publication, and after publication.

Analysis publishes exactly seven deterministic files as one rollback-safe
atomic group:

```text
primary_paired_splits.csv
primary_per_dataset.csv
ordinal_diagnostic_paired_splits.csv
ordinal_diagnostic_per_dataset.csv
paired_children.csv
summary.json
report.md
```

Custom output paths, partial publication, stale managed outputs, or a
provenance change during publication fail closed and preserve the previous
complete set.

## Frozen estimands and uncertainty

For every declared numerator/denominator contrast, ratios below one favor the
numerator. Each coordinate contributes `log(numerator / denominator)`.
Coordinates are averaged within each dataset and dataset estimates receive
equal weight: `1/13` in the primary lane and `1/2` in the ordinal diagnostic.
Raw-row pooling is forbidden.

The primary lane reports all pairwise engine contrasts (`D/M`, `D/C`, `M/C`)
for test RMSE, validation RMSE, training time, inference time, and incremental
memory (`peak_mem_cpu - min_mem_cpu`). Raw peak process RSS is secondary only
because it carries process-history/order contamination.

The diagnostic lane reports the same cross-engine ordinal contrasts and the
three within-engine ordinal/native contrasts on Airfoil and Diamonds. Native
and ordinal rows remain in separate lanes even when computing those explicitly
declared within-engine contrasts. Because the native and ordinal lanes execute
in separate schedule blocks, the within-engine cross-lane contrasts report
test and validation RMSE only; their training time, inference time, and memory
are order-confounded and must not be compared. Same-lane cross-engine ordinal
contrasts may report every declared metric.

An incremental-memory observation may legitimately be zero at the process
tracker's resolution. Raw zero values are retained, but any ratio or log-ratio
that depends on a zero is marked unavailable with an explicit reason and
count. The analyzer must not add an epsilon or let a memory measurement floor
invalidate otherwise valid accuracy and timing results.

Uncertainty uses **10,000 deterministic bootstrap draws with seed 20260719**.
Datasets remain fixed and are always equally weighted; coordinates are sampled
with replacement within each fixed dataset. Two-sided 95% intervals are
reported. Failures, imputations, timeouts, deadline hits, missing pairs, and
win/loss/tie counts are reported explicitly. There is no advancement gate and
no winner-selected configuration.

## Execution boundary

A dry run performs the complete source/default/import/grid/resource audit but
must create or modify no filesystem entry, including the requested output
directory, bytecode, warmup history, or TabArena result cache.

The expected full run is approximately 2.5–4 hours on the attested 18-CPU
machine, subject to ChimeraBoost's two-lane selection. The hard theoretical
ceiling is 135 job-hours if every outer job reaches its 3,600-second limit.

Example commands:

```bash
/Users/kmedved/.venvs/tabarena-darko312/bin/python \
  benchmarks/run_tabarena_regression_same_machine.py \
  --chimeraboost-path /Users/kmedved/.cache/chimeraboost-v0.14.1 \
  --output-dir .cache/tabarena-regression-same-machine-0.9.0-20260713 \
  --time-limit 3600 \
  --dry-run

/Users/kmedved/.venvs/tabarena-darko312/bin/python \
  benchmarks/run_tabarena_regression_same_machine.py \
  --chimeraboost-path /Users/kmedved/.cache/chimeraboost-v0.14.1 \
  --output-dir .cache/tabarena-regression-same-machine-0.9.0-20260713 \
  --time-limit 3600
```

Use the same full-run command plus `--resume` only after an interrupted run
whose complete manifest still matches.
