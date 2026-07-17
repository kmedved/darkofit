# Basketball native-ordinal no-engagement protocol

Status: **frozen before runner implementation and formal execution**.

Date frozen: 2026-07-17.

## Question and evidence boundary

This is the basketball-first fatal screen for Track C1 of
`BEYOND_PARITY_PLAN.md`. It asks whether the native ordinal-at-binning
implementation is an exact, acceptably cheap no-op when no ordinal feature is
declared or safely detectable.

The screen cannot establish categorical quality, authorize a default, or spend
any CTR23, TabArena, I3, fresh-confirmation, or lockbox coordinate. Passing
authorizes only the C2 categorical development tier. Failing exactness,
no-engagement, runtime, or memory closes this implementation shape before any
categorical outcome is inspected.

## Frozen source, runtime, and data

The implementation under test is clean, pushed DarkoFit `main` at:

```text
commit        ceb96d191e316ab5f88204cfe767bc96f78239e8
darkofit tree d4661e0d4a919d2e0f0da4385b12034a5c853a6c
```

The protocol commit must be an ancestor of the later runner commit. The runner
must bind its own normalized source hash, this protocol's byte hash, and these
forward-only support files:

```text
14df91eb9c99912bd0fdf5bce81434934ff2ae84d588006a34fb513743283433  benchmarks/basketball_campaign_harness.py
40360ca54d1954d540bd72bec47c891a179fb0f764a0cb6061d3a33b8593aba1  benchmarks/basketball_harness.py
4492a65161b2261e5d437b9733c286368534c54f7fd23b6f6b95a804507fff52  benchmarks/basketball_guardrails.py
9a2d6b3dc8f3a8586cd4ac20bdb3607c17675cab1d89e6bbd7c438d2bba53fec  benchmarks/run_basketball_creator_benchmark.py
```

Use `.cache/basketball-py312/bin/python` with:

```text
Python          3.12.13
macOS           26.5.2 arm64
logical CPUs    18
NumPy           2.4.6
pandas          3.0.3
scikit-learn    1.9.0
Numba           0.66.0
llvmlite        0.48.0
joblib          1.5.3
threadpoolctl   3.6.0
```

The immutable creator basketball boundary remains:

```text
raw CSV bytes          2549434
raw CSV SHA-256        43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2
training rows          5241
features               15
training X SHA-256     05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b
training y SHA-256     7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf
fold SHA-256           7270350a38a687b6e783d18d9c02e5a45f2c7367faa4a6955e74c710f5e8daea
cold-player mask       e17887c9240fd95aee21d37d7e1e8a533c38ef87f4492acd28fb3aa496a3dd19
cold-player rows       585
```

Fourteen features are `float64`; `Age` is `int64`. `cat_features` is `None`.
The safe auto rule may recognize integer codes only when the column is also
listed in `cat_features`, so all 15 features must remain on the existing
numeric path.

## Frozen arms

Both arms instantiate only:

```python
DarkoRegressor(random_state=4)
```

They differ only in the fit call:

```python
# control
model.fit(X_train, y_train)

# candidate
model.fit(X_train, y_train, ordinal_features="auto")
```

No arm receives a validation set, groups, sample weights, callbacks, manual
learning rate, thread count, categorical declaration, early stopping, refit,
ensemble, or other override. Product defaults must resolve to the same
1,000-tree CatBoost-mode boosting lane and 18 threads.

The control must report ordinal mode `off`, no ordinal records, and no
`auto_params_["ordinal_features"]` entry. The candidate must report mode
`auto`, no records or indices, and exactly this inactive metadata:

```text
active=False
feature_count=0
feature_indices=[]
feature_names=[]
sources=[]
nominal_categorical_count=0
added_columns=0
target_stat_blocks_added=0
target_used=False
unknown_policy="fail_closed"
missing_policy="numeric_missing_bin"
```

Both preprocessors must retain all 15 input columns as numeric, zero
categorical columns, the same feature map, bin counts, and borders.

## Execution

Run three reciprocal fresh-worker blocks:

```text
block 0: control, candidate
block 1: candidate, control
block 2: control, candidate
```

Every worker has a unique empty Numba cache, `DARKOFIT_WARMUP=0`,
`CHIMERABOOST_WARMUP=0`, `PYTHONHASHSEED=0`, and every numerical/Numba thread
limit set to 18. Import and one explicit `darkofit.warmup()` call occur before
the formal timer. Workers run alone and sequentially.

Each worker:

1. validates source, runtime, cache, raw data, processed data, folds, and
   player guardrail;
2. fits and predicts all ten unchanged creator folds;
3. fits the complete creator training view and predicts the complete
   alphabetical-team holdout, cold-player subset, and seen-player subset;
4. times 200 public prediction calls on the 2,409-row complete holdout and 500
   calls on the 585-row cold-player input;
5. records predictions, scores, hashes, fitted metadata, preprocessing state,
   feature importance, normalized model-state identity, wall time, phase time,
   peak RSS, cache state, warnings, environment, and versions.

The formal total-fit timer covers the eleven product fits but excludes import,
data loading, explicit warmup, serialization, repeated prediction timing, and
analysis. Prediction timers include the complete public wrapper path.

## Exactness and normalized model-state identity

Every candidate prediction array must be bitwise equal to its paired control
array on every fold and all three guardrail views. Both arms must reproduce
the previously established product-default identities:

```text
fold 0  6200db22da190d8c0787d7794c1fb8d859af737ab7e3026716e23aa1be95125f
fold 1  96ad500c63ac3701fe769b03a369d3a01ed1af9695d71c7ea68936d36479da44
fold 2  230b3cb530dee9ba8f5196b2b12b77f8d62751c545828ca13bad3fe04e54261b
fold 3  4603c6b3036bbdee060faaa92e6eee18a1f803e4abe9bc4aa7906745db5bd1c1
fold 4  e00b84d4aa7b8640aad72f5aed6e5e578cef2035459aa146b972145dc8d19fef
fold 5  12852587a9d1cd729cde1b28d714ff0c30b8051e806d6bb2f3f68088f22912d8
fold 6  514663b32f0adaf0fc7591def75632f5ea1103598b2d7aaeeaf37fdc2560bb04
fold 7  45374906a6931f90a6fff29ba0544c4d66311bb6152e3f250d54db55e0c03384
fold 8  32167d2ad1ba4ee34297a812be85ae67675f638383d61fe709b130bdbb3931a5
fold 9  f51972e8f896568291b259d698726b224a2399711f8e8cdf451e68b5090ae38d
held    5d910ae8f6b0dca563b99f9f881dcb17ee092711a46b2890452eaa3b8e68367a
cold    998a14f530ed284865a50726191da067f72d69da3001614d664a4b90e7aa6376
seen    c9b506afbfb3eb660dd918ee9635d996c0285b0320ba250cbf39c80df9122425
```

The creator-fold mean R² must remain `0.5267495183883605`, the resolved
learning rate must remain `0.052312`, and every fit must retain 1,000 trees
with stop reason `iteration_limit`.

Raw `.npz` archives are intentionally not byte-equal because the candidate
must persist its inactive ordinal declaration. Instead, the runner constructs
a normalized logical archive identity for every paired model:

1. parse the JSON header and retain every field except noisy timing;
2. remove only the candidate's expected
   `auto_params.ordinal_features`,
   `auto_params.diagnostics.ordinal_features`,
   `wrapper.state.ordinal_features_mode`, and
   `wrapper.state.ordinal_features` fields;
3. hash the canonical normalized header plus each non-header array's ordered
   name, dtype, shape, and raw contiguous bytes.

The normalized identities must match within every pair. Removing any other
field is forbidden. Raw archive hashes and sizes are still recorded.

## Preregistered gates

All gates are conjunctive:

1. all ten fold arrays and all guardrail arrays are bitwise equal within every
   block and match the frozen hashes above;
2. all normalized logical model-state identities, feature importance arrays,
   preprocessors, resolved learning rates, tree counts, selected modes and
   lanes, stop reasons, and timing-free fitted metadata are identical within
   every pair and reproduce across blocks;
3. control and candidate ordinal telemetry satisfy the exact off/inactive
   contracts above, with zero added columns, zero target-stat blocks, and no
   target use;
4. there are no failures, non-finite values, unexpected warnings, dirty or
   unpushed sources, source changes, cache reuse, missing rows, or changed
   runtime/data/support bindings;
5. using same-block candidate/control ratios, median total-fit ratio is at
   most `1.02`, median held-team prediction ratio is at most `1.05`, and
   median cold-player prediction ratio is at most `1.05`;
6. each of those three paired-ratio series has IQR/median at most `0.10`;
7. median paired peak-RSS ratio is at most `1.05`.

Timing stability is assessed only on paired ratios. Per-arm millisecond IQR is
reported diagnostically and cannot pass or fail the campaign.

## Artifact and analysis boundary

The runner writes one create-only raw JSON artifact and refuses an existing
file or symlink. A separate analyzer reads that raw file, reconstructs every
gate without importing executable campaign state, and writes distinct
create-only analyzed JSON and Markdown report paths. Input and output paths
must be lexically and physically distinct. The analyzer must verify the raw
file immediately before computation, immediately before publication, and
after atomic publication; ambiguous or changed inputs fail closed.

No result field, failed block, timing observation, or outlier may be discarded
or rerun after inspection. An interrupted campaign may restart only from an
entirely new raw output path; there is no partial resume.

## Decision

- **Pass:** authorize the frozen C2 categorical development panel. This does
  not authorize promotion, a default, or a categorical claim.
- **Exactness or engagement failure:** fix or close C1 before any categorical
  development outcome is inspected.
- **Runtime or memory failure:** close this implementation shape without
  weakening a threshold or rerunning the same campaign.

The basketball panel remains reusable for future fatal no-engagement screens;
it supplies no positive categorical evidence.
