# Basketball creator benchmark

This benchmark reproduces the small default-regressor comparison published by
the ChimeraBoost creator. It is the short-cycle quality target for DarkoFit;
the frozen TabArena and CTR23 work remains unchanged.

## Frozen source

- Creator script: [bbstats' basketball benchmark gist](https://gist.github.com/bbstats/b9f5c0c60a186f21d0574ad0220789c6), revision
  `cbaa9666f632a9891afb8e91959088d944d8c8b2`, SHA-256
  `40011048376dbc1af27c200568c0ba9c7608524c87503b8a5210cf689b98329b`.
- Dataset: the raw URL pinned by that script, SHA-256
  `43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2`.
- ChimeraBoost comparator: local upstream commit
  `29602d3452b1754042006ad2b14bca320c94b4b7`. The source reports version
  0.14.2, but this commit is 40 commits beyond tag `v0.14.2`; the commit is the
  authoritative identity.

The runner refuses a different ChimeraBoost commit or a dirty source tree by
default. Escape hatches exist for exploratory work, but their artifacts are not
the frozen baseline. Every artifact includes the expected comparator revision,
the active override flags, and an explicit baseline-eligibility verdict.

## Exact author lane

The `author` lane preserves the posted scoring call:

```python
cross_val_score(model, X_train, y_train, scoring="r2", cv=10, n_jobs=-1)
```

The integer CV is made explicit as `KFold(n_splits=10, shuffle=False)`. The
runner records every fold score and their arithmetic mean. Models retain their
quality-affecting product defaults. CatBoost receives only execution controls:
console/filesystem noise is disabled and `thread_count` is set to the lane's
recorded inner-thread limit. The four arms are:

1. `DarkoRegressor(random_state=4)`
2. `ChimeraBoostRegressor(random_state=4)`
3. `ChimeraBoostRegressor(random_state=4, n_ensembles=5)`
4. `CatBoostRegressor(random_state=4)`

Each arm runs in a fresh controlling subprocess. Dataset loading and imports in
that controller are outside the measured interval. In the author lane, however,
Loky fold workers start inside `cross_val_score`; their process startup, module
imports, and cache loading are included in `wall_seconds`. There is no explicit
warmup, matching the creator call shape. The runner forces ChimeraBoost's
opt-in import-time warmup hook off in every worker and records that setting.
The outer joblib folds have an explicit one-thread limit for OpenMP, BLAS,
NumExpr, and Numba, matching the usual anti-oversubscription policy for parallel
CV. Numba JIT and joblib/loky execution controls are normalized before the
worker imports Python packages. All inherited Numba, joblib, loky, OpenMP,
BLAS, TBB, and KMP controls are cleared; only the benchmark's canonical thread
and process controls are then set. The resulting values are recorded in every
arm. The runner also revalidates both source heads and worktrees before and
after every arm and proves imported local modules reside inside those attested
checkouts. Wall time can still depend heavily on pre-existing default Numba
caches and joblib process startup; R² is the primary comparison.

## Steady lane

The separate `steady` lane uses identical data, folds, scoring, constructors,
and random seed. It performs one complete first-fold fit and prediction outside
the timer, then evaluates the ten folds sequentially (`n_jobs=1`). This makes
each estimator the sole outer job. OpenMP, BLAS, NumExpr, and Numba thread
limits are set to the recorded logical CPU count. This gives a more
interpretable same-machine throughput measurement. It is a diagnostic lane,
not the creator's posted timing protocol.

## Data contract and limitations

The source has 11,893 rows. The creator keeps rows with `MP > 500`, recomputes
`MPG = MP / G`, then holds out the alphabetically first 12 of 37 team labels.
The score uses the remaining 5,241 rows and 15 numeric features. The processed
feature and target fingerprints are fixed in the runner.

Two important limitations are preserved, not hidden:

- The defined team holdout is not scored by `get_model_score`.
- The defined game-count weight `G` is not used.

The row-wise folds also contain repeated player identities and are not grouped
by team, player, or season. This is a useful fast default-quality target, not a
leakage-resistant claim of external generalization. Any held-team or grouped
diagnostic must be reported separately rather than blended into the author
score.

## Run

Use the existing Python 3.12 benchmark environment and both current source
trees:

```bash
PYTHONPATH="$PWD:$PWD/../chimeraboost" \
  /Users/kmedved/.venvs/darko312/bin/python \
  benchmarks/run_basketball_creator_benchmark.py --lane author

PYTHONPATH="$PWD:$PWD/../chimeraboost" \
  /Users/kmedved/.venvs/darko312/bin/python \
  benchmarks/run_basketball_creator_benchmark.py --lane steady
```

The default JSON outputs are ignored under
`.cache/basketball-creator-benchmark/`. A promoted baseline may be written to a
tracked path explicitly after the harness and source commits are clean.
