# Basketball categorical-combinations donor screen protocol

Status: frozen before any candidate outcome is observed.

Date frozen: 2026-07-17.

## Question

Does ChimeraBoost 0.15.0's pairwise categorical-combinations mechanism earn a
DarkoFit port on a small, noisy sports problem with a real unseen-player
boundary?

This is a donor screen, not a DarkoFit default-policy experiment. A passing
screen authorizes only an explicit, default-off DarkoFit implementation for
separate confirmation. It cannot authorize an automatic policy or a default
change.

## Why the creator lane needs a categorical companion view

The unchanged creator benchmark contains 15 numeric features and no
categorical feature. ChimeraBoost's `cat_combinations` route therefore cannot
engage there. The original numeric view remains in this protocol as an exact
non-engagement guard, but it cannot answer the quality question.

The quality screen uses the same immutable source rows, `MP > 500` filter,
`MPG` target, unshuffled ten outer folds, alphabetical team holdout, and
585-row cold-player mask. Only the model features change to four naturally
discrete source fields:

1. `Pos`, as the source position string;
2. `Age`, as the exact integer age and explicitly treated as categorical;
3. `Tm`, as the source team string;
4. `starter`, the existing binary `GS / G >= 0.5` derivation.

`Player` is never a model feature. It is used only to keep the model's internal
validation split group-aware and to identify the cold-player boundary. No
continuous feature is discretized for this screen.

The view is deliberately difficult. Every held-out team is unseen during the
full training fit, so team-bearing combinations must fall back to their prior.
Position, age, and starter combinations remain observable, including on cold
players.

## Immutable inputs

- DarkoFit source before the runner:
  `c34ab91d342f68073bbda0095b0204f3c02a8f4d`.
- ChimeraBoost source, local `main`, `origin/main`, and `upstream/main`:
  `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d` (`0.15.0`).
- ChimeraBoost package tree:
  `112078745db7f58b6d4399ecbb4ecebafe860256`.
- ChimeraBoost implementation hashes:
  - `chimeraboost/preprocessing.py`:
    `c3f062058a40df17b35b7a3c932d16173ba45c5294ca450b08ef447f563bcecb`;
  - `chimeraboost/sklearn_api.py`:
    `d354d360fea762be46a92e6cfaf9bc244c60690b7ecd0eebb8753aabc4b78c15`.
- Both projects are Apache-2.0. ChimeraBoost's `LICENSE` hash is
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
  This screen copies no donor code. Any later substantial or literal port must
  add a specific ChimeraBoost/bbstats entry to `NOTICE`.
- Raw basketball CSV:
  - 2,549,434 bytes;
  - SHA-256
    `43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2`.
- Existing creator numeric training matrix SHA-256:
  `05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b`.
- Training target SHA-256:
  `7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf`.
- Creator-fold fingerprint:
  `7270350a38a687b6e783d18d9c02e5a45f2c7367faa4a6955e74c710f5e8daea`.
- Cold-player mask SHA-256:
  `e17887c9240fd95aee21d37d7e1e8a533c38ef87f4492acd28fb3aa496a3dd19`.
- Canonical length-prefixed UTF-8 categorical-view hashes:
  - 5,241 training rows:
    `8f201e2c36b4addc6a223fb58b91912a5f2a0a6e732bea6558b0230c000cec17`;
  - 2,409 held-team rows:
    `ca708478f883aae7b2ebb1c01eea0ba6566af328505fd0810af152a3ac2ca18d`.
- Canonical length-prefixed player-sequence hashes:
  - training:
    `f59ca6aefdbdafb0ac6be4e9073bd5cbf5e5b0b8413004c30e776f7cae19c22d`;
  - held-team:
    `4f161a9233d4bfe5017c2f13e9b52d511c728e2b7c345479375b5acd3a8e995e`.
- Training levels in `(Pos, Age, Tm, starter)` are `(5, 23, 25, 2)`;
  held-team levels are `(5, 24, 12, 2)`.

The string hash writes every scalar as UTF-8 preceded by its unsigned
eight-byte little-endian length, in row-major order. This avoids delimiter
ambiguity.

## Frozen runtime

Use `.cache/basketball-py312/bin/python`:

- Python 3.12.13;
- macOS 26.5.2 arm64;
- NumPy 2.4.6;
- pandas 3.0.3;
- scikit-learn 1.9.0;
- Numba 0.66.0;
- llvmlite 0.48.0;
- joblib 1.5.3;
- threadpoolctl 3.6.0.

Every measured fit runs alone in a fresh worker with `PYTHONHASHSEED=0`,
`CHIMERABOOST_WARMUP=0`, and all numerical/Numba thread limits set to 18.
Unmeasured workers populate the Numba disk cache before timing. The parent
launches workers sequentially and alternates arm order.

## Isolated arms

Both quality arms use:

```text
ChimeraBoostRegressor(
    n_estimators=2000,
    learning_rate=None,
    depth=None,
    l2_leaf_reg=1.0,
    max_bins=128,
    cat_smoothing=1.0,
    cat_n_permutations=4,
    early_stopping_rounds=None,
    loss="RMSE",
    min_child_weight=1.0,
    thread_count=18,
    random_state=4,
    ordered_boosting=False,
    leaf_estimation_iterations=1,
    linear_leaves=False,
    cross_features=False,
    selection_rounds=100,
    early_stopping=True,
    validation_fraction=0.2,
    n_ensembles=None,
)
```

The fit receives all four column indices as `cat_features` and the aligned
player strings as `groups`. The only arm difference is:

- control: `cat_combinations=False`;
- candidate: `cat_combinations=True`.

The original numeric non-engagement check uses the same estimator settings and
creator features, without `cat_features`. It compares
`cat_combinations=None` with `False`.

## Execution

1. Validate every source, data, environment, feature, row, level, and split
   binding before a model fit.
2. Warm both categorical arms in unmeasured fresh workers.
3. Run both arms on every unchanged outer creator fold. Alternate which arm
   runs first by fold. Pass only the outer-training players as internal
   validation groups.
4. Run five reciprocal full-training blocks. Each worker records:
   fit time, public prediction time on the held-team and cold-player views,
   peak resident set size, fitted tree count, resolved learning rate, combo
   pairs, feature importance, predictions, hashes, and R2 scores.
5. Repeat the original numeric full-training non-engagement check in fresh
   workers and require exact predictions and fitted structure.
6. Write one create-only JSON artifact. Refuse an existing output, dirty
   source, a source change during execution, non-finite JSON values, or a
   failed binding.

## Predeclared gates

### Route and behavior

- The control has zero combination pairs.
- The candidate has exactly the six lexicographically ordered pairs from four
  categorical columns.
- Every worker resolves the same common estimator parameters and 18 threads.
- Within each arm, all five full-training blocks produce array-exact held-team
  and cold-player predictions, identical prediction hashes, identical fitted
  tree counts, identical resolved learning rates, and identical combination
  pairs.
- On the original 15-numeric-feature lane, `cat_combinations=None` and
  `False` produce array-exact train and held-team predictions, equal feature
  importance, equal fitted tree count, and no combination pairs.

### Quality

Let delta mean candidate R2 minus control R2.

- Mean ten-fold delta is at least `+0.002`.
- The candidate wins at least six of ten outer folds; exact ties are neutral.
- Worst outer-fold delta is at least `-0.010`.
- Full-training overlap-exposed held-team delta is at least `-0.001`.
- Full-training cold-player delta is at least `-0.001`.
- Full-training seen-player delta is at least `-0.005`.

### Runtime and memory

- Median full-training fit ratio, candidate/control, is at most `1.50`.
- Median public prediction ratio is at most `1.10` on both the complete
  held-team view and the cold-player subset.
- For each timed series, IQR/median is at most `0.20`.
- Median peak-RSS ratio is at most `1.50`, and the median absolute increase is
  at most 256 MiB.
- The candidate materializes exactly six combination-code columns; the screen
  records their cells and fitted category-map sizes.

All recorded numeric values must be finite. Every gate is conjunctive.

## Decision

- If every gate passes, authorize one isolated, explicit
  `cat_combinations=True` DarkoFit implementation. Keep it default-off. Before
  any automatic policy, require exact preprocessing/prediction/persistence
  tests, a new basketball confirmation, at least two declared categorical
  datasets, and a genuinely fresh preregistered promotion gate.
- If any quality gate fails, stop this donor mechanism without a DarkoFit
  port, retuning, or broader evidence.
- If quality passes but a route, runtime, or memory gate fails, do not port the
  donor implementation. A materially different design would be a new
  mechanism with a new protocol.

This screen does not use the 243 development coordinates, CTR23 lockbox,
TabArena lockbox, or any other promotion evidence. It cannot change a
DarkoFit default.
