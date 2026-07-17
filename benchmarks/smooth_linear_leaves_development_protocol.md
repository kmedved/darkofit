# Smooth linear-leaf development screen

## Question

Do DarkoFit's explicit local-linear leaves close a material part of the
smooth-simulation gap, and is the remaining gap to ChimeraBoost 0.15 caused by
the leaf mechanism or by its separate cross-feature selector?

This is development evidence only. All coordinates belong to the already
spent CTR23 confirmation panel. No lockbox task is loaded, and no default or
automatic selector can be promoted from this run.

## Frozen data boundary

- Tasks: grid stability `361251`, kin8nm `361258`, and space_ga `361623`.
- Coordinates: OpenML repeat 0, folds 3 through 9, sample 0: 21 total.
- Folds 0 through 2 were already consumed by the minimal CTR23 campaign and
  are excluded from this formal screen; they were used only for the declared
  mechanism probe.
- All three tasks must reproduce the CTR23-v3 partition metadata as numeric,
  complete, one-repeat/ten-fold tasks. Task, dataset, feature/target, and split
  index hashes are recorded.
- Explicit denylist: all nine lockbox task IDs in
  `ctr23_partition.json`. Encountering one is a hard failure.

## Frozen arms

| Arm | Policy |
|---|---|
| `darko_default` | Current `DarkoRegressor(random_state=4)` |
| `darko_linear_current` | Current defaults plus `linear_leaves=True` |
| `darko_linear_matched` | `linear_leaves=True`, `l2_leaf_reg=1`, `max_bins=128`, `learning_rate=0.1`, 1,000 rounds |
| `darko_linear_residual` | Current defaults plus `linear_residual=True` |
| `chimera_linear_only` | ChimeraBoost 0.15.0 with explicit `linear_leaves=True`, `cross_features=False` |
| `chimera_product` | Unmodified ChimeraBoost 0.15.0 product defaults |

The local ChimeraBoost source must be clean tag `v0.15.0`, commit
`851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`.

Each config/task worker warms fold 3 once outside timing, then evaluates folds
3–9 sequentially. Three task workers run concurrently with six threads each;
config waves remain sequential. Runtime and peak RSS are descriptive
same-machine evidence, not promotion gates.

## Development gates

A DarkoFit fixed-linear arm advances to benchmark-only selector design if:

1. its equal-task geometric-mean RMSE ratio versus `darko_default` is at most
   `0.98`;
2. it wins at least two of three datasets;
3. no dataset-level geometric-mean ratio exceeds `1.00`; and
4. it wins at least 14 of 21 splits.

If both current and matched arms pass, the lower equal-task ratio advances;
exact ties retain current defaults. This is an explicitly development-only
choice that must face basketball and fresh confirmation before any policy
claim.

ChimeraBoost product parity is reported against a strict `<=1.00` ratio but
does not determine whether the leaf mechanism advances. The
`chimera_linear_only` diagnostic separates local-linear capability from
ChimeraBoost's independently selected cross features.

`linear_residual` is recommended for 1.0 deprecation only if the advancing
local-linear arm beats it on every dataset. No deletion occurs in this run.

## Interpretation

Passing means "design and test a conservative selector." It does not mean
linear leaves should be on by default. Failure retires this linear-leaf policy
route without touching the lockbox.
