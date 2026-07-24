# Automatic linear-selector v3 2020 sports ship-check

This is the newest-complete-season sports half of the `SHIP_RULES.md`
ship-check for the automatic linear-leaf selector.

The owner clarified that unrelated external-comparator work does not count as
candidate-development contact. Accordingly, the GPBoost characterization
does not spend this DarkoFit selector holdout. The selector itself has not
been fit, selected, or tuned on 2020 outcomes.

## Data and split

Use the fixed Basketball Reference game-log export:

```text
bytes   214366516
sha256  96e0efffb09e27f64cee395faa1783b025757c88efb74f5fb98cbd82c583d826
```

Apply the existing sports-panel aggregation to complete season 2020:

- retain player/team aggregates above 500 total minutes;
- use the same 15 numeric predictors;
- score `minutes_per_game`, `game_score`, and `box_plus_minus`;
- sort the 30 team labels and hold out positions 10–19;
- fit on the remaining 20 teams with player IDs supplied as validation
  groups; and
- report all held-team rows plus seen- and cold-player subsets.

## Comparison and readout

Each target/arm runs in a fresh worker from one clean source:

- control: `linear_leaves=False`;
- candidate: `linear_leaves="auto"` with the fixed 2-SE selector.

All other model policies are identical. The runner checks source identity,
input and split fingerprints, safe-NPZ round-trip, selector metadata, ambient
Numba thread restoration, and exact prediction hashes.

The sports ship-check passes only if every candidate prediction vector is
bit-identical to its paired control or, if the selector legitimately engages,
every reported held-team view has aggregate candidate/control RMSE at most
`1.0` and no target-level ratio above `1.0`. For this small seasonal panel,
the expected product behavior is a recorded `below_min_samples` fallback and
bit-exact control parity.

The automatic default is eligible only after this result and the already
passed CTR23 result are both clean. `linear_leaves=False` remains the public
rollback.
