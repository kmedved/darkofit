# Basketball multi-season sports confirmation protocol

## Purpose

Track S4 needs a durable sports confirmation boundary that is broader than the
single 2000-01 creator dataset. This protocol freezes a DARKO-derived,
player-team-season panel before any candidate or comparator is fit on it.

The first candidate is the only survivor of the earlier basketball
random-strength screen:

- control: current `DarkoRegressor` defaults (`random_strength=0.0`);
- candidate: the same model with `random_strength=0.5`.

The confirmation can support an opt-in recommendation for noisy basketball
regression. It cannot change the global default. ChimeraBoost 0.15.0 and
CatBoost are same-machine external comparators, not tuning sources.

## Frozen source and transformation

The raw source is the pre-existing DARKO Basketball Reference game-log export:

```text
bytes   214366516
sha256  96e0efffb09e27f64cee395faa1783b025757c88efb74f5fb98cbd82c583d826
```

The source contains seasons 1985-2020 and predates this program by years. This
protocol uses complete seasons 2017, 2018, and 2019 only. Rows must have a
positive finite `Minutes`, a player ID, and a team. Player-game rows are
aggregated separately by `(bref_id, Player, year, Tm)`, preserving traded
players' multiple team rows. A player-team-season is eligible only when its
played-game minutes exceed 500, matching the creator benchmark's threshold.

Age strings of the form `YY-DDD` become `YY + DDD / 365.25`. The 15 shared
predictors are:

```text
age, games, start_rate, ts_pct, efg_pct,
orb_pct, drb_pct, trb_pct, ast_pct, stl_pct,
blk_pct, tov_pct, usg_pct, offensive_rating, defensive_rating
```

Age and the rate predictors are minutes-weighted within player-team-season.
When a game-level rate is structurally unavailable, that game is omitted from
that rate's numerator and denominator; every retained aggregate must still be
finite. `games` is the number of played-game rows and `start_rate` is the
arithmetic mean of `GS`. The three independently scored targets are:

1. `minutes_per_game`: total played minutes divided by games;
2. `game_score`: arithmetic mean game score;
3. `box_plus_minus`: minutes-weighted box plus/minus.

The canonical panel sorts within each season by player name, player ID, and
team. The builder writes deterministic float64 CSV bytes and a committed
manifest that binds the raw source, processed bytes, row identities, targets,
features, split fingerprints, protocol, builder, and power calculation.

## Nine primary cells and player guardrails

Each target-season pair is a separate cell, giving nine equal-weight cells.
Within each season, team labels are sorted and the first one third (10 of 30)
are held out. The remaining 20-team rows form the primary creator-style
sample. Ten-fold `KFold(shuffle=False)` evaluation follows the canonical player
order. Every arm receives byte-identical float64 matrices and identical fold
indices.

For each cell, one additional model fits all non-held teams and predicts all
held teams. The held rows are partitioned before fitting:

- **seen-player:** the player ID also occurs for a non-held team in that season;
- **cold-player:** the player ID does not occur in that season's training rows.

The overlap-exposed held-team score, seen-player score, and genuinely
cold-player score are reported separately. They are never blended into the
primary creator-fold score.

The primary estimand is the arithmetic mean of the nine cell-level mean R²
values. Every target and season therefore receives equal weight regardless of
row count or target scale. Leave-one-cell-out aggregates are the concentration
check.

## Contamination boundary

The candidate value `0.5` was selected on the creator's distinct 2000-01
season-level dataset. The three seasons, game-log export, aggregation, targets,
and S4 identities do not occur in any earlier DarkoFit benchmark artifact or
git history. Dataset construction may inspect schemas, row counts, missingness,
and identities, but no model may be fit and no candidate/comparator score may
be inspected until the protocol, builder, processed-data manifest, and formal
runner are committed.

The 2017-2019 panel becomes spent for future model selection after its first
formal use. It remains valid for regression tests and honest confirmation of a
candidate selected elsewhere.

## Candidate gate and preregistered power

Let each cell delta be candidate minus control mean creator-fold R². The
candidate confirms only if all conditions hold:

1. equal-cell mean delta is at least `+0.0005`;
2. every leave-one-cell-out equal-cell mean delta is nonnegative;
3. equal-cell overlap-exposed held-team R² delta is nonnegative;
4. equal-cell cold-player R² delta is nonnegative;
5. the median candidate/control total-fit-time ratio is at most `1.75`;
6. the median peak-RSS ratio is at most `1.10`; and
7. fit-time paired ratios have IQR/median at most `0.15`.

Seen-player quality and prediction time are diagnostic. Candidate/control
behavior fingerprints must reproduce across all timing repetitions.

The target-blind power design uses the ten paired fold deltas from the earlier
screen. Their mean is `0.002123761281670389` and sample SD is
`0.005482924018693017`. For each of 200,000 simulations, seed `20260717`:

1. draw nine true cell effects from a normal distribution centered at the
   earlier mean with conservative between-cell SD `0.004`;
2. draw ten observed fold deltas per cell using the earlier within-cell SD;
3. apply gates 1 and 2 above.

The design proceeds only if simulated primary-gate pass probability is at
least 80%. Safety and operating-cost gates are intentionally not assigned
invented probability models.

## Arms, execution, and external comparisons

All arms use random state 4 and 18 threads. DarkoFit control and candidate
otherwise retain product defaults. ChimeraBoost binds clean commit
`851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d` (0.15.0) and its product defaults.
CatBoost uses `CatBoostRegressor(random_seed=4, thread_count=18,
verbose=False, allow_writing_files=False)` and records its installed version.

Three reciprocal fresh-worker blocks run all four arms. Imports, source
validation, panel loading, and one explicit warmup fit are outside the formal
timer. The complete 90-fold-plus-9-guardrail workload is timed monotonically
inside each worker. Arm order reverses or rotates between blocks. Formal
candidate timing uses same-block paired ratios, never per-arm millisecond
spread.

The raw artifact is create-only and records predictions, hashes, scores, row
and fold identities, fitted metadata, timings, source states, environment,
versions, and peak RSS. A separate analyzer reconstructs all decisions from
the raw artifact and refuses an ambiguous input/output path.

If the candidate passes, document `random_strength=0.5` as a confirmed opt-in
for this noisy basketball regime; retain the global `0.0` default. If it fails,
close this candidate without retuning on S4.

For each external comparator, compare the eligible DarkoFit arm (candidate
only if it passes, otherwise control) using the same equal-cell, leave-one-cell-
out, held-team, and cold-player deltas. A "beats comparator on S4" claim
requires the same four quality conditions as gates 1-4. Otherwise report the
full Pareto result without a superiority claim. No external result can rescue
a failed candidate gate or authorize tuning on this spent panel.
