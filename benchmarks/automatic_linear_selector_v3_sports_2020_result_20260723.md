# Automatic linear-selector v3 2020 sports ship-check result

_Run 2026-07-23 from clean source and harness commit `ad9aa8e`, using the
newest complete season in the fixed Basketball Reference export._

## Result

The automatic selector **passes the sports holdout** and is eligible to
become the automatic default under `SHIP_RULES.md`.

- Targets: `minutes_per_game`, `game_score`, `box_plus_minus`
- Held-team automatic/control RMSE: **`1.000000x`**
- Cold-player automatic/control RMSE: **`1.000000x`**
- Seen-player automatic/control RMSE: **`1.000000x`**
- Exact automatic/control prediction vectors: **3/3 targets**
- Selector resolution: **`below_min_samples` on 3/3 targets**
- Integrity: **6/6 rows, 3/3 pairs**

The panel contained 325 eligible player/team rows. The deterministic
middle-third team split fit on 220 rows and scored 105 held-team rows,
including 104 cold-player rows. The selector correctly stayed out of this
small sports regime and reproduced `linear_leaves=False` bit for bit.

## Decision

The fixed 2-SE selector is now clearly positive in development, passed CTR23
observed release-validation without a task loss, and caused no change on the
newest complete sports season. It therefore satisfies the current ship rule.

`linear_leaves="auto"` is the automatic default. `linear_leaves=False` is
the documented rollback. The evidence must be labeled honestly: large smooth
tasks received the quality gain, while small sports fits were protected by
exact fallback.

The owner clarified that unrelated external-comparator characterization does
not count as candidate-development contact, so the separate GPBoost study
does not spend this selector holdout.

## Cost telemetry

On these short exact-fallback fits, automatic/control geometric-mean ratios
were `1.021381x` fit time, `0.964238x` prediction time, and `1.003174x` peak
RSS. These are single-run telemetry, not speed claims. The broader CTR23
selector cost remains the relevant disclosure (`2.200467x` fit,
`1.278040x` prediction).

## Artifacts

- Launch:
  `automatic_linear_selector_v3_sports_2020_20260723_launch.json`
  (`88a78fe8cf35cd151db0255bd3068cbd379aa5c6df3a178831276c93071f47dc`)
- Raw:
  `automatic_linear_selector_v3_sports_2020_20260723_raw.json`
  (`922e4bbe5fc133c588952a5201e9fdfe54f70b9c2a1a9528c5e4ce8901fe987b`)
- Result:
  `automatic_linear_selector_v3_sports_2020_20260723_result.json`
  (`5bd9623d5f35faa5bfe62aceb60a819bfdd473591c6ac0e940b626ea6c968786`)
