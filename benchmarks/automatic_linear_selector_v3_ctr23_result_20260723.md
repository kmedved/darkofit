# Automatic linear-selector v3 CTR23 ship-check result

_Run 2026-07-23 from clean source and harness commit `d148a84`, using the
nine previously opened CTR23 release-validation tasks and three official
folds per task._

## Result

The automatic selector **passes the general holdout half** of the
`SHIP_RULES.md` ship-check.

- Equal-task geometric-mean automatic/control RMSE: **`0.935614x`**
- Task-bootstrap 95th-percentile ratio: **`0.980274x`**
- Worst task ratio: **`1.000000x`**
- Leave-one-task-out maximum ratio: **`0.952634x`**
- Task wins / ties / losses: **3 / 6 / 0**
- Integrity: **54/54 rows, 27/27 pairs, 9/9 tasks**

The selector engaged on all three folds of `naval_propulsion_plant`,
`wave_energy`, and `sarcos`, producing task ratios of `0.811485x`,
`0.809957x`, and `0.835847x`. It declined the remaining 18 pairs: 12 were
below the minimum sample count and six failed the 2-SE gain guard. Every
declined pair fell back bit-exactly to the control prediction.

## Interpretation

The effect transferred cleanly from Protein to three distinct smooth
regression tasks, while the selector remained inert elsewhere. CTR23 is
observed release-validation, not pristine evidence, because the prior
automatic-depth ship-check already opened it. These outcomes must not be
used to retune the selector.

The automatic default is **not shipped yet**. `SHIP_RULES.md` also requires
the newest untouched sports season. That check is next; if it regresses, the
selector remains an explicit opt-in.

## Cost telemetry

Automatic/control geometric-mean ratios were:

- fit time: **`2.200467x`**
- prediction time: **`1.278040x`**
- peak process-tree RSS: **`1.044868x`**

The fit overhead is expected from the two selection auditions but is a real
product cost and must be disclosed. This holdout result establishes quality
and safety, not compute-frontier dominance.

## Artifacts

- Manifest:
  `automatic_linear_selector_v3_ctr23_20260723_manifest.json`
  (`2df8029359bb28955d553a7a32b1844964981833a3c676135fe23c18d89ce184`)
- Launch:
  `automatic_linear_selector_v3_ctr23_20260723_launch.json`
  (`34fb7c073245d127c3ba7e073ff61be78672d3a33787637b9c39cd23e743b109`)
- Raw:
  `automatic_linear_selector_v3_ctr23_20260723_raw.json`
  (`2fcf5aed312a3cc05574912a1fbce4784aca6ed40e8a05f7273b88da9256205c`)
- Result:
  `automatic_linear_selector_v3_ctr23_20260723_result.json`
  (`2c770f271a8571dec5dcbccba17f2a3a4bd147bbd2cbfc3adf10be41c665c30b`)
