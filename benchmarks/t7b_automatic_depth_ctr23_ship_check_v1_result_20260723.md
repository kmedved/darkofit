# Automatic-depth CTR23 ship-check

_Run 2026-07-23 from clean harness commit `3226b36`, with control
`e23d2b164f10374b1c0e02521c33fc96d48980da` and the unchanged automatic-depth
candidate `41e948f0c53b1d124e16071a7fa66eba47d084d3`._

## Result

The automatic-depth candidate is **not eligible to become the default** under
`SHIP_RULES.md`.

- Task-equal geometric-mean candidate/control RMSE: **`1.026662x`**
- Task bootstrap 95th-percentile ratio: **`1.062082x`**
- Leave-one-task-out maximum ratio: **`1.031063x`**
- Worst task ratio: **`1.165018x`**
- Task wins / ties / losses: **1 / 5 / 3**
- Integrity: **54/54 rows, 27/27 pairs, 9/9 tasks passed**

The clearest losses were `energy_efficiency` (`1.096127x`) and
`forest_fires` (`1.165018x`). The `forest_fires` loss was concentrated in
one official fold, where the candidate deterministically selected depth 4
instead of the control's depth 6. This is an observed holdout result, not a
prompt to retune the depth rule against these tasks.

## Interpretation

The candidate was clearly positive on the 32-lineage development panel but
did not transfer to CTR23. The automatic-default route is therefore closed.
The separately authorized public `depth="auto"` opt-in remains valid product
work on its correctness and honestly labeled development evidence.

The newest untouched sports season was not consulted: CTR23 already failed
the conjunctive holdout rule, so spending another untouched holdout could not
change the default decision.

CTR23 is now an **observed release-validation set**, not a pristine lockbox.
Future work must not tune this candidate from these outcomes and immediately
retest on the same tasks.

## Cost telemetry

Fit-time geometric-mean ratio was `0.922137x`; prediction-time ratio was
`1.048893x`. These are descriptive only. The launch recorded no competing
benchmark process, but macOS background activity produced load averages of
`10.81 / 12.09 / 9.02`, so this run is not release-grade timing evidence.

## Artifacts

- Manifest:
  `t7b_automatic_depth_ctr23_ship_check_v1_manifest_20260723.json`
  (`4edfd594ef967b383a75cdaab8caf8593c8f387f1d1a7741aee1666ab0db6cac`)
- Launch:
  `t7b_automatic_depth_ctr23_ship_check_v1_run1_launch_20260723.json`
  (`5bcdff4b305f4ccc6dfac1a7df11a86f4254d207a93fdc82983a2cc0f4078d9f`)
- Raw:
  `t7b_automatic_depth_ctr23_ship_check_v1_run1_raw_20260723.json`
  (`4bad8f98a80a0fac3769e7a3e9887491c9bd067fb757c6e7a7646c61e5927483`)
- Result:
  `t7b_automatic_depth_ctr23_ship_check_v1_run1_result_20260723.json`
  (`ceb1f6d4ee3feee4c850fa2632a8966603e98b453dc441d53764189d1616a553`)
