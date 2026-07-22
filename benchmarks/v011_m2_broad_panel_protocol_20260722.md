# v0.11 M2 broad-panel protocol

Status: **draft source contract; no outcome may be inspected until the
machine-readable contract generated from this source is committed.**

This is the first current-version calibrated-yardstick readout authorized by
the owner instruction dated 2026-07-21. It is descriptive, spent evidence. It
cannot select a default, expose the private ensemble, authorize TabArena-Lite,
or authorize a release.

## Frozen panel

The panel is the historical 13-dataset regression set, in its existing order,
at exactly `r0f0`, `r1f1`, and `r2f2`. The task IDs and registered split counts
come from the source-frozen historical campaign. There are 39 coordinate
groups, 117 outer jobs, and 936 eight-fold child fits.

The three adjacent jobs at each coordinate are:

- `D`: DarkoFit 0.10.1 from the eventual clean pushed contract commit, empty
  manual config, including the promoted fused-lane dispatch where eligible.
- `M`: ChimeraBoost 0.18.0 from exact Git commit
  `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`
  (`v0.18.0-6-gf14be60`), empty manual config. A moving checkout is forbidden.
- `C`: CatBoost 1.2.10 through the official AutoGluon adapter, empty manual
  config.

The evaluation framework is also fixed to the historical boundary: TabArena
commit `4cd1d2526874962daae048a6f2dcf34aa272f3fa` (tree
`a293df372a613c7358ba5fcd746f58d580cde7d6`, package 0.0.1) and AutoGluon
common/core/features/tabular `1.5.1b20260712`. A newer framework checkout or
distribution is not an equivalent run.

DarkoFit executes only from the pushed contract commit: it must be the direct
child of the recorded clean harness-freeze commit, must add only the
machine-readable contract, and must equal `origin/main`. A later descendant is
not an equivalent source pin even if its working tree is clean.

No private or public ensemble candidate is an arm. The historical TabArena
eight-fold evaluation is shared infrastructure, not a fourth product arm.

## Execution and resources

The complete 39-group sequence uses the historical continuous six-permutation
cycle `DMC, MCD, CDM, CMD, DCM, MDC`. Each engine occurs exactly 13 times in
each position. The source digest of this complete ordered grid is frozen.

Every outer job retains the historical settings: eight bag folds, one bag set,
fold-wise model seeds from base seed zero, sequential local fold fitting,
18 CPUs, zero GPUs, a 3,600-second per-job limit, failure propagation, and no
calibration. Outer jobs run sequentially, each in a newly launched Python
process; TabArena's in-process debug backend is confined inside that one-job
worker. Each worker first warms only its own engine's numeric and categorical
fit/predict routes, and publishes a PID-bound warmup/result attestation.
Warmup is excluded from measured records. Source trees, installed
CatBoost and AutoGluon distributions, runtime, hardware, job order, raw result
bytes, normalized finite JSON, per-worker warmup, and per-child telemetry are
attested.

A non-resume run requires a clean repository and unused output directory.
Resume is forbidden: any interrupted or failed run closes the campaign
identity and requires a prospectively frozen successor. Missing, duplicate,
failed, imputed, deadline-hit,
explicit-time-limit, nonfinite, misconfigured, or source-mismatched results
invalidate completion. There is no rerun to improve an outcome.

## Frozen measurements and reporting

Primary observations are test RMSE, validation RMSE, fit wall time, predict
wall time, incremental process peak RSS, and absolute process peak RSS. Full
per-coordinate rows are published.

For each ordered pair DarkoFit/ChimeraBoost, DarkoFit/CatBoost, and
ChimeraBoost/CatBoost, the point estimate is the equal-dataset geometric mean
of paired ratios: first average the three log ratios within each fixed dataset,
then average the 13 dataset values. Dispersion uses 10,000 seeded bootstrap
draws that resample the three registered coordinates within each fixed
dataset; datasets remain fixed and equally weighted.

The descriptive head-to-head supplement reports coordinate wins/losses/ties
and dataset wins/losses/ties using each dataset's three-coordinate geometric
mean. It is not an Elo estimate and is not an independent-dataset claim.

All results are characterization only. Prior panel artifacts remain untouched.
Fresh-confirmation and lockbox evidence are forbidden.
