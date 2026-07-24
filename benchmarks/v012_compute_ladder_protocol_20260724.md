# DarkoFit v0.12 release compute ladder

Date: 2026-07-24
Status: release-cadence characterization under `SHIP_RULES.md`

## Question

Where do the public DarkoFit v0.12.0 compute points sit against the current
public ChimeraBoost release, v0.23.0, on quality, fit time, prediction
throughput, and process-tree peak RSS?

This is the release scoreboard, not a tuning panel, default gate, fresh
confirmation, lockbox inspection, or TabArena placement.

## Immutable product and data sources

- DarkoFit `v0.12.0`, commit
  `a9eb4dbbf8af0e6db42e9ace433e7a267c80fca7`.
- ChimeraBoost `v0.23.0`, commit
  `6667843b8970454b0f582ffd1ab2be033989c578`, published
  `2026-07-24T01:06:38Z`. The runner verifies at worker zero that this remains
  the latest GitHub release.
- TabArena task/split source commit
  `4cd1d2526874962daae048a6f2dcf34aa272f3fa`, tree
  `a293df372a613c7358ba5fcd746f58d580cde7d6`.

All three source checkouts must be clean and checked out at the named commit.
The two model-library checkouts must also carry the named release tag.

## Public compute points

The runner constructs each public estimator directly. It does not wrap either
library in an AutoGluon bag.

| Code | Library | Public configuration |
| --- | --- | --- |
| D0 | DarkoFit | default |
| DA | DarkoFit | `preset="accuracy"` |
| D8 | DarkoFit | `ensemble_mode="v3", n_ensembles=8` |
| M0 | ChimeraBoost | default |
| MA | ChimeraBoost | `depth=10` |
| M8 | ChimeraBoost | `n_ensembles=8` |

Each estimator also receives the same coordinate seed and
`thread_count=14`. Current automatic behavior remains active: D0 includes the
v0.12 linear-leaf selector, and D8 may use the v0.12 ensemble-parallel route
when its public automatic policy engages.

## Grid and execution

- The same 13 fixed historical M2 regression datasets used by the v0.11
  ladder.
- Registered split coordinates `(0, 0)`, `(1, 1)`, and `(2, 2)` per dataset.
- 39 coordinates × 6 arms = 234 fresh worker processes.
- Deterministic, position-balanced arm ordering.
- One 14-core Apple-silicon host, 14 total CPU threads, zero GPUs.
- Workers run sequentially on an otherwise idle machine.
- Each route gets a reduced-iteration warmup outside the timed interval.
- Prediction timing uses three pilots, then at least three calls, targets one
  second, requires at least 0.5 seconds, and caps at 65,536 calls.
- RSS is sampled over the worker and recursive children at 5 ms intervals.
- Worker records retain split fingerprints, implementation paths and hashes,
  resolved model metadata, warnings, timing telemetry, and ambient Numba
  thread counts.

## Analysis and reporting

For each public point, ratios are reported against ChimeraBoost default.
Matched-profile DarkoFit/ChimeraBoost contrasts are also reported. Lower is
better for RMSE, fit time, prediction seconds per call, and RSS.

Each dataset is weighted equally after averaging its three registered split
log ratios. The interval resamples coordinates within each fixed dataset; it
does not pretend the 13 datasets are independent. The report includes
per-coordinate rows, per-dataset rows, head-to-head counts, stepwise fit and
prediction frontiers, and matched-profile peak RSS.

A harness bug may be fixed and the benchmark rerun under a new output
directory, with the material rerun noted in `benchmarks/TESTING_LOG.md`.
Results are read honestly; no threshold or certificate controls publication.
