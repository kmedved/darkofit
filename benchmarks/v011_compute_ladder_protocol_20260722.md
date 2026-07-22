# v0.11 release compute-ladder protocol

Status: **prospective source protocol; no model outcome may be inspected until
the machine-readable contract is committed and published.**

Contract v2 supersedes v1 after the published v1 dry-run stopped before any
worker or model fit: its exclusivity scan mistook the runner's own launch-shell
ancestor for a competing benchmark. The v1 terminal record is create-only.
V2 changes only that scan to ignore the current process's ancestor chain while
continuing to reject every unrelated matching benchmark process. Arms, data,
order, resources, measurements, analysis, and claims are unchanged.

This is the Phase E release-milestone characterization authorized by
[`BEAT_CHIMERABOOST_PLAN.md`](../BEAT_CHIMERABOOST_PLAN.md). It asks whether
the public DarkoFit v0.11 quality-versus-compute frontier dominates the current
ChimeraBoost release. It is spent, descriptive evidence. It cannot change a
default, authorize fresh or lockbox access, or substitute for a powered Tier-D
campaign.

## Product and data boundary

DarkoFit is the immutable GitHub release `v0.11.0`, commit
`0b820e332cec2c083b1dd89eef0fe306d69cfc0e`. ChimeraBoost is the latest
published rival release at protocol freeze, `v0.20.0`, commit
`7d48e053e5bd3c7aded1126871aeb0f1f6b84c46`, published 2026-07-21. The
runner rechecks the GitHub latest-release endpoint immediately before the
first worker; a newer release closes this identity before outcomes and
requires a new contract. The pin cannot change during a run.

The panel reuses the 13 historical M2 regression datasets and exactly the
registered `r0f0`, `r1f1`, and `r2f2` outer splits. TabArena is pinned only as
the task/split source at commit
`4cd1d2526874962daae048a6f2dcf34aa272f3fa` (tree
`a293df372a613c7358ba5fcd746f58d580cde7d6`). This campaign deliberately does
**not** wrap product estimators in AutoGluon's eight-fold bag: doing that to an
eight-member product ensemble would create a 64-model nested bag and would not
measure the public compute point a user selects. Each row therefore fits one
public estimator on the complete registered training split and scores the
registered test split.

## Frozen compute points

Every constructor also receives the same coordinate seed and `thread_count=14`.
No HPO is performed.

| Code | Arm | Public configuration and rationale |
| --- | --- | --- |
| `D0` | DarkoFit default | Empty public config. |
| `DA` | DarkoFit accuracy | `preset="accuracy"`, DarkoFit's documented accuracy-oriented single-model profile. |
| `D8` | DarkoFit ensemble | `ensemble_mode="v3", n_ensembles=8`, the public v0.11 recipe. |
| `M0` | ChimeraBoost default | Empty public config at v0.20.0. |
| `MA` | ChimeraBoost accuracy | `depth=10`, the rival's documented large/interaction-heavy accuracy setting; it is not represented as a universal preset. |
| `M8` | ChimeraBoost ensemble | `n_ensembles=8`, the rival's documented maximum-accuracy mode, retaining its public parallel-member default. |

The six jobs for each coordinate use a fixed cyclic Latin order. Across 39
coordinates every arm occupies each position six or seven times. Workers are
sequential and fresh. Each worker warms the relevant numeric/categorical and
tree/ensemble routes with a reduced iteration budget outside measurement,
then fits exactly one formal arm.

## Resources, measurements, and integrity

The common machine budget is 14 CPU threads and zero GPUs. ChimeraBoost's
public ensemble may divide that fixed total among member processes; DarkoFit's
public v3 fits members sequentially at the same total ceiling. That product
behavior is part of the measured frontier. Timed work requires exclusive
machine use.

Each of the 234 fresh workers records:

- test RMSE and a prediction hash;
- fit wall time;
- a repeated, warmed prediction interval on the actual registered test batch,
  including pilots, call count, seconds per call, and rows per second;
- aggregate process-tree RSS before fit, peak RSS, peak-minus-start RSS, and
  end RSS, sampled every 5 ms so parallel members are included;
- resolved member/tree/thread counts and selector/profile metadata;
- exact dataset/split fingerprints, product-source identities, environment,
  Python warnings, complete launcher stdout/stderr, and worker PID.

Prediction timing uses three pilots, targets at least one second of repeated
calls, requires at least 0.5 seconds, and caps at 65,536 calls. Warmup, OpenML
loading, and fingerprinting are outside fit and prediction measurements. A
two-hour worker timeout is an integrity ceiling, not a quality or materiality
gate.

The output directory, manifest, each worker record, aggregate raw artifact,
and terminal record are create-only. Resume is forbidden. Any source mismatch,
newer rival release before worker 0, duplicate/missing worker, nonfinite value,
failed fit, timeout, or contract drift closes the identity. There is no rerun
to improve an outcome.

## Frozen analysis

All ratios are lower-is-better and paired on coordinate. Point estimates give
each fixed dataset equal weight: average the three coordinate log ratios within
each dataset, then average the 13 dataset values. Seeded 10,000-draw dispersion
resamples the three registered coordinates within each fixed dataset; it does
not treat 13 datasets as an independent population. Full coordinate and
per-dataset tables and head-to-head coordinate/dataset win-loss-tie counts are
published.

For the empirical fit and prediction frontiers, every arm is normalized to
ChimeraBoost default on the same coordinates. Within each engine, dominated
points are removed separately for fit cost and prediction cost. At every union
budget where both engines have at least one observed configuration no slower
than the budget, the achieved quality is the best observed quality among
eligible configurations. There is no interpolation. DarkoFit has full-curve
dominance on an axis only if its stepwise quality envelope is no worse at every
comparable observed budget. The strict program verdict additionally requires
DarkoFit's matched default/accuracy/ensemble process-tree peak RSS to be no
worse on all three points. Fit, prediction, process-tree peak RSS, and
peak-minus-start RSS remain adjacent in the scoreboard; a quality win does not
erase a cost loss. The verdict is a pre-declared point-estimate readout with
paired uncertainty adjacent; it is descriptive evidence, not a certificate.

CatBoost and TabArena placement are intentionally outside this run. CatBoost
remains the separate quality ceiling; this release deliverable is the moving
two-library rivalry frontier, and the owner explicitly deferred M4.
