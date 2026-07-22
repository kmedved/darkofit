# Automatic linear-selector v2 Protein attribution protocol

_Frozen before any Protein attribution fit under candidate
`a53d4bf543534678189d87d88dcad87dd2a8bd8f`._

Contract identity:
`automatic-linear-selector-v2-protein-attribution-20260722`.

## Authority and evidence class

The automatic-selector development contract authorizes this run only because
M6 quality-successor-v3 inspection 1 returned `advance`. Protein and all three
coordinates below are already-spent development evidence. This run can close
the exact candidate or advance it to `ready_for_powered_fresh_design`; it
cannot merge, ship, change a default, inspect fresh confirmation, run
TabArena as a scoreboard, or access a lockbox.

The run binds to:

- selector candidate
  `a53d4bf543534678189d87d88dcad87dd2a8bd8f`;
- selector development contract
  `automatic-linear-selector-v2-development-20260722`;
- M6 result SHA-256
  `7445b70ca3bc727bb24f8990ceef590ca933eb1dd45ccefe9ee5788eff211948`;
- the v0.11 release-ladder v3 contract and its pinned TabArena split source;
  and
- OpenML task 363693, `physiochemical_protein`.

## Frozen coordinates and arms

Use the exact three release-ladder coordinates, in this order:

| Coordinate | Repeat | Fold | Fit seed |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 1 | 1 | 1 | 1001 |
| 2 | 2 | 2 | 2002 |

Each coordinate runs three candidate-source `DarkoRegressor` arms:

- `constant`: `linear_leaves=False`;
- `automatic`: `linear_leaves="auto"`; and
- `explicit_linear`: `linear_leaves=True`.

All other constructor policy is the public default, apart from the frozen
fit seed, `thread_count=14`, and `diagnostic_warnings="never"`. Arm order is
a cyclic Latin rotation so each arm occupies each execution position once.
Every arm/coordinate runs in a fresh process. A same-arm, two-iteration,
1,400-row warmup occurs outside fit timing. Fit uses the entire registered
training split and scoring uses the registered test split.

The prediction timing protocol is copied from the release ladder: three
pilots, at least three calls and 0.5 seconds, a one-second target, and at most
65,536 calls on the actual registered test batch. Fit RSS is sampled every
0.005 seconds over the worker and recursive children. The machine must expose
14 physical and 14 logical CPUs, with no competing declared benchmark job.

## Frozen invariants and decision rule

For every coordinate, the automatic arm must:

1. be eligible and resolve to `selected_linear`;
2. record a relative validation improvement at least `0.03`;
3. bind the selector seed to the release coordinate and record the expected
   weighted-target-stratified automatic holdout;
4. record disjoint automatic selection and validation rows;
5. finish with linear leaves active in the final booster;
6. produce the exact same test prediction bytes as `explicit_linear`; and
7. produce the exact same serialized core-booster state digest as
   `explicit_linear`.

The core-state digest covers every serialized booster array and the complete
header after removing only the automatic selector record and its duplicate
diagnostic entry. Those two records are validated separately above and are
expected provenance differences between an automatic wrapper and an explicit
wrapper; no predictive or fitted-policy field is excluded.

For quality, define each coordinate ratio as
`automatic RMSE / constant RMSE`. The aggregate is the geometric mean of the
three coordinate ratios. The exact candidate closes if the aggregate ratio or
any coordinate ratio exceeds `1.02`. There is no minimum improvement gate.
The result reports all RMSEs and ratios, the worst coordinate, every selector
margin, fit time, prediction throughput, absolute and delta RSS, and all
prediction/core-state hashes.

All seven invariants and both harm checks must pass. A failure is terminal for
this candidate identity. Passing advances only to the development disposition
`ready_for_powered_fresh_design`; it is not shipping evidence.

## Execution and no-rerun discipline

The protocol, runner, and tests must be committed and published before the
first fit. Execution requires clean, exact, published harness and candidate
sources plus the exact clean TabArena source. The output prefix is external to
the harness and create-only. The runner writes a create-only launch manifest
before worker zero; launch spends attempt 1 even if execution later fails.

On success it writes a create-only raw JSON artifact and terminal result. On a
caught failure it writes a terminal failure result without inventing missing
rows. No failed or inspected attempt may be rerun under this contract identity,
and no gate may be relaxed after inspection. A result note and 12-field
`TESTING_LOG.md` entry are required.
