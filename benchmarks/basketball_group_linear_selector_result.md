# Basketball group-aware linear selector result

_Run 2026-07-17 from clean `main` at `1dd1c36`, under the frozen
[`basketball_group_linear_selector_protocol.md`](basketball_group_linear_selector_protocol.md)._

## Decision

Advance the 3% group-validation-margin selector to the already-spent smooth
development tasks. Do not expose an API or change defaults.

The selector declined linear leaves on all ten creator folds and the held-team
guardrail. Its final constant-leaf models matched the control prediction
hashes, canonical serialized model-state hashes, fold R² values, held-team R²,
and cold-player R² exactly.

## Basketball boundary

The candidate's linear-over-constant validation margins ranged from
`-0.2011%` to `+1.9546%`, below the frozen 3% selection threshold everywhere.
Every internal validation split was player-group-disjoint.

| Metric | Control | Selector | Paired median ratio |
|---|---:|---:|---:|
| Mean creator-fold R² | 0.526750 | 0.526750 | Exact |
| Held-team R² | 0.531269 | 0.531269 | Exact |
| Cold-player R² | 0.500434 | 0.500434 | Exact |
| Steady wall time | — | — | 1.535× |
| Summed fit time | — | — | 1.549× |
| Summed prediction time | — | — | 1.043× |
| Peak RSS | — | — | 1.030× |

All three paired wall, fit, and prediction ratio series passed the frozen
stability criterion. The wall, prediction, and RSS ratios stayed within their
3.5×, 1.25×, and 2× budgets. The extra fit work is expected: the
benchmark-only selector fits constant and linear candidates on a group-held
validation split before the unchanged final fit.

## Scope

This is a basketball safety result, not evidence that the selector improves
unseen smooth data. It authorizes only the next development campaign on spent
CTR23 coordinates. The CTR23 lockbox remains sealed, and no public selector,
automatic policy, or default is authorized.

Two earlier formal invocations failed closed because observational NPZ archive
size and then process peak RSS leaked into the behavior fingerprint. Neither
wrote an artifact or supported a decision. The
[`invalid-attempt record`](basketball_group_linear_selector_invalid_attempt.md)
documents the corrections; resource telemetry remains in the paired timing
and memory gates.

## Evidence

- Raw artifact:
  [`basketball_group_linear_selector.json`](basketball_group_linear_selector.json),
  SHA-256
  `cb56ab34769609cd9639245f9ca6ea2012a0ea1a2b532aae458a9e6cfd9f2f25`.
- Protocol SHA-256:
  `0009475772c9cb9e31c74e05a72853d87b6c6a6a9c05791878fcdccce6aae0c9`.
- Runner SHA-256:
  `65bd0ff91cd0a9f9069f199f8d3e24cb20a47a6cc9260911bf44fbec727eea8a`.
