# Basketball packed-prediction routing result

_Formal result for
[`basketball_packed_prediction_protocol.md`](basketball_packed_prediction_protocol.md)._

## Decision

**Reject the forest-work-aware cutoff and restore the existing 8,192-row
boundary.** The candidate passed every correctness, source, configuration,
dispatch, storage, and real-basketball parity gate, but failed two frozen
timing gates. The protocol is fail-closed and forbids threshold retuning or a
second confirmation on fold 1.

The immutable artifact is
[`basketball_packed_prediction.json`](basketball_packed_prediction.json).
It binds candidate commit `e961bcc2ea64706169641722b5935f9f31402fa3`, the
DarkoFit package manifest
`6e80c24202ef503d43f6655ea66e866d7cb52ff670df8054fbf962483b8e9846`,
and ChimeraBoost 0.15.0 commit
`851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`.
The artifact's runner hash refers to the exact runner at candidate commit
`e961bcc`; the post-result copy on `main` only moves the rejected cutoff formula
out of production code so the source-gated historical command remains
importable.

## What passed

- Candidate, legacy-route, and ChimeraBoost predictions were array-exact on
  all six cases: the reserved creator fold, cold-player and held-team views,
  the 127-row serial control, and both repeated-row throughput cases.
- Both arms retained the complete frozen 1,000-tree, depth-six,
  constant-leaf configuration. Source manifests, support-file hashes, actual
  dispatched kernels, packed storage, output shape, and dtype all matched the
  protocol.
- At the 524-row confirmation fold, the parallel candidate core was **3.19x
  faster** than the legacy DarkoFit route. It was 1.044x ChimeraBoost core time
  and 0.954x ChimeraBoost public time.
- At the 585-row genuinely cold-player batch, the candidate core was **4.00x
  faster** than legacy. It was 0.917x ChimeraBoost core time and 0.884x
  ChimeraBoost public time.
- At the 2,409-row held-team batch, the diagnostic core speedup was **4.95x**;
  public DarkoFit time was 0.988x ChimeraBoost.
- The 8,192- and 100,000-row candidate cores both remained within the frozen
  ChimeraBoost parity bound.

## Why it failed

Two of thirteen gates failed:

1. At 8,192 repeated basketball rows, candidate/legacy packed-core time was
   **1.134**, above the 1.10 non-regression limit. Both routes invoked the same
   parallel Numba kernel over the same packed arrays, so this is consistent
   with timing noise rather than different computational work. The 100,000-row
   ratio was 0.880. The preregistered gate nevertheless fails.
2. The cold-player candidate-core IQR/median was **0.309**, just above the 0.30
   stability limit. All other gated timing series were within the bound.

The formal result is therefore `passed=false` with recommendation
`reject_work_router_without_threshold_retuning`. The strong small-batch result
is useful diagnostic evidence, but it is not permission to waive the frozen
large-batch and stability requirements after observation.

## Disposition and next step

The production routing change is reverted in the result commit. The protocol,
runner, and artifact remain as an auditable rejected experiment; their source
pins intentionally describe the candidate commit and are not a benchmark for
later `main` revisions.

This closes the packed-prediction routing attempt. The broader best-of-both
program should return to isolated product mechanisms on the fast basketball
and cold-player guardrails rather than spend more confirmation evidence on
this cutoff.
