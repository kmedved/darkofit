# Basketball automatic fused-oblivious promotion: ship the internal lane

## Decision

Promote the narrow fused-oblivious training lane through DarkoFit's internal
default dispatch. The final frozen recommendation is
`promote_internal_fused_lane`.

No public constructor parameter or model format changed. The automatic path
is limited to at least three threads, constant Hessian, all rows and features,
and no histogram injection, subtraction, row-parallel buffer, or random split
noise. Every ineligible path keeps the prior implementation.

## Final basketball result

| Measurement | Explicit reference | Automatic fused | Automatic / reference |
|---|---:|---:|---:|
| Mean 10-fold R² | 0.526749518388 | 0.526749518388 | exact |
| Median fit time | 28.926s | **19.314s** | **0.668** |
| Median steady wall | 29.464s | **19.832s** | **0.673** |
| Diagnostic prediction time | 18.648ms | 18.802ms | 1.008 |
| Median fresh-worker RSS | 256.9MB | 257.7MB | 1.003 |

Median fit time improved by 33.2% and steady wall time by 32.7%. Timing was
stable: reference max/min was 1.044 and automatic max/min was 1.021, below the
frozen 1.20 limit.

## Exactness and engagement

Every creator fold score and prediction hash matched exactly. The corrected
overlap-exposed held-team and 585-row cold-player outputs also matched, as did
feature importances, fitted metadata, behavior fingerprints, serialized model
bytes, and archive sizes. The explicit reference workers recorded zero fused
invocations; automatic workers recorded positive engagement, including 66,000
tree-level invocations in the canonical candidate worker.

An ordinary public fit without a benchmark override is separately tested to
engage the automatic lane. The benchmark's reference worker explicitly forces
the old path, preventing a fused-versus-fused comparison.

## Broader verification

Expanded candidate/reference tests are archive-exact for numeric and
categorical RMSE, MAE, Quantile, callbacks, and early stopping with exact
refit. Weighted RMSE and binary classification prove the nonconstant-Hessian
fallback, and one- and two-thread tests prove the low-thread fallback. Strict
prediction goldens and readable oblivious-tree oracles remain exact.

The complete suite produced 1,442 passes and 23 skips before the historical
evidence test was moved to a detached local clone of its frozen source commit.
That change preserves the production analyzer's fail-closed refusal to reuse
old TabArena evidence after package changes while allowing current product
development to validate the historical contract at its actual revision.

## Provenance

The final evidence used clean committed source at
`95476a8239a991f37bbd5a928c7c421f1185a6fe`, 18 threads, the unchanged
basketball folds, and the corrected player guardrails. Artifact publication
was atomic and create-only. No CTR23 development or lockbox data was used.
Apache-2.0 design attribution to ChimeraBoost commit
`a04430657fb82c806ee2a039506c99944a27accc` is recorded in `NOTICE`.

## Remaining gap

This is a meaningful engine win, not runtime parity. DarkoFit's basketball
steady time is now about 19.8 seconds versus the synced ChimeraBoost 0.15.0
diagnostic run at about 7.5 seconds. Future engine candidates should continue
to use basketball as the first fast fatal gate before broader work.

## Artifacts

- `basketball_fused_oblivious_automatic.json`: final clean-source evidence and
  promotion decision.
- `basketball_fused_oblivious_automatic_protocol.md`: frozen automatic-dispatch
  protocol and execution bindings.
- `basketball_fused_oblivious_confirmation.json`: private-candidate passing
  confirmation.
- `basketball_fused_oblivious.json`: immutable original campaign whose
  millisecond-scale prediction ratio failed.
