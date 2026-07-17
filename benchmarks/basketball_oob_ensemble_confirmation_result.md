# Basketball OOB ensemble stable confirmation result

## Decision

**Close the five-member OOB-ensemble attempt.** The candidate reproduced its
promising basketball quality exactly and passed every wall-time and cost gate,
but the frozen confirmation failed its absolute prediction-timing stability
gate. The protocol forbids rerunning, dropping a block, or weakening the
threshold. This result does not authorize a public ensemble API or a default
change.

The source-bound artifact is
[`basketball_oob_ensemble_confirmation.json`](basketball_oob_ensemble_confirmation.json),
produced from clean, pushed `main` at
`4b744c772b21b9f9b49289070e93cf8500d0f5eb`. The frozen protocol SHA-256 is
`a8fd26868471028fb9e652fee6153e2a7ba9f57fe558407f07d890d8917903bf` and
the executed runner SHA-256 is
`200fc8eae375133f52ad004790f6debd22d73e1f9187094ca9ab518c54fa1469`.

## Quality reproduced exactly

All six fresh-process repetitions for each arm produced one identical
timing-free behavior fingerprint. The ten creator-fold predictions, full
held-team predictions, 585-row cold-player predictions, seen-player
predictions, and bootstrap/OOB index plans matched the original screen's
frozen hashes. Fitted member count, validation source, early-stopping reason,
and model metadata invariants were revalidated under the current source-bound
runner.

| Metric | Default | OOB-5 | Delta |
| --- | ---: | ---: | ---: |
| Mean ten-fold R² | 0.526750 | 0.530625 | +0.003876 |
| Improved folds | — | 6 / 10 | passes |
| Held-team R² | 0.531269 | 0.537395 | +0.006126 |
| Cold-player R² | 0.500434 | 0.519783 | +0.019349 |
| Seen-player R² | 0.530247 | 0.532300 | +0.002052 |

Every leave-one-fold-out mean delta remained positive; the minimum was
`+0.001652`. All five preregistered quality gates passed.

## Timing result

The six reciprocal blocks alternated arm order and retained every observation.
Each worker performed one complete first-fold warmup outside timing.

| Gate | Observed | Limit | Result |
| --- | ---: | ---: | --- |
| Default wall IQR / median | 0.1276 | 0.20 | pass |
| OOB-5 wall IQR / median | 0.0149 | 0.20 | pass |
| Paired wall-ratio IQR / median | 0.0720 | 0.15 | pass |
| OOB-5 / default median wall | 2.4136x | 4.0x | pass |
| OOB-5 / default median prediction | 2.2593x | 6.0x | pass |
| Default prediction IQR / median | **0.2350** | 0.20 | **fail** |
| OOB-5 prediction IQR / median | 0.0982 | 0.20 | pass |

Default prediction totals across all ten folds were 34.2–50.5 ms, with a
40.7 ms median. OOB-5 totals were 79.0–114.5 ms, with a 91.9 ms median. The
failed series is tiny in absolute terms and the candidate series itself was
stable, but the decision rule was deliberately frozen before observation.

The fifth block also showed a shared whole-run slowdown: default wall time was
21.79 seconds and OOB-5 was 47.49 seconds. The position-balanced paired ratio
remained stable, confirming that the original campaign's wall-time failure was
not candidate-specific. That does not override the separate absolute
prediction-stability failure here.

## Consequence

No ensemble code enters `darkofit/`, no constructor parameter is added, and no
default changes. The quality evidence remains useful research evidence for
small noisy sports data, but this particular OOB-5 promotion path is closed.
Any future ensemble work must begin as a materially new mechanism with a new
protocol; it may not treat another timing run as a rescue of this decision.
