# Basketball fused-oblivious training confirmation: advance to expanded tests

## Decision

The one-shot training-only confirmation passed every frozen gate and advances
the private fused-oblivious kernel to expanded behavior testing. It does not
yet make the kernel automatic.

The original campaign remains a formal `advance_none` because its predeclared
prediction-time ratio failed. This successor used a separate output and a
reviewed, fail-closed policy bound to 18 threads, clean source, the exact
candidate subtree, and the frozen confirmation protocol hash.

## Results

| Measurement | Default median | Fused median | Fused / default |
|---|---:|---:|---:|
| Steady wall time | 27.964s | **19.479s** | **0.697** |
| Summed fit time | 27.435s | **18.960s** | **0.691** |
| Diagnostic prediction time | 18.701ms | 18.642ms | 0.997 |
| Fresh-worker peak RSS | 257.0MB | 256.2MB | 0.997 |

The fused kernel reduced median fit time by 30.9% and steady wall time by
30.3%. Default wall-time max/min was 1.015 and fused max/min was 1.012, well
inside the frozen 1.20 stability limit.

## Exactness and sports guardrails

Both arms produced mean 10-fold R² `0.5267495183883605`. Every fold score,
prediction hash, held-team score, cold-player score, feature importance,
fitted metadata payload, serialized model, and cross-worker behavior
fingerprint was identical. The default recorded zero fused invocations and
the candidate recorded 66,000 invocations in its canonical worker.

## Provenance and scope

The evidence used clean committed source at
`e01eb64488b845aff482bb4859fa56c69e5567a2`, 18 threads, and only the frozen
basketball folds and corrected player guardrails. No CTR23 development or
lockbox data was used. The confirmation policy recorded prediction timing as
diagnostic because this private switch changes training only and both arms
produce byte-identical model archives.

The next authorized step is expanded exactness testing for weighted and
categorical RMSE, MAE/Quantile, callbacks, early stopping/refit, classification
fallbacks, and supported thread counts. Automatic dispatch requires those
tests to pass and then requires basketball to remain the first fatal gate.

## Artifacts

- `basketball_fused_oblivious_confirmation.json`: clean-source predictions,
  hashes, fitted metadata, invocation telemetry, reciprocal timing, memory,
  and gate decisions.
- `basketball_fused_oblivious_confirmation_protocol.md`: frozen successor
  rules and exact execution binding.
- `basketball_fused_oblivious.json` and
  `basketball_fused_oblivious_result.md`: immutable original failed gate.
