# Basketball fused oblivious kernel: exact and faster, but first gate fails

## Decision

The frozen campaign recommendation is `advance_none`. The fused training
kernel preserved the current basketball model exactly and reduced median fit
time by 28.4%, but it failed the predeclared prediction-time ratio. This result
is preserved as a failed gate; it is not being relabeled as a pass.

The failure does not identify a prediction-code regression. Default and fused
models had byte-identical archives, identical behavior fingerprints, and used
the same untouched prediction implementation. The measured difference was
1.10 ms summed across all eleven fitted models, or 6.0% of a roughly 18 ms
measurement. A separately frozen confirmation may replace that noncausal
ratio with exact prediction-path evidence, but this artifact remains failed.

## Exact behavior

All predeclared behavior gates passed:

- mean 10-fold R² was exactly `0.5267495183883605` in both arms;
- every fold score, prediction hash, and test-index payload was identical;
- overlap-exposed team and cold-player guardrail predictions and scores were
  identical;
- feature importances, fitted metadata, and serialized model bytes were
  identical;
- behavior fingerprints matched within and across all six fresh workers; and
- the default recorded zero fused invocations while the candidate recorded
  66,000 fused tree-level invocations in its canonical worker.

## Runtime and memory

| Measurement | Default median | Fused median | Fused / default | Gate |
|---|---:|---:|---:|---:|
| Steady wall time | 29.773s | 21.502s | **0.722** | pass |
| Summed fit time | 29.250s | 20.951s | **0.716** | pass |
| Summed prediction time | 18.325ms | 19.426ms | 1.060 | **fail** |
| Fresh-worker peak RSS | 256.3MB | 256.7MB | 1.002 | pass |

Wall-time stability passed in both arms. Default max/min was 1.057 and fused
max/min was 1.145, below the frozen 1.20 limit. The fused steady times were
20.196s, 23.134s, and 21.502s; default times were 29.773s, 30.721s, and
29.065s.

## Test and provenance status

The evidence was produced from clean committed source at
`4d6832fddbc2f14884b9eac4246e62bef0da754b` with 18 threads per fit. Strict
prediction goldens and the fused/reference oracle set passed 30 tests. The
complete suite produced 1,428 passes and 23 skips; its only failure was the
intentional historical-evidence sentinel rejecting reuse of the prior
TabArena package subtree after this core-code change. No CTR23 development or
lockbox coordinates were used.

The implementation remains private and default-off at this decision point.
The result does not authorize automatic dispatch or broader performance
claims.

## Next step

Preserve this result, then freeze a basketball-only confirmation whose gates
match the candidate's causal scope: exact predictions, exact archives, exact
fit metadata, stable reciprocal fit/wall timing, and bounded memory. Because
the private switch changes training only and produces byte-identical models,
do not use a fresh-process ratio of millisecond-scale calls to decide whether
the training kernel advances to expanded behavior tests.

## Artifacts

- `basketball_fused_oblivious.json`: clean-source canonical results, all
  predictions and hashes, fitted metadata, invocation telemetry, reciprocal
  timing blocks, memory observations, and frozen gate decisions.
- `basketball_fused_oblivious_protocol.md`: the original frozen protocol,
  including the prediction ratio that failed.
- `run_basketball_fused_oblivious.py`: the clean-source runner used for the
  campaign.
- `test_fused_oblivious_kernel.py` and
  `test_basketball_fused_oblivious.py`: exactness, fallback, engagement, and
  harness tests.
