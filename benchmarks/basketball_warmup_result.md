# Basketball fresh-worker warmup result

## Decision

**Ship the explicit warmup API and opt-in environment dispatch.** The frozen
six-block basketball campaign passed every correctness, isolation, stability,
and timing gate. This authorizes `darkofit.warmup()`,
`DARKOFIT_WARMUP=1`, and `DARKOFIT_WARMUP=background`. It does not authorize
hidden import work, a model-default change, or any claim that warmup reduces
the total work of a single blocking cold start.

The source-bound artifact is
[`basketball_warmup.json`](basketball_warmup.json), produced from clean,
pushed `main` at `705b3677eb33b6b94d469f78616843a8fb686676`. The frozen
protocol SHA-256 is
`06a965820b237ce56b31465c002deb3b3eb230a4268e434c7957adeb7705764d`,
and the executed runner SHA-256 is
`e5060c6981e80951043f56007bb9934a84619ceaa2c07d93cd64306158ca1cd5`.
CTR23 was not used.

## Correctness and isolation

All 12 fresh processes used unique, initially empty Numba cache directories
and imported with `DARKOFIT_WARMUP=0`. Ordinary imports compiled no kernels.
Every control and candidate produced the same timing-free behavior
fingerprint and array-exact predictions for:

- creator fold 0;
- the complete held-team view; and
- the corrected 585-row cold-player view.

Every fit retained 1,000 CatBoost-mode trees, resolved learning rate
`0.052312`, and stopped at the iteration limit. Warmup restored the caller's
18-thread setting, produced no warning or output, and did not cause any
candidate fit to add a new compiled cache file.

## Timing result

The campaign retained all observations from six reciprocal,
position-balanced blocks.

| Metric | Control median | Warmup median | Candidate / control |
| --- | ---: | ---: | ---: |
| First fit | 3.1236 s | 1.6067 s | 0.5144x |
| First prediction | 99.67 ms | 3.82 ms | 0.0384x |
| Explicit warmup | — | 4.7186 s | — |

The explicit warmup reduced measured first-fit time by 48.6% and first
prediction time by 96.2%. Its maximum duration was 5.0503 seconds against the
15-second limit.

| Stability gate | Observed | Limit | Result |
| --- | ---: | ---: | --- |
| Control fit IQR / median | 0.0667 | 0.25 | pass |
| Warmed fit IQR / median | 0.0326 | 0.25 | pass |
| Paired fit-ratio IQR / median | 0.0444 | 0.20 | pass |
| Warmed prediction IQR / median | 0.1205 | 0.50 | pass |

## Interpretation

Warmup moves compilation and cache loading to an explicit startup phase; it
does not remove that work. A caller that blocks on warmup and immediately
runs only one basketball-sized fit pays approximately 6.33 seconds for
warmup, fit, and prediction versus 3.22 seconds without warmup. The API is
useful when a worker serves multiple jobs or background warmup can overlap
other startup work. Ordinary imports remain cold by default.

Basketball is the primary and fatal gate because it is fast and reflects the
project's noisy sports-data priority. The result closes this warmup item
without spending TabArena or CTR23 evidence.
