# Basketball automatic fused-oblivious promotion protocol

## Decision being confirmed

This final basketball gate asks whether enabling the already-confirmed fused
oblivious training lane through DarkoFit's internal default dispatch preserves
the reference implementation exactly. It is not a new kernel or parameter
search.

The private candidate passed the one-shot training confirmation with a 30.9%
median fit-time reduction, 30.3% wall-time reduction, exact serialized models,
and exact creator-fold, held-team, and cold-player outputs. Expanded tests then
proved exact behavior for numeric and categorical RMSE, MAE, Quantile,
callbacks, and early-stop/exact-refit fits. Weighted RMSE and binary
classification remained exact on the nonconstant-Hessian fallback, and one-
and two-thread fits remained on their existing kernels.

## Frozen implementation and comparison

The promoted DarkoFit package subtree is
`5d8d1f0e7c9edffcb1a8e03315f231ec3e30caf4`. Automatic dispatch is restricted
to the same proven lane:

- oblivious scalar trees;
- at least three Numba threads;
- constant Hessian;
- all rows and all features;
- no root-histogram injection, row-parallel buffers, level subtraction, or
  random split noise.

The benchmark reference worker explicitly passes
`fused_oblivious_kernel=false`; the automatic/candidate worker explicitly
passes `true` and records a positive invocation count. A separate integration
test proves that an ordinary public fit with no override engages the automatic
lane. This prevents a fused-versus-fused benchmark from passing silently.

## Basketball gates

Use the unchanged creator ten folds, corrected overlap-exposed team holdout,
and 585-row cold-player subset. Mean and every fold R², every prediction hash,
both player-guardrail outputs, feature importances, fitted metadata, behavior
fingerprints, and serialized model bytes must be identical. Default/reference
must record zero fused invocations and candidate/automatic must record a
positive count.

After per-worker warmup, use the same three reciprocal blocks. Both arms must
have max/min steady wall time at most 1.20. Automatic advances only if median
fit and wall ratios are each at most 0.85, archive bytes are exact, and median
fresh-worker RSS is at most 1.10 times reference. Prediction timing is recorded
but is diagnostic: the training dispatch does not enter prediction and archive
identity proves the same model representation reaches the same predictor.

## Execution and disposition

The runner must fail closed unless it records
`runtime_policy="automatic-training-only"`, uses exactly 18 threads, writes
only `basketball_fused_oblivious_automatic.json`, runs from clean committed
source, matches this protocol's exact SHA-256, and matches the frozen package
subtree above. Artifact publication is atomic and create-only. There is one
run; failure does not authorize another threshold change or repeat.

No CTR23 development or lockbox data is used. A pass authorizes merging this
narrow internal speed path, subject to documentation, packaging checks, and a
clean review. Basketball remains the first fatal gate for later engine work.

The frozen command is:

```bash
PYTHONPATH=. .venv/bin/python \
  benchmarks/run_basketball_fused_oblivious.py \
  --threads 18 \
  --runtime-policy automatic-training-only \
  --output benchmarks/basketball_fused_oblivious_automatic.json
```
