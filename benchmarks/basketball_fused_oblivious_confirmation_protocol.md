# Basketball fused-oblivious training-only confirmation protocol

## Purpose and prior result

This is one predeclared successor confirmation for the private fused-oblivious
training kernel. The original frozen campaign at commit `4d6832f` remains a
formal `advance_none`: all exactness, fit-time, wall-time, stability, archive,
and memory gates passed, but a prediction-time ratio failed. The original
artifact and result are immutable and continue to report that failure.

The observed prediction difference was 1.10 ms summed across eleven models.
The two arms had byte-identical archives and behavior fingerprints, and the
private switch changes only tree construction. A fresh-process prediction
ratio therefore does not identify candidate-caused work and is not a sound
fatal gate for this isolated training change.

## Frozen candidate and data

The candidate implementation, eligibility rules, exactness requirements, and
18-thread resource allocation are unchanged from
`basketball_fused_oblivious_protocol.md`. This confirmation again uses only
the fast basketball panel:

- the creator's unchanged ten unshuffled folds;
- the corrected overlap-exposed held-team view; and
- the 585-row cold-player subset.

No hyperparameter, split, representation, seed, quality selector, CTR23
development coordinate, or lockbox input may change. This is a measurement
policy correction, not another candidate search.

## Behavior and engagement gates

Default and fused arms must again match exactly on mean and every fold R²,
prediction hashes, held-team and cold-player predictions and scores, feature
importances, fitted metadata, behavior fingerprints, and serialized model
bytes. Default must record zero fused invocations and the candidate must
record a positive count. Any mismatch stops the campaign before timing
confirmation.

Prediction timing remains recorded for operational visibility. It is not a
decision gate because the switch does not enter prediction and archive
identity proves that both arms hand the same model representation to the same
prediction implementation.

## Runtime and resource gates

Use the same three reciprocal fresh-worker blocks and per-worker warmup as the
original campaign. Both arms must have max/min steady wall time at most 1.20.
The candidate advances only if:

- median summed fit time is at most 0.85 times default;
- median steady wall time is at most 0.85 times default;
- serialized model bytes are exactly equal; and
- median fresh-worker peak RSS is at most 1.10 times default.

The runner must record `runtime_policy="training-only"` and
`prediction_timing_is_decision_gate=false`. Missing measurements fail closed.
There is exactly one confirmation run; a failure does not authorize another
threshold change or repeat.

The training-only policy must also fail closed unless all frozen execution
bindings hold: exactly 18 threads, the confirmation-specific output path,
clean committed source, this protocol's exact SHA-256, and DarkoFit package
subtree `033ff90c60b01a30281ffb3b88729f30571ab246`. The original output path is
reserved for the immutable failed campaign and cannot be selected by this
policy.

## Advance path

A pass advances the private kernel only to expanded behavior tests covering
weighted RMSE, categorical RMSE, alternate scalar losses, callbacks, early
stopping/refit, and supported thread-count fallbacks. It does not make the
kernel automatic and does not authorize a broad accuracy or lockbox campaign.

The frozen command is:

```bash
PYTHONPATH=. .venv/bin/python \
  benchmarks/run_basketball_fused_oblivious.py \
  --threads 18 \
  --runtime-policy training-only \
  --output benchmarks/basketball_fused_oblivious_confirmation.json
```
