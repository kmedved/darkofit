# Basketball small-row serial leaf-descent protocol

## Decision being tested

This campaign asks whether DarkoFit should update oblivious-tree leaf IDs with
a serial in-place loop below 32,768 training rows instead of launching the
parallel kernel at every tree level. It is a training-engine dispatch change,
not a model or quality candidate. Every split, leaf value, prediction, fitted
metadata field, and serialized byte must remain unchanged.

Basketball is again the first fatal gate. The promoted fused histogram/split
kernel reduced median basketball fit time from 28.93 seconds to 19.31 seconds,
but a warmed first-fold phase profile still placed 1.906 of 1.923 fit seconds
inside tree construction.

## Frozen diagnostic and candidate

At DarkoFit commit `3f559b45c0b3d5938201316f7c01e574c18f91ca`,
the six fused histogram/split levels of a representative basketball tree took
about 0.944 ms in total. The six parallel leaf-ID updates took about 0.874 ms.
At 4,248 training rows, one parallel update took about 0.146 ms while the
existing serial level-split loop, used as a conservative proxy, took about
0.0033 ms including an extra leaf-local lookup.

A separate size/thread diagnostic found the serial proxy faster at 32,768
rows for 1, 2, 4, 8, and 18 threads; its serial/parallel ratios ranged from
0.11 to 0.19 there. The 32,768 cutoff is therefore conservative and matches
the independently synced ChimeraBoost 0.15.0 dispatch.

The candidate adds an exact serial twin of `_update_leaves_with_split` and a
private internal router:

- `n_rows < 32_768`: serial in-place update;
- `n_rows >= 32_768`: the existing parallel kernel unchanged.

No public parameter, estimator default, saved-model field, or prediction path
changes. The router may serve the oblivious builder and hybrid shared trunk
because both currently call the same update helper; all other tree mechanics
remain untouched. The parallel implementation remains the equality oracle.

The serial-twin design and cutoff are adapted from Apache-2.0 ChimeraBoost
commit `a04430657fb82c806ee2a039506c99944a27accc`; `NOTICE` must say so before
promotion.

## Behavior and dispatch gates

Before formal timing:

- serial and parallel kernels must be array-exact across zero rows, threshold
  ties, varied row counts, C- and Fortran-ordered binned matrices, and both
  `int64` and supported `uint32` leaf IDs;
- the router must prove serial engagement below the cutoff and parallel
  engagement at and above it without relying only on requested mode;
- oblivious scalar RMSE, weighted RMSE, categorical RMSE, MAE, Quantile,
  binary classification, callbacks, and early-stop/exact-refit models must
  remain archive-exact against the forced-parallel reference;
- the hybrid shared-trunk path must remain prediction- and archive-exact;
- strict prediction goldens, the readable oblivious oracle, and the complete
  suite must pass.

The clean basketball campaign then runs the unchanged creator ten folds plus
the corrected overlap-exposed held-team view and 585-row cold-player subset.
The forced-parallel reference and automatic serial candidate must match on
all fold and guardrail predictions, scores, feature importances, fitted
metadata, behavior fingerprints, and serialized model bytes. The artifact
must record zero serial calls for the reference and positive serial calls for
the candidate. Any mismatch is fatal and skips timing confirmation.

## Runtime and resource gates

After one complete first-fold warmup per fresh worker, run three reciprocal
blocks:

1. forced parallel, automatic serial;
2. automatic serial, forced parallel; and
3. forced parallel, automatic serial.

Each arm's steady max/min wall-time ratio must be at most 1.20. The candidate
advances only if:

- median summed fit time is at most 0.70 times the reference;
- median steady wall time is at most 0.70 times the reference;
- serialized model bytes are exactly equal; and
- median fresh-worker peak RSS is at most 1.05 times the reference.

Prediction timing is retained as diagnostic only: the candidate changes no
prediction code and archive identity proves the same model reaches the same
predictor. The small-kernel microbenchmark must also show median serial time
at most 0.10 times parallel time at the basketball training-row count. At and
above the cutoff, the router must call the existing parallel function itself,
so there is no separate large-row implementation to benchmark.

## Advance path

A pass authorizes the internal router and attribution update. It does not
authorize broader kernel deletion, a quality-policy change, or any CTR23
development or lockbox use. A failure stops this candidate; thresholds and
gates are not revised after observing the formal campaign.
