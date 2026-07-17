# Fused subset oblivious-tree gate

## Question

Can the proven fused CatBoost/oblivious histogram-and-split kernel cover
column sampling, row sampling, and their combination while preserving exact
model behavior and without slowing the already-shipped full-row/full-feature
lane?

This is E1 engine work. It changes no public estimator parameter, automatic
modeling policy, quality claim, or lockbox state. Basketball supplies the
small, latency-sensitive workload; exact reference/candidate identity is the
binding quality boundary.

## Frozen semantic boundary

- Pre-mechanism DarkoFit source: clean `main` at `c6cc551`.
- Reference: the existing selected-feature, selected-row, or combined
  histogram builder followed by the unchanged `_best_split`.
- Candidate: extend the existing fused unit- and variable-Hessian kernels with
  runtime-selected feature and row iteration.
- Feature order, row accumulation order, hessian handling, split legality,
  `min_child_weight`, gain arithmetic, tie-breaking, leaf routing, and final
  leaf values must remain unchanged.
- Unselected histogram and split scratch regions must retain the same contents
  as the reference path.
- The full-row/full-feature path remains eligible and is a binding
  no-regression control.
- Eligibility remains restricted to at least three threads, no row-parallel
  buffers, no histogram subtraction, no precomputed root histogram, and
  `random_strength=0`.
- One- and two-thread, subtraction, root-copy, row-parallel, and randomized
  split lanes remain on their existing kernels.

`min_child_samples` is not part of this mechanism. DarkoFit intentionally
applies that parameter only to leaf-wise and hybrid builders; adding count
semantics to oblivious trees would be a modeling change rather than a fused
kernel optimization.

## Exactness proofs

Focused kernel tests cover unit and variable Hessians for:

1. full rows and features;
2. selected features only;
3. selected rows only; and
4. selected rows and features together.

For every case, reference and candidate must have identical active and
inactive histogram buffers, split tuples, completed tree state, training
state, predictions, feature importances, and canonical serialized model
payloads. Integration tests must prove positive candidate engagement for
`colsample < 1`, `subsample < 1`, and their combination, while ineligible
lanes prove zero engagement.

## Basketball-scale performance workload

- Pinned creator basketball training matrix: 5,241 rows × 15 numeric features.
- Exactly 18 threads, 600 CatBoost-mode depth-six rounds, learning rate 0.1,
  L2 3, 128 bins, ordered boosting off, no validation or early stopping, and
  phase timing enabled.
- Hessian cases:
  1. unweighted RMSE (unit Hessian); and
  2. RMSE with deterministic positive nonuniform weights (variable Hessian).
- Sampling lanes:
  1. full: `subsample=1.0`, `colsample=1.0`;
  2. features: `subsample=1.0`, `colsample=2/3`;
  3. rows: `subsample=0.8`, `colsample=1.0`;
  4. both: `subsample=0.8`, `colsample=2/3`.
- Every fresh worker performs a same-lane 200-row, three-round JIT warmup
  outside timing.
- Three reciprocal blocks use reference/candidate, candidate/reference,
  reference/candidate order for every Hessian/sampling combination.

The artifact records fit and tree-build duration, predictions, canonical model
payload, engagement count, fitted metadata, peak RSS, data fingerprint, source
fingerprints, and process/thread environment. Timing telemetry is excluded
from the canonical model hash and retained separately.

## Gates

The candidate is retained only if:

1. candidate engagement is positive and reference engagement is zero in all
   eight workload cells;
2. prediction and canonical serialized-model hashes match within every cell
   and remain stable across blocks;
3. all fit and tree-build paired ratios are stable
   (`IQR / median <= 0.15`);
4. no cell regresses by more than 2% in fit or tree-build time;
5. geometric-mean subset-lane candidate/reference fit ratio is `<= 0.95`;
6. geometric-mean subset-lane tree-build ratio is `<= 0.90`;
7. the full-lane geometric-mean fit and tree-build ratios remain `<= 0.95`
   and `<= 0.90`, respectively; and
8. every paired peak-RSS ratio is stable and has median `<= 1.05`.

Failure leaves the current full-row/full-feature dispatch in place and closes
this subset expansion as shaped. Passing ships only the internal dispatch
extension and authorizes updating E1's ledger; it makes no external speed or
quality claim.
