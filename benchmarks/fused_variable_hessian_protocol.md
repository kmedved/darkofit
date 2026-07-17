# Fused variable-Hessian oblivious-tree gate

## Question

Can the proven fused oblivious histogram/split lane extend to binary Logloss
and weighted RMSE with byte-identical models and a material fit-time gain?

This is E1 engine work. It changes no public parameter, model policy, dataset
quality result, or lockbox state.

## Frozen mechanism

- Pre-mechanism DarkoFit source: clean `main` at `7097e7a`.
- Candidate: in the existing full-row/full-feature CatBoost-mode fused lane,
  accumulate each row's actual hessian before the unchanged shared split scan.
- Reference: the existing separate variable-Hessian histogram build followed
  by `_best_split`.
- Unit-Hessian dispatch is unchanged.
- Both lanes retain the same feature/row accumulation order, legality,
  min-child-weight behavior, tie-breaking, leaf routing, and final leaf-value
  code.
- Eligibility remains restricted to at least three threads, no selected rows
  or feature indices, no row-parallel buffers, no histogram subtraction, no
  precomputed root histogram, and `random_strength=0`.

Focused tests require exact histogram buffers and split tuples across L2,
minimum-child-weight, zero-hessian, and feature-mask cases; complete tree state,
public predictions, importances, and serialized archives must match. All
ineligible lanes must prove non-engagement.

## Performance workload

- Deterministic 50,000-row × 24-feature numeric matrix from the vector-path
  profile, exactly 18 threads.
- 300 CatBoost-mode depth-6 rounds, learning rate 0.1, L2 1, 128 bins,
  `min_child_samples=20`, full rows/features, ordered boosting off, no
  validation or early stopping, and phase timing enabled.
- Cases:
  1. binary Logloss; and
  2. RMSE with deterministic positive nonuniform sample weights.
- Each fresh worker performs a same-lane 5,000-row, three-round JIT warmup
  outside timing.
- Three reciprocal blocks use reference/candidate, candidate/reference,
  reference/candidate order for each case.

The artifact records total fit, tree-build phase, predictions, model archive,
engagement count, fitted metadata, behavior fingerprint, and peak RSS.
Canonical model-state exactness hashes every sorted `allow_pickle=False` NPZ
payload key, dtype, shape, and array byte after replacing only
`header.timing` with `null`; those phase durations are observational telemetry
and necessarily differ between the timed arms. The raw ZIP-file hash remains
diagnostic because both timing values and container timestamps vary across
fresh processes.

## Gates

The shared candidate is retained only if:

1. candidate engagement is positive and reference engagement is zero;
2. prediction and canonical serialized-model-state hashes match reference in
   both cases and remain stable across blocks;
3. fit and tree-build paired ratios are stable in both cases
   (`IQR/median <=0.10`);
4. neither case regresses by more than 2% in fit or tree-build time;
5. geometric-mean candidate/reference fit ratio is `<=0.95`;
6. geometric-mean tree-build ratio is `<=0.90`; and
7. paired peak-RSS ratio is stable and `<=1.05` in both cases.

Failure restores the reference dispatch while preserving the exactness tests
as research. Passing ships the internal lane but makes no external speed or
quality claim.
