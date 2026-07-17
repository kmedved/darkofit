# Matched prediction-throughput characterization

## Question

At production-sized batches, where does DarkoFit's remaining public prediction
gap to ChimeraBoost 0.15 come from, and does the current matched lane already
meet the ceiling program's `<=1.30x` target?

This is a characterization and mechanism-development harness. It cannot change
a default or reopen the closed basketball forest-work router.

## Frozen source and models

- DarkoFit source before protocol implementation: `bbdd5be`.
- ChimeraBoost comparator: clean `v0.15.0` / `851ab7f`, equal to both local
  `origin/main` and `upstream/main`.
- Both libraries fit constant-leaf oblivious forests with 1,000 trees, depth 6,
  learning rate 0.1, L2 1, 128 bins, full rows/features, ordered boosting off,
  no early stopping, and random state 4.
- DarkoFit additionally fixes `min_child_samples=1`, CatBoost tree mode, and
  disables diagnostics. ChimeraBoost disables linear leaves, cross features,
  and category combinations.
- Every worker fits two models outside timing:
  1. the creator basketball numeric first-fold training set; and
  2. a deterministic 20,000-row synthetic mixed matrix with six numeric and
     two declared categorical columns.

The learned forests need not be cross-library byte-identical for the mixed
case, but both retain exactly 1,000 depth-6 trees. Within each library, direct
packed-core output must be array-identical to its public prediction.

## Inputs and phases

Each fitted model predicts deterministic repeated-row matrices at:

```text
8,192 / 65,536 / 524,288 / 2,000,000 rows
```

Every case records:

- `cold_public`: the first public call after explicitly clearing the fitted
  forest cache;
- `warm_public`: repeated public calls after cache construction;
- `binning`: direct preprocessing/binning only; and
- `packed_core`: the fitted packed forest on an already-binned matrix.

Matrix construction, hashing, model fitting, imports, and JIT/cache warmup are
outside warm timing. Prediction arrays must be finite with the expected shape.

## Execution and gates

Three fresh-worker blocks use arm orders Darko/Chimera, Chimera/Darko,
Darko/Chimera at exactly 18 threads. Source identity is rechecked between
workers. Stability uses the Darko/Chimera median-time ratio paired inside each
block:

```text
IQR(paired ratio) / median(paired ratio) <= 0.10
```

No per-arm absolute millisecond dispersion can fail the campaign. The current
lane meets the program target only if every numeric and mixed `warm_public`
case is stable and has median paired ratio `<=1.30`. Cold, binning, and core
ratios are diagnostic. Peak RSS is reported as a paired ratio but has no
promotion threshold in this characterization.

The result selects the largest stable component of excess wall time as P2's
first implementation target. Any future mechanism reruns this exact harness
from a new, separately bound protocol; this artifact is never overwritten.
