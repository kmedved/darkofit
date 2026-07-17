# Large-n matched-core engine protocol

## Question

On a deterministic large numeric regression workload, does current DarkoFit
fit at least 1.30× faster than ChimeraBoost 0.15.0 at matched tree budgets and
core hyperparameters without material holdout-RMSE regret?

This is an engine/system certification on synthetic data. It changes no model
behavior, parameter, or default and uses no confirmation or lockbox data.

## Profile-selected mechanism

The pre-protocol 200,000-row profile compared DarkoFit's reference, current
fused, and forced sibling-subtraction lanes over 30 rounds. At 18 threads,
forced subtraction/current fused ratios were approximately `1.42×` total fit
and `2.02×` tree build; subtraction helped only at one thread. The indirect
smaller-child gathers and expand/subtract passes are therefore rejected for
the production-thread E2 candidate. No fused-plus-subtraction code will be
written from this evidence.

The retained system combines:

- the already-certified full-row/full-feature fused oblivious-tree lane;
- adaptive `uint8` bins at the frozen 128-bin budget; and
- capped 200,000-row numeric border construction above that size.

The 500k/1M diagnostic produced a DarkoFit/ChimeraBoost 300-round fit ratio of
`0.762×` geometric mean with a worst holdout-RMSE ratio of `1.00085×`. Those
diagnostics select the certification boundary; they are not formal evidence.

## Frozen workload and arms

- Deterministic generator:
  `benchmarks.run_vector_fit_profile._data`, seed `20260717`.
- Train sizes: 500,000 and 1,000,000 rows.
- Holdout: the following 100,000 generated rows at each size.
- Features: 24 continuous numeric columns.
- Target: the generator's scalar RMSE target.
- Both arms: 300 constant-leaf oblivious trees, learning rate 0.1, depth 6,
  L2 1, 128 bins, full rows/features, minimum child weight 1, ordered
  boosting off, early stopping off, random state 4, and 18 threads.
- DarkoFit additionally sets its distinct minimum-child-sample control to 1,
  `tree_mode="catboost"`, and disables diagnostic warnings.
- ChimeraBoost additionally disables product selectors: linear leaves, cross
  features, and categorical combinations.

DarkoFit uses its public 200,000-row border-sample policy; ChimeraBoost uses
its public full-data border construction. This is matched model capacity and
data, not byte-identical preprocessing above 200,000 rows. Holdout quality is
therefore a binding gate.

Each worker performs a 5,000-row, three-round same-arm JIT warmup outside
timing. Three reciprocal blocks use DarkoFit/ChimeraBoost,
ChimeraBoost/DarkoFit, DarkoFit/ChimeraBoost order at both sizes. Source state
is checked between every worker. Workers are fresh processes and bind all
thread-limit environment variables to 18.

## Gates

The current engine earns the narrowly worded claim “at least 1.30× faster
than ChimeraBoost 0.15 on this matched 500k–1M numeric lane” only if:

1. all 12 workers complete with 300 fitted trees, stable within-arm behavior
   fingerprints, and no stderr;
2. DarkoFit's fused lane engages in every DarkoFit worker;
3. DarkoFit/ChimeraBoost holdout-RMSE ratio is at most `1.002` at each size;
4. paired fit and peak-RSS ratios are stable at each size
   (`IQR / median <= 0.10`);
5. the equal-size geometric-mean fit ratio is at most `1 / 1.30`;
6. neither size's fit ratio exceeds `0.85`; and
7. peak-RSS ratio is at most `1.10` at each size.

Prediction time and DarkoFit phase attribution are report-only. Failure closes
this certification; it does not authorize a new optimization or weakened
threshold.
