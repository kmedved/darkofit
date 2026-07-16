# Best-of-both Phase 0: safety boundary established

## Decision

Phase 0 may advance. This change adds behavior tripwires and a forward-only
basketball harness boundary without changing any estimator, production
default, historical benchmark runner, prediction, or lockbox state.

The reviewed roadmap is `BEST_OF_BOTH_PLAN.md`. Its execution boundary is
stricter than the original draft: basketball is the first fatal screen for
every candidate, but never the sole evidence for a universal default.

## Proposal audit

The local ChimeraBoost checkout, `origin/main`, and `upstream/main` all resolve
to `29602d3452b1754042006ad2b14bca320c94b4b7`. DarkoFit Phase 0 starts from
`3295f70c231d4f7947e13a13ad77e3f2c19b3fe0`. Both repositories carry the
same Apache-2.0 `LICENSE` bytes (SHA-256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`).
No ChimeraBoost source is copied in Phase 0; substantial literal future ports
must add attribution in `NOTICE`.

Corrections applied to the draft include:

- ChimeraBoost has 4,029 Python package lines and 27 regressor constructor
  parameters; the six principal modules total 3,755 lines.
- DarkoFit has 22,057 Python package lines and 58 regressor constructor
  parameters.
- Safe ordinal is strong mechanism evidence but formally failed its frozen
  causal inference-time gate.
- The 243 unused CTR23 confirmation coordinates are development-only after
  the minimal confirmation exposed task identities and neighboring outcomes.
- Neither the fixed-LR nor current-auto-LR early-stop/exact-refit basketball
  result supports an early-stopping default. The latter was 11.7% faster but
  regressed mean and overlap-exposed holdout R².

## Prediction goldens

`prediction_goldens.py` freezes 12 deterministic single-thread cases:

- numeric RMSE regression in all four current tree modes;
- categorical ordered-boosting regression;
- categorical binary classification;
- numeric multiclass classification; and
- Gaussian, LogNormal, Student-t, Poisson, and Negative-Binomial regression.

The distributional cases cover `predict`, raw scores, every returned
distribution parameter, variance, central intervals, and seeded sampling.
Every public output must be byte-repeatable within a fit. The artifact records
both a twelve-decimal portable digest and the exact float64 byte digest.
Normal CI enforces the portable digest; controlled optimization lanes set
`DARKOFIT_STRICT_GOLDENS=1` to require exact bytes.

The exact digests reproduced across the local Python 3.11, 3.12, and 3.13
environments before publication.

## Kernel oracles

The existing suite already contains exact or reference comparisons for
binning, leafwise brute-force growth, serial/parallel split scans, histogram
layouts, selected rows and columns, constant Hessians, histogram subtraction,
multiclass root fusion, and packed-versus-tree-loop prediction.

Phase 0 adds a standalone readable Python oracle for the oblivious tree path,
which is the first proposed engine-consolidation target. It verifies shared
split features and thresholds, leaf routing, gradient/Hessian totals, Newton
values, and predictions across full data, selected columns, selected rows,
and combined row/column selection. Gain values use a 1e-12 absolute tolerance
because the readable row-order accumulation and binned-kernel accumulation
have intentionally different floating-point summation orders.

## Basketball harness boundary

`basketball_harness.py` is forward-only. It centralizes the immutable creator
split cross-check, cold-player view, prediction validation and hashes, fitted
route/stop/refit metadata, phase totals, behavior fingerprints, reciprocal
timing schedule, stability calculation, and clean worker thread environment.

Frozen historical runners and their source attestations are untouched. New
candidate screens must import this boundary rather than copy it.

The next candidate screen must run the unchanged basketball ten folds plus:

- overlap-exposed team-holdout R²;
- cold-player R² on 585 rows from 210 players absent from training;
- prediction and behavior fingerprints;
- resolved learning rate, selected/final tree counts, and stop reasons; and
- three reciprocal clean timing repetitions.

A candidate that regresses the basketball quality guardrails stops before any
larger panel. A survivor earns broader development testing but not automatic
default promotion.
