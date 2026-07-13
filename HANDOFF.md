# DarkoFit Historical Branch Handoff

> **Status: archival checkpoint.** The code parent of this branch is commit
> `18dad32`. It is already an exact ancestor of `kmedved/darkofit` `main`.
> Do not merge or replay this branch into current DarkoFit development.

I want to discuss and possibly work on: understanding the preserved
pre-DarkoFit performance history and deciding whether any current DarkoFit
optimization work remains.

## Context

- The canonical project is `kmedved/darkofit` and the package is `darkofit`.
- This branch is named `codex/pre-upstream-merge-20260712`. Its code parent,
  `18dad32`, still uses the earlier `chimeraboost` package name because the
  repository-wide rename happened later on the direct `darkofit/main` line.
- `kmedved/chimeraboost` is the old fork repository. `bbstats/chimeraboost` is
  a separate upstream project. Neither should be used as the base for current
  DarkoFit work.
- This checkpoint preserves 122 local commits that once appeared to be missing
  when compared with `bbstats/chimeraboost`. They include performance kernels,
  flattened prediction, serialization, tuning, native leaf-wise tree work, and
  distributional regression.
- The apparent loss was a repository-identity mistake. A history audit proved
  that `darkofit/main` contains `18dad32` as an exact ancestor and adds seven
  commits on top:

| Commit | Continuation on `darkofit/main` |
| --- | --- |
| `92049d1` | Add linear residual boosting wrapper |
| `dc95e61` | Standardize distributional targets for 0.7.0 |
| `ad3304e` | Fix calibrated SearchCV refit horizon |
| `c00f8e1` | Rename `chimeraboost` to `darkofit` while preserving file history |
| `21c7ec4` | Prepare DarkoFit 0.9.0 defaults and hardening |
| `f2624dc` | Relax cross-backend softmax parity tolerance |
| `b89faad` | Fix empty-child handling in shared split search |

- Git recognized the package rename in `c00f8e1` as mostly 98-100% file
  renames, so blame and file history remain available across the rename.
- The archive branch has this document as its only change after `18dad32`.
  It exists as a named historical marker, not as an integration branch.

## What the earlier audit established

- The archived implementation passed 448 tests, with one skipped test and five
  failures caused only by the optional Optuna dependency being absent.
- Recorded historical wins included 6.6s to 2.0s preprocessing, 2-3x flattened
  batch prediction, 4-12x loss-kernel improvements, and substantial histogram
  reuse gains. Treat these as historical evidence, not current-main proof.
- A controlled comparison found a stale semantic difference in the archived
  shared-split legality rule: empty children were rejected under
  `min_child_weight`, causing RMSE to move from 12.78 to 16.67 in that lane.
- Current `darkofit/main` commit `b89faad` fixes that exact empty-child issue.
  Preserve the current behavior; never restore the archived rule.
- Several early optimization concepts were also independently present in the
  separate bbstats project. Commit-title similarity is not evidence that the
  projects or implementations should be merged.

## Before doing any implementation

- Find the canonical `kmedved/darkofit` repository and read its current local
  agent and repository instructions.
- Fetch live remote state and verify the ancestry rather than trusting this
  document indefinitely. The expected merge base between this branch's code
  parent and current `main` is `18dad32`.
- Inspect current DarkoFit code, tests, benchmarks, recent commits, and CI before
  deciding that an archived optimization is still missing.
- Decide independently whether the proposed work is useful, stale, already
  solved, over-scoped, or better implemented another way.
- Call out benchmark confounders, behavior changes, unsupported model families,
  and compatibility risks before editing hot paths.

## Task

- Use current `darkofit/main` for all new work.
- Use this branch only for historical comparison, `git show`, blame, or
  performance archaeology.
- Do not merge this branch, rebase it onto `main`, or cherry-pick the 122-commit
  stack wholesale. The stack is already in DarkoFit history and later commits
  contain necessary fixes and the package rename.
- If investigating performance, establish a fresh current-main baseline and
  port only a demonstrably absent concept in a small, independently reviewable
  change.
- Keep flattened prediction, training kernels, serialization, and
  distributional modeling as separate review boundaries.

## Validation

- Require behavior or prediction parity for performance-only work, with explicit
  fallback tests for unsupported model and tree families.
- Run focused tests after each logical change and the full current DarkoFit suite
  before handoff.
- Use repeated warm-cache benchmarks, retain raw timings, report medians and
  environment details, and include a quality metric so speed cannot hide a
  modeling regression.
- Re-check serialization compatibility whenever tree representation or fitted
  state changes.

## Output

- Start with independent findings and a recommendation.
- Clearly distinguish current measurements from historical results recorded on
  this branch.
- If code is edited, keep it scoped and report the exact tests and benchmarks
  run.
- Do not push, merge, open or close pull requests or issues, label anything, or
  post public comments unless explicitly authorized.
