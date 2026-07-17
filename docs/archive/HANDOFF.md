# DarkoFit TabArena Handoff

> **Status: closed on 2026-07-12.** This handoff is retained as historical
> investigation context. Ordered boosting policy was corrected, and a later
> 13-dataset run found the remaining dominant issue: shared symmetric splits
> incorrectly treated an already-pure leaf's empty child as a veto. The fix and
> final learning-rate decision are recorded in
> `benchmarks/tabarena_regression_default_check.md`. The automatic learning-rate
> default was retained.

I want to discuss and possibly work on: fixing DarkoFit's weak numeric-regression default exposed by the TabArena-Lite smoke, then validating the fix broadly enough to decide whether it belongs in the TabArena adapter or DarkoFit's public defaults.

## Context

- This is the DarkoFit project. It began as an early fork, but it is now a distinct project. The bbstats ChimeraBoost project and the ChimeraBoost entry already on TabArena are unrelated and must never be described as this project's prior submission or identity.
- The local repository is in the middle of a broad rename from `chimeraboost` to `darkofit` plus substantial unrelated development. The worktree is intentionally very dirty. Preserve all existing user changes, inspect overlaps carefully, and do not reset, delete, or casually reformat unrelated files.
- Project metadata was updated to identify the package as `darkofit` and point its URLs at `kmedved/darkofit`. Re-check the live repository state because these edits may still be uncommitted.
- A completed local TabArena integration added the `benchmarks.tabarena_adapter` module with `DarkoFitModel` and the `benchmarks.run_tabarena_smoke` module. The runner targets the official Lite quickstart datasets `blood-transfusion-service-center`, `QSAR_fish_toxicity`, and `anneal`.
- Run the smoke with `python -m benchmarks.run_tabarena_smoke`. It is resumable and defaults to one random HPO configuration in addition to the default. Use `--n-configs 0` for the default-only integration check.
- The final local leaderboard export deliberately marks DarkoFit as unverified. Do not present this as an official TabArena submission.
- The three-dataset smoke completed successfully for binary classification, multiclass classification, and regression, with zero failed or imputed DarkoFit tasks.
- Local aggregate results were: default Elo 872 / average rank 68.00; tuned Elo 885 / average rank 67.33; tuned-plus-ensembled Elo 987 / average rank 61.50. With only three datasets, the aggregate ranking and confidence interval are diagnostic, not statistically meaningful.
- The default QSAR regression result was RMSE `1.0951381254510744`, versus `0.9199405100682811` for the unrelated ChimeraBoost 0.13.0 default. The latter was reproduced exactly with its official TabArena adapter and package version, so the comparison is apples-to-apples for this split.
- QSAR has 907 rows, six entirely numeric features, and no categorical features. TabArena's outer smoke split has 604 training rows and 303 test rows; the AutoGluon job uses eight inner bagging folds.

## Established diagnostic evidence

- DarkoFit's adapter leaves `ordered_boosting` unspecified. DarkoFit resolves `ordered_boosting="auto"` to true for CatBoost/depthwise modes, including entirely numeric regression.
- The relevant code anchors are `DarkoFitModel._set_default_params`, `_BaseBooster._resolve_ordered_boosting`, and the scalar boosting loop's `ordered_leaf_update_inplace` call.
- DarkoFit's implementation keeps ordinary Newton leaf values for inference but uses leave-one-out leaf updates only for the evolving training predictions. On small numeric QSAR data there is no categorical target-statistic leakage to offset, and this training/inference discrepancy harms learned structure.
- Prediction diagnostics ruled out simple output bias or scale calibration: optimal affine recalibration barely improved RMSE. The current default's test prediction correlation was `0.689162`; the one sampled HPO configuration reached `0.776728`.
- Raising the DarkoFit iteration cap from 1,000 to 10,000 did not help: RMSE became `1.100521`. The automatic learning rate shrinks as the cap grows, yielding roughly the same effective horizon. Early stopping was active and individual fold best iterations ranged widely, so a broken stopping hook was not the cause.
- Focused fold-0 ablations produced:

| Configuration | QSAR RMSE |
| --- | ---: |
| Current DarkoFit default | 1.095138 |
| 10,000-round cap | 1.100521 |
| Learning rate 0.1 only | 1.073297 |
| L2=1 only | 1.150108 |
| Minimum child samples=1 only | 1.095138 |
| `max_bins=128` only | 0.968051 |
| `ordered_boosting=False` only | 0.932008 |
| Ordered off + 128 bins | 0.925248 |
| Ordered off + learning rate 0.1 | 0.935172 |
| Ordered off + 128 bins + learning rate 0.1 | 0.922775 |
| ChimeraBoost-like settings including L2=1 and min child samples=1 | 0.926303 |
| Unrelated ChimeraBoost 0.13.0 default | 0.919941 |

- Therefore disabling ordered boosting alone closed about 93% of the original gap. Coarser 128-bin quantization was the secondary improvement. Lower L2 and a smaller leaf-row constraint were not fixes.
- A five-outer-split QSAR check compared current DarkoFit, the proposed numeric-regression policy (`ordered_boosting=False`, `max_bins=128`, `learning_rate=0.1`), and unrelated ChimeraBoost 0.13.0:

| Model/policy | Mean RMSE over five splits |
| --- | ---: |
| Unrelated ChimeraBoost 0.13.0 default | 0.871403 |
| Proposed DarkoFit numeric-regression policy | 0.878624 |
| Current DarkoFit default | 0.921193 |

- The proposed policy beat current DarkoFit on all five splits. The original smoke split was unusually punitive, but it exposed a real default-policy regression.
- The one random TabArena HPO configuration sampled `ordered_boosting=False` and achieved test RMSE `0.947278`, but its validation RMSE was slightly worse than the default, so the one-config HPO selector retained the default. One HPO sample is not informative about full tuning potential.
- The adapter honors TabArena's CPU allocation, validation set, and sample weights. DarkoFit still lacks a wall-clock callback, so the adapter cannot strictly enforce TabArena's per-fit `time_limit`. This remains a blocker for a submission-grade full benchmark.

## Before doing any implementation

- Find the correct DarkoFit repository from the current directory, a parent directory, or the usual workspace.
- Read all local agent/repository instructions.
- Inspect the current worktree, relevant modules, tests, benchmark notes, recent commits, and any live GitHub/CI state. Assume the handoff may be stale after the workspace restart.
- Independently reproduce or audit the key evidence rather than accepting this handoff as the final technical judgment.
- Decide whether this remains a real problem and whether the proposed direction is sound, already solved, over-scoped, or better addressed with a narrower policy.
- Specifically assess whether ordered boosting itself is incorrect, merely over-applied by the `auto` policy, or appropriate only when target-derived categorical statistics are present.
- Call out hidden classification, categorical-regression, serialization, reproducibility, and backward-compatibility risks before editing defaults.

## Task

- If the review supports a change, determine the safest scope:
  - Immediate benchmark-only option: give `DarkoFitModel` regression-specific defaults such as `ordered_boosting=False`, `max_bins=128`, and possibly `learning_rate=0.1`.
  - Product-level option: make `ordered_boosting="auto"` task/data-aware, likely disabling it for purely numeric RMSE regression while preserving explicit `True` and behavior supported by categorical evidence.
- Do not make a global-default change based solely on QSAR. Run a broader representative regression matrix first, including numeric and categorical datasets, small and larger row counts, and weighted regression if practical.
- Preserve explicit user settings. Any `auto` policy change must remain observable in fitted metadata and covered by tests.
- Keep the TabArena adapter optional: importing DarkoFit must not require AutoGluon or TabArena.
- Do not conflate DarkoFit with the unrelated ChimeraBoost project in code, docs, metadata, or result interpretation.
- Non-goals for this continuation: an official leaderboard submission, a full benchmark campaign without first addressing wall-clock enforcement, or unrelated cleanup of the dirty rename worktree.

## Validation

- Add focused unit tests for the resolved ordered-boosting policy and explicit override behavior.
- Prove prediction/serialization behavior remains stable for explicit configurations.
- Re-run the three-dataset TabArena smoke and verify all six default-plus-one-HPO jobs succeed without imputation.
- Run the broader regression matrix and report per-dataset deltas, not only an aggregate mean. Include current default, proposed policy, and strong CPU baselines where available.
- Check classification and categorical-regression guardrails before recommending a product default.
- Run relevant project tests, Python compilation/static checks, and diff checks. If repository review tooling is available, use it as a closeout check.
- Keep local leaderboard artifacts labeled unverified.

## Output

- Start with independent review findings and a recommendation about adapter-only versus product-level scope.
- Then provide the proposed plan or patch summary and the evidence that supports it.
- If code is edited, keep the changes scoped, identify every affected public behavior, and report exact tests and benchmark commands run.
- Clearly separate established measurements from inference.
- Do not commit, push, merge, open or close pull requests/issues, label anything, or post public comments unless explicitly authorized.
