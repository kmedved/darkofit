# Basketball auto-LR early-stop/refit: do not advance

## Decision

Keep the current default unchanged. Combining the current automatic learning
rate with internal early stopping, best-prefix selection, and exact full-data
refit did not preserve broad quality and did not reach the predeclared 20%
speed threshold.

The run used the frozen creator data, features, seed, and ten unshuffled folds.
It also scored the unchanged alphabetical team holdout under its corrected
**player-overlap-exposed** label and the supplemental 585-row cold-player
subset. No lockbox data was used.

## Results

| Arm | Mean 10-fold R² | Fold wins | Overlap-exposed team R² | Cold-player R² | Median steady wall |
|---|---:|---:|---:|---:|---:|
| Current default | **0.526750** | — | **0.531269** | 0.500434 | 27.178s |
| Auto LR + early stop + exact refit | 0.525114 | 3 / 10 | 0.522336 | **0.511811** | **24.005s** |
| Candidate delta | -0.001635 | — | -0.008933 | +0.011377 | -11.7% |

The candidate lost 7 of 10 folds. All ten leave-one-fold-out mean deltas were
negative, ranging from -0.002559 to -0.000986, so the mean loss is not driven
by one bad fold. It improved the truly unseen-player subset, but this does not
offset its broader fold and overlap-exposed holdout regressions.

## Fitted behavior

- The default resolved learning rate to 0.052312–0.052314, selected no prefix,
  and fit all 1,000 trees on every fold.
- The candidate's selection fits resolved learning rate to
  0.063893–0.063896. The higher value follows the current automatic-LR logic
  on the smaller internal-selection training partition; exact refit preserves
  that fitted rate.
- Selection stopped early on all ten folds. Selected tree counts were
  286–583 (median 418); every exact refit retained exactly the selected count
  and stopped at that requested iteration limit.
- Prediction hashes and all non-timing fitted metadata were identical across
  the three fresh-process repetitions for each arm.

## Timing and profile disposition

The reciprocal timing schedule was stable:

| Arm | Three steady times | Median | Max / min |
|---|---|---:|---:|
| Current default | 26.851s, 27.178s, 29.075s | 27.178s | 1.083 |
| Candidate | 24.005s, 23.933s, 25.324s | 24.005s | 1.058 |

Both arms passed the 1.20 stability bound. The candidate runtime ratio was
0.883, so it failed the required ratio of at most 0.80. It also failed four of
five quality gates: mean score, fold breadth, leave-one-fold-out stability,
and the overlap-exposed team holdout. Only the cold-player no-regression gate
passed.

The protocol required a new basketball-scale tree-kernel profile only if all
quality and timing gates passed. Profiling was therefore intentionally skipped
rather than optimizing a rejected policy. The prior frozen ablation already
located roughly 99% of fit time in tree construction at this data shape.

## Recommendation

Do not promote automatic early stopping and exact refit as a basketball or
sports-data default. Preserve the result as evidence that the mechanism can
help unseen-player generalization, but it needs a different selection policy
before another speed/quality claim. Any follow-up should remain explicit and
must explain why it is expected to preserve seen-player and general fold
quality; this result does not justify more tuning on the frozen folds.

## Artifacts

- `basketball_auto_lr_refit.json` records source attestation, data and fold
  fingerprints, every prediction and hash, fitted metadata, per-fold and
  guardrail scores, all six clean timing runs, and every decision gate.
- `basketball_auto_lr_refit_protocol.md` is the frozen protocol.
- `run_basketball_auto_lr_refit.py` is the clean-source runner.
- `basketball_darkofit_cold_player_guardrail.json` records the player-overlap
  disclosure and rescoring of the prior ablation predictions.
