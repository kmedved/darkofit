# Basketball DarkoFit ablation: no default change

## Decision

Do not advance any tested configuration. The frozen screen reproduced the
current default's mean 10-fold R² exactly, and none of the four candidates met
the predeclared quality, fold-breadth, held-team, and runtime gates.

`linear_residual=True` was the screen winner, but its +0.000402 mean R² gain
closes only 4.2% of the gap to CatBoost. It won 5 of 10 folds, and its mean gain
became negative when any of four favorable folds was omitted. This is useful
opt-in evidence, not a default-policy result.

## Results

| Configuration | Mean 10-fold R² | Δ vs default | Fold wins | Held-team R² | Δ held | Steady wall |
|---|---:|---:|---:|---:|---:|---:|
| Current default | **0.526750** | — | — | 0.531269 | — | 27.72s confirmed |
| A10 numeric | 0.514541 | -0.012209 | 2 / 10 | 0.516310 | -0.014959 | 48.57s¹ |
| A10 numeric, 2,000 rounds | 0.503587 | -0.023163 | 0 / 10 | 0.496402 | -0.034867 | 55.24s |
| A10 + early stop + exact refit | 0.523909 | -0.002840 | 5 / 10 | 0.527851 | -0.003419 | **12.72s** |
| Linear residual | **0.527151** | **+0.000402** | 5 / 10 | **0.539212** | **+0.007943** | 27.88s |

¹ The first two arms ran during a transient slow interval: the instrumented
default took 49.09s in the campaign. A clean isolated rerun immediately after
the campaign reproduced the same prediction score in 27.72s, consistent with
the frozen 28.30s baseline. The A10 arm is rejected on quality and guardrails
regardless, so its contaminated time does not affect the decision.

The external quality targets remain CatBoost at 0.536308 and the current
five-member ChimeraBoost ensemble at 0.540159. The external speed target
remains single-model ChimeraBoost at 9.29s on the same frozen steady protocol.

## Guardrail correction and cold-player supplement

The alphabetical team holdout is not independent at the player level. Of its
767 player identities, 557 (72.6%) also occur in training; those repeated
players account for 1,824 of 2,409 holdout rows (75.7%). `Player` is not a
model feature, so this is not direct feature leakage, but the previous generic
"held-team" label overstated the split's independence. This report now calls
it the **overlap-exposed team holdout**.

A reproducible supplement rescored the already-persisted predictions on the
585 holdout rows belonging to 210 players absent from training. It did not
refit any model or change the frozen creator benchmark.

| Configuration | Overlap-exposed team R² | Cold-player R² | Seen-player R² |
|---|---:|---:|---:|
| Current default | 0.531269 | 0.500434 | 0.530247 |
| A10 numeric | 0.516310 | 0.499691 | 0.510409 |
| A10 numeric, 2,000 rounds | 0.496402 | 0.479899 | 0.490004 |
| A10 + early stop + exact refit | 0.527851 | **0.521139** | 0.519058 |
| Linear residual | **0.539212** | 0.507523 | **0.538647** |

The correction does not reverse the frozen decision. A10 remains rejected;
the early-stop arm improved cold-player R² but regressed the overlap-exposed
team and mean fold scores, while linear residual still lacks fold breadth.
There is no season or date field in the source, so this is an unseen-identity
diagnostic and not a temporal-generalization claim.

## What the fitted metadata says

- The default and linear-residual arms resolved learning rate to
  0.052312–0.052314 and retained all 1,000 trees on every fold.
- The two fixed-LR arms used 0.1 and retained all requested 1,000 or 2,000
  trees. More horizon consistently worsened generalization: the 2,000-round
  arm lost all 10 folds and reduced the held-team score by 0.034867.
- The early-stop arm stopped selection on all 10 folds, then performed an
  exact full-data refit of 103–290 trees (median 207.5). It is 2.2× faster than
  the confirmed current default, but its quality and held-team regressions
  disqualify it.
- Linear residual improved the alphabetical held-team guardrail substantially,
  but its row-wise fold result is evenly split and too small to establish a
  broad gain.

## Profile

Fresh-process profiles warmed a full fold before measurement and reproduced
the exact stored fold-0 prediction hashes. For both the default and the
linear-residual screen winner, about 99% of fit time was recorded in tree
building. Prediction, preprocessing, gradient/Hessian, and wrapper overhead
were immaterial.

| Profiled arm | Fold-0 fit | Tree building | Share |
|---|---:|---:|---:|
| Default | 2.778s | 2.752s | 99.1% |
| Linear residual | 2.785s | 2.757s | 99.0% |

Any future speed work should therefore target the oblivious-tree construction
path and prove prediction equivalence. Optimizing estimator wrappers or
prediction will not close the 27.72s-to-9.29s training gap.

## Gate audit

The predeclared gates required all of:

- mean R² gain of at least 0.002;
- at least 6 of 10 fold wins;
- positive mean gain after omitting any single fold;
- no held-team R² regression; and
- no steady runtime above the frozen 28.299s DarkoFit baseline.

A10 at 1,000 and 2,000 rounds failed every gate. The early-stop/refit arm
passed only the runtime gate. Linear residual passed the held-team and runtime
gates but failed material gain, fold breadth, and leave-one-fold-out stability.

## Recommendation

Keep the current default unchanged. Preserve `linear_residual=True` as an
opt-in option and the early-stop/refit mechanism as a promising speed lever,
but do not combine or promote them from this dataset. If work resumes, the
smallest defensible next quality experiment is early stopping plus exact refit
with the current auto-resolved learning rate, isolated from A10's harmful 0.1
learning rate. Any production policy still requires unseen-dataset validation.

## Artifacts

- `basketball_darkofit_ablation.json` contains all fold scores, predictions,
  prediction hashes, held-team predictions, fitted metadata, phase timings,
  decision gates, and profiles.
- `basketball_darkofit_cold_player_guardrail.json` pins the source artifact and
  records the overlap disclosure plus cold/seen-player rescoring.
- `analyze_basketball_cold_player_guardrail.py` reproduces that supplement
  without refitting models.
- `basketball_darkofit_default_timing_confirmation.json` is the clean-source
  isolated timing confirmation.
- `run_basketball_darkofit_ablation.py` is the frozen five-arm runner.
