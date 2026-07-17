# Best-of-both-worlds completion audit

_Audited 2026-07-17 against DarkoFit `main` and the synced ChimeraBoost
0.15.0 comparator. This is a completion ledger, not a new benchmark result._

## Decision

Close the current best-of-both execution program at this boundary.

Every model or engine mechanism selected and authorized during this execution
has a terminal decision at its declared frozen gate: the passing mechanisms
are shipped, and the failing mechanisms are closed without a default change
or a broader evidence spend. Basketball supplied the primary fast gate. Safe
ordinal retains its separate 33-coordinate TabArena decision and is not
basketball evidence. There is no remaining candidate in this execution that
is authorized to enter the 243 development coordinates or the CTR23 lockbox.

Future mechanism research is a new campaign. Basketball remains its mandatory
fast first screen: the unchanged ten creator folds, the overlap-exposed
held-team view, the 585-row cold-player subset, behavior fingerprints, and
reciprocal clean timing. Only a basketball survivor may receive broader
confirmation.

## Repository, license, and comparator boundary

- DarkoFit development is on `main`; the final completion and CI-hygiene
  commits are descendants of every shipped mechanism and frozen decision
  artifact listed below.
- `/Users/kmedved/Code/GitHub/chimeraboost` is clean at
  `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`, tagged `v0.15.0`.
  Its local `main`, `origin/main`, and `upstream/main` agree.
- Both projects use Apache-2.0. DarkoFit's `NOTICE` attributes the adapted
  linear-leaf/LU design, fused histogram-and-split design, serial leaf
  descent, exact TreeSHAP, warmup, and input-validation patterns to
  ChimeraBoost, including the donor commits.
- No rejected donor screen copied ChimeraBoost source into DarkoFit.

## Requirement ledger

| Program requirement | Authoritative evidence | Terminal status |
| --- | --- | --- |
| Critically validate the proposal and establish safe execution boundaries | [`best_of_both_phase0_result.md`](best_of_both_phase0_result.md), [`BEST_OF_BOTH_PLAN.md`](../BEST_OF_BOTH_PLAN.md) | Complete |
| Freeze representative direct-prediction behavior, including all five distributional heads | `prediction_goldens.py`, `golden_predictions.json`, and strict `test_prediction_goldens.py` | Complete for the declared Phase-0 manifest |
| Retain exact/readable kernel oracles | `test_darkofit.py`, `test_fused_oblivious_kernel.py`, `test_fused_oblivious_expanded.py`, `test_serial_leaf_descent.py`, `test_linear_leaves_core.py`, and `test_tree_shap.py` | Complete |
| Use a shared forward-only basketball/cold-player boundary and audit larger runners for copy drift | `basketball_harness.py`, `basketball_guardrails.py`, their focused tests, and the runner audit below | Shared basketball boundary complete; historical runners remain frozen and forward-only generalization is deferred |
| Test current auto LR with early stopping and exact refit | [`basketball_auto_lr_refit_result.md`](basketball_auto_lr_refit_result.md) | Closed; quality and speed gates failed |
| Test OOB ensembles without assuming default promotion | [`basketball_oob_ensemble_confirmation_result.md`](basketball_oob_ensemble_confirmation_result.md) | Closed; absolute prediction-stability gate failed |
| Test the three authorized calibration candidates separately | [`basketball_quantile_calibration_result.md`](basketball_quantile_calibration_result.md), [`basketball_temperature_scaling_result.md`](basketball_temperature_scaling_result.md), and [`basketball_gaussian_scalar_calibration_result.md`](basketball_gaussian_scalar_calibration_result.md) | These candidates are closed; width or sports-quality gates failed. Distributional conformal, affine/grouped calibration, and other distributional heads were not tested |
| Add explicit warmup without hidden import work | [`basketball_warmup_result.md`](basketball_warmup_result.md) | Shipped; opt-in only |
| Add input validation and sklearn compliance | [`basketball_input_validation_result.md`](basketball_input_validation_result.md) | Shipped |
| Implement and gate per-leaf linear leaves | [`basketball_linear_leaves_result.md`](basketball_linear_leaves_result.md) | Explicit default-off research API shipped; automatic selector closed |
| Screen numeric crosses and categorical combinations independently | [`basketball_cross_features_donor_screen_result.md`](basketball_cross_features_donor_screen_result.md) and [`basketball_categorical_combinations_result.md`](basketball_categorical_combinations_result.md) | Both closed before a DarkoFit port |
| Fuse the proven training hot path with behavior identity | [`basketball_fused_oblivious_automatic_result.md`](basketball_fused_oblivious_automatic_result.md) | Shipped |
| Route small-row leaf descent serially with behavior identity | [`basketball_serial_leaf_descent_result.md`](basketball_serial_leaf_descent_result.md) | Shipped |
| Establish same-machine low-level parity with current ChimeraBoost | [`basketball_chimera_v015_result.md`](basketball_chimera_v015_result.md) | Complete; matched-core fit parity achieved |
| Improve packed prediction only where measured | [`basketball_packed_prediction_result.md`](basketball_packed_prediction_result.md) and [`basketball_leafwise_packed_prediction_result.md`](basketball_leafwise_packed_prediction_result.md) | New oblivious router rejected; bounded scalar leafwise route shipped |
| Add exact TreeSHAP without changing modeling defaults | [`basketball_tree_shap_result.md`](basketball_tree_shap_result.md) | Shipped for the supported scalar-oblivious lanes |
| Delete or consolidate complexity only after replacement proof | Phase 3 and Phase 4 decisions in [`BEST_OF_BOTH_PLAN.md`](../BEST_OF_BOTH_PLAN.md) | No speculative deletion authorized; unsupported fallbacks retained |
| Preserve DarkoFit's differentiators | Public API and focused suites for distributional regression, serialization, tuning, callbacks, auto-LR with ES off, and exact refit | Complete |
| Keep noisy-data defaults unchanged unless basketball passes | Constructor defaults plus every rejection artifact above | Complete; early stopping, linear leaves, crosses, combinations, and calibration were not promoted |
| Protect the CTR23 lockbox | [`tabarena_ctr23_minimal_confirmation_result.md`](tabarena_ctr23_minimal_confirmation_result.md) and its independent review | Lockbox sealed; confirmation result cannot authorize tuning or promotion |
| Attribute substantial donor adaptations | `LICENSE` and `NOTICE` | Complete |

## Preserved product behavior

The public regressor still defaults to:

- `learning_rate=None` (automatic learning rate);
- `early_stopping=False` and `early_stopping_rounds=None`;
- `refit=False`, with `refit_strategy="exact"` available explicitly;
- `tree_mode="catboost"`; and
- `linear_leaves=False` and `linear_residual=False`.

The five distributional losses and
`predict_dist`/`predict_variance`/`predict_interval`/`sample`, safe
serialization, `darkofit.tuning`, fit callbacks and `WallClockStopper`, the
ES-off automatic-learning-rate path, and exact refit remain present and
covered.

## Harness audit disposition

The live basketball work now shares one forward-only data/split,
cold/seen-player guardrail score, fitted-metadata, prediction-hash, and
reciprocal-timing boundary. Ordinary creator-fold scoring remains
campaign-local in the frozen runners and is not claimed as a consolidated
primitive.
Six larger historical campaign runners (20,565 lines total) were also
inspected for copy drift: the CTR23 minimal confirmation and the TabArena
accuracy-shootout, cap-horizon, follow-on, ordinal-confirmation, and
same-machine runners. They still duplicate campaign-specific manifest,
warmup, provenance, wave, resume, and attestation logic, and they intentionally
do not import the basketball harness because they bind different frozen
campaigns.

Those source-attested runners were not rewritten or relocated: doing so after
the fact would weaken reproducibility without supplying a live campaign that
could prove equivalent CI and release-verifier behavior. A general campaign
library and verifier extraction are therefore forward-only infrastructure
work for the next live non-basketball campaign, not an unfinished model change
or permission to alter a frozen artifact.

## Safety and verification

- The Phase-0 manifest contains 12 deterministic cases covering RMSE in all
  four tree modes, categorical regression, binary and multiclass
  classification, and Gaussian, LogNormal, Student-t, Poisson, and
  Negative-Binomial regression.
- Stable CI digests cover the Phase-0 manifest's direct `predict`, raw-score,
  probability, distribution-parameter, variance, interval, and sample
  outputs. Staged prediction and SHAP use their separate focused oracle suites
  rather than these goldens. Exact byte digests remain available under
  `DARKOFIT_STRICT_GOLDENS=1`. Student-t interval endpoints use an explicitly
  recorded eight-decimal portable digest because supported SciPy releases
  differ below that precision; all other manifest outputs retain twelve
  decimals.
- Exact or independent oracles cover binning, shared and leafwise splits,
  fused/reference histograms, serial/parallel descent, packed/tree-loop
  prediction, linear leaves, and TreeSHAP coalition enumeration.
- The shared sports boundary proves that the creator training rows are
  unchanged and labels the alphabetical held-team result as
  player-overlap-exposed. Its cold subset contains 585 rows from 210 players
  absent from training.
- The current local full suite passes on Python 3.13, and the CI-failure set
  passes locally on Python 3.9, 3.11, and 3.13. GitHub Actions run
  `29571900341` passed the full matrix on Python 3.9, 3.11, and 3.13 at the
  final code-bearing head `da3cbaa`.

## Evidence discipline

The nine-task minimal CTR23 run was a confirmation panel, not the lockbox. It
failed its registered uncertainty and default-guardrail gates and disclosed
two protocol deviations. No extra folds, post-outcome tuning, or default
change is permitted from it. The remaining 243 exposed neighboring
coordinates are development-only, not fresh confirmation. The separate
270-coordinate lockbox remains unopened.

No current Phase-2 candidate passed basketball, so the program does not run
the 243 development coordinates, simulate a lockbox pass probability, or
spend the lockbox.

## Work deliberately left for a new goal

Distributional conformal correction, affine/grouped distributional
calibration, calibration of the other distributional heads, safe-ordinal
mechanism redesign, the generic validation-selection framework, the deferred
mode-mix diagnostic, the optional accuracy preset, and general campaign
library work were not executed here. The plan's Phase-5 decision suite, Pareto
release chart, and kill calendar are likewise independent R&D-infrastructure
proposals rather than unfinished model mechanisms in this execution. These
items remain eligible future work, but none authorizes reopening a failed
candidate, weakening a frozen gate, or making a default claim from the
existing evidence.

The next model experiment must begin with a materially new mechanism and a new
protocol. For iteration speed and sports relevance, basketball is the primary
development dataset again; broader panels are confirmation only.
