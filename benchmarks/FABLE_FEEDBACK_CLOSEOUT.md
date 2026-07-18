# Fable feedback closeout ledger

This ledger reconciles the feedback in the July 18, 2026 Fable review with the
repository evidence. It is a closure map, not a new benchmark result. Frozen
campaign protocols, raw artifacts, and historical decisions remain unchanged.

The labels below mean:

- **Resolved / proved:** current product code, tests, or immutable campaign
  evidence directly supports the statement.
- **Corrected / rejected:** the original wording was stale, too broad, or
  contradicted by the bound evidence.
- **Research-only:** the finding is useful but cannot promote a default or
  support a broader product claim.
- **Pending:** required work or fresh evidence does not yet exist. A pending
  item must not be described as complete elsewhere.

## Resolved / proved

| Feedback item | Resolution and controlling evidence |
| --- | --- |
| Split-conformal intervals crashed when `random_state=None`. | Fixed. Seed normalization now preserves `None`, and `test_gaussian_conformal_holdout_accepts_default_random_state_none` exercises the formerly crashing public fit path. See [`darkofit/sklearn_api.py`](../darkofit/sklearn_api.py) and [`tests/test_distributional.py`](../tests/test_distributional.py). |
| Grouped ensembles did not forward groups into member fits. | Fixed. Group-bootstrap members receive their sampled groups, and their explicit OOB validation remains group-disjoint. `test_group_bootstrap_forwards_groups_to_explicit_oob_member_fits` covers the interaction. |
| Row bootstrap silently discarded supplied groups. | Fixed fail-closed. Supplying `groups=` with `ensemble_bootstrap="rows"` now raises with migration guidance to use group bootstrap. `test_row_bootstrap_rejects_groups_instead_of_silently_splitting_entities` covers the public behavior. |
| `preset="accuracy"` silently overrides explicit values for its managed fields. | The override is intentional product behavior, not an unresolved implementation accident. [`docs/parameters.md`](../docs/parameters.md) lists the managed fields and states that the preset temporarily overrides them; other fields remain caller-controlled. `test_accuracy_preset_temporarily_overrides_only_managed_fields` binds both the resolved fit policy and restoration of constructor values. |
| An expired shared deadline can retain the capped `selection_rounds` audition. | The behavior is intentional and now directly documented and tested. The fitted model records `final_refit_performed=false` and `final_refit_status="skipped_deadline"` rather than starting a new fit after the deadline. See [`README.md`](../README.md), [`docs/parameters.md`](../docs/parameters.md), and `test_selection_rounds_retains_audition_when_deadline_blocks_refit`. |
| Ensemble preprocessing might leak targets between members. | The shipped shared path is restricted to target-free numeric preprocessing. Categorical and ordinal fits use member-local preprocessing, so ordered target statistics are fitted independently. The ensemble API and persistence tests bind this distinction. |
| Conformal calibration rows might participate in fitting or model selection. | The conformal holdout is isolated before fitting and selection. Existing distributional tests bind the split, score construction, persistence, and calibrated interval path. |
| The Tier-E accuracy preset must reproduce the frozen A10 configuration. | `test_accuracy_preset_matches_the_frozen_a10_configuration` proves prediction equality to the explicit frozen profile and persistence equality after round trip. |
| Eight campaign evidence packages, 21 evidence bindings, T9 re-execution, and the generated measurements page verified at Fable's checkpoint. | Treat this as a verified historical checkpoint, not as a substitute for final-commit CI. The frozen raw bytes did not change afterward; hardened reports distinguish original run-time hashes from current source hashes, campaign tests bind the raw/derived boundaries, and the benchmark-status tests require deterministic regeneration. A new full-suite run remains a separate pending release gate below. |
| Post-run hardening appeared after a “clean worktree” closeout. | The statement was stale at the moment Fable checked it. The hardening was subsequently landed in labeled commits, beginning with `dca503e` (`fix: harden product APIs and benchmark evidence`), with original run-time hashes retained separately from current hardened source hashes in regenerated reports. The later Fable-gap tests and target-preflight primitive are isolated in `0dd5cae`. |
| The T8 coverage result deserved prominent, bounded documentation. | The README now reports the exact frozen equal-dataset result: conformal mean absolute 90% coverage gap `0.0110`, versus NGBoost `0.0824`, with conformal width `0.9831x` DarkoFit's parametric width. The text explicitly limits this to descriptive marginal-coverage evidence and disclaims conditional coverage and universal superiority. See [`t8_distributional_flagship_result.md`](t8_distributional_flagship_result.md). |
| A future confirmation registry needs a target-validity gate before authorization. | The reusable primitive now exists in [`confirmation_target_preflight.py`](confirmation_target_preflight.py). It binds task metadata and the exact dataset fingerprint, requires a nonempty one-dimensional target with the expected row count, verifies float64 coercibility and finiteness, and persists neither values nor target statistics. Its unit tests cover valid nullable/numeric inputs, non-finite and complex values, shape/row drift, metadata drift, and fingerprint drift. |

## Corrected / rejected wording

| Original framing | Correction |
| --- | --- |
| “The T5 registry never opened the target column.” | Too strong. The frozen registry protocol explicitly allowed reading target bytes to build opaque contamination hashes; it prohibited target statistics and candidate outcomes. It did not preflight finiteness, which is why the control wave later found two invalid targets. The accurate description is **outcome-blind but not target-unread**. |
| “Target-finiteness preflight reveals nothing about outcomes.” | Too broad. It necessarily reveals eligibility facts: numeric coercibility, row alignment, and whether every target value is finite. It does not reveal target distributions, associations, model scores, or candidate/control outcomes. Future protocols should call this **outcome-blind, eligibility-aware**. |
| “T7 eliminated every classic CatBoost mechanism.” | T7 cleanly shows that CatBoost Plain mode equals its product default on all measured coordinates and that the tested ordered, border, CTR-complexity, and depth arms do not explain DarkoFit's gap. It did **not** isolate leaf-estimation iterations cleanly because that arm changed learning rate too. T7b subsequently removed that confound with frozen per-coordinate learning rates and three model seeds, and tested stochastic, L2, one-hot, and leaf-step directions. It still attributed none of the gap. `l2_leaf_reg=1` was the sole promising configuration; one-hot 255 had a large mean gain but failed its uncertainty and worst-task gates. See [`t7b_catboost_gap_attribution_result.md`](t7b_catboost_gap_attribution_result.md). |
| “The current full suite is 1,972 passed / 27 skipped.” | That was a valid local count at Fable's checkpoint, not a permanent repository invariant. New tests were added afterward. Final closeout must report a fresh suite count and CI result for the exact shipped commit. |
| “The composite candidate failed T5.” | Incorrect. T5 failed during the current-default control wave because two frozen tasks had non-finite targets. The composite and comparator waves never ran. The candidate remains untested, but all 25 lineages are spent because control outcomes were already scored. See [`t5_composite_confirmation_failure.md`](t5_composite_confirmation_failure.md). |
| “The five-member sports ensemble had sound quality evidence and was blocked only by ritual timing gates.” | Refuted by fresh player-disjoint evidence. Panel 2 measured a `1.023115x` equal-lineage RMSE ratio, losing eight of nine lineages, with worse held-team and cold-player aggregates. The automatic policy is closed; the Tier-E ensemble API remains a valid opt-in capability. See [`basketball_sports_panel_v2_result.md`](basketball_sports_panel_v2_result.md). |

## Research-only findings

| Finding | Proper scope |
| --- | --- |
| RSSI's large default gap is policy rather than missing tree or linear-leaf machinery. | On one spent OpenML coordinate, matched DarkoFit and ChimeraBoost constant- and linear-leaf lanes are byte-identical, including borders, histories, trees, predictions, retained rounds, and RMSE. Validation fraction, early stopping, and lane selection explain the observed product gap there. This is strong mechanism evidence, not a broad quality or timing claim. See [`rssi_linear_leaf_diagnosis_result.md`](rssi_linear_leaf_diagnosis_result.md). |
| ChimeraBoost's 100-round lane audition can select the wrong full-budget winner. | On the RSSI diagnosis coordinate, the capped audition selected linear leaves while the full-budget race selected constant leaves; the selected linear test RMSE was `1.2047x` the constant lane. This justifies avoiding that exact selector in a composite default, but one coordinate does not estimate its general failure rate. |
| The T7 samples-per-feature depth rule improves CatBoost itself. | The guarded `depth_by_n_p` policy produced a `0.962248x` CatBoost/default aggregate on the eight spent development tasks, while declining on five. It widened rather than explained the historical DarkoFit/CatBoost gap and authorizes no DarkoFit change. See [`t7_catboost_attribution_result.md`](t7_catboost_attribution_result.md). |
| T7b narrowed the CatBoost search space without identifying the gap driver. | The fixed-learning-rate, three-seed follow-on found no supported contributor among stochastic regularization, row sampling, one-hot thresholds, L2, or extra leaf-estimation steps. `l2_leaf_reg=1` passed its promising-configuration gates. One-hot 255 improved the equal-dataset mean sharply but failed the Bonferroni upper and worst-task gates, so it is not a general policy. This is spent development evidence only. See [`t7b_catboost_gap_attribution_result.md`](t7b_catboost_gap_attribution_result.md). |
| DarkoFit beat ChimeraBoost again on the fresh sports control, while CatBoost remained ahead. | These are descriptive same-machine comparisons from the frozen sports panel. They validate the measured product ordering on that panel, not universal superiority or a new automatic policy. |
| Split conformal is the strongest current distributional product result. | The five-dataset T8 panel supports a marketable marginal-coverage claim with width shown alongside it. NLL and CRPS remain dataset-dependent; the result does not claim DarkoFit dominates NGBoost or CatBoost on every probabilistic metric. |
| Subset-fused kernels and the measurements page are Tier-E evidence. | Behavior-exact engine work and hash-bound timing measurements ship as engineering facts under [`SHIPPING_POLICY.md`](SHIPPING_POLICY.md). The old binary engineering gate remains historically immutable but is no longer the prospective policy. |

## Pending work

| Item | Required evidence before closure |
| --- | --- |
| Fresh panel 3 | The prospective protocol, exact 12-primary/three-stratum design, candidate contract, deterministic declarations, contamination rules, target-finiteness preflight, and implementation are prepared. The old `0.963085` exchangeable-lineage calculation is invalidated and is not design power. Before any fresh target access, the exact-policy calibration must run on all 39 spent coordinates; a candidate survives only when both its simulated pass probability and preregistered Wilson lower bound are at least `0.80`. The resulting create-only power decision is the explicit go/no-go. No such decision exists yet. |
| Composite and guarded-cross candidates | Resolved prospectively, not yet adjudicated. They are two separately tested candidates. The contract freezes Bonferroni one-sided inference when both survive calibration, a preregistered singleton recalculation when exactly one initially survives, and fixed T5 precedence if both later pass. Post-outcome winner selection is prohibited. |
| Deterministic primaries and reserves | Resolved prospectively. The exact first four declarations in each stratum are the power-authorized primaries. A failed primary stops registry v1; reserves may diagnose or support a newly frozen future design but may not substitute into this one. |
| Panel-3 execution | Commit the completed executable H1 sources, create the H2 calibration freeze, run and publish the spent-data calibration, and surface its explicit power go/no-go. Only a passing decision may authorize fresh target preflight, registry creation, and the one-shot lockbox run. Once any selected-task arm is scored, all 12 lineages are spent and only prediction-identical immutable spools may be resumed. |
| Full verification | Run the complete library and campaign suites on the exact final commit, regenerate only derived evidence whose protocols permit regeneration, verify all original/frozen versus current hashes, run independent review, and report the fresh pass/skip counts. |
| Release integration | Merge the completed branch to `main`, push it, confirm `origin/main` equals local `main`, verify a clean worktree, and record the exact final commit. Until then, the goal is active rather than closed. |

## Closure rule

This ledger may be updated from **Pending** only when the named artifact or
test exists and has been verified against the exact final commit. Historical
campaign outcomes do not change category merely because the shipping policy
changed. No old lockbox, spent panel, or failed campaign may be relabeled as
fresh confirmation.
