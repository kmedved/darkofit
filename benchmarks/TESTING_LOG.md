# DarkoFit testing log

_Canonical navigation ledger. Updated 2026-07-22 after the GitHub-only
`v0.11.0` release, public ensemble-v3 exposure, and owner-promoted static
fused/unfused dispatch._

This file records what DarkoFit has tested, why it was tested, what happened,
and which artifact controls the conclusion. It is intentionally broader than
the generated release frontier: correctness suites, performance
characterization, spent development screens, fresh confirmation campaigns,
lockbox decisions, and release verification all belong here.

This log is a map, not a substitute for the evidence it links. A frozen
protocol, raw artifact, analyzer, or result remains authoritative for its
campaign. Adding a newer comparison never rewrites an older result or changes
the version boundary under which it was true.

## How to read the record

| Label | Meaning |
| --- | --- |
| Product verification | Unit, integration, compatibility, serialization, packaging, or CI evidence. |
| Tier-E measurement | Descriptive performance or behavior-exact engine evidence. It can support a scoped engineering fact or opt-in product surface, not a universal default claim. |
| Spent development | Outcomes were inspected and may guide mechanism research, but cannot provide fresh confirmation. |
| Tier-D confirmation | Prospectively frozen evidence for a default or automatic policy, including uncertainty, concentration, harm, and cost gates. |
| Closed | The tested candidate has a terminal negative decision for the declared route. It is not silently retuned on the same evidence. |
| Sealed | Registered outcomes were not accessed. A lockbox remains available only for a future candidate that earns authorization. |

The binding prospective rules are in
[`SHIPPING_POLICY.md`](SHIPPING_POLICY.md). The generated high-level frontier
is [`benchmark_status.md`](benchmark_status.md), and the longer numerical
checkpoint is [`../BENCHMARK_NOTES.md`](../BENCHMARK_NOTES.md).

## Current checkpoint

| Boundary | Current state |
| --- | --- |
| DarkoFit release | GitHub-only `v0.11.0`, exact tag commit `0b820e332cec2c083b1dd89eef0fe306d69cfc0e`; not published to PyPI |
| Release verification | GitHub Actions run `29942771031` passed at the exact tag commit: each Python 3.9/3.11/3.13 library lane reported `1,329 passed / 11 skipped / 4 deselected`; the campaign lane reported `1,692 passed / 29 skipped`, then 19 contract checks passed |
| GitHub integration | Campaign, strict documentation, generated-evidence, package-build, and all supported-Python lanes passed before the annotated tag and GitHub release were published |
| v0.10.0 final release CI | GitHub Actions run `29686258603` passed for that tagged source |
| Historical broad comparator | ChimeraBoost 0.14.1 on the frozen 13-dataset same-machine regression panel |
| Historical sports comparator | ChimeraBoost 0.15.0 on the frozen player-disjoint sports panel |
| Latest descriptive comparator | ChimeraBoost `v0.18.0-6-gf14be60`, exact commit `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`, on spent basketball data |
| CTR23 lockbox | Sealed; Panel 3 retained zero candidates and did not authorize fresh access |

The two final local-suite totals differ only because 25 environment-gated
tests passed in one environment and skipped in the other. Both collected the
same 2,673 tests. The reconciliation and final CI boundary are recorded in
[`FABLE_FEEDBACK_CLOSEOUT.md`](FABLE_FEEDBACK_CLOSEOUT.md).

## Chronological campaign record

### 1. Core correctness and scalar-regression policy

The first release-hardening pass established the scalar-regression behavior
that later campaigns treated as the product control.

| Question | Test and result | Disposition |
| --- | --- | --- |
| Should ordered leaf updates remain automatic for scalar regression? | Six-case, three-seed shared-split guardrail with numeric, weighted, and categorical regression. Plain boosting improved five case means; House Prices exposed a catastrophic ordered-on tail on one seed despite healthy validation error. [`ordered_boosting_policy_check.md`](ordered_boosting_policy_check.md) | `ordered_boosting="auto"` resolves off for scalar regression. Ordered target-statistic preprocessing remains active for categoricals. |
| Does the policy resolver reach every public fit path? | Scalar, categorical, multiclass, distributional, MAE, and Quantile tests checked the resolver and call sites; `auto_off_scalar_regression` is persisted in fitted metadata. | Product verification passed. |
| Did ndarray-subclass extraction preserve one-dimensional columns? | An `np.matrix` round-trip reproduced the two-dimensional slice defect. Coercion through `np.asarray` restored the public one-dimensional contract. [`tests/test_distributional.py`](../tests/test_distributional.py) | Fixed and covered. |
| Can malformed serialized headers crash before validation? | Crafted unhashable loss names and huge JSON integers exercised string guards and `OverflowError` handling. [`tests/test_payload_hardening.py`](../tests/test_payload_hardening.py) | Fixed and covered. |
| Is tuner learning-rate fallback observable? | `refit_learning_rate="fold_median"` without usable metadata now warns and records the fallback source. [`tests/test_tuning.py`](../tests/test_tuning.py) | Fixed and covered. |

The phase reproduced 532 core tests plus 18 LightGBM-comparison tests—550
passes—on both Python 3.11 and 3.13.13. That was an intermediate checkpoint,
not the final release-suite count.

### 2. TabArena integration smoke

On 2026-07-09, the local TabArena adapter completed the official Lite
quickstart's three task types:

- `blood-transfusion-service-center` for binary classification;
- `QSAR_fish_toxicity` for regression; and
- `anneal` for multiclass classification.

The default and one sampled HPO configuration completed for all three. The
test established adapter integration, validation-set and sample-weight
forwarding, cooperative wall-clock callbacks, and fitted-child telemetry. A
three-task Elo or rank was explicitly treated as diagnostic only. Reproduction
instructions and scope are in
[`../BENCHMARK_NOTES.md`](../BENCHMARK_NOTES.md).

### 3. Thirteen-dataset regression diagnosis

The initial 13-dataset work found a real shared-split legality defect:
already-pure leaves with an empty child could incorrectly veto an otherwise
legal symmetric-tree split. The corrected default matched the diagnostic
`min_child_weight=0` proxy on all 13 datasets and reduced the geometric-mean
RMSE gap to the then-current ChimeraBoost default from 5.14% to 1.25%.

A fixed `learning_rate=0.1` arm improved aggregate RMSE by 0.40% but reduced
head-to-head dataset wins against ChimeraBoost from five to four. The automatic
learning-rate default was retained. Per-dataset evidence is in
[`tabarena_regression_default_check.md`](tabarena_regression_default_check.md).

### 4. Horizon and isolated policy ablations

On 2026-07-13, the frozen horizon campaign ran 444/444 outer jobs and
3,552/3,552 fitted children.

| Experiment | Principal result | Decision |
| --- | --- | --- |
| 10,000 versus 1,000 rounds | 10,000 rounds improved equal-dataset RMSE by 0.453%, below the declared 0.5% bar, while training rose 12.88% and inference 10.65%. All eight cap-active datasets improved. [`tabarena_regression_cap_horizon_result.md`](tabarena_regression_cap_horizon_result.md) | Retain the 1,000-round default; longer horizon remains opt-in research. |
| `tree_mode="auto"` | 3.10% aggregate RMSE improvement, but 2.57× inference time. | Do not promote automatically. |
| Four target-stat permutations | 0.21% aggregate improvement on five applicable datasets, with a Diamonds regression and failed quality/harm gates. | Closed as a broad policy. |
| Safe source-declared ordinal representation | 19.50% screen improvement across Airfoil and Diamonds. | Advanced to mechanism confirmation only. |
| Safe one-hot | 4.00% screen improvement but failed the declared breadth gate. | Closed for this route. |
| Global linear residual | 1.07% aggregate improvement but failed the dataset-harm gate. | Closed for this route. |

The isolated screen is
[`tabarena_regression_followon_screen_result.md`](tabarena_regression_followon_screen_result.md).

The ordinal confirmation then improved all 33 coordinates and reduced
equal-dataset RMSE by 17.29%, but its causal inference-time ratio was
`1.265169×`, above the frozen `1.25×` ceiling. The formal decision remained
`do_not_advance`; source-declared ordinal support later shipped only as an
explicit Tier-E surface. See
[`tabarena_regression_ordinal_confirmation_result.md`](tabarena_regression_ordinal_confirmation_result.md).

### 5. Same-machine product-default comparison

On 2026-07-14, DarkoFit 0.9.0, ChimeraBoost 0.14.1, and CatBoost 1.2.10 ran on
the same Apple-silicon machine and the same `r0f0`, `r1f1`, and `r2f2`
coordinates across 13 regression datasets.

| Contrast | Equal-dataset test RMSE ratio | 95% interval | Train ratio | Inference ratio | Incremental-memory ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| DarkoFit / ChimeraBoost | 1.012523 | [1.008196, 1.017260] | 0.9093 | 1.4332 | 0.5660 |
| DarkoFit / CatBoost | 1.053834 | [1.051124, 1.056798] | 0.3729 | 1.2561 | 0.1526 |
| ChimeraBoost / CatBoost | 1.040800 | [1.037356, 1.044255] | 0.4101 | 0.8765 | 0.2696 |

The primary panel contained 117 outer jobs. A separate 18-job
Airfoil/Diamonds representation diagnostic brought the execution total to 135
outer jobs and 1,080 fitted children, all completed without failure or
imputation. The representation diagnostic is non-poolable and repaired known
source semantics; it is not generic ordinal evidence.

The authoritative result and protocol are
[`tabarena_regression_same_machine_result.md`](tabarena_regression_same_machine_result.md)
and
[`tabarena_regression_same_machine_protocol.md`](tabarena_regression_same_machine_protocol.md).

### 6. Accuracy profile A10

On 2026-07-15, the frozen A10 development profile combined:

- a 10,000-round safe cap;
- fixed learning rate 0.1;
- 128 bins;
- `l2_leaf_reg=3`;
- one target-stat permutation; and
- validation-selected CatBoost, LightGBM, or hybrid tree mode.

It completed 78 outer jobs, 624 selected children, and 936 internal mode
candidates without failure, imputation, deadline, restart, recovery, or
time-limit stop.

| Contrast | Equal-dataset RMSE | Interpretation |
| --- | ---: | --- |
| A10 / ChimeraBoost 0.14.1 | 0.9756 | 2.44% better on spent development data |
| A10 / DarkoFit product default | 0.9636 | 3.64% better |
| A10 / fixed CatBoost-mode B10 | 0.9722 | 2.78% better |
| A10 / CatBoost 1.2.10 | 1.0154 | 1.54% worse |

Diamonds supplied 87.6% of A10's mean-log advantage over ChimeraBoost.
Removing it left only a 0.33% advantage and a 6–6 dataset split. A10 therefore
shipped as `preset="accuracy"` under Tier-E rules, not as a product default.
See
[`tabarena_regression_accuracy_shootout_result.md`](tabarena_regression_accuracy_shootout_result.md).

### 7. Minimal CTR23 confirmation

The nine-task minimal CTR23 panel was confirmation data, not the 270-coordinate
lockbox. It failed its registered uncertainty and default-guardrail gates and
disclosed two protocol deviations. No extra folds, post-outcome tuning,
default change, or lockbox access was permitted.

The reviewed publication is
[`tabarena_ctr23_minimal_confirmation_result.md`](tabarena_ctr23_minimal_confirmation_result.md).
The remaining neighboring coordinates became development-only; the lockbox
remained sealed.

### 8. Basketball guardrail correction

The original creator split held out teams but still shared many players
between training and evaluation. Later basketball work corrected that
boundary:

- creator folds remain a fast, reproducible compatibility screen;
- held-team results are labeled overlap-exposed;
- a 585-row cold-player subset contains 210 players absent from training; and
- the later T10 primary folds are player-disjoint by `bref_id`.

The shared data, split, scoring, fingerprint, metadata, and timing primitives
live in `basketball_harness.py` and `basketball_guardrails.py`. The completed
program's boundary audit is
[`best_of_both_completion_audit.md`](best_of_both_completion_audit.md).

### 9. ChimeraBoost 0.15 engine and product characterization

The frozen 0.15 comparison separated product policy from matched engine
behavior.

| Lane | Result |
| --- | --- |
| Matched 1,000-tree core | DarkoFit and ChimeraBoost predictions were byte-identical; fit-time ratio was 0.975×. |
| Product defaults | Quality remained close, but tree count, automatic learning rate, early stopping, and lane selection differed. |
| Prediction | The then-current DarkoFit path remained slower, motivating later packed/contiguous prediction work. |

See [`basketball_chimera_v015_result.md`](basketball_chimera_v015_result.md).

### 10. Best-of-both mechanism program

Basketball was used as the mandatory fast gate because it is inexpensive,
noisy, and directly relevant to the owner's sports workloads. The terminal
ledger is
[`best_of_both_completion_audit.md`](best_of_both_completion_audit.md).

| Mechanism or product surface | Evidence | Terminal result |
| --- | --- | --- |
| Auto LR + early stopping + exact refit | [`basketball_auto_lr_refit_result.md`](basketball_auto_lr_refit_result.md) | Closed; quality and speed gates failed. |
| Row-OOB ensemble candidate | [`basketball_oob_ensemble_confirmation_result.md`](basketball_oob_ensemble_confirmation_result.md) | Automatic route closed; opt-in ensemble API remained eligible. |
| Quantile, temperature, and Gaussian scalar calibration | [`basketball_quantile_calibration_result.md`](basketball_quantile_calibration_result.md), [`basketball_temperature_scaling_result.md`](basketball_temperature_scaling_result.md), [`basketball_gaussian_scalar_calibration_result.md`](basketball_gaussian_scalar_calibration_result.md) | Closed at their declared width or sports-quality gates. |
| Explicit warmup | [`basketball_warmup_result.md`](basketball_warmup_result.md) | Shipped opt-in; no hidden import work. |
| Input validation and sklearn compliance | [`basketball_input_validation_result.md`](basketball_input_validation_result.md) | Shipped after compatibility decisions and overhead fixes. |
| Per-leaf linear leaves | [`basketball_linear_leaves_result.md`](basketball_linear_leaves_result.md) | Explicit default-off API shipped; automatic selector closed. |
| Numeric crosses and categorical combinations | [`basketball_cross_features_donor_screen_result.md`](basketball_cross_features_donor_screen_result.md), [`basketball_categorical_combinations_result.md`](basketball_categorical_combinations_result.md) | Closed before copying donor code. |
| Fused oblivious training | [`basketball_fused_oblivious_automatic_result.md`](basketball_fused_oblivious_automatic_result.md) | Behavior-exact internal lane shipped. |
| Small-row serial leaf descent | [`basketball_serial_leaf_descent_result.md`](basketball_serial_leaf_descent_result.md) | Behavior-exact internal router shipped. |
| Packed prediction | [`basketball_packed_prediction_result.md`](basketball_packed_prediction_result.md), [`basketball_leafwise_packed_prediction_result.md`](basketball_leafwise_packed_prediction_result.md) | New oblivious route rejected; bounded scalar leafwise route shipped. |
| Exact TreeSHAP | [`basketball_tree_shap_result.md`](basketball_tree_shap_result.md) | Supported scalar-oblivious API shipped with exact coalition-oracle coverage. |

The program also froze 12 representative direct-prediction goldens spanning
all four tree modes, categorical regression, binary and multiclass
classification, and all five distributional heads. Independent oracles cover
binning, split finding, fused histograms, leaf descent, packed prediction,
linear leaves, and TreeSHAP.

### 11. Engine ceiling measurements

These are scoped Tier-E measurements, not a claim that every DarkoFit workload
beats every ChimeraBoost workload.

| Track | Measured result | Evidence status |
| --- | --- | --- |
| Large-n matched core | DarkoFit/ChimeraBoost fit ratio `0.7817×`, or a `1.2793×` speedup; RMSE ratios `0.99998–1.00085×`. [`large_n_engine_result.md`](large_n_engine_result.md) | Published measurement; missed the old frozen `1.30×` certification bar. |
| Public prediction | DarkoFit won 8/8 median cases; ratios `0.805–0.987×`, with 6/8 also meeting the old conjunctive stability rule. [`predict_throughput_integrated_result.md`](predict_throughput_integrated_result.md) | Published measurement; no universal prediction claim. |
| Selected-row/feature fusion | Exact in all eight cells; fit ratio `0.5348×`, tree-build ratio `0.5265×`. [`fused_subset_oblivious_result.md`](fused_subset_oblivious_result.md) | Shipped as behavior-exact Tier-E engine work after the shipping-policy correction. |

The deterministically generated summary is
[`../docs/measurements.md`](../docs/measurements.md).

### 12. Product Offense campaigns

The Product Offense program separated opt-in/Tier-E shipping from
default/Tier-D confirmation. Its execution ledger is
[`../PRODUCT_OFFENSE_PLAN.md`](../PRODUCT_OFFENSE_PLAN.md).

| Campaign | What was tested | Result |
| --- | --- | --- |
| T5 composite | Approximately 25-task fresh panel for the combined tabular policy. | Closed fail-closed when the control wave found two non-finite-target tasks. Candidate and comparator waves did not run, but all lineages became spent. [`t5_composite_confirmation_failure.md`](t5_composite_confirmation_failure.md) |
| T6 RSSI and smooth crosses | Matched linear-leaf diagnosis followed by diff/product feature development. | RSSI proved the engine was not missing; validation and lane selection explained the gap. A 5%-guarded cross selector retained a `0.9591×` development ratio with exact declines and no observed split harm. [`rssi_linear_leaf_diagnosis_result.md`](rssi_linear_leaf_diagnosis_result.md), [`smooth_cross_features_result.md`](smooth_cross_features_result.md) |
| T7 CatBoost attribution | Ordered/plain, border count, leaf estimation, CTR complexity, and depth by `(n,p)`. | Plain equaled CatBoost default; tested mechanisms did not explain the gap. A samples-per-feature depth policy improved CatBoost itself and was retained as research only. [`t7_catboost_attribution_result.md`](t7_catboost_attribution_result.md) |
| T7b CatBoost attribution | Fixed-learning-rate, three-seed stochastic, L2, one-hot, and leaf-step follow-on. | No gap contributor identified. `l2_leaf_reg=1` was promising; one-hot 255's large mean failed uncertainty and worst-task gates. [`t7b_catboost_gap_attribution_result.md`](t7b_catboost_gap_attribution_result.md) |
| T8 distributional flagship | Five datasets, 75 coordinates, DarkoFit Gaussian/conformal, NGBoost, CatBoost uncertainty, and LightGBM quantiles. | DarkoFit conformal had mean absolute 90%-coverage gap `0.0110` versus NGBoost `0.0824`, at `0.9831×` DarkoFit parametric width. Descriptive marginal-coverage evidence only. [`t8_distributional_flagship_result.md`](t8_distributional_flagship_result.md) |
| T9 SynthGen ledger | Retrospective synthetic direction finder against corrected real outcomes. | Reached 8/9 only after two later outcomes replaced development labels. Adopted for probe ranking only, never confirmation. [`t9_synthgen_corrected_ledger_result.md`](t9_synthgen_corrected_ledger_result.md) |
| T10 sports panel #2 | Fresh 2014–2016, nine-lineage, player-disjoint panel for a five-member row-OOB automatic policy. | Candidate lost eight of nine lineages: `1.023115×` aggregate RMSE and `1.028733×` cold-player RMSE. Automatic policy closed. [`basketball_sports_panel_v2_result.md`](basketball_sports_panel_v2_result.md) |

The T10 control was 2.81% better than ChimeraBoost 0.15 on equal-lineage RMSE
and 5.26% worse than CatBoost 1.2.10. Those are panel-scoped descriptive
comparisons, not universal rankings.

### 13. Panel 3 lockbox preparation and no-go

Panel 3 was the prospective route for the T5 composite and guarded
cross-feature candidates. Preparation catalogued benchmark exposure,
excluded contaminated lineages, quarantined inadvertent Parquet-footer target
exposure, unified fitted-metadata validation, and froze a spent-data power
calibration before any fresh target access.

The repaired calibration completed all 117 spent coordinates and fed 5,000
simulated 12-task panels:

| Candidate | Simulated pass probability | One-sided Wilson lower bound | Required |
| --- | ---: | ---: | ---: |
| T5 composite | 50.00% | 48.61% | 80.00% |
| Guarded cross features | 10.64% | 9.82% | 80.00% |

Zero candidates were retained and
`confirmation_run_authorized` remained false. T5's point-effect and
leave-one-favorable-dataset gates passed in all 5,000 simulations, but its
worst-dataset harm bound passed only 2,500. The measured research bottleneck
is tail harm, not mean effect.

The binding record is
[`panel3_power_design_decision.json`](panel3_power_design_decision.json).
No fresh registry, target preflight, confirmation fit, or lockbox spool was
created. The lockbox remains sealed.

### 14. ChimeraBoost 0.18 post-release diagnostic

On 2026-07-19, ChimeraBoost `f14be60`—version 0.18.0 plus six audit
commits—was run against DarkoFit 0.10 on the spent creator and player-disjoint
basketball protocols.

| View | DarkoFit 0.10 | ChimeraBoost 0.18 default | Interpretation |
| --- | ---: | ---: | --- |
| Creator mean R2 | **0.526750** | 0.525993 | Effective single-model tie; DarkoFit had the small numerical edge. |
| Creator default wall | **9.73s median** | 12.92s median | DarkoFit was 24.70% lower on this small ten-fold workload. |
| Player-disjoint primary RMSE | **1.963257** | 2.021610 | DarkoFit was 2.89% lower. |
| Cold-player RMSE | **1.807506** | 1.841097 | DarkoFit was 1.82% lower. |
| Player-disjoint fit | 95.99s | **34.72s median** | ChimeraBoost remained 2.77× faster. |

Turning `quantize_gradients=False` restored ChimeraBoost 0.15's
player-disjoint quality values exactly to printed precision. The actual 0.18
default was 0.0763% worse on primary RMSE and 0.0389% worse on cold players
than that float lane. ChimeraBoost's recommended eight-member ensemble reached
creator-fold R2 `0.543327` at 54.78 seconds, the highest score and highest cost
in that diagnostic.

The complete source, environment, data hashes, folds, repeat timings,
limitations, and disposition are recorded in
[`basketball_chimera_v018_diagnostic.md`](basketball_chimera_v018_diagnostic.md).
The broad 13-dataset panel has not been rerun against 0.18, so the old
same-machine regression comparison must not be relabeled as current-version
evidence.

### 15. Wave 1 Q0 scalar-path attribution

1. **Execution boundary:** 2026-07-20; DarkoFit package source
   `726e5d8e6131c580bce948db833a5007d0692dca`; harness
   `18bc48c7778eed0980efa430ad6fa722310919bb`.
2. **Comparator:** ChimeraBoost
   `f14be606b641f1bf0dc92bb14b3951f1fe631c6b` was source-attested for the
   shared Wave 1 boundary but not executed in Q0.
3. **Evidence class:** Tier-E engineering profile on deterministic synthetic
   data; spent; no fresh or sealed data.
4. **Data:** seed 20260717, 24 numeric features, scalar RMSE target, 500,000
   and 1,000,000 training rows plus the following 100,000-row holdout.
5. **Arms:** current fused DarkoFit production path and private
   behavior-exact `fused_oblivious_kernel=False` reference; 40 trees.
6. **Resources:** arm64 macOS 26.5.2, 14 logical CPUs, fixed 14-thread budget,
   three reciprocal fresh-worker blocks, same-path 5,000-row/three-tree fit
   plus 256-row prediction warmup.
7. **Command:** `python benchmarks/run_m1_q0_wave1.py --campaign q0
   --darkofit-source /private/tmp/darkofit-wave1-source-726e5d8
   --chimeraboost-source /Users/konstantinmedvedovsky/code/chimeraboost`.
8. **Artifacts:** raw
   [`q0_wave1_profile.json`](q0_wave1_profile.json) SHA-256
   `9111f14ae4d0d89e122f541b53f85c76c6bd5e76f4fa781c69039c1020c04e1c`;
   protocol SHA-256
   `7b25851753f83916c8dd542d8dd0f8d569c5b871b9ef38cb8e933f0f46ff2a34`;
   executed runner/analyzer SHA-256
   `793f764c7287a3007b20d83dc452917fd1ed56339195d508db71a5544ab8f179`;
   summary
   [`q0_wave1_profile_result.md`](q0_wave1_profile_result.md).
9. **Primary result:** eligible fused share was 52.37% at 500k and 62.61% at
   1M; the frozen 1.30x-kernel projection implied ratio `0.867242`, or 13.28%
   lower end-to-end fit time, clearing the 10% screen.
10. **Gates:** all behavior, engagement, decomposition, sibling-inactivity,
    component-accounting, metadata, and stderr integrity checks passed.
11. **Limitations/deviation:** 40-tree numeric scalar profile only. The
    unfused reference was unexpectedly faster but cannot enter the production
    share. All workers and the immutable artifact write completed; a
    post-write disposition-print bug then produced a nonzero exit. The
    artifact was not rerun or replaced, reanalysis matched exactly, and
    `bb40018` fixes only that presentation path.
12. **Terminal decision:** Q is eligible for the G-M funding decision; no
    prototype, public option, or default change is authorized.

### 16. Wave 1 M1 current large-n comparison

1. **Execution boundary:** 2026-07-20; DarkoFit package source
   `726e5d8e6131c580bce948db833a5007d0692dca`; harness
   `c39c15e26ea545e19c822505ff0fbc345815aec2`.
2. **Comparator:** exact ChimeraBoost source
   `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`
   (`v0.18.0-6-gf14be60`).
3. **Evidence class:** Tier-E matched-capacity product-path characterization
   on deterministic synthetic data; spent; no fresh or sealed data.
4. **Data:** seed 20260717, 24 numeric features, scalar RMSE target, 500,000
   and 1,000,000 training rows plus the following 100,000-row holdout.
5. **Arms:** DarkoFit float, ChimeraBoost quantized, and the same
   ChimeraBoost source with `quantize_gradients=False`; 300 matched-capacity
   symmetric trees and all product selectors disabled.
6. **Resources:** arm64 macOS 26.5.2, 14 logical CPUs, fixed 14-thread budget,
   all six arm-order permutations at both sizes, fresh workers, same-arm
   5,000-row/three-tree fit plus 256-row prediction warmup.
7. **Command:** `python benchmarks/run_m1_q0_wave1.py --campaign m1
   --darkofit-source /private/tmp/darkofit-wave1-source-726e5d8
   --chimeraboost-source /Users/konstantinmedvedovsky/code/chimeraboost`.
8. **Artifacts:** raw [`m1_wave1.json`](m1_wave1.json) SHA-256
   `74fd4c9c85948a4c19664a57534e19be3efb0483c78c13767c2521194626eb7a`;
   protocol SHA-256
   `7b25851753f83916c8dd542d8dd0f8d569c5b871b9ef38cb8e933f0f46ff2a34`;
   runner/analyzer SHA-256
   `83690fa0873f017512e9d9c82f42a6be464547832b935786f627debbbb6ab2ab`;
   summary [`m1_wave1_result.md`](m1_wave1_result.md).
9. **Primary result:** DarkoFit/current-quantized-Chimera equal-size fit ratio
   `0.844722`; DarkoFit/float-Chimera `0.762876`; quantized/float Chimera
   `0.903595`.
10. **Gates:** all source, metadata, behavior, engagement, stderr, and timing
    stability checks passed. Quantized/float RMSE was within `1.002`, but the
    fit ratio missed the predeclared `0.90` material-donor threshold.
11. **Limitations:** current-machine matched capacity, not byte-identical
    preprocessing; 14 rather than the historical 18 threads; no 0.15-era arm,
    so no release-movement attribution.
12. **Terminal decision:** `no_material_quantization_donor_signal`; publish
    once, do not rerun or relax the threshold. Combined with Q0, the
    provisional G-M Q disposition is close/do-not-fund.

### 17. M6 release-anchor establishment

1. **Execution boundary:** 2026-07-20; clean harness `d509111`; ChimeraBoost
   `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`; CatBoost 1.2.10 with
   installed-wheel `RECORD` SHA-256
   `9c20fb35750d9ff814309323b225e836b538c1496745f357c8fd50187e7824ed`.
2. **Evidence class:** Tier-E spent release-anchor establishment; no fresh or
   sealed data and no shipping/default authority.
3. **Data:** the ten M6 adapter datasets at small and medium sizes, seeds
   0–2, unweighted and deterministic stress-weighted; 120 matched cells.
4. **Arms:** ChimeraBoost and CatBoost product defaults, fixing only four
   threads and the random seed.
5. **Resources:** one sequential fresh worker per product/cell and a
   same-product three-tree warmup outside timing.
6. **Artifact:** [`m6_release_anchors.json`](m6_release_anchors.json),
   SHA-256
   `59747bc08d48a2ddad9b3cec05c965ecbd9edf21025c537f17dc58d816385409`;
   summary
   [`m6_release_anchors_result.md`](m6_release_anchors_result.md).
7. **Integrity:** 240/240 rows succeeded, data fingerprints matched within
   every product pair, and worker stderr was empty.
8. **Descriptive result:** CatBoost/ChimeraBoost primary-loss geometric mean
   `0.841814`; CatBoost/ChimeraBoost fit-time geometric mean `3.343943`.
9. **Decision:** exact release anchors are established and hash-bound, so the
   M6 contract is frozen. M6 remains ineligible to rank candidates until its
   separately predeclared historical backtest completes.

### 18. M6 historical-backtest failure

1. **Execution boundary:** 2026-07-20; frozen executor commit `59f7613`;
   exact historical DarkoFit sources `1016e7e`, `e961bcc`, and `29bd30c`;
   ChimeraBoost 0.15 at `851ab7f`.
2. **Evidence class:** Tier-E spent infrastructure backtest; no shipping or
   default authority.
3. **Declared subset:** fused variable Hessian (`advance`), forest-work
   packed router (`kill`), and 3% linear-leaf selector (`kill`).
4. **Execution:** three no-outcome launch failures were recorded separately.
   On the only outcome-bearing launch, the exact fused replay completed and
   the frozen analyzer returned `kill`, disagreeing with the known positive
   verdict.
5. **Packed boundary:** the exact runner called
   `numba.set_num_threads(18)` before loading data; the current runtime allows
   at most 14 and raised `ValueError`. This is `lacks_power`, not agreement.
6. **Selector boundary:** not run after the terminal prerequisite failure.
7. **Artifact:** [`m6_historical_backtest_failure.json`](m6_historical_backtest_failure.json),
   SHA-256
   `18b902e6099a4686b8eda71fac9ac327a0b5243872b80b5da79c5e01e5e2c201`;
   summary
   [`m6_historical_backtest_result.md`](m6_historical_backtest_result.md).
8. **Limitation:** the combined fail-closed executor kept replay shards in a
   temporary directory and the later packed exception removed the fused raw
   shard. The emitted analyzer disposition is preserved; it is not
   reconstructed by rerunning spent evidence.
9. **Terminal decision:** `backtest_complete=false`,
   `backtest_terminal=true`; M6 remains non-ranking and v3 reruns are closed.

### 19. M5 diversity-sentinel baseline

1. **Execution boundary:** 2026-07-20; clean harness `682dddf`; exact
   post-H1 control `726e5d8`; control/candidate package tree
   `e1fe956f32df0440e321805511ae2d96e383735c`.
2. **Evidence class:** Tier-E non-ranking drift baseline; no fresh or sealed
   data and no shipping/default authority.
3. **Data:** 19 cells across grouped/entity, smooth, noisy,
   categorical-missing, high-row, binary, multiclass, weighted regression,
   and weighted classification domains; 38 paired fresh-worker rows.
4. **Invariants:** finite task-appropriate output, normalized loss at most
   `1.10` times a train-only trivial predictor, exact save/load predictions,
   pinned data/splits, resolved metadata, and behavior fingerprints.
5. **Known floors:** binary df1/647 mean/worst excess Brier
   `0.00314483/0.00343480`; multiclass df1/077
   `0.00009772/0.00013069`, both inside mean `0.005` and worst `0.01`.
6. **Resources:** four threads, one fresh worker per arm/cell, alternating
   order, same-source three-tree warmup outside timing. Median
   candidate/control fit, predict, and RSS ratios were
   `0.998818/1.001768/0.995723`.
7. **Integrity:** 38/38 rows succeeded, paired behavior fingerprints matched,
   all serialization roundtrips were exact, and worker stderr was empty.
8. **Artifact:** [`m5_sentinel_baseline.json`](m5_sentinel_baseline.json),
   SHA-256
   `0971e06d4ed307d352d75e1e6400b849c0001b5e11f40243173d7080b6c5859d`;
   summary
   [`m5_sentinel_baseline_result.md`](m5_sentinel_baseline_result.md).
9. **Decision:** M5 v1 is frozen. It detects hard failure and unexplained
   drift; it does not rank or accept mechanisms.

### 20. Wave 1 M3a shipped-ensemble comparison

1. **Execution boundary:** 2026-07-20; exact post-H1 DarkoFit package source
   `726e5d8e6131c580bce948db833a5007d0692dca`; clean frozen harness
   `dae36ac435064722f955ed8d0d4586dae1a26d2d`.
2. **Comparator:** exact ChimeraBoost source
   `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`, version 0.18.0.
3. **Evidence class:** Tier-E quality-first characterization on spent data;
   no fresh or sealed data, no default or cross-season claim.
4. **Data:** frozen sports-panel-v2 SHA-256
   `8f7eab3765b4166740b150ed372f9607bcd6dd9673e0e73cc6541583230a59e6`;
   three seasons x three targets, exact player-disjoint folds, creator-fold
   diagnostic, held-team/seen/cold views. General context used six fixed
   medium M6-adapter regression cells (three datasets x seeds 0/1).
5. **Arms:** DarkoFit single, group8, row5/8, and group5; ChimeraBoost
   quantized single/ensemble8 and float single/ensemble8. DarkoFit row arms
   and ChimeraBoost subagging are player-overlap exposed internally; DarkoFit
   group arms are player-disjoint.
6. **Resources:** arm64 macOS, 14 logical CPUs, fixed 14-thread budget, one
   fresh worker per arm, same-arm warmup, sampled aggregate parent-plus-child
   RSS. Repeat timing was conditional on the frozen group8 quality gate.
7. **Commands:** `python benchmarks/run_m3a_wave1.py --phase
   primary-quality --output
   /private/tmp/darkofit-m3a-primary-quality-dae36ac.json`; then the
   analogous `--phase diagnostics` command. The analyzer combined the two
   shards; no `primary-repeats` command was authorized or run.
8. **Artifacts:** contract SHA-256
   `930d405e75947747892337851d2a767a0a789069ad86eaf726d7396fb0a435b8`;
   protocol `965296a3ef1bd1547a93e5061cacad20885ac6c015ae00ce73a48c816f2833a6`;
   runner `f971b28a83d6a14291141dbc0a657ab28f95fb31e9459a77bd72a42c942c8cd7`;
   analyzer `876e79aab5f05263d8acb779d9004891785f0ef1a72358f832c26ca04f176dba`;
   primary shard
   `3671d12eaa2e2647ff2677304173fec339eb2c5197c39bf8211a1ff5042e00fd`;
   diagnostic shard
   `a9452207e81a4891d1ba47592a5be488998e782f4dc1d54f01e885d8ea4eb9bf`;
   combined [`m3a_wave1.json`](m3a_wave1.json) SHA-256
   `c811c8b04cbbaff6edb8226d7e8f5dbac3f9229adf18c3f8b658129ba7fc459a`;
   summary [`m3a_wave1_result.md`](m3a_wave1_result.md).
9. **Primary result:** DarkoFit group8/single player-disjoint ratio
   `1.025482`, clustered p95 `1.032391`, held `1.016048`, cold `1.015661`;
   fit/predict/model-bytes/RSS ratios
   `4.770486/3.898647/3.929268/1.091085`. ChimeraBoost ensemble8/single was
   `0.950230` on sports with 9/9 wins and `0.947797` on the six general cells
   with 6/6 wins.
10. **Gates:** all source, target, grid, RSS, and stderr integrity checks
    passed. DarkoFit group8 failed the aggregate, clustered, held, cold,
    worst-season, and worst-cell quality gates; all four bounded-cost checks
    passed.
11. **Limitations/non-claims:** only three spent season clusters; creator
    folds are overlap exposed; the general slice is small and M6 remains
    non-ranking; primary costs are single descriptive observations because
    quality failed; no unseen-season, broad-panel, or default claim.
12. **Terminal decision:** M3a closes the current DarkoFit ensemble route and
    forbids repeat timing. G-M separately funds a new private B0/B1/B2
    mechanism attribution based on the predeclared ChimeraBoost donor arm;
    see [`wave1_gm_decision.md`](wave1_gm_decision.md).

### 21. Wave 2 M3b private ensemble-v3 attribution

1. **Execution boundary:** 2026-07-20; exact corrected private-model source
   `6d063f98128d457f8b8bbf610c7aec46e675d844`; successor harness source
   `74ac6cc32adcc3ece3179ffd9a77d34517906c6d`; clean execution head
   `826fe82d3738d1a5dd57f4fb3e2fab79fa83ea8e`, which adds only the frozen
   create-only contract to that harness.
2. **Comparator:** internal private controls at the same source pin: an
   eight-member bootstrap/base-policy control plus a single-model reference.
   No external-library outcome is part of the acceptance decision.
3. **Evidence class:** Tier-E spent private mechanism attribution; no fresh,
   TabArena, or sealed-lockbox data and no public/default claim.
4. **Data:** 13 fixed medium cases: nine player-disjoint sports cells (three
   seasons x three targets) and four general numeric/categorical,
   regression/binary/multiclass cells with stress weights. Exact data, split,
   weight, case, and panel-cache fingerprints are bound in
   [`m3b_ensemble_v3_r3_contract.json`](m3b_ensemble_v3_r3_contract.json).
5. **Arms:** single reference; bootstrap/base-policy control; B1 80%
   without-replacement sampling with base policy; B2 bootstrap with named
   `donor_balanced_v1` member policy; and the combined B1+B2 arm. Group cases
   use group-disjoint train/OOB partitions; all eight members fit
   sequentially.
6. **Resources:** arm64 macOS; fixed four-thread worker contract; fresh worker
   per case/arm/repeat; same-arm two-round warmup outside measurement;
   600-round fits with patience 30; peak self-worker-process RSS sampled at
   10 ms. Quality ran once for all 65 rows; two timing repeats produced 130
   rows after every candidate cleared the frozen quality gate.
7. **Execution:** source-attested
   [`run_m3b_ensemble_v3_r3.py`](run_m3b_ensemble_v3_r3.py) ran `quality`
   then, after create-only gate analysis, `timing`; source-attested
   [`analyze_m3b_ensemble_v3_r3.py`](analyze_m3b_ensemble_v3_r3.py) created
   the gate and final result. All formal outputs were create-only in
   `/private/tmp` before their exact bytes were copied into this directory.
8. **Artifacts:** protocol SHA-256
   `c7663573fb2f49ccc6ba42e4b633577192c8961c4001f4c86aeb341d1b264409`;
   runner `c37b445fcb4ba9959cb972562f1d60cef4f9385a945f5362e6f5c42674cc1b15`;
   analyzer `e7a2c801e3b42c5a851ad64b186f7fc7311a308fe10070d83b16505ceceae3ef`;
   contract `5889e130c7afbadbc8e0f082673eb1a80961b5cb396ca906efbe9a5d32ea8b50`;
   quality `5fec218cbc0ec97ef4b3fec10f65a89131a377cf026dbb80da809d6396ead6c3`;
   gate `68c40a92a75ed9c8288445bb0fd46677e1bc9178d468ff4c354da02492f2ba68`;
   timing `7048041c7a5edb2ec83c34920a657d6f0286946d216837bc2d482c36117e032e`;
   result `3e6d0750e772c156b6c4daed948eb6baa640564ce87fe1ffee7414b3fe03c8bc`;
   summary `3095b84ee93edbbb53e8c01ae635591ea1a5400945f34054563216e618070421`.
   Attempt-1 and attempt-2 terminal artifacts and failure records are also
   preserved beside the final evidence.
9. **Primary result:** B1/control aggregate loss `0.987744`, fit `1.041601`,
   archive/single `7.842030`; B2/control loss `0.996393`, fit `0.690735`,
   archive/single `5.905469`; combined/control loss `0.979638`, general loss
   `0.984542`, sports cold `0.977466`, sports held `0.976667`, worst cell
   `1.008081`, fit `0.557873`, predict `0.753550`, archive `0.708856`, and
   RSS `0.976841`. Combined/single archive was `5.534767` and RSS `1.069201`.
10. **Gates:** source, grid, fingerprint, runtime, prediction, probability,
    serialization, fitted-metadata, OOB, warning, RSS, and quality gates all
    passed in attempt 3. B1 and B2 failed both their arm-specific value and
    common archive/single checks. Combined passed every quality/value and
    other resource check but failed archive/single (`5.534767` observed;
    `4.0` maximum). No candidate survived all final checks.
11. **Limitations/non-claims:** all data are spent; the general slice has four
    medium cells and the sports slice has only three seasons; RSS covers the
    sequential worker process because B3 child workers do not exist; the
    prototype is private; the result neither validates a general ensemble
    default nor authorizes tuning on these outcomes. Attempt 1 ended before
    model fit on sandbox-denied process-tree RSS. Attempt 2 discarded one
    completed, uninspected row after a group-bootstrap safe-load defect; the
    corrected loader received a new source and contract identity.
12. **Terminal decision:**
    `close_b1_b2_preserve_existing_opt_in`. Retain no private arm; preserve
    the existing shipped opt-in; do not proceed to B3, a public/default
    surface, fresh confirmation, TabArena, or lockbox access. Any successor
    requires a distinct mechanism and a new prospective contract.

### 22. Wave 3 B-archive exact-factoring feasibility

1. **Execution boundary:** 2026-07-21; exact model source
   `858ac14c30e280491d7bd5232da56f7050561782`; v1 harness
   `7232c42984b838403b1941a2e73022fcc412a0ee` and clean execution head
   `239cf58fdff112ac8ba7eace2eaeaa689c254153`; corrected v2 harness
   `05086b6a2dfd90db2c88886ed30f9f1b8a5558c1` and clean execution head
   `8fcc478fa1b7eebfec47f9bf0c6c852dfc7cf1f5`. The model source ran from a
   detached, clean worktree.
2. **Comparator:** a newly fitted matched single model for each case. The
   candidate is not a serializer: it is the non-loadable byte-size model
   obtained by factoring only the complete canonical numeric target-free
   preprocessing section from the newly fitted eight-member combined B1+B2
   archive. Member-local cases retain current archive bytes.
3. **Evidence class:** prospectively frozen Tier-E spent size feasibility.
   The M3b r3 result remains immutable; the dated matched-single readout is
   post-hoc development context, not a frozen M3b promotion comparator. No
   fresh, TabArena, or lockbox data were accessed.
4. **Data:** exactly the 13 M3b cases and frozen data/split/weight/case
   fingerprints: nine player-disjoint cold-player sports cells within held
   teams and four medium general regression/binary/multiclass cells with the
   original stress weights. Eleven cases carry `numeric_target_free`
   provenance; the two categorical cases are member-local.
5. **Arms:** one-member reference and eight-member combined B1+B2: 80%
   without-replacement row/group sampling, `donor_balanced_v1`, group-safe
   OOB where applicable, requested shared preprocessing, seed 4, 600 maximum
   rounds, patience 30, sequential fitting. The effective simulation factors
   exactly seven declared `prep__*`/`bin__*` arrays and the exact three-field
   NumPy-input preprocessing header; it uses no categorical, encoder, tree,
   SHAP, wrapper, target, or generalized delta state.
6. **Resources:** arm64 macOS; frozen Python 3.11.8, NumPy 2.2.6,
   scikit-learn 1.7.1, Numba 0.61.2, pandas 2.2.3, and SciPy 1.15.1; fixed
   four-thread paired-evidence environment; one fresh worker per case. This
   campaign measures deterministic archive bytes, not elapsed time or RSS.
7. **Execution:** v1 ran source-attested `run_barchive_v1.py` and terminated
   on its first post-fit header invariant before emitting a row. V2 bound that
   terminal lineage under a new identity and ran
   `conda run -n darko311 env PYTHONPATH=. python benchmarks/run_barchive_v2.py --source /private/tmp/darkofit-barchive-source-858ac14`;
   after committing the raw artifact, `analyze_barchive_v2.py` validated and
   decided it from a clean tree. Outputs were create-only.
8. **Artifacts:** v1 protocol SHA-256
   `85feb7e2e804d666eddea412a0d80e502d099cfbc19ec2559af4e4f7f75c0c22`,
   runner `a9885c319a69c7dd1cd839d51de71974f1f4620bf0e49d8022eb72bb1962f729`,
   analyzer `f1cb8da0b30bb695174a843057b0aad31984bad9830aa9e039def4e70b6d62f4`,
   contract `1cddcbe679521053f8912c21730fc940e581944d35bb895f45694fc2c49676a5`,
   terminal `11e06490d1c9e90a2bac5c46f01009b3348942e8ad0a83606db08551aa1697d0`,
   and failure record
   `457aa71e5d60580c9e1875b44ab61b7676e68dd02ad5da1dba2681a15e29a803`.
   V2 protocol SHA-256
   `dee7c4423b5908b16f839cb28f3f097882f1e1a7fa3e1da5723f2ac4101c289f`,
   runner `e0b1abb6986ae601e86b6b99bba4213cfc2d7e760c0ba06c2b9794bc65e6befb`,
   analyzer `a1a47c7589ffb0ad4b371d5927b2d1deb10f37ee66f69286ce22fedf5535363f`,
   contract `1466342e58a0b480a94121023184674d4ec10654ef19bb80a2b03c5f6742e9f0`,
   raw `3b8eebc0140dc399a995089dee1097627f0c8cc13ff7d8fd37d00e0524355509`,
   result `8773d152b81bc5236e66f5cdfc1deb60d5059e7d01582a7a6721f3128ce9ff7b`,
   and original Markdown note
   `2a3a0e9529ac70ec97ff8624f06fef6fef03f3036e847c00f8292200431e7e6f`.
   The note's inherited v1 heading is preserved and narrowly corrected by the
   hash-bound erratum
   `ed8d7cb62827379d55391e07f3e9cd03278cc0132a862c12784477ec96570234`.
9. **Primary result:** all 13 current combined/single archive ratios had
   median `6.032405`. Applying only the allowed complete canonical section to
   the 11 eligible cases and leaving both member-local cases unchanged gave
   median effective archive/single `4.152525`. Per-case effective ratios
   ranged from `2.193001` to `9.748762`; the frozen decision uses only the
   predeclared median, not that range.
10. **Gates:** v2 passed source, clean-tree, runtime, case-manifest,
    fingerprint, thread, safe-load prediction/probability, feature-schema,
    constructor/fitted-metadata, deterministic-resave, canonical provenance,
    exact seven-array/eight-member byte-identity, header, no-out-of-scope,
    non-growth, member-local, completeness, and create-only publication
    checks. It failed the sole size decision gate: `4.152525 > 4.0`.
11. **Limitations/non-claims:** the simulation is intentionally non-loadable;
    data are spent; the general slice has only four medium cases and sports
    only three seasons; archive ratios depend on newly fitted stopping
    horizons and matched-single sizes. V1's only observed campaign fact was
    the header-contract mismatch; it published zero rows and was not rerun.
    V2 does not validate or authorize a serializer, ensemble default, B3,
    M2, TabArena, fresh confirmation, or lockbox access.
12. **Terminal decision:**
    `close_barchive_nominate_fused_lane_dispatch`. B-archive is closed; do not
    implement canonical serialization or generalized member deltas. Promote
    behavior-exact fused-lane dispatch to the next Track I mechanism slot,
    where it still requires a new prospective contract before implementation.

### 23. Wave 4 fused-lane dispatch staged implementation and calibration freeze

1. **Execution boundary:** 2026-07-21; staged product implementation
   `eb2b6cf4b0f927316d82450eeaeef60e961cfad8`; outcome-blind harness and
   measured-source pin `0e67eb157c79e2e42171bd1c779210d6cf1909ec`;
   create-only calibration-contract commit `518aede`. No calibration or
   validation worker has run and no campaign outcome has been opened.
2. **Comparator:** the existing forced-fused histogram-plus-split kernel is
   the prospective control; the existing forced-unfused builder-plus-search
   kernel is the prospective candidate lane. This checkpoint contains no
   measured comparison and makes no speed claim.
3. **Evidence class:** product verification plus a prospectively frozen,
   outcome-unopened Tier-E kernel-calibration identity. It is generic
   synthetic infrastructure, not sports, quality, M2/M4, Q re-entry, fresh
   confirmation, or lockbox evidence.
4. **Data:** the unexecuted calibration grid is the frozen 30-coordinate
   Cartesian product of five row counts, three feature/thread shapes, and
   unit/variable Hessians at depth 6 and 128 realized bins, seed `20260721`.
   Generators and required per-array hashes are source-bound; no raw artifact
   exists.
5. **Arms:** seven paired alternating-order repetitions of forced fused and
   forced unfused after two untimed warmups per lane. The product surface is
   `oblivious_kernel={"auto","fused","unfused"}` with an absent automatic
   threshold, so effective `auto` behavior remains fused.
6. **Resources:** exact frozen fingerprint: Apple M4 Pro, hardware model
   `Mac16,7`, 14 physical/logical CPUs, 24 GiB RAM, Darwin `25.5.0`, Python
   3.11.8, NumPy 2.2.6, Numba 0.61.2, and llvmlite 0.44.0. Workers fix every
   common thread variable to 4, 9, or 14 by coordinate and require unchanged
   Numba ceiling/current state.
7. **Execution:** only
   `python benchmarks/freeze_fused_lane_dispatch_calibration.py` ran, from a
   clean tree, to create the non-authorizing contract. Formal execution would
   require a separate clean detached source worktree, the contract's exact
   authorization/raw/terminal paths, and an explicit hash-bound owner record.
   Alternate output names and copied authorization paths are rejected.
8. **Artifacts:** design v1 SHA-256
   `68d0dd6ef42f29d164943ef16e766821c5bd53319840b22a59b1bd449191cf1a`;
   design v2 `ed032758dfa5829766ae324bdde54b9a1724ed0063d3997f55f3d72f7907240e`;
   bin erratum `18847b3118f6873d68bed57b9730ebd18fb07b8418260a37059dc1b3700217db`;
   protocol `4f85112d89845f13320fd1758e98dbaae59b9ca7cc870203177c0ceee0f93d76`;
   campaign/analyzer `034893ef265f8c3c2a9d6ab62368d0b9c47076e23656005b31a21551ba81ac73`;
   runner `721a8d71654b732c346e47ad81ca6b14317c8af213529d0938b0c8bf0762a691`;
   freezer `41ef4bcea83da7d772d9865f782be6401cd8953f983bec5b1404a0a63c3ffc2c`;
   and execution contract
   `3d7f8a653a71d6a9712f57f51bb01421765b42fcd105902f1fb0c6a611f7712d`.
9. **Primary result:** none. Calibration fit ratios, regret, threshold, and
   qualification are unopened. Product/golden verification currently passes
   109 focused dispatch, kernel, thread, loss, and evidence-contract tests;
   the committed contract reloads with all bound hashes and the exact runtime.
   The full local checkout run passed 2,915 tests with 30 skips; its sole
   failure was the pre-existing Panel 3 historical-sibling guard because the
   clean neighboring ChimeraBoost checkout is now at `919a80a`, not Panel 3's
   frozen `851ab7f` source. The guard was preserved and no sibling state was
   changed.
10. **Gates:** staged API, eligibility, deterministic resolution, persistence,
    tamper rejection, exactness projection, thread restoration, source pin,
    runtime fingerprint, formal-path, create-only, and owner-authorization
    invariants pass. The calibration exactness, stability, `0.97` geomean,
    `1.02` worst-cell, mixed-lane, and every validation gate remain untested.
11. **Limitations/non-claims:** no speed, crossover, portability, quality,
    sports, release, or default-change claim follows. The campaign is scoped
    to the recorded Apple-arm64 host and declared envelope. B remains closed;
    Q must rebase only after a retained dispatch; the next mechanism slot is
    quality-first regardless of this campaign's result.
12. **Current decision:** `await_explicit_calibration_authorization`. The
    authorization, raw, terminal, analysis, threshold, and validation-contract
    artifacts do not exist. Without a matching owner record, the runner must
    refuse execution and Wave 4 remains paused at this frozen checkpoint.

_The execution identity in this checkpoint was superseded pre-outcome by the
corrected v2 identity below. Its frozen contract remains immutable._

### 24. Wave 4 calibration-v2 pre-outcome CI supersession

1. **Execution boundary:** 2026-07-21; GitHub Actions run `29851074232`
   exposed the v1 issue before any formal calibration worker started. The
   host-independent product test is commit `5292c7d`; corrected harness/source
   is `cf6a667cff5eaa2d36b9c16c9304470e0feac083`; v2 contract commit is
   `4ff11fe`.
2. **Comparator:** unchanged prospective forced-fused control and
   forced-unfused candidate. No timing comparison occurred.
3. **Evidence class:** product-test portability correction and
   prospectively frozen, outcome-unopened Tier-E execution identity. The v1
   scientific campaign remains the same; only its formal execution identity
   advances to `calibration_v2`.
4. **Data:** unchanged 30 synthetic calibration coordinates, generators,
   seed, array fingerprints, and six separately frozen validation cells. No
   raw row exists under v1 or v2.
5. **Arms:** unchanged forced fused/unfused calibration lanes, warmups, paired
   order, repeat count, selector family, and absent automatic threshold.
6. **Resources:** the v1 contract's M4 Pro runtime remains the exact v2 formal
   runtime. The discovered failure was on a Linux Python 3.13 CI library lane:
   the test expected `rows_outside_envelope` without pinning the platform, so
   Linux correctly returned the earlier `unsupported_platform` reason.
7. **Execution:** the corrected test pins Darwin, arm64, and 14 logical CPUs
   before fitting. Only product/infrastructure tests and
   `python benchmarks/freeze_fused_lane_dispatch_calibration_v2.py` ran. V2
   adds a required authorization `execution_identity` field and unique
   create-only v2 authorization/raw/terminal/analysis paths.
8. **Artifacts:** immutable v1 contract SHA-256
   `3d7f8a653a71d6a9712f57f51bb01421765b42fcd105902f1fb0c6a611f7712d`;
   v2 protocol `c07e9565e8f337317c6c564b03cf9d8cd60af684be4ed7c487114d17c1e91e8c`;
   v2 freezer `44e746f117151ff199ee9fd5428bec88fe5dae93e9f0d8f52a6610263adcbf30`;
   corrected runner `0248d8182758d95d0c37ef96d3591530170cf12735b4728f4ad7e6f93d9c6a0f`;
   product test `b08e55a77dff29d5317b61fb1ecd8896da989953b7004681af7e103c9cb8c3e5`;
   campaign-contract tests
   `9f478a02affb02bf64edf844634309a95b6b3d4d06b8ee44cf6070e8b751f169`;
   and v2 execution contract
   `b2075f9c45df3b3fb674c74fe0b47cd9ddd1ec3bae790f5379308e15a327061a`.
9. **Primary result:** none; every performance outcome remains unopened. The
   corrected focused matrix passes 110 tests. The committed v2 contract loads
   with its exact hashes/runtime, while the superseded v1 contract is rejected
   by current code because its bound product-test hash no longer matches.
10. **Gates:** the platform-independent expected-reason test, distinct
    execution identity, immutable v1 binding, unique paths, authorization
    identity, contract load, and non-authorization checks pass. Every
    calibration/validation outcome gate remains untested.
11. **Limitations/non-claims:** this is not a failed calibration attempt and
    does not justify a rerun allowance, speed claim, threshold, default
    change, portability claim, release, M2/M4, Q, fresh data, or lockbox use.
    No scientific threshold or coordinate changed in response to an outcome.
12. **Current decision:** `await_explicit_calibration_v2_authorization`. V1 is
    closed before execution. V2 records `execution_authorized=false` and
    `outcomes_opened=false`; no authorization or result artifact exists. The
    next mechanism slot remains quality-first regardless of Wave 4's result.

_This execution identity was also superseded before authorization, formal
worker execution, or outcome access. Its immutable contract remains part of
the v3 lineage below._

### 25. Wave 4 calibration-v3 pre-outcome evidence-gate supersession

1. **Execution boundary:** 2026-07-21; independent review occurred before any
   v2 authorization or formal worker. Gate repair is commit
   `a8cec9b4efbfd4afdcf1345c4088c7d17162eeb8`; prospective v3 protocol/freezer
   is `691f5a3a110c6d5a0f7a17ed3b6f04e296c97419`; immutable v3 contract commit
   is `4a07594716cabffba8fab8d4ea981480bb3e1b0e`.
2. **Comparator:** unchanged prospective forced-fused control and
   forced-unfused candidate. No timing comparison occurred.
3. **Evidence class:** pre-authorization harness/product provenance repair and
   a prospectively frozen, outcome-unopened Tier-E execution identity. This is
   neither campaign performance evidence nor permission to execute.
4. **Data:** unchanged 30 synthetic calibration coordinates, generator seed,
   array fingerprint rules, and separately frozen six-cell validation design.
   No raw row exists under v1, v2, or v3.
5. **Arms:** unchanged lanes, two untimed warmups, seven paired
   alternating-order repetitions, selector family, timing region, threshold
   candidates, tie rule, acceptance limits, and absent automatic threshold.
6. **Resources:** v3 carries forward v2's exact M4 Pro runtime and complete
   per-thread environment records, including absolute cache paths ending in
   `calibration/threads-{4,9,14}`. The repair enforces those records rather
   than introducing per-coordinate cache directories.
7. **Execution:** only correctness/infrastructure tests and
   `python benchmarks/freeze_fused_lane_dispatch_calibration_v3.py` ran. The
   freezer ran in a clean detached worktree at the exact source pin because
   unrelated documentation edits were present in the main checkout. No
   calibration or validation worker ran. Direct workers now self-load the
   canonical contract and owner authorization, reject off-grid coordinates,
   and validate the exact frozen environment before product import or timing.
8. **Artifacts:** immutable v2 contract SHA-256
   `b2075f9c45df3b3fb674c74fe0b47cd9ddd1ec3bae790f5379308e15a327061a`;
   v3 protocol `ee60b26183cc7003234c90301b7148f4661955664843c76162cb224a11e973fe`;
   v3 freezer `4410a18d76a3a89ec4154017d313a6912df7fa9602c115eaf0ba0c2b6443ab36`;
   repaired runner `6a6f8fbb3026ebad7f9158e62fc97b723343406d6d0ae837f17551911fa355d7`;
   and v3 execution contract
   `c55ee50fccda5b9ba24e004ae8a27285e4db92e52a9c17a668bc1b417b0fa648`.
9. **Primary result:** none; every speed ratio, regret, crossover threshold,
   and qualification outcome remains unopened. The repaired focused matrix
   passes 120 dispatch, kernel, private-provenance, and contract tests. The
   CI-equivalent local library partition passes 1,210 tests with 2 skips and
   the four immutable historical M3b modules excluded exactly as in CI. The
   campaign partition passes 1,693 tests with 27 skips after deselecting the
   one documented Panel 3 historical-sibling guard; the unfiltered run
   reproduced only that guard because the neighboring ChimeraBoost checkout
   differs from its frozen historical HEAD.
10. **Gates:** direct-worker authorization, copied-authorization rejection,
    exact coordinate binding, exact complete worker environment, actual
    builder-counter truth, zero-iteration semantics, hash-bound threshold
    analysis, normalized group-code persistence, plausible group-count
    forgery rejection, deterministic safe re-save, and all prior product
    invariants pass. All performance and validation outcome gates remain
    untested.
11. **Limitations/non-claims:** private group codes establish structural
    consistency, not cryptographic authenticity of original-fit history. No
    speed, crossover, portability, quality, sports, release, public/private
    ensemble promotion, default change, M2/M4, Q, fresh-data, or lockbox claim
    follows. V1 and v2 remain immutable pre-outcome predecessors.
12. **Current decision:** `await_explicit_calibration_v3_authorization`. V3
    records `execution_authorized=false` and `outcomes_opened=false`; no v3
    authorization, raw, terminal, analysis, threshold, or validation-contract
    artifact exists. The next mechanism slot remains quality-first regardless
    of Wave 4's result.

_This execution identity was also superseded before authorization, formal
worker execution, or outcome access. Its immutable contract remains part of
the v4 lineage below._

### 26. Wave 4 calibration-v4 pre-outcome capability/layout supersession

1. **Execution boundary:** 2026-07-21; a second independent review occurred
   before any v3 authorization or formal worker. Correctness repair is commit
   `1115bdd621d3dfe5612f22f9198c5a995b8eaed6`; prospective v4
   protocol/freezer is `05a7a0d996fcd797cbc925026f7eba48db4becd5`;
   immutable v4 contract commit is
   `6230993635bd20c645457be27225dc6b40de9e7e`.
2. **Comparator:** unchanged prospective forced-fused control and
   forced-unfused candidate. No timing comparison occurred.
3. **Evidence class:** pre-authorization worker-capability, fitted-constructor,
   weighted-class, and timing-layout repair plus a prospectively frozen,
   outcome-unopened Tier-E execution identity. This is neither performance
   evidence nor permission to execute.
4. **Data:** unchanged 30 synthetic calibration coordinates, generator seed,
   array fingerprints, and separately frozen six-cell validation design. No
   raw row exists under v1, v2, v3, or v4.
5. **Arms:** unchanged lanes, two untimed warmups, seven paired
   alternating-order repetitions, selector family, threshold candidates, tie
   rule, acceptance limits, and absent automatic threshold. Both timed lanes
   now share production's Fortran routing/histogram view; the canonical input
   and fingerprint are unchanged.
6. **Resources:** v4 carries forward v3's exact M4 Pro runtime and complete
   per-thread environment records, including the same absolute cache paths.
   No runtime or resource limit changed.
7. **Execution:** only correctness/infrastructure tests and
   `python benchmarks/freeze_fused_lane_dispatch_calibration_v4.py` ran. The
   freezer ran in a clean detached worktree at source `05a7a0d` because
   unrelated documentation edits remained in the main checkout. Each formal
   parent must now revalidate contract, authorization, clean source, unused
   raw/terminal paths, and coordinate before issuing a one-use inherited-pipe
   capability. Workers reject authorization alone before case generation.
8. **Artifacts:** immutable v3 contract SHA-256
   `c55ee50fccda5b9ba24e004ae8a27285e4db92e52a9c17a668bc1b417b0fa648`;
   v4 protocol `828d063b524fe7dd622905dcfb076269c71c75ebe0582b53b820dfd1bbb3eb33`;
   v4 freezer `db7e842cfd2689d04a9c36f9cbb2acbcb930dfc65f7583255fa7666860e6dae6`;
   repaired runner `4133a24e749689de393878c1f1845714e6f5b8224fc0e80ff41719147d7ec97e`;
   repaired public API
   `80a66c7abe5ec6971bb74850530817a2de08e3232783b77484a88217f519e1fa`;
   and v4 execution contract
   `fab0784beee165b4643b817f12076b79ff832d95224469bc244cc15c839e9c7f`.
9. **Primary result:** none; every speed ratio, regret, crossover threshold,
   and qualification outcome remains unopened. The focused repaired surface
   passes 206 tests. V4/v3 freezer plus campaign-contract tests pass 25 tests.
   The CI-equivalent local library partition passes 1,217 tests with 2 skips
   and 4 deselections. The complete local suite passes 2,936 tests with 30
   skips after deselecting only the documented Panel 3 historical-sibling
   HEAD guard.
10. **Gates:** parent capability issuance and binding, authorization-only
    rejection, exact coordinate/environment/source/output binding,
    production-equivalent routing layout, wrapper/member/booster
    `oblivious_kernel` consistency, positive-mass-only weighted class safety,
    actual builder counters, safe serialization, and all prior product
    invariants pass. All performance and validation outcome gates remain
    untested.
11. **Limitations/non-claims:** the inherited capability enforces the formal
    runner boundary; it is not a security boundary against an operator who can
    rewrite and execute arbitrary repository code. Safe-NPZ integrity remains
    structural rather than externally signed. No speed, crossover,
    portability, quality, sports, release, default, M2/M4, Q, fresh-data, or
    lockbox claim follows. V1--v3 remain immutable pre-outcome predecessors.
12. **Current decision:** `await_explicit_calibration_v4_authorization`. V4
    records `execution_authorized=false` and `outcomes_opened=false`; no v4
    authorization, raw, terminal, analysis, threshold, or validation-contract
    artifact exists. The next mechanism slot remains quality-first regardless
    of Wave 4's result.

_This pre-authorization decision was superseded by the create-only owner
authorization checkpoint below; the frozen scientific contract did not
change._

### 27. Wave 4 calibration-v4 owner authorization and pre-execution pause

1. **Execution boundary:** 2026-07-21; owner authorization was recorded in
   commit `9ed122facc849ecb2816e240e15d2fc07e1def93` and published to
   `origin/main`. No formal calibration worker started.
2. **Comparator:** unchanged prospective forced-fused control and
   forced-unfused candidate. No timing comparison occurred.
3. **Evidence class:** create-only owner authorization for the already-frozen,
   outcome-unopened Tier-E `calibration_v4` execution identity. This is
   permission to execute, not performance evidence.
4. **Data:** unchanged 30 synthetic calibration coordinates, generator seed,
   and array fingerprints. No raw row exists under v4.
5. **Arms:** unchanged lanes, two untimed warmups, seven paired
   alternating-order repetitions, selector family, threshold candidates, tie
   rule, and acceptance limits.
6. **Resources:** unchanged M4 Pro runtime and exact per-thread worker
   environments. The formal run remains subject to the exclusive-machine rule.
7. **Execution:** the authorization artifact was created, validated against
   the frozen contract, committed, and pushed. A prospective launch was not
   allowed to open calibration while Chrome Remote Desktop caused material
   background CPU contention.
8. **Artifacts:** frozen execution contract SHA-256
   `fab0784beee165b4643b817f12076b79ff832d95224469bc244cc15c839e9c7f`;
   create-only authorization SHA-256
   `42fb0ab01f8a7b271cda2610c59a953d5815e93657ca0a5ab3a003e38dfea775`;
   source `05a7a0d996fcd797cbc925026f7eba48db4becd5`. No raw, terminal,
   analysis, selected-threshold, or validation-contract artifact exists.
9. **Primary result:** none. Every speed ratio, regret, crossover threshold,
   and qualification outcome remains unopened.
10. **Gates:** authorization identity, source pin, contract hash, owner
    decision, and create-only path are valid. Performance and validation gates
    remain untested.
11. **Limitations/non-claims:** authorization does not imply qualification,
    speed, portability, quality, sports, release, default, M2/M4, Q,
    fresh-data, or lockbox evidence. The frozen contract correctly retains
    `execution_authorized=false`; the separate artifact carries authorization.
12. **Current decision:** `authorized_await_exclusive_machine`. Execute the
    one-shot `calibration_v4` identity only after material background CPU
    contention is absent; otherwise preserve the unused identity.

### 28. Post-authorization Wave 4 fresh-eyes audit

1. **Execution boundary:** 2026-07-21; repeated fresh-eyes review of the exact
   nine-commit range `369d512..9ed122f` and its 22 changed files, followed by
   a second pass over the resulting documentation correction.
2. **Comparator:** current disk state against the frozen v3/v4 contracts,
   create-only v4 authorization, source-bound implementation, campaign
   protocols, and regression tests.
3. **Evidence class:** correctness and evidence-contract audit. No benchmark
   timing or campaign outcome was generated or opened.
4. **Data:** synthetic unit fixtures only. The formal 30-coordinate calibration
   grid remained unused.
5. **Arms:** product dispatch, wrapper/booster persistence, private group
   provenance, worker capability, analyzer, freezer, and documentation status
   surfaces were inspected; no comparative model arm ran.
6. **Resources:** `darko311`; ordinary test execution only. No formal timed
   worker or exclusive-machine claim follows from these tests.
7. **Execution:** scoped harness/freezer/regression-test Ruff checks and
   repository `compileall` passed; the focused dispatch/private-ensemble
   matrix passed 220 tests; the CI-equivalent library partition passed 1,217
   tests with 2 skips and 4 expected deselections. The campaign partition
   passed 1,693 tests with 27 skips and reproduced only the documented
   historical Panel 3 sibling-HEAD guard.
8. **Artifacts:** all 32 v4 bound-file size/hash records, contract SHA-256
   `fab0784beee165b4643b817f12076b79ff832d95224469bc244cc15c839e9c7f`,
   authorization SHA-256
   `42fb0ab01f8a7b271cda2610c59a953d5815e93657ca0a5ab3a003e38dfea775`,
   authorization identity/source, and unused formal output paths were
   revalidated from disk.
9. **Primary result:** one confirmed documentation defect was found and fixed:
   the live plan still described v4 as unauthorized after the authorization
   commit had been published. No implementation defect was confirmed.
10. **Gates:** contract lineage, unchanged science, exact bound files,
    separate authorization semantics, unused create-only outputs, dispatch
    counters, persistence, group provenance, and worker capability checks pass.
11. **Limitations/non-claims:** the preserved Panel 3 failure reflects the
    neighboring ChimeraBoost checkout differing from its frozen historical
    HEAD; the audit did not weaken that guard. Calibration and validation
    remain unopened, and no speed, threshold, or shipping claim follows.
12. **Current decision:** `fresh_eyes_clean_authorized_unrun`. Preserve the
    authorized v4 identity until exclusive machine access is available, then
    execute it once under the frozen contract.

### 29. Wave 4 calibration-v4 binding execution and close

1. **Execution boundary:** 2026-07-21; the create-only owner-authorized
   `calibration_v4` identity ran exactly once from clean harness commit
   `9ed122facc849ecb2816e240e15d2fc07e1def93` against clean detached product
   source `05a7a0d996fcd797cbc925026f7eba48db4becd5`. No rerun occurred.
2. **Comparator:** forced-unfused candidate versus the forced-fused current
   production lane, paired inside each fresh coordinate worker.
3. **Evidence class:** binding Tier-E synthetic kernel calibration under the
   prospectively frozen v4 contract. It is generic engineering evidence, not
   sports, quality, portability, release, M2/M4, fresh-data, or lockbox
   evidence.
4. **Data:** all 30 frozen coordinates completed: five row counts, three
   feature/thread shapes, and unit plus positive-variable Hessians under seed
   `20260721`. Every frozen array and combined dataset fingerprint validated.
5. **Arms:** each coordinate used two untimed warmups per lane and seven
   paired timed repetitions in the frozen alternating order. Timing covered
   only `build_oblivious_tree`; no coordinate, threshold candidate, tie rule,
   acceptance limit, or analyzer changed after outcome access.
6. **Resources:** the frozen Apple M4 Pro / 14-CPU `darko311` fingerprint and
   exact per-thread environments were revalidated. Chrome Remote Desktop was
   absent before launch; there was no concurrent timed repository job. The
   owner accepted ordinary host background activity because lane pairing,
   alternating order, repetitions, and the frozen stability gate adjudicate
   timing noise rather than permitting a rerun.
7. **Execution:** the formal runner command was
   `python benchmarks/run_fused_lane_dispatch.py calibration` with the v4
   contract, v4 authorization, detached source worktree, and declared raw
   path. It returned zero, published all rows atomically, and emitted no
   terminal-failure record. The frozen analyzer then ran once without a
   threshold argument and created the declared analysis artifact.
8. **Artifacts:** contract SHA-256
   `fab0784beee165b4643b817f12076b79ff832d95224469bc244cc15c839e9c7f`;
   authorization SHA-256
   `42fb0ab01f8a7b271cda2610c59a953d5815e93657ca0a5ab3a003e38dfea775`;
   raw
   [`fused_lane_dispatch_calibration_raw_v4.json`](fused_lane_dispatch_calibration_raw_v4.json)
   SHA-256
   `27a94aa8b93626ec1ae5db329d281b528b52e62beaf0ba3f416d0877a203fea0`;
   analysis
   [`fused_lane_dispatch_calibration_analysis_v4.json`](fused_lane_dispatch_calibration_analysis_v4.json)
   SHA-256
   `c47314191eaec43e6ceb5fa7a2eca870b7af2308cc736dae23c12b9735f3bf9b`.
9. **Primary result:** all 30 cells were behavior-exact. Minimum-regret
   selection chose threshold `1048576`, with 18 fused and 12 unfused cells,
   geomean regret `1.002833`, worst selected/current-fused ratio `1.0`, and
   selected/current-fused geomean `0.973846`.
10. **Gates:** exactness, both-lanes-selected, and worst-ratio gates passed.
    The selected geomean missed its `<=0.970000` gate, and `all_stable=false`
    because six cells exceeded the `IQR / median <=0.10` limit. Qualification
    is conjunctive, so `qualifies=false`.
11. **Limitations/non-claims:** this calibration does not establish a retained
    crossover, speed claim, default change, or portability. Instability is a
    binding failure, not grounds to discard rows or rerun. No selected-threshold,
    validation contract, validation authorization, or validation result exists.
12. **Current decision:** `close_dispatch_campaign`. Keep the effective
    `auto` behavior fused, retain the already-authorized explicit overrides,
    perform no validation phase, and return the next mechanism slot to the
    quality-first shortlist. Q remains closed unless independently re-funded.

### 30. Post-close owner product promotion (2026-07-21)

After section 29 and its immutable artifacts were published, the owner
directed **promote**. The separate create-only
[`owner decision`](fused_lane_dispatch_owner_promotion_20260721.md) activates
the selected `scan_work` threshold `1048576` for new
`oblivious_kernel="auto"` fits inside the existing macOS-arm64 automatic
envelope. Values below the threshold use fused; ties and values above it use
unfused. Explicit overrides, persisted metadata, safe-load validation, and
fused fallback outside the envelope are unchanged.

This product override does not rewrite section 29: the calibration remains
`qualifies=false` with disposition `close_dispatch_campaign`. No selected-
threshold evidence artifact, validation phase, 3% speed claim, portability
claim, release authorization, fresh-data access, or lockbox access is created.
No validation is authorized or created by this product decision.
The owner accepts the marginal and timing-stability risk on the strength of
30/30 behavior exactness, both-lane selection, worst ratio `1.0`, and the
bounded explicit-fused rollback. The next mechanism slot remains
quality-first.

### 31. Gate-reform and ensemble-v3 preparation authority (2026-07-21)

The owner adopted [`NEXT_STEPS.md`](../NEXT_STEPS.md) at `671f2e0` and the
create-only
[`owner adoption note`](gate_reform_owner_adoption_20260721.md) resolves the
adoption commit's wording: the commit itself changed documentation only, while
the §6 matrix authorizes the named future preparation work.

The forward decision retracts archive-size ratios as product gates for this
work without changing any frozen M3b or B-archive result. It authorizes the
ensemble-v3 public-contract freeze, a private/non-exported release-candidate
implementation plus correctness suite, and a new M6 successor build/backtest.
It does not authorize public ensemble parameters/exports, M2, M4/TabArena-Lite,
v0.11, B3, Q work, fresh confirmation, CTR23, or lockbox access. B1 and B2
alone remain unselected because the combined arm Pareto-dominates them; the
B-archive simulation remains non-loadable, optional size telemetry.

### 32. M6 quality-successor v1 binding failure (2026-07-21)

1. **Source:** clean detached DarkoFit contract checkpoint `8abf0b8`; result
   binding is a later state-only checkpoint.
2. **Evidence class:** Tier-E infrastructure backtest over spent, immutable
   artifacts; no new model fit, fresh data, TabArena, or lockbox access.
3. **Declared subset:** combined B1+B2 ensemble-v3 (known advance) and the 3%
   linear-leaf selector (known kill), pinned by SHA-256 in
   [`m6_quality_successor_contract.md`](m6_quality_successor_contract.md).
4. **Analyzer rule:** advance only with quality geomean `<=0.98`, wins in at
   least 60% of cases, and no case above `1.02`; development ranking only.
5. **Execution:** one clean outcome-bearing artifact-only launch through
   [`run_m6_quality_successor_backtest.py`](run_m6_quality_successor_backtest.py).
6. **Result:** both replays agreed. B1+B2 advanced at `0.965513×`, 13/13 wins,
   worst `0.991888×`; the selector was killed at `0.989264×`, 2/14 wins,
   worst `1.000000×`.
7. **Artifact:**
   [`m6_quality_successor_backtest_result.json`](m6_quality_successor_backtest_result.json),
   SHA-256
   `360a60130c99220a3466ff0fab40b54ead99a2d0a29a2bde3a33a12e38500baa`.
8. **Binding audit:** before activation, review found that changing the
   analyzer's embedded false completion flag would change the whole-file hash
   bound by the result; the future CSV also did not attest three repeats, and
   `--datasets all` was not an exact durable coordinate list.
9. **Decision:** the calculation is immutable but grants no ranking authority.
   V1 will not be rebound or rerun. V2 keeps the thresholds/subset unchanged,
   separates immutable rule code from activation, and binds an exact
   repeat-attested command; M6 stays non-ranking until v2 passes.

### 33. M6 quality-successor v2 backtest (2026-07-21)

1. **Source:** clean detached contract checkpoint `e5ff51d`.
2. **Evidence class:** structural Tier-E backtest over the same spent artifacts;
   outcomes were already known from v1, and v2 makes no blindness claim.
3. **Unchanged rule/subset:** `0.98` geomean, 60% wins, `1.02` worst cell;
   combined B1+B2 known advance and 3% selector known kill with the same hashes.
4. **Structural corrections:** immutable rule separated from activation; exact
   ten-dataset command rather than `all`; three repeats owned and recorded by
   the wrapper; rule/contract/wrapper/comparison/paired-execution hashes bound.
5. **Execution:** one clean launch; both replays agreed with the same `0.965513`
   advance and `0.989264` kill readouts.
6. **Artifact:**
   [`m6_quality_successor_v2_backtest_result.json`](m6_quality_successor_v2_backtest_result.json),
   SHA-256
   `6880c679cd5f16aa61d13c2e57282e3f162769be87e478a6ddf18d8958c9cf57`.
7. **Decision:** v2 may rank or kill quality mechanisms on its medium generic
   slice. Every result is inspection-indexed and spent. It cannot rank speed,
   ship, change defaults, or reduce sports/Tier-D/milestone/owner gates. V3 and
   v1 retain their terminal/invalid statuses.

### 34. Ensemble-v3 characterization freeze (2026-07-21)

1. **Model sources:** published DarkoFit
   `c5e66ef7e6bdcf5665b55b81c6b870f42d76237b`; ChimeraBoost 0.18 pin
   `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`.
2. **Evidence class:** Tier-E characterization on spent/frozen inputs; no M2,
   M4, fresh-confirmation, lockbox, public API, default, or release authority.
3. **Quality plan:** reuse immutable M3b r3 point estimates; 100,000-draw
   three-season cluster bootstrap plus leave-one-season-out; separate
   four-general-case bootstrap and leave-one-case-out sensitivity.
4. **Current performance grid:** four frozen medium general tasks, four batch
   sizes (`8,192` through `2,000,000`), DarkoFit single/private v3 and pinned
   ChimeraBoost single, 14 threads, three balanced fresh-worker blocks.
5. **Resources:** fit wall time, worker-plus-recursive-child process-tree RSS,
   safe-NPZ bytes, raw absolute memory deltas, fitted tree/member/thread
   telemetry, and exact safe-load predictions.
6. **Prediction timing:** complete public `predict`; untimed warm call selects
   a bounded integrated loop targeting one second; every formal interval
   declares a `0.75` second minimum and retains paired series plus dispersion.
7. **Correctness bridge:** a named test proves array-exact predictions between
   the historical combined mechanics and the release-candidate wrapper on a
   fixed synthetic case. The complete focused suite passed 66 tests.
8. **Contract:**
   [`ensemble_v3_characterization_contract.json`](ensemble_v3_characterization_contract.json),
   SHA-256
   `f8f7b780c6dc915926a33262e24545696754221ef310d76c01da6f9df3b00103`.
9. **State:** prospectively frozen before any formal current-source
   measurement. One raw artifact or one terminal record is allowed; no
   outcome-driven rerun or threshold is available.

### 35. Ensemble-v3 characterization result (2026-07-21)

1. **Execution:** one complete run from clean harness `6a61fb6`, clean
   DarkoFit model source `c5e66ef`, and clean ChimeraBoost 0.18 source
   `f14be60`; no terminal artifact or rerun.
2. **Evidence:** spent M3b quality plus four frozen medium general performance
   tasks. No M2, M4, fresh-confirmation, lockbox, public API, default, or
   release action occurred.
3. **Quality:** exact M3b reproduction in analysis, 13/13 wins and `0.965513x`
   overall; sports `0.961077x` with season-cluster interval
   `[0.958861x, 0.962867x]`; general `0.975569x` with descriptive interval
   `[0.963303x, 0.987718x]` and leave-one-case-out range
   `[0.970189x, 0.981160x]`.
4. **Current fit:** private v3/single equal-case geomean `6.142053x`; case
   medians ranged `4.051x` to `8.584x`.
5. **Current memory:** aggregate process-tree peak-RSS ratio `1.135581x`;
   median v3 peak-minus-start deltas ranged `14.3 MB` to `67.6 MB`. The older
   `1.074015x` M3b value remains labeled self-worker RSS.
6. **Archives:** safe-NPZ ratio `8.125239x`; v3 archives ranged `0.762 MB` to
   `1.783 MB`. Size is telemetry, not a validity gate.
7. **Prediction:** DarkoFit single/pinned ChimeraBoost `0.485145x` (16/16 no
   slower); v3/ChimeraBoost `3.013607x`; v3/DarkoFit single `6.207940x`.
   V3 was slower in all 16 coordinates against both single arms.
8. **Dispersion/limitations:** 47/48 paired series had IQR/median `<=0.10`.
   Nine of 144 intervals missed `0.75 s`, all DarkoFit-single 8,192-row
   intervals; every 65,536-row-and-larger interval cleared it. No subset or
   favorable rerun replaces the frozen aggregate.
9. **Artifacts:** raw
   [`ensemble_v3_characterization_raw.json`](ensemble_v3_characterization_raw.json)
   SHA-256
   `005c50a89a06e100aa95cb6a776dd7f67026786de6f261470e808a39f9310a9b`;
   result
   [`ensemble_v3_characterization_result.json`](ensemble_v3_characterization_result.json)
   SHA-256
   `5cfd7b40382187aebed43798715017e1e2867744c5c40f66a00e935f6acefeed`;
   generated note
   [`ensemble_v3_characterization_result.md`](ensemble_v3_characterization_result.md)
   SHA-256
   `bef08bf9f972eba7ebfd9b2f51ce1d42828b9444c6e4697063166351ed21b0e4`.
10. **Interpretation:** quality survived; cost is material. The evidence
    supports only an honestly described explicit opt-in and awaits the
    separately gated public-ship decision.

### 36. Ensemble-v3 characterization post-run audit (2026-07-21)

1. **Scope:** fresh-eyes audit of the frozen contract, runner, analyzer, raw
   artifact, generated result, interpretation, tests, and live plan state; no
   model rerun, M2/M4, fresh data, lockbox, API, default, or release action.
2. **Immutable evidence:** contract/raw/result/note/interpretation retain their
   published hashes. The audit is additive and create-only.
3. **Memory correction:** the protocol-promised but omitted equal-case
   peak-minus-start RSS ratio is `3.262867x` v3/single, computed from the four
   case-median paired ratios retained in the immutable result. This sits beside,
   rather than replacing, the `1.135581x` absolute peak-RSS ratio and absolute
   v3 deltas.
4. **Prediction limitation:** nine of 144 intervals missed `0.75 s`; the
   minimum was `0.006492584 s` after an anomalous first warm call selected only
   eight formal calls. The full-grid aggregate remains descriptive and is not
   timing-decision eligible or certified. No subset or rerun replaces it.
5. **Harness audit:** the v1 loader did not assert every declarative contract
   field, create-only write failures could leave partial targets, and RSS
   teardown could mask a primary exception. None occurred in the completed
   run; v1 is retired rather than amended.
6. **Disposition:** preserve all v1 artifacts. Any successor requires a new
   identity with full contract validation, failure-safe output/telemetry, and
   stabilized timing calibration or a fail-closed duration floor.
7. **Record:**
   [`ensemble_v3_characterization_post_run_audit_20260721.json`](ensemble_v3_characterization_post_run_audit_20260721.json)
   SHA-256
   `6fcccf098c217e07513a02f6ca588f95deb883f22d92600976077cead406fbdf`;
   and
   [`ensemble_v3_characterization_post_run_audit_20260721.md`](ensemble_v3_characterization_post_run_audit_20260721.md)
   SHA-256
   `2726b699193eb669e2840bba527743c07d77505dfb8ed782a0fd8076cd94287c`.

### 37. v0.11 private ensemble evidence v2 (2026-07-22)

1. **Execution date/source:** 2026-07-22; DarkoFit model source was the clean
   published commit `543604dd9860a28c30912f914b2cfccfcb99d783`. The formal
   harness was clean published commit `eab7e78`; 177/177 fresh workers
   completed and no terminal artifact or retry occurred.
2. **Comparators:** ChimeraBoost was the exact clean 0.18 pin
   `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`; CatBoost was the exact
   `1.2.10` wheel. Moving checkout heads were not used.
3. **Evidence class:** Tier-E spent private-release-candidate characterization.
   No fresh-confirmation, lockbox, M2, M4, public API, default, or release
   authority was used.
4. **Data/splits:** the exact 13-case M3b r3 grid: nine player-disjoint
   cold-player sports cells within held teams, clustered by three seasons, and
   four fixed seeded 75/25 medium general cells. The contract stores every
   case, dataset, split, and weight fingerprint.
5. **Arms/policies:** quality/cost ran DarkoFit single, existing bootstrap8,
   and private v3 (eight sequential members, 0.8 without replacement,
   `donor_balanced_v1`). Prediction added ChimeraBoost single/ensemble8 and
   CatBoost single. Eight remains the only evaluated member count.
6. **Environment/repeats:** one Apple-silicon machine, 14-thread ceiling,
   three complete fresh-worker blocks, same-case/same-arm two-iteration warmup
   outside measurement. Prediction used four fixed batches, five post-warm
   pilots, a 2.0-second target, at least three public calls, and a fail-closed
   1.0-second interval floor. All 240 intervals cleared; minimum was
   `1.605673208 s`.
7. **Runner:** source-attested
   [`run_v011_ensemble_evidence_v2.py`](run_v011_ensemble_evidence_v2.py).
   V1 was retired before formal execution because a synthetic smoke exposed
   warmup-only warning stderr; v2's sole amendment captures unmeasured warmup
   warnings while preserving formal-fit warning disclosure.
8. **Hashes:** protocol `319ae5e7ef0cecd86d6ccbf752fdba93dfde290d75e65eed4dc4a13589e06a91`;
   runner `09065c51cfc86e31b1914e6349f5f6701eec28563387fb218b3cc5c6d2b51573`;
   analyzer `50cd9948006e8661a3dc1ace5cb771106d6d2fdb7d3c1252bc72f621553edf16`;
   contract `96d85870b9fdb02e0e62e0d9a1386ba22d1f1027a481d652852607cb443ef35f`;
   raw `d6c0b794db4ce4bdd1e393f2b23546f1351a051f1f66fa7438175f826454171e`;
   result `edb35694a6b6d19aa9b320545b759603a7e5a99c34165dd9f1a0ebe66937dabc`;
   note `8c0fc244cf3eb5b9e63b2803d2d8d20b7e66e9b375c15c78cc7dee064c7baee4`.
9. **Primary results:** reproduction passed at absolute ratio tolerance
   `1e-10` (maximum difference `7.78e-16`). V3/single primary loss was
   `0.965513x` pooled; sports `0.961077x` with season-cluster interval
   `[0.958861x, 0.962867x]`; general `0.975569x` with case-bootstrap interval
   `[0.963303x, 0.987718x]` and leave-one-case-out range
   `[0.970189x, 0.981160x]`.
10. **Cost/prediction:** v3/single was `5.030x` fit, `1.090x` absolute peak
    RSS, `3.539x` peak-minus-start RSS, and `6.181x` safe-NPZ bytes. Versus
    existing bootstrap8 it was `0.578x`, `0.999x`, `0.935x`, and `0.706x`.
    Prediction seconds ratios were Darko single/Chimera single `0.478x`
    (16/16 faster), Darko single/CatBoost `0.871x` (9/16 faster), v3/Darko
    single `6.251x`, and v3/Chimera ensemble8 `0.126x` (16/16 faster). Every
    declared integrity/reproduction gate passed; performance and cost had no
    gate by owner authorization.
11. **Limitations/non-claims:** three sports seasons and four fixed general
    cases are not 13 independent datasets; ratios are hardware/grid scoped;
    costs are disclosures; archive size has no reinstated gate; no general
    superiority, certification, public exposure, default, M4, or v0.11 release
    is claimed.
12. **Terminal decision/next action:** no correctness or unresolved
    reproduction stop condition is present. This evidence does not itself
    expose v3. Phase 2 M2 remains the next authorized non-overlapping action;
    public exposure remains a separate Phase 3 owner decision.

### 38. v0.11 M2 defaults-only broad panel v3 (2026-07-22)

1. **Execution date/source:** 2026-07-22; the successful formal campaign ran
   once from clean published DarkoFit commit
   `a2983ce97c2be30199054f30915d7788420cf330`. V1 stopped before output on an
   18-vs-14 host-CPU preflight mismatch; v2 stopped before warmup or fitting on
   the warmup module's independent 18-thread constant. Neither identity was
   rerun, and v3 changed no scientific protocol field from v2.
2. **Comparators:** exact clean ChimeraBoost commit
   `f14be606b641f1bf0dc92bb14b3951f1fe631c6b` (`0.18.0`) and CatBoost wheel
   `1.2.10`; TabArena was exact commit
   `4cd1d2526874962daae048a6f2dcf34aa272f3fa` and AutoGluon was
   `1.5.1b20260712`.
3. **Evidence class:** spent, descriptive M2 evidence. No fresh-confirmation,
   lockbox, private ensemble, public API, default, M4, or release authority was
   used.
4. **Data/splits:** 13 fixed TabArena regression datasets, three registered
   `(repeat, fold)` coordinates per dataset, equal-dataset aggregation, and
   no claim that the fixed datasets are independent random draws.
5. **Arms/policies:** official-default single-model DarkoFit `0.10.1`, pinned
   ChimeraBoost, and CatBoost; eight sequential framework bag folds, one bag
   set, model seed zero varied across folds, no calibration, no manual model
   configuration, and no private ensemble.
6. **Environment/repeats:** one Apple-silicon machine, common 14-CPU/thread
   allocation, one fresh same-arm-warmed process per each of 117 outer jobs,
   936 child fits, continuous balanced arm order, one-hour per-job limit, no
   resume, and no favorable rerun.
7. **Runner/commands:** source-attested
   [`run_v011_m2_broad_panel_v3.py`](run_v011_m2_broad_panel_v3.py) followed
   once after completion by
   [`analyze_v011_m2_broad_panel_v3.py`](analyze_v011_m2_broad_panel_v3.py).
   The same runner's dry run passed first and created no output directory.
8. **Hashes:** contract
   `719213fd993b8626d7ece192fa9b9581ffa4ea6220d0f7d94a598683e098f846`;
   protocol `d4c8bc3fbe980149a3528d13a7f9fd6393f4690517a862cef31d8e622796e403`;
   runner `6eac56b1b0e8fe60e6539f67d413e263c430c7147279bb21067cb1dc9e6fad68`;
   analyzer `24ea34080486bc9d46180bef98bfdcfc132c8134d515d4f35f62af4ba6928f24`;
   raw result set
   `81ee5327e7e2e4997af421ad6ab5579bbd12e1099552898c76552483c217cda3`;
   analysis payload
   `327f24f90383865ea8502118ea622ecf3b983a34926826c4ba96998bccb11f8d`;
   completion attestation
   `1fbd09e4e71e537d58479b4343e3269a1cb7d1a8b56e6f8d23a59aa4b96c4b5c`;
   committed summary
   `e995b96760f0f48eff6ca0745a45055128c10c9a4b73bb0c7b25c55402157af0`;
   LF-normalized paired/per-dataset tables
   `0ca6c1d139c138ca48d116332c56bbb747d16aa6b14e973c3960d0d8befa8020`
   and
   `9b0dcab3baccad428dbef40be82cab219d1a6c2fb7751413f947470cb5301ab3`;
   result note
   `89abd0606b940a5ca7ef3ebeed54ed5ef2da1e066ee38adf3e63926ecadcc49b`.
9. **Primary results:** DarkoFit/ChimeraBoost test-RMSE ratio `1.017433x`
   (descriptive interval `[1.013494x, 1.021558x]`, dataset W-L-T `6-7-0`);
   DarkoFit/CatBoost `1.053834x` (`[1.051130x, 1.056878x]`, `1-12-0`);
   ChimeraBoost/CatBoost `1.035778x` (`[1.033043x, 1.038693x]`, `2-11-0`).
10. **Cost results/integrity:** DarkoFit/ChimeraBoost was `0.812620x` fit,
    `1.316591x` prediction, `0.842460x` incremental RSS, and `0.962665x`
    peak RSS; DarkoFit/CatBoost was `0.091327x`, `1.270594x`, `0.368938x`,
    and `0.707062x`, respectively.
    All 117 jobs and 936 child fits completed with zero failures, imputations,
    known deadlines, or known time-limit stops. All 117 worker attestations and
    the exact resource/order/provenance bindings validated.
11. **Limitations/non-claims:** one hardware/software stack, fixed spent data,
    workload-specific framework timings, and partially unresolved competitor
    stop-reason metadata (`443` child fits labeled `unknown`). The harness
    verified that no such fit was a known time-limit/deadline stop. No general
    superiority, certification, default, public exposure, M4, or release is
    claimed.
12. **Terminal decision/next action:** Phase 2 is complete and honestly shows
    DarkoFit's speed/memory advantage alongside its quality/prediction deficit.
    The authorized evidence phase stops here. Phase 3 public exposure, M4, and
    v0.11 release remain separate owner decisions.

### 39. v0.11 release compute ladder v3 (2026-07-22)

1. **Execution date/source:** 2026-07-22; the measured DarkoFit product was
   clean public `v0.11.0`, commit
   `0b820e332cec2c083b1dd89eef0fe306d69cfc0e`. The successful v3 harness was
   clean published commit `bda98dc91e98023cef6efc1d3f47fdc06ff22f33`.
   V1 and v2 each stopped before worker zero on exclusivity-topology checks and
   received terminal records; neither fit a model or was rerun.
2. **Comparators:** latest upstream ChimeraBoost release `v0.20.0`, exact
   commit `7d48e053e5bd3c7aded1126871aeb0f1f6b84c46`, published
   2026-07-21T02:44:50Z and reverified as latest at worker zero and result
   close. Moving checkout heads were not used.
3. **Evidence class:** spent, descriptive release-cadence compute-ladder
   evidence. It is neither a Tier-D certificate nor authorization for tuning,
   defaults, fresh confirmation, lockbox, classification, CatBoost, or
   TabArena placement.
4. **Data/splits:** 13 fixed historical M2 regression datasets, three
   registered `(repeat, fold)` coordinates per dataset, with the source split
   tree pinned to TabArena commit
   `4cd1d2526874962daae048a6f2dcf34aa272f3fa`. Worker records bind the target,
   feature, train, and test fingerprints.
5. **Arms/policies:** six direct public-estimator arms: DarkoFit default,
   `preset="accuracy"`, and public ensemble8; ChimeraBoost default, `depth=10`,
   and public ensemble8. There was no AutoGluon outer bag. The common resource
   contract was 14 CPU threads and zero GPUs.
6. **Environment/repeats:** one 14-core Apple-silicon host; 234 sequential
   fresh workers; continuous deterministic arm order; engine/route warmup
   outside measurement; three prediction pilots followed by at least three
   calls, a one-second target, a 0.5-second fail-closed floor, and a 65,536-call
   cap. All intervals cleared the floor; minimum was 0.598 seconds.
7. **Runner/commands:** source-attested
   [`run_v011_compute_ladder.py`](run_v011_compute_ladder.py), followed once
   after terminal completion by
   [`analyze_v011_compute_ladder.py`](analyze_v011_compute_ladder.py). The
   frozen v3 contract was generated by
   [`freeze_v011_compute_ladder.py`](freeze_v011_compute_ladder.py). V3 ran
   under a standalone `caffeinate` keeper rather than wrapping the runner.
8. **Hashes:** protocol
   `2b48ebe91ffe8586cad69c1abecafc14fc01dcb895c346c97d78a166c20a5e23`;
   runner `db5b47af68fa0d74458c9d48d0c441caee8621cf1922542df2a27668118d14fb`;
   analyzer `d65c84b16c1f43499687771ddb07e9f6dc23a5a1af09ba177f520733f05abf9b`;
   contract `61e788f06b88eefcc2e3c08a38402bf93246e7334980a77061b46763650b581a`;
   raw `96f594da1a0ea885aa55d45636049d97b9b6e1a7f56d85679dfe879420636f79`;
   manifest `01fbb053d1390c43758adc4f47da38e39b6beb53be26ed13548a5eb399d485d4`;
   summary `28c904e4585d343d96366bf998edd39034795ab18f092a8765b0efe7049543d6`;
   coordinate table
   `8887ba02c2ae2189907e4afc3064d3c262030432a776ccd9597985088f2d35df`;
   per-dataset table
   `546592592a3a70720fa214245451374982f2e17f783341c07fbe03b97682dd10`;
   analyzer-generated report
   `e23fc2d6b32a3cf22373227ce0a7bcd3604a4b9fb21383c40ac659af0362db11`;
   result note
   `4e99756f84de137c147790f3205d5d2b00fc7e71d67e6049102982aa2eee12f6`.
9. **Primary results:** versus ChimeraBoost default, quality ratios were
   DarkoFit default `1.0145x` `[1.0080, 1.0207]`, DarkoFit accuracy `1.0038x`
   `[0.9934, 1.0153]`, DarkoFit ensemble8 `0.9996x` `[0.9937, 1.0054]`,
   ChimeraBoost depth10 `1.0159x` `[1.0085, 1.0229]`, and ChimeraBoost
   ensemble8 `0.9646x` `[0.9596, 0.9697]`. Matched DarkoFit/ChimeraBoost
   quality ratios were `1.0145x`, `0.9881x`, and `1.0363x` at default,
   accuracy, and ensemble8.
10. **Costs, passed and failed conditions:** DarkoFit matched-profile fit
    ratios were `1.3796x`, `1.3019x`, and `6.1036x`; prediction ratios were
    `2.2127x`, `2.4221x`, and `0.3663x`; peak-RSS ratios were `0.9739x`,
    `0.9146x`, and `0.1606x`. Peak RSS was no worse at all three points, but
    fit-frontier dominance failed at all three comparable budgets and
    prediction-frontier dominance held at only one of four. All 234 workers,
    source attestations, ambient-thread restores, prediction floors, and RSS
    samplers passed. Known OpenMP deprecation noise and ChimeraBoost's public
    ensemble-default warning were retained in raw telemetry.
11. **Limitations/non-claims:** fixed spent regression tasks on one hardware
    stack and fixed test-batch shapes; the uncertainty resamples coordinates
    within each fixed dataset and does not imply 13 independent datasets. No
    classification, CatBoost, fresh, lockbox, TabArena-placement, universal
    superiority, or hardware-portable timing claim is made.
12. **Terminal decision/next action:** the predeclared strict Pareto victory is
    **false**. ChimeraBoost v0.20's ensemble owns the measured quality/training
    frontier; DarkoFit retains the measured memory and ensemble-prediction
    advantages. The scoreboard is terminal and spent. Phase F's historical
    kill-rule audit is the next authorized action, followed by the automatic
    smooth-data selector campaign.

### 40. Phase F kill-rule audit and M6 quality-successor v3 (2026-07-22)

1. **Execution date/source:** 2026-07-22; the 41-disposition Phase F audit and
   v3 harness were frozen and published at clean commit
   `f3d19ebb4d9306e278a52534a7856650675d1166`. The v3 historical backtest ran
   once from that exact clean commit.
2. **Comparators:** no live model comparator. The outcome-known replays bind
   the combined B1+B2 readout, native ordinal C2 development result, and old
   3% selector result at their exact recorded hashes.
3. **Evidence class:** governance and development-ranking infrastructure over
   spent historical evidence. It is not new quality evidence and grants no
   shipping, default, speed-ranking, fresh-confirmation, TabArena-placement,
   or lockbox authority.
4. **Data/splits:** the positive replay has 13 fixed cases, the valid negative
   has four fixed C2 tasks, and the retired-verdict tripwire has 14 fixed
   smooth/process lineages. Cases are not claimed as independent draws. A
   future live v3 inspection retains the exact ten-dataset, medium, 60-cell
   paired grid with dataset-level concentration.
5. **Arms/policies:** v3 removes v2's 60% win-count and 0.98 minimum-effect
   gates. Development advance requires aggregate ratio at most 1.000, worst
   dataset/group at most 1.020, and worst leave-one-group-out ratio at most
   1.003. These are non-harm/concentration triage only.
6. **Environment/repeats:** one outcome-known, artifact-only execution in the
   `darko311` environment; no model fit, timing, worker, fresh data, or rerun.
   The harness source state was clean and already published before execution.
7. **Runner/commands:**
   `python benchmarks/run_m6_quality_successor_v3_backtest.py --output benchmarks/m6_quality_successor_v3_backtest_result.json`.
   After the runner returned and created its fsynced result, the surrounding
   zsh wrapper failed while assigning to reserved variable `status`; the
   Python runner was not rerun, and the post-run shell mistake is disclosed in
   the result note.
8. **Hashes:** Phase F audit
   `3717a080030788ded9fa12101dfad7e1b87ac811f517f1b1e1e16fb0fa35769f`;
   contract
   `1fedb2d2d2e043f56c8547fd67bf32ef028f98866f7455c05c2e8fa6c9d0e2b3`;
   rule
   `2415c7a7bde2bed23283067fdfe200892c15cf1c70d869153cc9cade81f9694c`;
   execution runner
   `950c3867f387112a65a5dd103f830cce71f7e74af42c5f5208499e787e609d39`;
   backtest runner
   `bcb733bdefd36fe4e6052f91f3453f42f7a21dd45011bd5705e7c91a098dd019`;
   result
   `35cc54acfeb7de7950966445ed8248654f945072e5e5900e3333fff4b15129b6`.
9. **Primary results:** combined B1+B2 reproduced `advance` at aggregate
   `0.965513`, worst group `0.991888`, worst LOO `0.968329`; native ordinal C2
   reproduced `kill` at `0.992755`, `1.317510`, and `1.090069`; the retired
   selector verdict correctly changed from v2's kill to v3 `advance` at
   `0.989264`, `1.000000`, and `0.998504`.
10. **Passed and failed conditions:** all three expected dispositions agreed,
    every artifact/code/audit binding matched, and
    `candidate_ranking_eligible=true`. No scientific or integrity gate failed.
    The disclosed shell-wrapper error occurred strictly after result creation
    and did not enter the runner.
11. **Limitations/non-claims:** the backtest is outcome-known and uses spent,
    heterogeneous historical summaries. It validates triage behavior, not
    statistical calibration, selector shipping, or default safety. V1/v2
    artifacts and outcomes remain immutable; only forward ranking authority
    is superseded.
12. **Terminal decision/next action:** Phase F is complete. V3 is the sole M6
    quality-development ranking path. Begin the separately authorized short
    causal engagement check for the automatic smooth-data selector, then
    freeze its new-identity invariants-first Tier-D campaign if the mechanism
    is confirmed.

### 41. Phase B-1 smooth-selector engagement check (2026-07-22)

1. **Execution date/source:** 2026-07-22; read-only extraction from a worktree
   based on published DarkoFit commit
   `5f98d7b0087746bd7349927c96dc368ea872b870`. No library source changed and
   no model was fit.
2. **Comparators:** the completed M2 cache pins ChimeraBoost
   `f14be606b641f1bf0dc92bb14b3951f1fe631c6b` (`v0.18.0-6-gf14be60`);
   the release compute ladder pins public ChimeraBoost v0.20.0 commit
   `7d48e053e5bd3c7aded1126871aeb0f1f6b84c46`.
3. **Evidence class:** spent descriptive fitted-metadata inspection. It is a
   causal funding check, not new quality, shipping, or default evidence.
4. **Data/splits:** M2 task ids 363612 (`airfoil_self_noise`) and 363693
   (`physiochemical_protein`), three registered outer coordinates and eight
   child fits per coordinate; plus the three direct default coordinates per
   dataset from the v0.11 release compute ladder. Child fits are not treated
   as independent datasets.
5. **Arms/policies:** ChimeraBoost's default fitted metadata only. Inspected
   selected lane, whether linear selection ran, resolved linear/cross/category
   policies, and current v0.20 member selector summaries. No candidate was
   tuned or executed.
6. **Environment/repeats:** `darko311`; one deterministic extraction of six
   hash-attested gzip pickles and three committed v0.20 artifacts. No warmup,
   thread, timing, fresh-worker, or favorable-rerun issue applies.
7. **Runner/command:**
   `python benchmarks/extract_smooth_selector_engagement.py --m2-cache <v011-m2-v3-cache> --output benchmarks/smooth_selector_engagement_check_20260722.json`.
   One pre-output development attempt failed closed on a CSV schema assumption;
   it did not create the canonical output. The corrected extractor wrote the
   create-only record once.
8. **Hashes:** extractor
   `368764dcf102d79a37b0cb16156a1fd56192de899f5d1ec7e001d688389876cf`;
   result
   `878ffdc0bfb615714b5acd0ea0c1d09f63604d4d423d57ab0898f9bd377ab3d1`;
   M2 completion attestation
   `1fbd09e4e71e537d58479b4343e3269a1cb7d1a8b56e6f8d23a59aa4b96c4b5c`;
   current raw ladder
   `96f594da1a0ea885aa55d45636049d97b9b6e1a7f56d85679dfe879420636f79`.
9. **Primary results:** M2 Airfoil selected constant in 24/24 child fits and
   never performed linear selection; M2 Protein selected linear and performed
   selection in 24/24. Crosses were null and category combinations false in
   all 48 M2 children. Current v0.20 again reports linear selection on all
   three Protein defaults (crosses on one) and none on Airfoil. Current D0/M0
   RMSE ratios are 0.953347 on Airfoil and 1.067903 on Protein.
10. **Passed and failed conditions:** the Protein selector signature passed
    across both rival pins. The original two-dataset hypothesis failed because
    Airfoil had no selector engagement. This is a narrowing, not a favorable
    relabeling: Airfoil is removed from the selector causal claim.
11. **Limitations/non-claims:** v0.20's compact member summary does not expose
    categorical-combination selection; the M2 full resolved metadata does.
    The evidence is spent, workload-specific, and cannot establish default
    safety or independent statistical breadth. No fresh/lockbox evidence was
    accessed.
12. **Terminal decision/next action:** fund a new-identity automatic-selector
    development campaign only for Protein and the generic smooth/process
    signature, using spent development data. Airfoil remains outside that
    causal claim. A default-on policy still needs separately authorized,
    prospectively frozen and powered Tier-D evidence.

### 42. Automatic linear-selector v2 M5 check (2026-07-22)

1. **Execution date/source:** 2026-07-22; clean, published candidate
   `a53d4bf543534678189d87d88dcad87dd2a8bd8f` on
   `codex/smooth-selector-20260722`, with the unchanged runner executed from
   that same clean checkout.
2. **Comparators:** exact frozen M5 control
   `726e5d8e6131c580bce948db833a5007d0692dca`; hash-bound M5 baseline
   `0971e06d4ed307d352d75e1e6400b849c0001b5e11f40243173d7080b6c5859d`.
3. **Evidence class:** spent, non-ranking M5 correctness and diversity-drift
   evidence. It grants no quality-ranking, shipping, default, fresh,
   TabArena, or lockbox authority.
4. **Data/splits:** the frozen 19-cell M5 grid across grouped, smooth, noisy,
   categorical/missing, high-row, binary, multiclass, weighted-regression,
   and weighted-classification domains. Dataset and split hashes are recorded
   per row in the raw artifact.
5. **Arms/policies:** frozen control defaults versus the candidate default
   automatic linear selector. Explicit classification and ensemble behavior
   remained outside selector eligibility as designed; no cell or threshold
   was tuned after inspection.
6. **Environment/repeats:** Python 3.11.8 in `darko311` on
   `macOS-26.5.2-arm64-arm-64bit`, 14 logical CPUs, four fixed worker threads,
   one fresh worker per arm/cell, alternating arm order, and same-source
   three-tree warmup outside timing.
7. **Runner/command:** `python benchmarks/run_m5_sentinels.py --control
   /private/tmp/darkofit-wave1-source-726e5d8 --candidate
   /private/tmp/darkofit-smooth-selector-20260722 --baseline
   benchmarks/m5_sentinel_baseline.json --output
   /Users/konstantinmedvedovsky/code/darkofit/benchmarks/automatic_linear_selector_v2_m5_check_20260722.json`,
   with the output path in the main checkout so the clean candidate harness
   remained immutable throughout execution.
8. **Hashes:** selector development contract
   `fe2d476417e8e8087a3c7342eee0d5cb82a6b8a4ee3f360a1806ee4c0922163b`;
   M5 runner
   `5975d8037e3d94c54b63611b9eb50b28e1098e23f7e9f064608b97141cec61ca`;
   machine-readable M5 contract
   `71b9ab84af20663ca86725bbaf3328541623e33afb2724e28d9f6ed85542a8f0`;
   raw result
   `1c765589ed303432d87009ca0330db8dcf35e3651fbd9b93d2f8bc576f9e494a`.
9. **Primary results:** all 38 rows passed. Both earned classification floors
   passed. Eighteen of 19 paired behavior fingerprints were identical; noisy
   numeric regression seed 0 changed at a candidate/control primary-loss
   ratio of `1.004434950`.
10. **Costs, passed and failed conditions:** no baseline drift and no
    advancement block were reported. Median candidate/control ratios were
    `1.031226` fit, `1.007107` prediction, and `1.002855` peak RSS. Maximum
    fit ratio was `7.734942`. These costs are telemetry, not M5 gates.
11. **Limitations/non-claims:** M5 is a small fixed sentinel grid, not an
    independent dataset panel or quality scoreboard. Its timing cells are too
    short for portable performance claims, and its one changed noisy cell is
    neither a kill nor an acceptance result.
12. **Terminal decision/next action:** M5 invariants and drift checks pass.
    Run exactly one M6 quality-successor-v3 inspection with mechanism id
    `automatic_linear_selector_v2` and inspection index 1, reporting every
    selector engagement reason. M6 failure is terminal for this identity.

### 43. Automatic selector M6 engagement companion (2026-07-22)

1. **Execution date/source:** 2026-07-22; clean published companion harness
   `56a66700a354dfe90d4cfd72d4254a7d8e22b351` and clean published selector
   candidate `a53d4bf543534678189d87d88dcad87dd2a8bd8f`.
2. **Comparators:** no quality comparator was executed. The companion binds to
   the immutable `m6-quality-successor-v3` grid and is paired prospectively
   with mechanism `automatic_linear_selector_v2`, inspection index 1.
3. **Evidence class:** spent mechanism-engagement provenance only. The output
   expressly contains no quality metric or ranking/default/shipping authority.
4. **Data/splits:** all 60 exact M6 v3 medium cells: ten datasets, seeds 0--2,
   weight modes `none` and `stress`. Case, dataset, split, and weight hashes
   are recorded for every cell.
5. **Arms/policies:** candidate public default only. Regression exposes the
   complete fitted selector record; classification must expose no selector
   state and is labeled `classification_not_applicable`.
6. **Environment/repeats:** `darko311`, four fixed threads, strict
   `paired-evidence-v1` worker environment, one deterministic fit in one fresh
   worker per cell. Benchmark timing and RSS are discarded.
7. **Runner/command:** `python
   benchmarks/run_automatic_linear_selector_v2_m6_engagement.py --candidate
   /private/tmp/darkofit-smooth-selector-20260722 --output
   /Users/konstantinmedvedovsky/code/darkofit/benchmarks/automatic_linear_selector_v2_m6_engagement_20260722.json`.
8. **Hashes:** companion protocol
   `f9dbb9bb93c4e71d7670f2e6c0ac8100c7a49a7b6541f8858466d1571167314b`;
   runner
   `50d69b1c372b4e6849796c85f42395651d19ebabbe184f993e5addfa5e864969`;
   selector contract
   `fe2d476417e8e8087a3c7342eee0d5cb82a6b8a4ee3f360a1806ee4c0922163b`;
   M6 rule
   `2415c7a7bde2bed23283067fdfe200892c15cf1c70d869153cc9cade81f9694c`;
   comparison runner
   `0fcd849a13c0348c4c6802556d9a3d3b9f1d5b02c8c47a4e82c3e744f358760f`;
   paired foundation
   `63c63d4f0b7c6f649b7155325ee064faf6e5981094ed3cb79ac91b6b8fefedf9`;
   output
   `6120efe99421403de1d64e7bff594bcf51d3aba18d8851de2e9f728860952405`.
9. **Primary results:** 36 classification cells were non-applicable, six
   diabetes cells used exact `below_min_samples` fallback, and all 18 eligible
   regression cells returned `margin_below_threshold`. No cell selected linear
   leaves; eligible margins ranged from `-0.056974` to `0.028148`.
10. **Passed and failed conditions:** all workers, source stability, fixed
    environment, implementation path, thread state, prediction/probability
    validation, and selector schemas passed. No condition failed.
11. **Limitations/non-claims:** the companion intentionally omits quality and
    cannot substitute for M6 inspection 1. Its 60 cells are dependent fixed
    development coordinates, not independent datasets or fresh confirmation.
12. **Terminal decision/next action:** selector engagement provenance is
    complete. Launch the frozen M6 quality-successor-v3 inspection 1 once;
    its result alone determines `advance` or terminal `kill` for this rung.

### 44. Automatic selector M6 v3 inspection 1 (2026-07-22)

1. **Execution date/source:** 2026-07-22; clean published M6 harness
   `56a66700a354dfe90d4cfd72d4254a7d8e22b351`, clean control
   `b11f013f7ba926e533c38db8261f1a569ebce6c6`, and clean published candidate
   `a53d4bf543534678189d87d88dcad87dd2a8bd8f`.
2. **Comparators:** control public defaults versus candidate public defaults,
   under mechanism id `automatic_linear_selector_v2`, inspection index 1.
3. **Evidence class:** spent general-quality development ranking under
   `m6-quality-successor-v3`. It grants no shipping, default, fresh,
   TabArena, or lockbox authority.
4. **Data/splits:** 60 paired medium cells across ten frozen synthetic/real
   regression and classification datasets, seeds 0--2, and weight modes
   `none` and `stress`; exact case, dataset, split, and weight hashes are in
   each raw row.
5. **Arms/policies:** unchanged control defaults versus the candidate's
   automatic-selector default. The pre-run engagement companion recorded no
   selected-linear cell; no threshold, cell, or policy changed after that
   observation.
6. **Environment/repeats:** `darko311`, 14 physical/logical CPUs, four fixed
   threads, fresh workers, alternating source order, same-source warmup, and
   three repeats per exact cell under `paired-evidence-v1`.
7. **Runner/command:** `python benchmarks/run_m6_quality_successor_v3.py
   --control /private/tmp/darkofit-selector-control-b11f013 --candidate
   /private/tmp/darkofit-smooth-selector-20260722 --mechanism-id
   automatic_linear_selector_v2 --inspection-index 1 --raw-csv
   benchmarks/automatic_linear_selector_v2_m6_v3_inspection1_raw_20260722.csv
   --output
   benchmarks/automatic_linear_selector_v2_m6_v3_inspection1_result_20260722.json`
   (the two output paths resolved to the main checkout).
8. **Hashes:** development contract
   `fe2d476417e8e8087a3c7342eee0d5cb82a6b8a4ee3f360a1806ee4c0922163b`;
   M6 contract
   `1fedb2d2d2e043f56c8547fd67bf32ef028f98866f7455c05c2e8fa6c9d0e2b3`;
   rule
   `2415c7a7bde2bed23283067fdfe200892c15cf1c70d869153cc9cade81f9694c`;
   runner
   `950c3867f387112a65a5dd103f830cce71f7e74af42c5f5208499e787e609d39`;
   backtest result
   `35cc54acfeb7de7950966445ed8248654f945072e5e5900e3333fff4b15129b6`;
   raw CSV
   `e30d089e79d177eb866514e45a0a9ec921a25e46f15e293d72d95525a86cec66`;
   result
   `7445b70ca3bc727bb24f8990ceef590ca933eb1dd45ccefe9ee5788eff211948`;
   manifest
   `601f069896cdf664fcab470abe8c3643f0c0aacf5f79572a6663e304af3d7782`.
9. **Primary results:** all 60 quality ratios were exactly `1.000000`.
   Aggregate, worst dataset, worst coordinate, and worst leave-one-dataset-out
   ratios were all `1.000000`.
10. **Costs, passed and failed conditions:** aggregate `<=1.000`, worst
    dataset `<=1.020`, and worst leave-one-dataset-out `<=1.003` all passed;
    no gate failed. Geometric-mean candidate/control ratios were `1.196564`
    fit, `0.998746` prediction, and `1.022557` peak RSS; maximum fit ratio was
    `3.087023`. Costs were adjacent telemetry, not gates.
11. **Limitations/non-claims:** this is one spent, dependent, fixed medium
    panel whose cells did not engage linear leaves. Exact preservation here
    does not demonstrate Protein benefit or generalize selector engagement.
    Inspection 1 is spent and cannot be rerun favorably.
12. **Terminal decision/next action:** M6 disposition `advance`. Run the
    contract's three-coordinate spent Protein attribution with constant,
    automatic, and explicit-linear arms. Any Protein harm above `1.02` or
    failure to select/match explicit linear is terminal for this identity.

### 45. Automatic selector Protein attribution attempt 1 (2026-07-22)

1. **Execution date/source:** 2026-07-22; clean published harness
   `19bdef2f27496ff4312c1a156d2f6198d358184e`, clean published candidate
   `a53d4bf543534678189d87d88dcad87dd2a8bd8f`, and clean TabArena source
   `4cd1d2526874962daae048a6f2dcf34aa272f3fa`.
2. **Comparators:** planned candidate-source `constant`, `automatic`, and
   `explicit_linear` arms. No arm completed and no comparator outcome exists.
3. **Evidence class:** spent Protein development attribution attempt under
   `automatic-linear-selector-v2-protein-attribution-20260722`; execution
   failure only, with no quality or product evidence.
4. **Data/splits:** planned OpenML task 363693 (`physiochemical_protein`) at
   exact release coordinates r0f0, r1f1, and r2f2. The task loader failed
   before loading worker zero's split, so no split fingerprint was produced.
5. **Arms/policies:** planned public-default DarkoRegressor policy with only
   `linear_leaves=False`, `"auto"`, or `True` differing. No model was built or
   fit.
6. **Environment/repeats:** `darko311`, 14 physical/logical CPUs, exact frozen
   14-thread worker variables, no competing benchmark process, planned one
   fresh worker per arm/coordinate. Zero workers completed and zero repeats
   were observed.
7. **Runner/command:** `python
   benchmarks/run_automatic_linear_selector_v2_protein_attribution.py
   --candidate-source /private/tmp/darkofit-smooth-selector-20260722
   --tabarena-source /private/tmp/tabarena-m2-4cd1d25 --output-prefix
   benchmarks/automatic_linear_selector_v2_protein_attribution_attempt1_20260722`
   (the output prefix resolved to the main checkout).
8. **Hashes:** protocol
   `e231ab25297cb61280ed72716a423d2ec86c71403a5521d40c3ea5d346580d8f`;
   runner
   `c6b6f65dcc1a0f5916a04ebb7daaf7e60ea99631221d82b2d201eaad8d9955c1`;
   tests
   `cf8cbd249f4ec6791b303eebf666809de6d7e0d48b00d5641996b9d0b24e94fa`;
   launch manifest
   `4b4471cdba3beab6cc9dc2cce8d1c8835bfa01cebc986321b9541f89e191def4`;
   terminal result
   `e4bb44356c90d18e88c252bc2a9c8d197303e4a4cb750daacee6eda3c104ab0f`.
9. **Primary results:** none. Worker zero raised
   `ModuleNotFoundError: No module named 'autogluon'` while importing the
   pinned TabArena task loader. `completed_worker_count=0`; no raw artifact
   exists.
10. **Passed and failed conditions:** clean published source pins, evidence
    bindings, 14/14 hardware, exclusivity, and create-only output checks
    passed. The worker dependency/data-loader precondition failed before any
    fit. No scientific gate was evaluated.
11. **Limitations/non-claims:** this record says nothing about Protein
    quality, selector engagement, cost, or safety. The missing outcome cannot
    be imputed from M6 or historical rival metadata. Attempt 1 is spent and
    cannot be favorably rerun.
12. **Terminal decision/next action:** contract disposition
    `terminal_execution_failure`; close `automatic_linear_selector_v2` without
    a scientific Protein verdict. Fix the harness forward with a pre-launch
    exact data-loader probe, but do not reopen this candidate. Proceed to the
    next one-mechanism quality slot only after that repair is verified.

### 46. T7b automatic scalar-RMSE L2 invariants and M5 (2026-07-22)

1. **Execution date/source:** 2026-07-22; clean published harness
   `454fee09794a5090c68614b46a2f2be455a53b38`, clean pre-mechanism
   invariant control `370b8924c034de0332a4b990817972cf0e876f3e`, frozen M5
   control `726e5d8e6131c580bce948db833a5007d0692dca`, and clean published
   private candidate `4bf425fcf3ef095679176b8326ea6621830b64cc`.
2. **Comparators:** control automatic/explicit L2 behavior versus the exact
   candidate that changes only automatic scalar-RMSE CatBoost's unweighted
   L2 base from `3.0` to `1.0`; M5 uses explicit L2 `3.0` in both arms.
3. **Evidence class:** pre-quality correctness invariants and non-ranking M5
   sentinels under `t7b-automatic-scalar-rmse-l2-v1-20260722`; spent
   development infrastructure, no shipping/default evidence.
4. **Data/splits:** five deterministic 200-row invariant probes; frozen M5
   v1's 19 paired generic/SynthGen/adapter cells, exact registered seeds and
   baseline fingerprints.
5. **Arms/policies:** exact control and candidate revisions; invariant no-op
   families were explicit CatBoost RMSE, CatBoost classification, CatBoost
   MAE, LightGBM RMSE, and hybrid RMSE. No depth-policy change was present.
6. **Environment/repeats:** `darko311`; invariant workers used two Numba
   threads and restored the ambient mask; M5 used four threads in fresh
   workers under its frozen repeat/seed policy after a manual
   no-conflicting-benchmark preflight.
7. **Runner/command:**
   `python benchmarks/check_t7b_automatic_l2_invariants.py --control
   /private/tmp/darkofit-t7b-auto-l2-control-370b892 --candidate
   /private/tmp/darkofit-t7b-auto-l2-v1-20260722 --output
   /private/tmp/t7b_automatic_l2_v1_invariants_20260722.json`; then
   `python benchmarks/run_m5_sentinels.py --control
   /private/tmp/darkofit-wave1-source-726e5d8 --candidate
   /private/tmp/darkofit-t7b-auto-l2-v1-20260722 --baseline
   benchmarks/m5_sentinel_baseline.json --output
   /private/tmp/t7b_automatic_l2_v1_m5_20260722.json`.
8. **Hashes:** invariant artifact
   `c3dee2ecb521648e2f9521e280267d41301361cf9aeccfd84ef77b817f4443f9`;
   M5 artifact
   `3bc489a9304ccd0021ed936b8eeec3bcfb1ab6b37476b8ecc87ffb3943a3c747`;
   candidate contract
   `96bb3123b093213ad5657dfd714557e9e011cbb73a17b723fb11cc4ef2f77913`;
   invariant runner
   `231624f22403550ed08408a9656efec3b81834d48cb45a64a87b8789a8a3bfa0`.
9. **Primary results:** all five no-op families had exact predictions and
   exact logical fitted state. All 19 M5 behavior fingerprints matched;
   baseline drift was empty and every earned classification floor passed.
10. **Costs, passed and failed conditions:** all invariant and M5 conditions
    passed. M5 candidate/control medians were `0.998607` fit, `1.015070`
    prediction, and `1.000056` peak RSS; these are non-ranking telemetry.
11. **Limitations/non-claims:** M5 deliberately fixes L2 at `3.0`, so it is a
    no-op regression check rather than evidence that the candidate improves
    quality. Neither artifact authorizes sports, fresh, TabArena, lockbox,
    shipping, or default claims.
12. **Terminal decision/next action:** pre-quality disposition `pass`; spend
    exactly one frozen M6 v3 inspection through the mechanism-specific
    wrapper. No favorable rerun is permitted after launch-manifest creation.

### 47. T7b automatic scalar-RMSE L2 M6 v3 inspection 1 (2026-07-22)

1. **Execution date/source:** 2026-07-22; clean published harness
   `454fee09794a5090c68614b46a2f2be455a53b38`, clean pre-mechanism control
   `370b8924c034de0332a4b990817972cf0e876f3e`, and clean published private
   candidate `4bf425fcf3ef095679176b8326ea6621830b64cc`.
2. **Comparators:** public-default control versus public-default candidate;
   the only intended behavior change was automatic scalar-RMSE CatBoost L2.
3. **Evidence class:** one spent general-development inspection under M6
   quality-successor-v3 and candidate contract
   `t7b-automatic-scalar-rmse-l2-v1-20260722`; ranking-eligible for this
   terminal development decision, never shipping/default evidence.
4. **Data/splits:** exact frozen 60-cell medium grid: ten regression and
   classification datasets, three seeds, unweighted and stress-weighted
   policies, fixed adapter splits and fingerprints.
5. **Arms/policies:** `control_default` and `candidate_default`; 1,000-round
   public defaults, four threads, with six classification datasets retained
   as exact quality no-ops. The samples-per-feature depth idea was excluded.
6. **Environment/repeats:** `darko311`, macOS arm64, four threads, three
   repeats in each fresh worker; exclusive-machine audit found no conflicting
   benchmark process before the create-only launch manifest was written.
7. **Runner/command:** `python benchmarks/run_t7b_automatic_l2_v1.py
   --control /private/tmp/darkofit-t7b-auto-l2-control-370b892 --candidate
   /private/tmp/darkofit-t7b-auto-l2-v1-20260722 --output-prefix
   /private/tmp/t7b_automatic_l2_v1_m6_inspection1_20260722`.
8. **Hashes:** launch manifest
   `593d44e331b9be14f1683947315e60eaefbe947b4460e7d99354074564fc4e1f`;
   raw CSV
   `dfa5560d752f1c17fa8dea0b497d90ebfaf1cb63275f2d69e7ca0afd57677a3a`;
   result
   `6fc5ececda62da257fd3e00fce7df1b8dba2978501e689d0d2a2ca678f296f26`;
   generic manifest
   `034bfbc47a2ef1fe872efa57cc52f3eb97d5986e269cc219e0bde802eab558d8`;
   terminal attestation
   `6bc045080cb6db0a38f912d5d7b31d10d5483e392520dcf682d057db43d05419`.
9. **Primary results:** equal-cell geometric-mean primary-loss ratio
   `1.000818`; worst dataset-group ratio `1.010896` on
   `diabetes_resampled`; worst leave-one-dataset-out ratio `1.001370` when
   omitting `wide_numeric_reg`; worst individual cell `1.032550` on
   `diabetes_resampled/medium/1/stress`.
10. **Costs, passed and failed conditions:** aggregate `<=1.000000` failed;
    worst dataset group `<=1.020000` and worst LOO `<=1.003000` passed.
    Non-gating geometric-mean ratios were `1.001920` fit, `1.003624`
    prediction, and `1.001976` worker peak RSS.
11. **Limitations/non-claims:** this is one dependent, fixed, spent medium
    panel. It is not sports or fresh confirmation, and it supports no public
    automatic-L2, release, TabArena, lockbox, or shipping claim. The separate
    depth-policy hypothesis cannot inherit either this failure or its data.
12. **Terminal decision/next action:** frozen disposition `closed_in_m6`.
    Inspection 1 is spent; rerun is false. Do not merge candidate `4bf425fc`.
    Preserve the artifacts and leave the existing public L2 policy unchanged.

### 48. T7b automatic scalar-RMSE depth invariants and M5 (2026-07-22)

1. **Execution date/source:** 2026-07-22; clean published harness
   `a1eb76071ca72a52494f6cca6022ea8ace8d5394`, clean pre-mechanism
   invariant control `e23d2b164f10374b1c0e02521c33fc96d48980da`, frozen M5
   control `726e5d8e6131c580bce948db833a5007d0692dca`, and clean published
   private candidate `41e948f0c53b1d124e16071a7fa66eba47d084d3`.
2. **Comparators:** public-default control versus the exact candidate that
   chooses CatBoost scalar-RMSE depth 4/6/8 from effective rows per original
   input feature; all excluded families and explicit depth values are no-ops.
3. **Evidence class:** pre-quality correctness invariants and non-ranking M5
   sentinels under `t7b-automatic-scalar-rmse-depth-v1-20260722`; spent
   development infrastructure, no shipping/default evidence.
4. **Data/splits:** seven deterministic 200-row no-op probes, three branch
   probes at the frozen boundaries, and frozen M5 v1's 19 paired cells.
5. **Arms/policies:** exact control and candidate revisions; no-op families
   include explicit CatBoost RMSE, literal `auto`, classification, MAE,
   depthwise, LightGBM, and hybrid. Automatic L2 is unchanged.
6. **Environment/repeats:** `darko311`; invariant workers used two Numba
   threads and restored the ambient mask; M5 used four threads in fresh
   workers after a no-conflicting-benchmark preflight.
7. **Runner/command:** `python benchmarks/check_t7b_automatic_depth_invariants.py
   --control /private/tmp/darkofit-t7b-auto-depth-control-e23d2b1 --candidate
   /private/tmp/darkofit-t7b-auto-depth-v1-20260722 --output
   /private/tmp/t7b_automatic_depth_v1_invariants_20260722.json`; then the
   contract wrapper's frozen M5 precondition.
8. **Hashes:** invariant artifact
   `02362e5d7080c155add0846a58b6960db997bd29a0374e936a16a5a5364e5aff`;
   M5 artifact
   `1d3eac70f81babeb628850cf19844d7b4c590c6df67ded723fcf7caba019bca1`;
   contract `83ea767e5d0060f7ff1c129c6ee84e6d0be3b28236da4e1d0c94369ee6c6b000`.
9. **Primary results:** all seven no-op families had exact predictions and
   exact logical fitted state; depths 4/6/8 engaged at their declared
   boundaries. All 19 M5 fingerprints matched and every floor passed.
10. **Costs, passed and failed conditions:** every invariant and M5 condition
    passed. M5 resource readings are non-ranking telemetry.
11. **Limitations/non-claims:** these checks establish mechanism scope and
    regression safety, not quality. They authorize no sports, fresh, TabArena,
    lockbox, shipping, release, or default claim.
12. **Terminal decision/next action:** pre-quality disposition `pass`; spend
    exactly one frozen M6 v3 inspection. No favorable rerun is permitted after
    launch-manifest creation.

### 49. T7b automatic scalar-RMSE depth M6 v3 inspection 1 (2026-07-22)

1. **Execution date/source:** 2026-07-22; clean published harness
   `a1eb76071ca72a52494f6cca6022ea8ace8d5394`, clean pre-mechanism control
   `e23d2b164f10374b1c0e02521c33fc96d48980da`, and clean published private
   candidate `41e948f0c53b1d124e16071a7fa66eba47d084d3`.
2. **Comparators:** public-default control versus public-default candidate;
   the only intended behavior change was automatic scalar-RMSE CatBoost depth.
3. **Evidence class:** one spent general-development inspection under M6
   quality-successor-v3; ranking-eligible for this development decision,
   never shipping/default evidence.
4. **Data/splits:** exact frozen 60-cell medium grid: ten regression and
   classification datasets, three seeds, unweighted and stress-weighted
   policies, fixed adapter splits and fingerprints.
5. **Arms/policies:** `control_default` and `candidate_default`; 1,000-round
   public defaults, four threads, with six classification datasets retained
   as exact quality no-ops. Automatic L2 was unchanged.
6. **Environment/repeats:** `darko311`, macOS arm64, four threads, three
   repeats in each fresh worker; exclusive-machine audit found no conflicting
   benchmark process before the create-only launch manifest was written.
7. **Runner/command:** `python benchmarks/run_t7b_automatic_depth_v1.py
   --control /private/tmp/darkofit-t7b-auto-depth-control-e23d2b1 --candidate
   /private/tmp/darkofit-t7b-auto-depth-v1-20260722 --output-prefix
   /private/tmp/t7b_automatic_depth_v1_m6_inspection1_20260722`.
8. **Hashes:** launch manifest
   `7eb95710c761f0682c00cf4b5971233089c70e654c5e5adc316d5388d933dc46`;
   raw CSV `e8e651459fafdea7ace0d298ccedd2c8d87145b945928111d475a007b955bafe`;
   result `7af0c480221b5886c7bbf41f810147663d9da6e2c4171a70bc9db3a431eebb28`;
   generic manifest `dbb47702f4e7992f34e653ea1155a8638e4e1945dbda0da1eb582345c73c32c7`;
   terminal attestation
   `b925aab09fdd71ca0f8887e1d3a4023c20412b2eefc337f2a2a7c1d5a267f598`.
9. **Primary results:** equal-cell geometric-mean ratio `0.992921`; worst
   dataset-group ratio `1.011124` on `diabetes_resampled`; worst LOO ratio
   `1.001230` when omitting `wide_numeric_reg`; worst cell `1.037673`.
10. **Costs, passed and failed conditions:** aggregate `<=1.000000`, worst
    group `<=1.020000`, and worst LOO `<=1.003000` all passed. Non-gating
    geomeans were `0.849501` fit, `0.933229` predict, and `0.993834` RSS.
11. **Limitations/non-claims:** one dependent, fixed, spent medium panel; no
    fresh, sports, public, release, TabArena, lockbox, or shipping claim.
12. **Terminal decision/next action:** frozen disposition `advance`.
    Inspection 1 is spent and rerun is false. Keep candidate `41e948f0`
    private and unmerged; freeze a separate spent-sports contract before any
    sports outcome is inspected.

### 50. T7b automatic depth spent-sports successor (2026-07-22/23)

1. **Execution date/source:** 2026-07-22/23; clean published contract harness
   `ac51c3e3379f855ba960f684375bee49cf0910e4`, clean control
   `e23d2b164f10374b1c0e02521c33fc96d48980da`, and exact clean private
   candidate `41e948f0c53b1d124e16071a7fa66eba47d084d3`.
2. **Comparators:** the same scalar-RMSE CatBoost policy in both arms except
   that the candidate's automatic samples-per-feature rule resolved depth 4;
   control retained depth 6 and both retained L2 `3.0`.
3. **Evidence class:** one spent, player-disjoint, Tier-E sports-development
   successor under `t7b-automatic-depth-spent-sports-v1-20260722`; it can
   prioritize fresh Tier-D design but cannot ship or change a default.
4. **Data/splits:** exact frozen sports-panel-v2 cache, seasons 2014--2016
   crossed with three targets. Fit rows exclude held teams and use player
   groups for validation; primary RMSE uses cold-player held-team rows.
5. **Arms/policies:** one control and one candidate fit per case; 600 maximum
   rounds, patience 30, `use_best_model=True`, `refit=False`, 0.15 group
   validation, random state 4, and no sample weights.
6. **Environment/repeats:** `darko311`, macOS arm64, four fixed threads,
   fresh worker per arm/case, same-arm two-round warmup, one quality fit only;
   no-conflicting-benchmark audit passed before manifest creation.
7. **Runner/command:** `python
   benchmarks/run_t7b_automatic_depth_sports_v1.py --control
   /private/tmp/darkofit-t7b-auto-depth-control-e23d2b1 --candidate
   /private/tmp/darkofit-t7b-auto-depth-v1-20260722 --panel-cache
   /Users/konstantinmedvedovsky/code/darkofit/.cache/basketball-sports-panel-v2/panel.csv
   --cache-dir
   /private/tmp/t7b_automatic_depth_sports_v1_cache_20260722 --output-prefix
   /private/tmp/t7b_automatic_depth_sports_v1_inspection1_20260722`.
8. **Hashes:** contract
   `ac5a745378a086ed119af1d55a68e961ea95e1f74c4f307dce72ad9b6717fe1b`;
   launch `07567f2585df0183bbd0f6dee9b3c18d678e28b3280ddd41c21331a23439bac1`;
   raw `31b4d18576ed35efae3fe89e07375f18b82c02668586a562aa9969d1c9f0830d`;
   result `1ec0d2d37ef75195b66b779ec94920e05f5047147538de6eb17622947fd1a0da`;
   terminal `180e7ea418b4a5e53c0672c2c5b5c1672824dc83fc3f9b3e279bca0cd19d9644`.
9. **Primary results:** cold-player equal-lineage ratio `0.950266`, held-team
   `0.951078`; all nine cold-player lineages improved. Season ratios were
   `0.972028`, `0.923605`, and `0.955809`; clustered p95 was `0.966591` and
   worst leave-one-season-out was `0.963884`.
10. **Costs, passed and failed conditions:** all six frozen quality,
    uncertainty, concentration, and harm gates passed. Single-run non-gating
    telemetry was `0.601281` fit, `1.038138` predict, `0.988196` RSS, and
    `0.732452` archive bytes; prediction ratios were visibly noisy.
11. **Limitations/non-claims:** three dependent, already-spent seasons, not
    nine independent datasets or fresh confirmation. No speed, merge,
    default, public API, M2, TabArena, release, or lockbox claim is authorized.
12. **Terminal decision/next action:** disposition
    `eligible_for_fresh_tier_d_design`; inspection 1 is spent and no rerun is
    allowed. Candidate remains private/unmerged. A powered fresh Tier-D design
    requires a separate owner authorization before data access.

### 51. Automatic linear-selector v2 Protein attempt 2 and terminal replay (2026-07-22/23)

1. **Execution date/source:** 2026-07-22/23; clean published attempt-2 harness
   `4cb9debef576025590ca69a10fe3dd85080fbb6a`, published artifact-only replay
   analyzer `f72b549`, and exact clean private candidate
   `a53d4bf543534678189d87d88dcad87dd2a8bd8f`.
2. **Comparators:** constant leaves, the automatic 3% validation-margin
   selector, and explicit linear leaves under the same Protein configuration.
3. **Evidence class:** three already-spent Protein release-ladder coordinates,
   followed by a dependent, prior-outcome-known historical artifact replay.
   Neither is fresh or independent confirmation.
4. **Data/splits:** OpenML task 363693 (`physiochemical_protein`), coordinates
   `(repeat, fold, seed)` `(0,0,0)`, `(1,1,1001)`, `(2,2,2002)`; all direct-
   loader split fingerprints matched the immutable v0.11 ladder rows before
   manifest creation. Replay covered 14 smooth/process, three categorical,
   three noisy-tabular, and one group-safe sports lineage.
5. **Arms/policies:** cyclic Latin rotation of constant, automatic, and
   explicit-linear arms; automatic selects linear only at relative validation
   improvement `>=0.03`, then refits from scratch under original full-fit
   semantics.
6. **Environment/repeats:** isolated Python 3.11 run environment with OpenML
   0.15.1, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.5.2; 14 threads, fresh
   worker per arm, one completed cell per arm/coordinate, and exclusive-machine
   audit passed. The replay fitted no model.
7. **Runner/command:** `/private/tmp/darkofit-selector-protein-runenv-20260722/bin/python
   benchmarks/run_automatic_linear_selector_v2_protein_attribution_attempt2.py
   --candidate-source /private/tmp/darkofit-smooth-selector-20260722
   --tabarena-source /private/tmp/tabarena-m2-4cd1d25 --output-prefix
   /private/tmp/automatic_linear_selector_v2_protein_attribution_attempt2_20260722`;
   then `python benchmarks/analyze_automatic_linear_selector_v2_guardrail_replay.py
   --output /private/tmp/automatic_linear_selector_v2_guardrail_replay_20260722.json`.
8. **Hashes:** attempt-2 manifest
   `d6cbee2249046bc8eca05080eea38457c21d0d130076a59839b313c52c8b54b7`;
   raw `0caaa2f97fd527976233f6511267c3df2b6487bc8d5a665d87c9fad2c3b11be7`;
   result `4b75f4ae048e926ec07bf3a17c4a9e9356b52a7adfd869409594cc3878f7e61c`;
   replay `1d0ac7eedbcc86dd83b47f826e77efb70071381d88a27a68c6dc61d31e707122`;
   terminal attestation
   `e35b6907ce1872a9f01ce5359d2b49064ef2bcd112b5952fcdd53db6a166387a`.
9. **Primary results:** Protein automatic/constant equal-coordinate ratio
   `0.968638`; coordinate ratios `0.951434`, `1.000000`, `0.955225`. Coordinate
   1 margin was `0.025179`, so automatic declined and did not match explicit
   linear. Historical combined ratio was `0.962739` across 21 lineages, with
   worst lineage/split `1.0` and worst LOO `0.989084`.
10. **Costs, passed and failed conditions:** both Protein harm gates passed;
    the required all-coordinate selector/exactness gate failed. Automatic fit,
    predict-call, and peak-RSS geomeans were `9.7597 s`, `0.01442 s`, and
    `340.65 MB` versus constant `2.2863 s`, `0.00872 s`, and `323.95 MB`;
    descriptive only.
11. **Limitations/non-claims:** attempt 1's environment failure stays terminal;
    attempt 2 is spent and may not rerun. Historical replay has prior outcome
    knowledge and dependence, ran no candidate code, and cannot reverse the
    Protein close. No fresh, merge, default, public, release, TabArena,
    lockbox, quality, or speed claim.
12. **Terminal decision/next action:** exact candidate disposition `killed`.
    Preserve its branch and artifacts; do not rerun or merge. The next
    separately governed mechanism slot is categorical crosses.

### 52. Group-centered categorical crosses v1 general development (2026-07-22/23)

1. **Execution date/source:** 2026-07-22/23; clean published evidence harness
   `9c63f8171d3e57add4e3eb33681c5b6b764ff628`, clean control
   `01ae675bcebdf435988ce9e0d493d0fc0017f54a`, and exact clean private
   candidate `c3f2608cd3033cfc00aa0737897a92ed868b5865`.
2. **Comparators:** default scalar-RMSE CatBoost policy versus the same policy
   with the frozen automatic group-centered-cross selector; all ineligible
   cells executed the exact control lane.
3. **Evidence class:** invariants and non-ranking M5, one metadata-only M6
   engagement companion, and one spent general-development inspection under
   M6 quality-successor-v3. It ranks only this development decision.
4. **Data/splits:** exact frozen 60-cell medium grid: ten regression and
   classification datasets, three seeds, unweighted and stress-weighted
   policies, fixed adapter splits and fingerprints. Engagement occurred only
   on the six `categorical_reg` cells.
5. **Arms/policies:** `control_default` and `candidate_default`; 1,000-round
   public defaults and four threads. Eligible candidates used a deterministic
   15% audition holdout, bounded top-four-numeric by top-three-categorical
   target-free crosses, and strict lower validation RMSE selection.
6. **Environment/repeats:** `darko311`, macOS arm64, four threads, three
   repeats in each fresh worker; exclusive-machine audit found no conflicting
   benchmark process before the create-only launch manifest was written.
7. **Runner/command:** `python
   benchmarks/run_group_centered_categorical_crosses_v1.py --control
   /private/tmp/darkofit-catcross-control-01ae675 --candidate
   /private/tmp/darkofit-catcross-v1-20260722 --invariants
   /private/tmp/catcross_v1_invariants_20260722.json --m5-result
   /private/tmp/catcross_v1_m5_20260722.json --engagement
   /private/tmp/catcross_v1_m6_engagement_20260722.json --output-prefix
   /private/tmp/catcross_v1_m6_inspection1_20260722`.
8. **Hashes:** invariants
   `9cf91f57894cc66776a4128826398dd8fe37ba4ef79466094afe2b62db9b95b4`;
   M5 `254fcc551668a619d8589b617a55a67210f17459b65323e2117929bcda406c62`;
   engagement `b1b7f1c372bfb502fa6d66ece6def5cdbe305b8e311fc59addc342cca0f78940`;
   launch `45ec167150ecece277dd72474cc256fba8f0ae18ef0a5eabbe3f14dd0864277a`;
   raw `3b9c4301b0d593cea03634d8ecd9aa34f151fface8453d650b820f0742fe84ec`;
   result `fa2bdf46e1439a5b1cbc043541db133c1d6a854fc04d26e40709fbcf838f6fa4`;
   generic manifest
   `75ddda2ecf21d2ea7e65847688206ee56caddafb17ec4605a3db027333342837`;
   terminal attestation
   `bd02744c4afcd2117cd2f9db333bc33f2aa877402a162382936d96c9ab23cef7`.
9. **Primary results:** equal-cell geometric-mean ratio `0.992606`; selected
   `categorical_reg` group `0.928473`; worst dataset-group and LOO ratios both
   `1.000000`. Its six selected coordinates ranged `0.900282`--`0.964085`;
   every other coordinate was exactly `1.000000`.
10. **Costs, passed and failed conditions:** aggregate `<=1.000000`, worst
    group `<=1.020000`, and worst LOO `<=1.003000` all passed. Non-gating
    all-cell fit/predict/RSS geomeans were `1.083311`, `1.004975`, and
    `1.008625`; on engaged cells they were `2.145607`, `1.045341`, and
    `1.096926`.
11. **Limitations/non-claims:** one dependent, fixed, spent medium panel. The
    mechanism engaged on one synthetic family and its two-audition fit cost is
    material. No fresh, sports, public, merge, default, release, TabArena,
    lockbox, or general quality claim is authorized.
12. **Terminal decision/next action:** frozen disposition `advance`, meaning
    only `eligible_for_mechanism_specific_spent_attribution`. Inspection 1 is
    spent and rerun is false. Keep candidate `c3f2608c` private and unmerged;
    any attribution needs a separate frozen contract.

### 53. B3 parallel ensemble-members v1 (2026-07-22/23)

1. **Execution date/source:** 2026-07-22/23; clean published evidence harness
   `5a236e4f37d429fa55c40a6ebc65dc9b2b6d00f5`, clean sequential control
   `c4dae58fcf7a8d456533ba2d9b469f039adc453c`, and exact clean private
   candidate `5116470e21675f8a869ee7a84145eb2a663ed809`.
2. **Comparators:** public sequential ensemble-v3 `1 worker × 14 threads`
   versus the private process-parallel route `7 workers × 2 threads`; both
   used the same eight deterministic member plans.
3. **Evidence class:** Tier-E behavior and resource invariants plus one spent
   general-development timing inspection. It ranks only the frozen B3 v1
   engineering decision.
4. **Data/splits:** four already-spent general cases from the ensemble-v3
   characterization: Friedman numeric, categorical regression, numeric
   binary, and categorical multiclass, with exact frozen generators, splits,
   weights, and fingerprints.
5. **Arms/policies:** eight public v3 members, 600 maximum rounds, patience
   30, validation fraction 0.15, random state 4; sequential `1x14` control and
   private `7x2` candidate. No sampling, member-policy, preprocessing,
   prediction, or archive-format change.
6. **Environment/repeats:** `darko311`, macOS arm64, 14 physical cores and 24
   GiB RAM; three paired blocks with rotated arm order, fresh outer workers,
   and both first-use cold and immediately repeated steady fits. The
   exclusive-machine audit found no conflicting benchmark process.
7. **Runner/command:** `python benchmarks/run_b3_parallel_ensemble_v1.py
   --control /private/tmp/darkofit-b3-control-c4dae58 --candidate
   /private/tmp/darkofit-b3-candidate-20260723 --invariants
   /private/tmp/b3_parallel_ensemble_v1_invariants_20260723.json
   --output-prefix
   /private/tmp/b3_parallel_ensemble_v1_inspection1_20260723`.
8. **Hashes:** contract
   `306fbea95a1e33e0bee22b937d9dd15b2ff205f3479b2c39fa116786f6d5b662`;
   invariants
   `9797ebc23bbc790835c2f88428129746c0cbb7744adb158d80f763db7c62e9db`;
   launch `cdf93e46af80c560d7e809f51bb97d053981738b69cc75d9d55ac014f68ee5dd`;
   raw `7ba73e1d113d8cf412318201268ecc768cfc0102e61ed66696fd473112d344cc`;
   result `9d1e97e23e1bec0ae4449e4c0a9c842bddaf87d45adc2fea6a8e827791d7bb35`;
   terminal attestation
   `2b6a43dbf71435c87dab16ba48b77dfb606fb5f343f5ce4f966498a04921025e`.
9. **Primary results:** cold equal-case fit ratio `0.684187`, cold case
   medians `0.497480`/`0.867852`/`1.075049`/`0.462523`, and cold worst LOO
   `0.770631`. Steady equal-case fit ratio `0.260379`, worst case median
   `0.362653`, and worst LOO `0.286485`.
10. **Costs, passed and failed conditions:** behavior exactness, execution
    integrity, hybrid RSS, and every steady speed gate passed. The cold all-
    case gate failed because Friedman numeric was 7.5% slower. Maximum
    candidate process-tree peak RSS was `2,399,551,488` bytes, below 6 GiB;
    archive bytes and predictions were exact. Prediction time is raw-artifact
    telemetry.
11. **Limitations/non-claims:** one dependent, fixed, spent four-case panel on
    one 14-core ARM machine. No rival, fresh, sports, public, merge, default,
    release, TabArena, lockbox, or portable-speed claim is authorized.
12. **Terminal decision/next action:** frozen disposition `kill`. Inspection
    1 is spent, rerun is false, and private candidate `5116470e` stays
    unmerged. A warm-worker lifecycle or short-fit activation rule is a new
    mechanism requiring a new identity and owner authority; proceed now only
    to the authorized powered fresh Tier-D panel design.

### 54. T7b automatic-depth shared Tier-D power design (2026-07-23)

1. **Execution date/source:** 2026-07-23; clean published power-design source
   `f895e480fcd2ffc117dc85fd9bd0b9bf0d492414`, exact private candidate
   `41e948f0c53b1d124e16071a7fa66eba47d084d3`, and control
   `e23d2b164f10374b1c0e02521c33fc96d48980da`.
2. **Comparators:** no fresh model comparison occurred. The design contrasts
   the frozen automatic-depth candidate/control estimand under a synthetic
   structured effect alternative sized from already-spent profiles.
3. **Evidence class:** Tier-D design-time power only. It qualifies a panel
   template, not the candidate's quality and not fresh access or a default.
4. **Data/splits:** no prospective identities or targets. Sizing used the two
   complete changed general group profiles and three spent season clusters
   (five lineage clusters, 21 nested coordinates), then proposed an anonymous
   32-lineage template balanced across depth 4/depth 8 and numeric/
   categorical-or-grouped strata.
5. **Arms/policies:** primary alternative scales every spent log effect and
   within-lineage deviation to 20% of observed magnitude; sensitivities use
   10%, 15%, and 25%. The exact Tier-D aggregate, cluster-bootstrap,
   leave-one-favorable-out, worst-lineage, and two branch-direction gates are
   applied.
6. **Environment/repeats:** `darko311`, deterministic NumPy simulation; 5,000
   outer panels, 5,000 lineage-bootstrap draws, seeds `20260723`/`20260724`,
   32 lineage clusters and three coordinates per lineage.
7. **Runner/command:** `python -m benchmarks.tier_d_fresh_power_design`. One
   prior direct-file invocation failed before importing the contract or
   running the simulator; it produced no result. The unchanged published
   module entry point produced the sole create-only outcome.
8. **Hashes:** general sizing input
   `7af0c480221b5886c7bbf41f810147663d9da6e2c4171a70bc9db3a431eebb28`;
   sports sizing input
   `1ec0d2d37ef75195b66b779ec94920e05f5047147538de6eb17622947fd1a0da`;
   contract `1aa89083b16ed31ee816f005bc961751b3b22785b2ec1ec2a54fc1e2d0d94595`;
   builder `f1482d20fbc6ad2f84d4bdc9a338adf4d6d87cb7a4fe640d997aeb9f9ee93fce`;
   result file
   `5b767ce0a27e09d479bb18d6314d9adce3bbac78380aeff481639b13152714ad`;
   result self-hash
   `735604d24828f6294e60e023ceda053caf272095c50ae83310593833ccdd07d1`.
9. **Primary results:** the 20% scenario implies true geometric-mean ratio
   `0.991077`; 4,990/5,000 panels passed, for power `0.998000` and one-sided
   95% Wilson lower `0.996657` against the required `0.800000`.
10. **Costs, passed and failed conditions:** primary point and Wilson power
    floors passed. Component pass probabilities were `0.9984` point,
    `1.0000` bootstrap, `1.0000` LOO, `0.9996` worst-lineage, and `1.0000`
    branch direction. The 10% sensitivity failed power at `0.217600`; 15%
    and 25% passed at `0.957600` and `0.991800`. Operational costs are not
    simulated and must receive separately harm-justified frozen budgets.
11. **Limitations/non-claims:** the sizing evidence selected the candidate;
    joint 20% shrinkage is a structured alternative, not a conservative bound
    or effect estimate. DarkoFit evidence exercised depth 4, so depth 8 is a
    prospective transport assumption protected by its own fresh direction
    gate. No fresh, quality, public, merge, default, release, TabArena,
    lockbox, or product claim is authorized.
12. **Terminal decision/next action:** `design_power_qualified`. The design
    work is complete. A separate owner decision is required before exact
    registry/execution-contract work, and another explicit authorization must
    precede any fresh target access or confirmation run. Candidate `41e948f0`
    stays private and unmerged.

### 55. T7b automatic-depth fresh Tier-D registry/contract freeze (2026-07-23)

1. **Execution date/source:** 2026-07-23; contract worktree based on published
   authorization commit `9f0662e1c87f992c474fa37ebcaf30054c669231`,
   candidate `41e948f0c53b1d124e16071a7fa66eba47d084d3`, and control
   `e23d2b164f10374b1c0e02521c33fc96d48980da`.
2. **Comparators:** no model comparison. This checkpoint binds the future
   control/candidate pair and exact one-shot analyzer.
3. **Evidence class:** prospective Tier-D execution infrastructure and
   target-blind contamination review. It is not quality evidence.
4. **Data/splits:** OpenML task/dataset metadata only; no feature matrix or
   target value was downloaded. The frozen registry has 32 primaries plus two
   ordered same-stratum reserves in each of four 8-lineage strata, three
   coordinates per active lineage, four group-safe low-density primaries,
   frozen row/group hashes, weights, and the 3,250-row-per-feature high-density
   cap.
5. **Arms/policies:** unchanged automatic-depth candidate versus its exact
   control. The owner addendum makes terminal GO authorize v0.12 default
   promotion and terminal NO-GO close the default candidate while preserving
   P3.
6. **Environment/repeats:** registry generation used `darko311` and exact Git
   history at DarkoFit `9f0662e1` and ChimeraBoost `6a76586d`. The future run
   is frozen to fresh 14-thread workers, 600 maximum rounds, three coordinates,
   and three fixed 50,000-row prediction repeats.
7. **Runner/command:** registry:
   `python benchmarks/build_t7b_automatic_depth_fresh_tier_d_registry.py`;
   verification adds `--verify-existing`. Focused verification:
   `python -m pytest -q` over the fresh/power/development/sports/thread suites.
8. **Hashes:** registry file
   `ce539f5fefdba07a4983904921a5841e35371765304cd11f92f3b5759db25878`;
   registry self-hash
   `704393992270b80679a9ad76c0f65174df1da87b87e5ca8cdfe0b7ff7ad5cb48`;
   execution contract
   `04156b35517a5701ca7e0cb7a8aad92cb6ea696a09587148dd19dd939c24f25b`;
   runner
   `ce35cf557f1f38939d774afb3bb762967f605362a7eeb578a54c381af7ff8c72`;
   analyzer
   `2e230316513486563961467e113e6e490355c903f6ceede3015d68e00c818dd8`.
9. **Primary results:** all 40 declared identities passed exact pre-freeze
   name/alias/task-ID/dataset-ID history checks in both repositories; the
   registry reproduced byte-for-byte. The 51-test focused suite passed.
10. **Costs, passed and failed conditions:** no model cost was measured.
    Quality gates are copied unchanged by hash from the qualified power design.
    Fit/predict Pareto and hybrid RSS gates are frozen prospectively; archive
    bytes are telemetry.
11. **Limitations/non-claims:** near-lineage feature/target fingerprinting,
    target validity, realized splits/branches, and model outcomes remain
    unopened until this checkpoint is clean, committed, and published. No
    default has changed; no v0.12 release, TabArena, CTR23, or lockbox access is
    authorized.
12. **Terminal decision/next action:** publish this single freeze commit, run
    the create-only target preflight, and launch exactly one inspection only
    if all 32 slots resolve cleanly. Launch-manifest creation is terminal and
    permits no rerun or partial read.

### 56. T7b fresh preflight v1 harness failure and v2 freeze (2026-07-23)

1. **Execution date/source:** 2026-07-23; published v1 freeze
   `987b1c900667519583438900f8306a19b4ac37e5`, unchanged candidate
   `41e948f0c53b1d124e16071a7fa66eba47d084d3`, and unchanged control
   `e23d2b164f10374b1c0e02521c33fc96d48980da`.
2. **Comparators:** no model comparison and no dataset comparison.
3. **Evidence class:** pre-launch harness failure plus prospective Tier-D
   execution-infrastructure supersession. It is not quality evidence and did
   not spend the fresh inspection.
4. **Data/splits:** the v1 command failed while importing the local
   contamination helper, before the first OpenML lineage load. No feature
   matrix or target value was loaded and no split was realized. V2 reuses the
   immutable 32-primary/8-reserve v1 registry and every frozen split rule.
5. **Arms/policies:** candidate, control, automatic-depth policy, terminal
   decisions, gates, power assumptions, and cost rules are byte-for-byte
   unchanged from v1. The only repair places the published DarkoFit root first
   on `sys.path` before local `benchmarks` imports in the runner and analyzer.
6. **Environment/repeats:** `darko311`, 14 logical CPUs planned; no worker,
   warmup, repeat, or model fit started.
7. **Runner/command:** failed v1 command:
   `python benchmarks/run_t7b_automatic_depth_fresh_tier_d.py preflight
   --output benchmarks/t7b_automatic_depth_fresh_tier_d_preflight_20260723.json`.
   V2 verification used an isolated module-load probe plus the same 51-test
   focused fresh/power/development/sports/thread suite.
8. **Hashes:** v1 failure record
   `37208219b2af77a89f2855981b9b793d39834a13fd6fb0dfd7341684b0f302cc`;
   v2 protocol
   `2c1151ef742592682acd90d8aa7247b43fa17ddd1667ada56874369a917e7d4f`;
   v2 contract
   `1a8e8f5e68fb557c504ddc25688ff9aa319436282cbc925b37df5a87e880dac1`;
   repaired runner
   `f7d3dc9651a4559470921b9b9d686cff8fbe9961e378939009cb697974032deb`;
   repaired analyzer
   `498671de591cb51b998dcf6c07d842d4ce7070f630c1eac8648eb3c1b3ce5274`.
9. **Primary results:** the import probe resolved `benchmarks` to this
   worktree, not the unrelated `sift` checkout. The complete focused suite
   passed: 51 tests.
10. **Costs, passed and failed conditions:** no quality or resource gate was
    evaluated. V1 preflight failed its harness-integrity prerequisite; v2's
    source-hash and contract checks passed.
11. **Limitations/non-claims:** v2 still has not read a feature matrix or
    target value, fitted a model, created a launch manifest, or inspected a
    result. No default, public API, release, TabArena, CTR23, or lockbox claim
    follows.
12. **Terminal decision/next action:** v1 preflight identity is closed by a
    create-only failure record. Publish the v2 freeze, run one create-only v2
    preflight, and create the sole launch manifest only if all 32 slots pass.
    The one-shot remains unspent.

### 57. T7b automatic-depth fresh Tier-D preflight closure (2026-07-23)

1. **Execution date/source:** 2026-07-23; published v2 harness
   `383fc4c5518766f905911ac4657fed9a309bb375`, candidate
   `41e948f0c53b1d124e16071a7fa66eba47d084d3`, and control
   `e23d2b164f10374b1c0e02521c33fc96d48980da`.
2. **Comparators:** no model comparison; neither arm was fitted.
3. **Evidence class:** value-free fresh-registry preflight under the
   prospectively frozen Tier-D execution contract. It is neither development
   nor confirmation-quality evidence.
4. **Data/splits:** the preflight followed the frozen 32-slot OpenML registry,
   exact/near-lineage fingerprint review, and exact branch/split attestations.
   It terminated because slot `high_density_numeric_02` had no eligible frozen
   identity. The contract's all-32 requirement forbids a smaller or recomposed
   panel.
5. **Arms/policies:** unchanged automatic-depth candidate and control were
   bound but never run. P3 remains unaffected.
6. **Environment/repeats:** `darko311`; no 14-thread worker, warmup, model
   repeat, or timed operation started.
7. **Runner/command:**
   `python benchmarks/run_t7b_automatic_depth_fresh_tier_d.py preflight
   --output benchmarks/t7b_automatic_depth_fresh_tier_d_preflight_v2_20260723.json`.
8. **Hashes:** v2 execution contract
   `1a8e8f5e68fb557c504ddc25688ff9aa319436282cbc925b37df5a87e880dac1`;
   create-only terminal record
   `36447f32695910d62b76b22d3f73d17346391244c55a9ff80cd9c875c9a995cf`.
   No preflight output, launch manifest, raw result, or analyzer result exists.
9. **Primary results:** `all_32_slots_required` failed at
   `high_density_numeric_02`. No target statistic, candidate metric, control
   metric, partial panel, or quality outcome was inspected.
10. **Costs, passed and failed conditions:** no fit, prediction, RSS, or
    archive cost was measured. Registry completeness failed; all quality and
    cost gates are unevaluated.
11. **Limitations/non-claims:** this is not a GO or NO-GO result about the
    automatic-depth mechanism. It supplies no shipping evidence and does not
    justify a default, public API, or release change. The prior development
    evidence remains historical only.
12. **Terminal decision/next action:** close this fresh execution identity
    before launch with no rerun, registry expansion, or panel recomposition.
    Automatic depth remains private and unpromoted; P3 remains available under
    its existing Tier-E basis. Any future confirmation requires a genuinely
    new prospective design and explicit owner authorization.

### 58. T7b automatic-depth P1-v3 enumeration harness (2026-07-23)

1. **Execution date/source:** 2026-07-23; clean branch based on published
   `main` at `974b5eeaad96df28f3577335a0defab6147428e7`.
2. **Comparators:** no model arms or model comparison. The resource audit will
   inspect the 40 exact identities already named in the immutable v1 registry.
3. **Evidence class:** prospective pre-design fillability infrastructure
   authorized by `R2_PLAN.md` P1-v3. It is not quality evidence and freezes no
   confirmation panel.
4. **Data/splits:** no data was loaded while authoring this checkpoint. The
   future create-only enumeration verifies each concrete identity
   independently: loadability, history/fingerprint contamination, all three
   deterministic split coordinates, group disjointness, and the realized
   depth branch.
5. **Arms/policies:** no candidate/control fit. Existing value-free v1/v2
   resource contact is disclosed; no abstract-slot replacement or outcome
   selection is permitted.
6. **Environment/repeats:** future enumeration is pinned to `darko311` and
   records exact module paths/versions plus DarkoFit and ChimeraBoost commits.
   No timed workers or repeats are involved.
7. **Runner/command:** after clean commit and publication:
   `python benchmarks/enumerate_t7b_automatic_depth_fresh_tier_d_v3.py
   --output benchmarks/t7b_automatic_depth_fresh_tier_d_v3_enumeration_20260723.json`.
   Focused verification ran the new tests plus the existing fresh execution,
   power-design/result, and thread-state suites.
8. **Hashes:** protocol
   `f352081f33a5729849c2fa727437f02aa92c08e831e09b728350719aeaeddbf7`;
   runner
   `df077068a2938602ab4bc03b55d628dd761d6a5044d12bf073bb3b38dd8c71ed`;
   R2 authorization
   `edc583db280f2706e656ae635a82b76e5ac2a3da9a0a5b6d241b854a093918dd`;
   v1 registry file
   `ce539f5fefdba07a4983904921a5841e35371765304cd11f92f3b5759db25878`;
   reused v2 eligibility helpers
   `f7d3dc9651a4559470921b9b9d686cff8fbe9961e378939009cb697974032deb`.
9. **Primary results:** harness validation passed, including independent
   identity evaluation and narrow disclosure-path classification: 27 tests
   passed.
10. **Costs, passed and failed conditions:** no model cost or evidence gate
    was evaluated. The future audit fails as a whole on unexpected harness,
    dependency, network, or environment errors rather than mislabeling them as
    dataset ineligibility.
11. **Limitations/non-claims:** this checkpoint proves only that the
    enumeration machinery is internally consistent. It makes no claim about
    how many resources are fillable, whether an as-built panel has 80% power,
    or whether automatic depth should ship.
12. **Terminal decision/next action:** publish the clean harness, run the
    create-only enumeration once, and write a dated pre-design note naming the
    verified resources. Only then may prospective power be recomputed; the
    confirmation freeze and fresh run remain separately owner-gated.

### 59. T7b P1-v3 enumeration v1 failure and v2 supersession (2026-07-23)

1. **Execution date/source:** 2026-07-23; published v1 enumeration harness
   `44cc2b086289218e9674b9e9a8eeac51cf5304d8`.
2. **Comparators:** no model arms, model fits, or quality comparisons.
3. **Evidence class:** pre-design harness failure and prospective
   infrastructure supersession. Neither v1 nor v2 is confirmation evidence.
4. **Data/splits:** v1 rejected all 40 declarations on repository-history
   self-matches before loading any resource, so no feature matrix, target, or
   split was accessed. V2 retains the same 40 identities and every substantive
   loadability, contamination, split, and branch check.
5. **Arms/policies:** no automatic-depth candidate or control was run.
6. **Environment/repeats:** `darko311`; no timed worker or repeat. V1's failure
   occurred inside local path classification.
7. **Runner/command:** v1:
   `python benchmarks/enumerate_t7b_automatic_depth_fresh_tier_d_v3.py
   --output benchmarks/t7b_automatic_depth_fresh_tier_d_v3_enumeration_20260723.json`.
   V2 changes only removal of a valid 40-hex revision prefix from `git grep`
   output before applying the unchanged disclosure allowlist.
8. **Hashes:** immutable v1 raw artifact
   `0077494a73eb82f3454fd0a4f2ededbd833e35884b82a4dc3666f805496dfdfa`;
   create-only failure record
   `717e1b33a48701a9e5f6218c73f4125ec107514983629b210f311592832d4f90`;
   v2 protocol
   `9d6e76bac717f9671b02b8c76fad3466515d63a6407c4c889ee594ee06c94f89`;
   v2 runner
   `c883a831db59bcf4da23d38bba2e14c4484d0b4295c300f944f8b656e61e2fb7`.
9. **Primary results:** v1's reported 0/40 was invalid: each rejection named
   only the P1 registry/declaration files with a revision-prefixed path. The
   added regression tests cover both prefixed disclosure and non-disclosure
   paths; the focused suite passed 27 tests.
10. **Costs, passed and failed conditions:** no data or model cost was
    measured. V1 failed path-normalization integrity; no dataset eligibility
    condition was genuinely evaluated.
11. **Limitations/non-claims:** the v1 0/40 count cannot inform panel
    composition or power. V2 still has not run and supplies no fillability,
    power, quality, or shipping claim.
12. **Terminal decision/next action:** close v1 as an invalid harness result,
    publish the v2 path-normalization repair, and run a distinct create-only
    v2 enumeration. The candidate pool is unchanged and the fresh
    confirmation inspection remains unspent.

### 60. T7b P1-v3 as-built resource enumeration (2026-07-23)

1. **Execution date/source:** 2026-07-23; published DarkoFit enumeration head
   `a83530fe13a80be9d74dad9dc7d943b636ec1922`; ChimeraBoost history pin
   `6a76586dfdff90275e7e816f25e35c927b8527fb`.
2. **Comparators:** no candidate/control fit and no model comparison.
3. **Evidence class:** P1-v3 pre-design fillability audit. It is fresh
   resource metadata/fingerprint evidence but not quality evidence.
4. **Data/splits:** all 40 concrete v1-registry identities were evaluated
   independently. Thirty-two loaded and passed exact history/fingerprint,
   target-validity, schema, three-coordinate split/group, and branch checks:
   9 low numeric, 8 low categorical/grouped, 5 high numeric, and 10 high
   categorical/grouped; 17 depth-4, 15 depth-8, and three group-safe.
5. **Arms/policies:** no model arms. No abstract-slot replacement occurred;
   the eight rejected resources remain rejected under their declared roles.
6. **Environment/repeats:** `darko311`, Python 3.11, NumPy 2.2.6, pandas
   2.2.3, sklearn 1.7.1, Numba 0.61.2, OpenML 0.15.1, macOS arm64, 14 logical
   CPUs. No timed repeat.
7. **Runner/command:**
   `python benchmarks/enumerate_t7b_automatic_depth_fresh_tier_d_v3.py
   --output benchmarks/t7b_automatic_depth_fresh_tier_d_v3_enumeration_v2_20260723.json`.
   The sandboxed first invocation failed on DNS before output; the unchanged
   published identity then completed with OpenML network access.
8. **Hashes:** create-only enumeration
   `c7c76259823d6ee4d3ce6202b127d4bed984493e6153775dfb0f05a105b22851`;
   dated pre-design note
   `12b824867fbca67bf91f4a3106290b4dec0342b4e5cbbacf793749be7a3310f8`;
   v2 protocol
   `9d6e76bac717f9671b02b8c76fad3466515d63a6407c4c889ee594ee06c94f89`;
   v2 runner
   `c883a831db59bcf4da23d38bba2e14c4484d0b4295c300f944f8b656e61e2fb7`.
9. **Primary results:** 32 eligible and 8 ineligible. Rejections: four
   declared-numeric roles had categorical inputs, three targets contained
   non-finite values, and one OpenML task target had drifted. All eligible
   identities loaded; no unexpected dependency or environment failure
   remained.
10. **Costs, passed and failed conditions:** only download, normalization,
    fingerprint, and deterministic split work was performed. No fit,
    prediction, RSS, archive, quality, or confirmation gate was evaluated.
11. **Limitations/non-claims:** the 32 identities are verified resources, not
    a frozen panel and not evidence that the candidate works. The old 99.8%
    power figure does not transfer to the new 17/15 branch composition.
12. **Terminal decision/next action:** publish the enumeration and dated
    pre-design note, then prospectively recompute power on exactly these 32
    identities. Design/execution freeze and the fresh run remain owner-gated.

### 61. T7b P1-v3 as-built power-contract freeze (2026-07-23)

1. **Execution date/source:** 2026-07-23; contract branch based on published
   as-built enumeration commit `ce0ba0be55ede9a7ff10a49949664facd76a4d19`.
2. **Comparators:** no model comparison. The future simulation sizes the
   unchanged candidate/control decision only.
3. **Evidence class:** prospective design-time Tier-D power infrastructure,
   not quality evidence and not confirmation authorization.
4. **Data/splits:** exact hash-bound 32-identity census from enumeration v2:
   9/8/5/10 across the four strata, 17 depth-4, 15 depth-8, three group-safe,
   and three already-attested coordinates per lineage.
5. **Arms/policies:** unchanged candidate `41e948f0` and control `e23d2b16`.
   The same v1 spent-effect derivation, 20% retained primary alternative,
   sensitivities, true-ratio cap, and quality gates bind.
6. **Environment/repeats:** design simulation only; 5,000 outer panels and
   5,000 lineage-bootstrap draws with the original seeds and 95% one-sided
   Wilson decision.
7. **Runner/command:** after clean commit/publish:
   `python benchmarks/tier_d_fresh_power_design_v3.py`. Focused validation
   covered exact registry/branch binding and a small both-branch simulation.
8. **Hashes:** contract
   `bcf44533c94312b41ff3efdb2a6d08639ccd69ac22cc8a0050fda021111ed82b`;
   protocol
   `41ea9ac266642ed1e5329aa8fcc1d777b0fc07e7222340b41083a5aae7ed0657`;
   runner
   `4f3de13fcffa3d43b2747093f60845268a5d91b3bea5151e4abde2f4601b5763`;
   tests
   `4e545f4b3d867edcb0b0876185c9612b95e707ea13e244dd4c1aa1304d1d7e71`;
   verified enumeration
   `c7c76259823d6ee4d3ce6202b127d4bed984493e6153775dfb0f05a105b22851`;
   reused simulation engine
   `f1482d20fbc6ad2f84d4bdc9a338adf4d6d87cb7a4fe640d997aeb9f9ee93fce`.
9. **Primary results:** contract and harness verification passed: 22 focused
   tests.
10. **Costs, passed and failed conditions:** no power result or quality gate
    has been evaluated yet. The contract will qualify only if both simulated
    pass probability and one-sided Wilson lower bound reach 80%.
11. **Limitations/non-claims:** effect inputs selected the candidate and the
    depth-8 alternative remains a transport assumption. Power qualification
    would size the panel, not confirm the mechanism or authorize a run.
12. **Terminal decision/next action:** publish this power contract, execute
    its create-only simulation once, and return the result for owner freeze
    review. Combined design/execution freeze and fresh access remain false.

### 62. T7b P1-v3 as-built power result (2026-07-23)

1. **Execution date/source:** 2026-07-23; published power-contract head
   `9c58065bfbd1844448bd1bc8b142e8ae2c6d1060`.
2. **Comparators:** no model fit; the design simulates the future unchanged
   candidate/control paired decision.
3. **Evidence class:** prospective Tier-D design-time power result, not
   candidate-quality or shipping evidence.
4. **Data/splits:** exact 32 verified identities, 17 depth-4 and 15 depth-8,
   with three already-attested coordinates per lineage.
5. **Arms/policies:** candidate `41e948f0`, control `e23d2b16`; unchanged
   spent-effect derivation, gates, and branch-direction rule.
6. **Environment/repeats:** 5,000 outer panels, 5,000 lineage bootstrap
   draws, original deterministic seeds.
7. **Runner/command:**
   `python benchmarks/tier_d_fresh_power_design_v3.py`.
8. **Hashes:** result file
   `d6d572e47c672262b007c436cc048b6259a753097e860357523bcec033085ba8`;
   result self-hash
   `78e74a48e060edfe09e371a4d1b5355a684847c4c2dba16e3966ae5c6ac858c1`;
   dated result note
   `9a0d0802f4d3bc69b9cde886b5bd3199e8e994b9915f0c49e9bbd1aee781275d`;
   contract
   `bcf44533c94312b41ff3efdb2a6d08639ccd69ac22cc8a0050fda021111ed82b`.
9. **Primary results:** `design_power_qualified`; primary pass probability
   `0.998000`, one-sided 95% Wilson lower `0.996657`, both above `0.800000`.
   Retained-effect sensitivities: 10% `0.217600` (lacks power), 15%
   `0.957600` (passes), 25% `0.991800` (passes).
10. **Costs, passed and failed conditions:** primary power and Wilson gates
    passed. The 10% sensitivity failed and is disclosure only. No fit,
    prediction, RSS, archive, or observed-quality gate was evaluated.
11. **Limitations/non-claims:** spent inputs selected the candidate; depth-8
    behavior is transported for sizing. The result does not confirm the
    mechanism or authorize a model run, merge, default, or release.
12. **Terminal decision/next action:** prepare the combined design/execution
    freeze over these exact identities for owner review. Every fresh,
    confirmation, merge, default, release, and lockbox authority remains
    false until separately granted.

### 63. T7b P1-v3 combined execution freeze review (2026-07-23)

1. **Execution date/source:** 2026-07-23; freeze-review branch based on
   published qualified-power commit
   `ff7de35ead18184fbffc505ef4b2912fee8904a8`.
2. **Comparators:** future unchanged candidate `41e948f0` versus control
   `e23d2b16`; no model run in this checkpoint.
3. **Evidence class:** prospective Tier-D design/execution freeze package.
   It is not fresh confirmation or shipping evidence.
4. **Data/splits:** exact hash-bound 32-identity enumeration with 96 fixed
   coordinates, 17 depth-4 and 15 depth-8 lineages, and three group-safe
   lineages. Execution-time discovery and substitution are absent.
5. **Arms/policies:** scalar-RMSE CatBoost `depth=None`, 600 maximum rounds,
   early stopping 30, validation 0.15, best model, no refit, seed 20260723.
6. **Environment/repeats:** future fresh `darko311` workers, 14 threads,
   alternating arm order, two-round same-source warmup, and three fixed
   50,000-row prediction repeats.
7. **Runner/command:** preflight after publication:
   `python benchmarks/run_t7b_automatic_depth_fresh_tier_d_v3.py preflight
   --output benchmarks/t7b_automatic_depth_fresh_tier_d_v3_preflight_20260723.json`.
   Execution has no authorized command until a later owner record exists.
8. **Hashes:** execution contract
   `12ff0db7553b2748eaa75b2e0f0610fa423abc3112df79fb061bb4b59a4dc34d`;
   protocol
   `f6afe07c839aaf2f136e7ec987a0440315a1ebff8dd867f0b4b2199098ec944f`;
   runner
   `a214d392746342cb58468980aae6cdf323ac4670097e28b92926c2b141f66986`;
   analyzer
   `bef784604e1e9da50180b104e6b9fce012ab15953a9c3e3e45b9b0c097e10c36`;
   focused tests
   `bccb68b93ee5b0ae043584bf67aa63463db347bca880cd27705280efbc654d93`;
   owner freeze-review note
   `0097f56c74136c7ea406a8f9a7c2d647077b2c5ba263b7101d44c05c01ace60f`.
9. **Primary results:** exact contract, registry, qualified-power, future
   authorization, preflight, and analyzer bindings passed 32 focused tests.
10. **Costs, passed and failed conditions:** no observed gate was evaluated.
    The frozen future gates require all quality gates plus non-regressing fit
    and prediction geomeans, hard/hybrid RSS, and all integrity checks.
11. **Limitations/non-claims:** the package cannot execute itself. It grants
    no fresh access, model fitting, default, release, TabArena, CTR23, or
    lockbox authority. A later exact owner record is mandatory.
12. **Terminal decision/next action:** publish the freeze-review package,
    create and publish its data-free execution preflight, then stop for the
    owner's explicit one-shot decision. Launch-manifest creation remains
    forbidden.

### 64. T7b P1-v3 data-free execution preflight (2026-07-23)

1. **Execution date/source:** 2026-07-23; published non-executable freeze head
   `d4078fadef239d1e0878a62a0c660e0a06de6f72`.
2. **Comparators:** no model arms were run.
3. **Evidence class:** data-free execution-integrity preflight, not fresh
   confirmation or quality evidence.
4. **Data/splits:** exact projection of the hash-bound verified enumeration:
   32 lineages, 96 coordinates, 17 depth 4, 15 depth 8, three group-safe.
5. **Arms/policies:** future candidate/control bindings only; no fit.
6. **Environment/repeats:** no OpenML access, worker, thread allocation,
   warmup, or repeat.
7. **Runner/command:**
   `python benchmarks/run_t7b_automatic_depth_fresh_tier_d_v3.py preflight
   --output benchmarks/t7b_automatic_depth_fresh_tier_d_v3_preflight_20260723.json`.
8. **Hashes:** preflight
   `ea496a2851c29bf3d254af49057daf94cf2c8cd5b912e59e00962b5e0b068f22`;
   dated preflight note
   `3fc92e4c0ec9441840f7243bdb00780e67349ac853858e284620fcd59e95c03e`;
   execution contract
   `12ff0db7553b2748eaa75b2e0f0610fa423abc3112df79fb061bb4b59a4dc34d`;
   enumeration
   `c7c76259823d6ee4d3ce6202b127d4bed984493e6153775dfb0f05a105b22851`;
   power result
   `d6d572e47c672262b007c436cc048b6259a753097e860357523bcec033085ba8`.
9. **Primary results:** `preflight_passed`; every resource, fingerprint,
   branch, split, group, contract, and power binding is present.
10. **Costs, passed and failed conditions:** no observed quality or resource
    cost gate was evaluated. Preflight integrity passed.
11. **Limitations/non-claims:** the preflight does not authorize launch and
    cannot predict the confirmation result. No fresh inspection was spent.
12. **Terminal decision/next action:** publish the preflight and stop for the
    owner's exact one-shot authorization. Without that create-only record, the
    harness refuses before launch-manifest creation.

### 65. T7b P1-v3 fresh one-shot terminal integrity failure (2026-07-23)

1. **Execution date/source:** 2026-07-23; published harness
   `37bf561a1415cef072c767a2a5240d10849f905d`.
2. **Comparators:** exact control
   `e23d2b164f10374b1c0e02521c33fc96d48980da` and candidate
   `41e948f0c53b1d124e16071a7fa66eba47d084d3`; the run stopped on the
   first candidate worker before a control comparison.
3. **Evidence class:** authorized prospective Tier-D fresh one-shot,
   terminally failed after launch. The sole inspection is spent; this is
   neither quality nor shipping evidence.
4. **Data/splits:** the first bound lineage,
   `airlines_departure_delay_10m` / OpenML task `359929`, coordinate 0,
   ordinary weights, split SHA-256
   `a55dcff590b2441d7fc8bc6584a5a358665fb5fa0ab85ca3a82f0ceb74a68d1f`.
   No other lineage was fitted.
5. **Arms/policies:** candidate scalar-RMSE CatBoost `depth=None`, 600
   maximum rounds, early stopping 30, validation 0.15, best model, no
   refit, seed 20260723. The registry expected depth 8; the actual
   post-validation policy resolved depth 6.
6. **Environment/repeats:** `darko311`, macOS arm64, 14 logical CPUs,
   Python 3.11.8, NumPy 2.2.6, Numba 0.61.2, sklearn 1.7.1; exclusive
   audit found no conflicting benchmark process. The first worker completed
   its frozen warmup/prediction/serialization checks before integrity
   rejection.
7. **Runner/command:**
   `python benchmarks/run_t7b_automatic_depth_fresh_tier_d_v3.py execute
   --preflight benchmarks/t7b_automatic_depth_fresh_tier_d_v3_preflight_20260723.json
   --owner-authorization benchmarks/t7b_automatic_depth_fresh_tier_d_v3_owner_run_authorization_20260723.json
   --control /private/tmp/darkofit-t7b-auto-depth-control-e23d2b1
   --candidate /private/tmp/darkofit-t7b-auto-depth-v1-20260722
   --output-prefix /private/tmp/t7b_automatic_depth_fresh_tier_d_v3_inspection1_20260723`.
8. **Hashes:** owner authorization
   `775cdd0d3ff2f7913470e2d2badc35cbcd1b78ce72630ed6e8be4df60baf5bda`;
   launch manifest
   `cb0198d3bf42224ef1ca7c2e7feed9e2145ca72d9c8f85b43544a2e6203f1b54`;
   terminal failure
   `10b0b225c16a3f8c1039ada13fbb4884379d4ae7c982fc3c4963f1d72c17aeae`;
   dated terminal note
   `91010904b561070e7262eaab0d39919c2ac582db5b2cbd3138972a5f5ed5fe6f`;
   focused test file
   `4a9be9d3ca382bc0cf44124fd2fda82991c2e4736c6cad93f46f20354e34ef61`;
   frozen contract, enumeration, and power hashes remain those in entry 64.
9. **Primary results:** no paired result and no quality verdict. Registry
   rows per feature were `2,597` from 23,373 outer-training rows; the
   candidate used 19,867 post-validation rows, or `2,207.444444` per
   feature, resolved `middle_density`, and fit depth 6 rather than the
   frozen expected depth 8.
10. **Costs, passed and failed conditions:** the branch-integrity condition
    failed. No quality, fit, prediction, RSS, or archive gate was evaluated.
11. **Limitations/non-claims:** the runner published no raw/result artifact;
    one completed row remains unpublished and unread except for the terminal
    failure payload emitted by the runner. No inference about transfer
    quality is permitted. P3's explicit opt-in basis is unaffected.
12. **Terminal decision/next action:** campaign closed; rerun forbidden.
    Automatic depth remains private and unpromoted. Any successor requires a
    new identity, fresh owner authorization, contamination review, and branch
    verification against the exact post-validation fit population.

### 66. Automatic-depth 32-lineage development rerun (2026-07-23)

1. **Execution date/source:** 2026-07-23; clean local harness
   `ce7962f20923c98333512c1a80a46810a5795d0e`.
2. **Comparators:** exact control
   `e23d2b164f10374b1c0e02521c33fc96d48980da` and unchanged candidate
   `41e948f0c53b1d124e16071a7fa66eba47d084d3`.
3. **Evidence class:** paired development benchmark under `SHIP_RULES.md`;
   not holdout or shipping evidence.
4. **Data/splits:** all 32 verified P1-v3 lineages, three deterministic
   coordinates each, with the historical selected views and split hashes.
5. **Arms/policies:** scalar-RMSE CatBoost, 600 maximum rounds, early
   stopping 30, validation 0.15, no refit, seed 20260723; control depth 6,
   candidate `depth=None`.
6. **Environment/repeats:** `darko311`, macOS arm64, 14 logical CPUs, fresh
   worker per arm/coordinate, alternating paired arm order, three 50,000-row
   prediction repeats.
7. **Runner/command:**
   `python benchmarks/run_t7b_automatic_depth_development_v1.py execute
   --preflight /private/tmp/t7b_auto_depth_dev_v1_preflight_20260723.json
   --control /private/tmp/darkofit-t7b-auto-depth-control-e23d2b1
   --candidate /private/tmp/darkofit-t7b-auto-depth-v1-20260722
   --output-prefix /private/tmp/t7b_auto_depth_dev_v1_run1_20260723`.
8. **Hashes:** preflight `cfc94b9c57f86bc30b3654052490406c027b292002e27a2b10c0f3f441770334`;
   launch `987f71bb45f19fa0a76bcb91b0478760eb8e5ad2a74b377f98f5088a5dc18b2d`;
   raw `db7b96cbeec9ee21f1696453e16792560d57a6d6fe9ab5c7eae0f1fded19b30e`;
   result `7e92d584b4adb8a96675d1f116a35682ddc2d4e2adc43051eadbca316d5c3307`.
9. **Primary results:** equal-lineage RMSE `0.996860×`, bootstrap upper
   `0.999869×`, leave-one-favorable-out `0.998192×`, worst lineage
   `1.016344×`; low-density panel `0.994098×`, high-density panel
   `1.000000×`.
10. **Costs and integrity:** 192/192 rows, 96 pairs, and 32 lineages passed;
    fit `0.807402×`, predict `0.943500×`, RSS `0.991148×`, mean RSS delta
    `-15,380,480` bytes.
11. **Limitations/non-claims:** development evidence only; no CTR23,
    newest-season sports, TabArena, or release claim. Desktop activity makes
    cost ratios telemetry rather than a release timing claim.
12. **Decision/next action:** development is clearly positive under
    `SHIP_RULES`; proceed once to the CTR23 and newest-untouched-season
    ship-check, without tuning from holdout outcomes.

### 67. Automatic-depth CTR23 holdout ship-check (2026-07-23)

1. **Source:** clean harness `3226b36`; control
   `e23d2b164f10374b1c0e02521c33fc96d48980da`; unchanged candidate
   `41e948f0c53b1d124e16071a7fa66eba47d084d3`.
2. **Comparator:** the fixed-depth-6 control and automatic-depth candidate
   from the paired development run; no rival library arm.
3. **Evidence:** deliberate `SHIP_RULES` holdout ship-check on the nine sealed
   CTR23 lockbox tasks. CTR23 is observed release-validation after this run.
4. **Data/splits:** OpenML tasks `361247`, `361253`, `361254`, `361261`,
   `361264`, `361272`, `361616`, `361617`, and `361618`; official repeat 0,
   folds 0–2, sample 0. Exact train/test sizes and index SHA-256 values were
   checked against `ctr23_suite_snapshot.json` before every fit.
5. **Arms:** scalar RMSE CatBoost mode, 600 maximum iterations, early stopping
   30, validation fraction 0.15, best-model retention, no refit, seed
   `20260723`; control depth 6 versus candidate automatic depth.
6. **Environment:** `darko311`, Python 3.11.8, NumPy 2.2.6, Numba 0.61.2,
   pandas 2.2.3, sklearn 1.7.1, OpenML 0.15.1, macOS 26.5.2 arm64, 14 logical
   CPUs and 24 GiB RAM. The audit found no competing benchmark process; high
   macOS background load makes cost ratios telemetry only.
7. **Execution:** `python
   benchmarks/run_t7b_automatic_depth_ctr23_ship_check_v1.py execute` with
   the hash-bound manifest and exact control/candidate worktrees; 54 isolated
   fresh workers with alternating arm order.
8. **Artifacts/hashes:** manifest
   `4edfd594ef967b383a75cdaab8caf8593c8f387f1d1a7741aee1666ab0db6cac`;
   launch `5bcdff4b305f4ccc6dfac1a7df11a86f4254d207a93fdc82983a2cc0f4078d9f`;
   raw `4bad8f98a80a0fac3769e7a3e9887491c9bd067fb757c6e7a7646c61e5927483`;
   result `ceb1f6d4ee3feee4c850fa2632a8966603e98b453dc441d53764189d1616a553`.
9. **Primary results:** task-equal RMSE ratio `1.026662x`, task-bootstrap
   upper `1.062082x`, leave-one-task-out maximum `1.031063x`, worst task
   `1.165018x`, and 1/5/3 task wins/ties/losses.
10. **Costs and integrity:** 54/54 rows, 27/27 pairs, and 9/9 tasks passed;
    fit `0.922137x` and predict `1.048893x` are descriptive only.
11. **Limitations/non-claims:** this is holdout evidence for the unchanged
    automatic-depth candidate, not a general CTR23 tuning surface or clean
    timing result. The newest untouched sports season was not consulted.
12. **Decision/next action:** automatic depth is closed for the public
    default because it is worse on CTR23. Preserve P3's explicit opt-in path,
    do not tune from these holdout outcomes, and move the mechanism slot to
    catcross.

### 68. Group-centered categorical-cross v1 spent attribution (2026-07-23)

1. **Source:** clean harness
   `1b2f6b6f81bcf0a7ad6c9ca593cf18684c6c1e27`; clean private candidate
   `c3f2608cd3033cfc00aa0737897a92ed868b5865`.
2. **Comparator:** the candidate's private explicit-off lane as constant
   control; automatic v1 selector; and an exact forced-pair lane.
3. **Evidence:** mechanism-specific spent development attribution on M2
   coordinates already observed in the v0.11 campaign; no holdout or fresh
   data.
4. **Data/splits:** OpenML tasks `363631` (`diamonds`) and `363675`
   (`healthcare_insurance_expenses`), official repeat/fold coordinates
   `0/0`, `1/1`, and `2/2`, sample 0. Exact index hashes are in the raw
   artifact.
5. **Arms/policies:** scalar RMSE CatBoost mode, 1,000 iterations, seeds 0,
   1001, and 2002. Automatic supplied its exact candidate pairs when
   eligible; below the sample floor, forced pairs came from that coordinate's
   full-train constant-model importance.
6. **Environment:** `darko311`, macOS arm64, 14 logical CPUs, fresh process
   per coordinate, 14 threads per arm. No competing benchmark process;
   timing is telemetry only.
7. **Execution:** `python
   benchmarks/run_group_centered_categorical_crosses_v1_attribution.py
   execute --manifest
   /private/tmp/catcross_v1_attribution_manifest_20260723.json --source
   /private/tmp/darkofit-catcross-v1-20260722 --output-prefix
   /private/tmp/catcross_v1_attribution_run1_20260723`.
8. **Artifacts/hashes:** manifest
   `3c4897e2165c769ab4cb2df8f65f515149370781d55846a1fa22c4a8f8150819`;
   launch `4f024f8bf0fce8e5378e37602f18c5995146c7f83558990821ac16bd7c28a2df`;
   raw `7679133a3740f6998067366a0b1205a7c19a4ff84d546bf864c450e8913dc5d4`;
   result `c9d7f4268a3018aeb518cf215b7b1fa39532a8e268fbd33786c3cd95eeb851f4`.
9. **Primary results:** Diamonds automatic/control and forced/control both
   `0.724496x`, with 3/3 engagement. Healthcare automatic/control
   `1.000000x` via exact ineligible fallback; forced/control `1.008072x`.
   The pooled automatic/control `0.851173x` is descriptive, not universal.
10. **Costs and checks:** 6/6 workers and 18/18 arms passed; every arm
    resolved 14 threads and restored the ambient Numba mask. Per-dataset
    automatic quality and the `1.02` harm check passed; all-coordinate
    eligibility failed at 3/6. Eligible Diamonds automatic fit was
    `2.440972x` constant because it includes two auditions and the final fit.
11. **Limitations/non-claims:** no sports, holdout, prediction-throughput,
    memory, archive, release-ladder, or default claim. Forced healthcare
    pairs probe latent mechanism value under a different pair-source path.
12. **Decision/next action:** continue to the cold-player sports guardrail
    for an honestly scoped large-data opt-in. Record a small-data selector
    successor; do not claim the current automatic policy solves healthcare.

### 69. Catcross v1 mixed-feature basketball guardrail (2026-07-23)

1. **Source:** clean harness
   `6d76dfae694c621a7dbf05861755d6f0a4638a94`; clean private candidate
   `c3f2608cd3033cfc00aa0737897a92ed868b5865`.
2. **Comparator:** the candidate's explicit-off control lane versus the
   unchanged private automatic selector.
3. **Evidence:** spent basketball development guardrail; no fresh, holdout,
   or default evidence.
4. **Data/splits:** pinned creator CSV
   `43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2`;
   ten established unshuffled creator folds plus the alphabetical held-team
   split and its 585-row genuinely cold-player subset.
5. **Arms/policies:** the established 15 numeric features plus categorical
   `Pos`, categorical `Age`, `Tm`, and derived `starter`; scalar RMSE CatBoost,
   1,000 iterations, seed 4, player-group-aware internal validation, 14
   threads. `Player` was never a model feature.
6. **Environment:** `darko311`, macOS arm64, 14 logical CPUs, fresh process
   per arm/coordinate with alternating order. No competing benchmark process;
   timing is telemetry only.
7. **Execution:** `python
   benchmarks/run_group_centered_categorical_crosses_v1_sports_guardrail.py
   execute --manifest
   /private/tmp/catcross_sports_guardrail_manifest_20260723.json --source
   /private/tmp/darkofit-catcross-v1-20260722 --data-cache
   /private/tmp/darkofit_basketball_reference_20260723.csv --output-prefix
   /private/tmp/catcross_sports_guardrail_run1_20260723`.
8. **Artifacts/hashes:** manifest
   `6343e8aff8042efa7cd0be108fadba96002047bba78e7bc8a52da679982f9bae`;
   launch `af0d1c90b21d97fa8bf24d76d238d80a58114eca114089419381fdefef7ffc40`;
   raw `7405b6a827caf296693003705fc6c6d155dbdd18bf1f0c5bb028986af60a40c1`;
   result `b7c0b76f32f7a66294b29497415dc533eeab67e7fcce9cbf705b3409621a7359`.
9. **Primary results:** equal-fold RMSE `0.996016x`, 8/0/2 fold
   wins/ties/losses, worst fold `1.010136x`; all-held-team `0.996891x`,
   seen-player `0.997861x`, and cold-player `0.993971x`.
10. **Costs and checks:** 22/22 workers and 11/11 pairs passed; automatic
    was eligible and selected 12 pairs on every coordinate. All five declared
    checks passed. Fit `1.597123x` and single-call predict `0.860144x` are
    telemetry only.
11. **Limitations/non-claims:** one spent basketball workload and one mixed
    view; no holdout, default, prediction-throughput, memory, release-ladder,
    or release claim.
12. **Decision/next action:** expose the validated automatic selector as the
    explicit `categorical_crosses=True` opt-in with honest large-data,
    small-data-fallback, sports, and selection-cost documentation. Do not
    change the default.

### 70. Catcross v1 public opt-in exposure (2026-07-23)

1. **Source:** clean product commit
   `2249d13db72fa58b7b124820b7a50d39f5b7a3cd`.
2. **Comparator:** the prior public `DarkoRegressor` path, preserved by the
   new default `categorical_crosses=False`.
3. **Evidence:** correctness and honest product characterization only; the
   previously recorded spent attribution and sports guardrail remain the
   quality evidence. No new benchmark or holdout contact.
4. **Data/splits:** deterministic synthetic mixed, numeric-only, and
   all-categorical test fixtures; the selector tests include group-disjoint
   validation.
5. **Arms/policies:** default-off exact control; explicit automatic audition;
   selected, control-win, and data-ineligible fallback states; incompatible
   requested modes; repeated true-to-false fit.
6. **Environment:** macOS arm64; focused verification in `darko311`, broader
   core sweep in the local Python 3.12 environment.
7. **Execution:** `python -m pytest -q` over the five group-centered suites;
   the focused suites plus thread/input validation in `darko311`; broader
   `tests/test_darkofit.py` core subset; strict MkDocs in `darko311`.
8. **Artifacts/hashes:** implementation commit above; no generated benchmark
   artifacts.
9. **Primary results:** 84 focused/API/thread/input tests passed in
   `darko311`; 387 broader core tests passed; strict MkDocs passed.
10. **Failed as well as passed checks:** the full local non-campaign sweep
    reached 1,569 passes and three unrelated historical-evidence failures:
    unavailable M3b sports cache, a one-ULP frozen power-result mismatch, and
    an old compute-ladder governing-plan hash invalidated by later plan edits.
11. **Limitations/non-claims:** no new quality, speed, memory, holdout,
    release-ladder, default, or release claim. Classification and the listed
    multi-fit or alternate-head modes remain unsupported by this opt-in.
12. **Decision/next action:** the default-off public opt-in is complete. Keep
    the small-data selector successor separate and advance to B3's
    deterministic minimum-work threshold.

### 71. B3-v2 activation-gated parallel ensemble characterization (2026-07-23)

1. **Source:** clean candidate
   `b35c092bbdfef45f2ac4d5b0cc16eaaf1c89bf55`.
2. **Comparator:** public ensemble-v3's existing sequential 1×14 fit versus
   the static B3-v2 route, which either stays sequential or uses 7×2 process
   workers under the same 14-CPU budget.
3. **Evidence:** behavior-exact Tier-E engineering characterization on the
   spent B3-v1 grid; no fresh, holdout, sports, or rival evidence.
4. **Data/splits:** B3-v1's four fixed 7,500-fit-row general cases:
   Friedman numeric regression, categorical regression, numeric binary, and
   categorical multiclass. Existing fingerprints are paired within every
   block and arm.
5. **Arms/policies:** identical eight-member public ensemble-v3 policies.
   Candidate dispatch score is sampled rows × input features × 600 planned
   iterations × output width, with the frozen `80,000,000` threshold.
6. **Environment:** `darko311`, macOS arm64, 14 logical CPUs; three paired
   blocks, cold and steady fits, alternating arm order, process-tree RSS;
   exclusive-machine audit found no competing benchmark process.
7. **Execution:** `python benchmarks/run_b3_parallel_ensemble_v2.py --source
   /private/tmp/darkofit-depth-light --output-prefix
   /private/tmp/b3_parallel_ensemble_v2_20260723`.
8. **Artifacts/hashes:** launch
   `707a4fc3d7283023721ed61417b20e52254d4a7b417e35696e6fe052fbf040a3`;
   raw
   `1d48276bcde51e9fedd778d35ba521a954ac40f6e927626442c627e0a52b7be1`;
   result
   `f2e34bcb695f28ceea8309177a86e239ae170f53b8c66da7b9d29b55006f7c9c`;
   runner
   `f90bc7d5d0e4d3aaa1429d65c91fc0e67e06d3d91e79a02607b55136c1d88495`.
9. **Primary results:** engaged cold geomean `0.487217x`, worst `0.491280x`;
   engaged steady geomean `0.235154x`, worst `0.283799x`. Fallback cold
   geomean/worst `1.015415x`/`1.032163x`; fallback steady
   `1.019149x`/`1.049417x`.
10. **Failed as well as passed checks:** all eight declared checks passed:
    exact behavior, routes, resource sampling, memory, engaged cold/steady
    direction, and fallback cold/steady bound. No declared check failed.
    Parallel process-tree RSS ratios exceeded 5×; maximum peak was about
    2.28 GiB and every paired absolute delta remained below 2 GiB, so the
    standing conjunctive hybrid-RSS rule passed.
11. **Limitations/non-claims:** only four spent general shapes on this
    14-CPU ARM host. The result makes no quality, holdout, sports, rival,
    portability, default-quality, or release claim. Fallback ratios measure
    timing noise around identical code.
12. **Decision/next action:** disposition `ready_to_productize`. Integrate
    the deterministic activation into public eight-member ensemble-v3 fit,
    persist its resolution, retain rollback, then rerun correctness and
    focused performance checks before moving to selector-v3.

### 72. B3-v2 public automatic activation and rollback (2026-07-23)

1. **Source:** product integration commit
   `dc468c34b19d385669ed85ec0524527dea52674a`, layered on the immutable
   characterization above.
2. **Comparator:** no new timed comparator; this checkpoint wires the
   characterized process route into public ensemble-v3.
3. **Evidence:** correctness and compatibility verification for
   behavior-exact Tier-E engineering; no fresh data or new benchmark result.
4. **Data/splits:** deterministic synthetic regression, binary, multiclass,
   and archive fixtures in the named test suites.
5. **Arms/policies:** public
   `ensemble_parallelism={"auto","sequential","parallel"}`. Automatic
   activation is limited to the measured macOS-arm64 14-thread envelope and
   `member_work >= 80,000,000`; sequential is the rollback and parallel is an
   explicit research escape hatch.
6. **Environment:** `darko311`, macOS arm64, real process workers where the
   tests engage parallel fitting.
7. **Execution:** focused public/B3/contract suite; broader ensemble,
   serialization, archive, and release-candidate suite; core
   ensemble/clone/thread subset; strict MkDocs.
8. **Artifacts/hashes:** source commit above; the characterization artifacts
   and hashes remain entry 71. No new generated benchmark artifact.
9. **Primary results:** 56 focused tests passed; 221 broader ensemble and
   serialization tests passed; four selected core tests and strict MkDocs
   passed.
10. **Failed as well as passed checks:** no check failed after the final
    fresh-eyes pass. During review, incorrect fallback thread provenance,
    an inconsistent private-fixture request label, and an overbroad legacy
    schema repair were found, fixed, and covered by regression tests.
11. **Limitations/non-claims:** automatic activation makes no portability or
    unmeasured-shape speed claim. The explicit parallel option is a research
    surface. The measured peak remains about 2.28 GiB and is documented.
12. **Decision/next action:** B3-v2 is complete in source with deterministic
    activation, persisted resolution, safe-NPZ validation, backward archive
    compatibility, and a documented rollback. Move to selector-v3 after the
    required fresh-eyes stage boundary.

### 73. Selector-v3 non-Protein noise calibration (2026-07-23)

1. **Source:** initial 1-SE source
   `ca2f29f31bad0457af93b7cf007e5880ce953547`; revised 2-SE source
   `b3c006ffef8bb9f191edb273184a308fb38f6439`.
2. **Comparator:** constant- versus linear-leaf auditions inside one
   automatic selector; no external model comparator.
3. **Evidence:** spent synthetic development calibration, engagement
   statistics only; no quality outcome, Protein, holdout, or fresh evidence.
4. **Data/splits:** 24 regression cells from the M6-v3 medium grid: four
   datasets, three seeds, and unweighted/stress-weighted variants. Protein
   was excluded.
5. **Arms/policies:** paired per-row MSE gain and its weighted or unweighted
   standard error. The initial rule required gain z at least 1.0; the
   predeclared noisy-engagement contingency revised it to 2.0.
6. **Environment:** `darko311`, four threads, fresh worker per cell; process
   audit found no competing Python or benchmark worker.
7. **Execution:** `python
   benchmarks/run_automatic_linear_selector_v3_noise_calibration.py
   --candidate /private/tmp/darkofit-depth-light --output ...`.
8. **Artifacts/hashes:** 1-SE artifact
   `626814e8522415d52085779c61935669efa3122c71313fbae8d5578735070b9f`;
   2-SE artifact
   `d179aa46ac3787be1f50f8b6fc11ede52745e0ddae4d7ba9ede37e901db1c7da`.
9. **Primary results:** 18 eligible and six below-minimum-sample cells.
   The 1-SE rule engaged three seed-fragile cells; the 2-SE rule engaged
   none. Maximum z was `1.974423049` in both runs.
10. **Failed as well as passed checks:** the initial rule failed its stated
    no-obviously-noisy-engagement condition. The revised rule completed all
    cells and passed; no cell changed its underlying gain statistic.
11. **Limitations/non-claims:** this is selector calibration, not quality
    evidence. It supports no default, shipping, sports, or rival claim.
12. **Decision/next action:** retain the 2-SE rule and run the spent Protein
    development comparison. Consult the holdout only if development is
    clearly better.

### 74. Selector-v3 Protein development comparison (2026-07-23)

1. **Source:** clean candidate and harness commit
   `0df50bef3a42b0d5b22cd50ad2e2c29ea5005b56`.
2. **Comparator:** constant leaves and explicit linear leaves from the same
   source; no external model comparator.
3. **Evidence:** spent Protein development evidence. This is not holdout,
   shipping, or rival evidence.
4. **Data/splits:** OpenML task 363693 (`physiochemical_protein`), three
   historical TabArena coordinates with split fingerprints reproduced by the
   direct OpenML 0.15.1 loader.
5. **Arms/policies:** constant leaves, automatic selector with the fixed
   2-SE paired-MSE-gain guard, and explicit linear leaves. Each automatic
   fit must be exact to its recorded explicit arm.
6. **Environment:** `darko311`, macOS arm64, 14 threads, fresh worker per
   arm; warmup disabled and ambient Numba masks checked after fit/predict.
7. **Execution:** `python
   benchmarks/run_automatic_linear_selector_v3_protein.py
   --candidate-source /private/tmp/darkofit-depth-light
   --tabarena-source /private/tmp/tabarena-m2-4cd1d25 --output ...`.
8. **Artifacts/hashes:** result JSON
   `b14cdde34d2a938c845ee31fb900ad16202dee7ce9c2617e753258374d859a72`;
   the artifact binds the calibration inputs, source trees, split
   fingerprints, row-level outputs, and environment.
9. **Primary results:** automatic/constant RMSE geometric mean
   `0.951039707x`, worst coordinate `0.955224848x`, 3/3 coordinates improved,
   3/3 selected linear, and 3/3 exact to explicit linear.
10. **Failed as well as passed checks:** no benchmark or integrity check
    failed. The previously missed coordinate engaged at z `2.957600` with
    `2.517852%` validation improvement.
11. **Limitations/non-claims:** the three coordinates are spent and all come
    from one smooth regression dataset. Costs are telemetry. No default or
    general-quality claim follows from this result alone.
12. **Decision/next action:** development is clearly positive. Run the fixed
    SHIP_RULES holdout ship-check; only a non-regressing holdout result can
    authorize automatic engagement, with `linear_leaves=False` as rollback.

## Product behavior established by the testing

### Defaults retained

- `learning_rate=None` uses DarkoFit's transparent automatic rule.
- Scalar regression keeps ordered leaf updates off by default.
- Categorical regression still uses ordered target-statistic preprocessing.
- The default horizon remains 1,000 rounds.
- `tree_mode="catboost"` remains the default.
- Early stopping, exact refit, linear leaves, global linear residuals, cross
  features, safe ordinal representation, ensembles, and the accuracy preset
  remain explicit rather than silently automatic.
- No noisy-data or sports default was changed after failing its basketball
  guardrail.

### Tier-E surfaces and engine work shipped

- behavior-exact selected-subset fused training;
- bounded serial leaf descent and leafwise packed prediction routes;
- explicit warmup;
- hardened input validation with empty prediction-batch support;
- safe `.npz` serialization;
- opt-in accuracy preset;
- row- and group-aware ensemble API;
- explicit per-leaf linear leaves;
- source-declared ordinal features;
- capped lane audition with full-budget refit semantics;
- split-conformal Gaussian intervals;
- exact supported-lane TreeSHAP;
- owner-promoted, statically recorded macOS-arm64 fused/unfused dispatch
  (behavior-exact, without a validated speed or portability claim); and
- generated, hash-bound engineering measurements.

### Closed automatic/default routes

- global 10,000-round default;
- automatic safe-ordinal promotion under the frozen confirmation;
- automatic linear leaves;
- automatic OOB sports ensemble;
- auto-LR + early-stopping + exact-refit sports policy;
- the tested cross-feature and categorical-combination donor routes;
- the tested standalone calibration candidates;
- T5 composite confirmation as originally registered;
- both Panel 3 candidates;
- the historical private B1/B2 ensemble-v3 campaign disposition (no arm
  cleared every frozen M3b gate; section 31 supersedes only the archive gate's
  forward product effect); and
- B-archive canonical-preprocessor serialization (the non-loadable effective
  median `4.152525×` simulation missed the frozen `4.0×` archive limit; section
  31 makes it optional telemetry but still authorizes no serializer format);
- any CTR23 lockbox run without a newly powered candidate.

## Verification and release log

| Checkpoint | Verification |
| --- | --- |
| Scalar-policy hardening | 550/550 on both Python 3.11 and 3.13.13, including 18 optional LightGBM-comparison tests |
| Best-of-both final code-bearing head | Full matrix passed locally; GitHub Actions run `29571900341` passed Python 3.9, 3.11, and 3.13 |
| Final integrated branch | `2,671 passed, 2 skipped`; strict MkDocs, generated benchmark status, source distribution, and wheel builds passed |
| Independent environment reconciliation | `2,646 passed, 27 skipped`; same 2,673 collected tests and zero failures |
| PR #1 | Five GitHub lanes passed, including 1,692 campaign verifiers and package/docs/generated-evidence checks |
| GitHub-only v0.10 release | Tag and `main` agree at `ec66a64`; wheel, source archive, and checksums were attached to the GitHub release and public URL installation was verified |
| GitHub-only v0.10.1 release | Annotated `v0.10.1` resolves to `d3aba3d`; the user-visible thread-mask and sklearn-tag fixes are recorded in the release CHANGELOG |
| GitHub-only v0.11.0 release | Annotated `v0.11.0` resolves to `0b820e3`; GitHub Actions run `29942771031` passed all five lanes; a clean detached worktree produced the universal wheel and source archive; the wheel passed isolated regression/classification ensemble-v3 fit/predict and exact safe-NPZ round-trip smoke; downloaded release assets matched the attached `SHA256SUMS` |

Intermediate suite totals are useful progress markers but are not permanent
repository invariants. The release claim always binds to the exact source
commit and its final CI run.

## Where to look for detail

| Need | Start here |
| --- | --- |
| Governing evidence policy | [`SHIPPING_POLICY.md`](SHIPPING_POLICY.md) |
| Current generated frontier | [`benchmark_status.md`](benchmark_status.md) |
| Long-form current benchmark notes | [`../BENCHMARK_NOTES.md`](../BENCHMARK_NOTES.md) |
| Engineering-only measurements | [`../docs/measurements.md`](../docs/measurements.md) |
| Best-of-both terminal ledger | [`best_of_both_completion_audit.md`](best_of_both_completion_audit.md) |
| Product Offense execution ledger | [`../PRODUCT_OFFENSE_PLAN.md`](../PRODUCT_OFFENSE_PLAN.md) |
| Review correction and final integration | [`FABLE_FEEDBACK_CLOSEOUT.md`](FABLE_FEEDBACK_CLOSEOUT.md) |
| Historical LightGBM/speed investigation | [`FINDINGS.md`](FINDINGS.md) |
| Every benchmark protocol and result | [`README.md`](README.md) and this log's campaign links |

## Update rule for future work

Every material test added after this checkpoint should append or update one
entry here with:

1. execution date and exact DarkoFit commit;
2. exact comparator commit or wheel version;
3. evidence class and whether the data were fresh, spent, or sealed;
4. dataset identity, split strategy, and target/feature fingerprints where
   available;
5. model arms and every material policy difference;
6. environment, thread policy, warmup policy, and repeat count;
7. exact command or source-attested runner;
8. raw, summary, protocol, runner, and analyzer hashes for frozen campaigns;
9. primary metric, uncertainty, concentration, harm, and cost results;
10. failed as well as passed gates;
11. limitations and non-claims; and
12. terminal decision and the next authorized action, if any.

Frozen artifacts are never edited to make a later story cleaner. If a new
version supersedes a descriptive comparison, add a new dated record and label
the old version boundary explicitly.
