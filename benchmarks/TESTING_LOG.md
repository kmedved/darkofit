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
