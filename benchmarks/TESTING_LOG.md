# DarkoFit testing log

_Canonical navigation ledger. Updated 2026-07-19 at DarkoFit `v0.10.0`,
commit `ec66a64654becaf948592588a047bfb8205decc8`._

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
| DarkoFit release | GitHub-only `v0.10.0`, exact commit `ec66a64654becaf948592588a047bfb8205decc8`; not published to PyPI |
| Release verification | 2,673 tests collected with zero failures in both recorded environments: `2,671 passed / 2 skipped` and `2,646 passed / 27 skipped` |
| GitHub integration | PR #1 passed Python 3.9, 3.11, and 3.13 library lanes, 1,692 campaign verifiers with expected optional skips, and the documentation/generated-evidence/package lane |
| Final release CI | GitHub Actions run `29686258603` passed for the tagged source |
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
- exact supported-lane TreeSHAP; and
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
- both Panel 3 candidates; and
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
