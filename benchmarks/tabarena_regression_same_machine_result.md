# Same-machine TabArena regression comparison

The primary panel compares the official DarkoFit 0.9.0, ChimeraBoost 0.14.1, and CatBoost 1.2.10 defaults on identical r0f0/r1f1/r2f2 coordinates across 13 datasets. Ratios below one favor the numerator. Each dataset receives equal weight; the bootstrap keeps datasets fixed and resamples the three coordinates within each.

This is descriptive characterization only. The Airfoil/Diamonds safe-ordinal lane is reported separately, is never pooled with the primary panel, and cannot revive or advance the rejected ordinal policy.

## Material default-policy differences

These are product-default comparisons, not hyperparameter-parity comparisons. The material frozen differences are:

| Engine | Cap | Learning rate | Depth/mode | L2 | Bins | Categorical permutations | Ordered boosting | Linear lane | Early stopping |
| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |
| DarkoFit | 1,000 | auto | auto depth / catboost mode | auto → 3 | 254 | 1 | auto → off for scalar regression | off | on |
| ChimeraBoost | 10,000 | 0.1 | depth 6 | 1 | 128 | 4 | off | auto-select (`None`) | on; patience 50 |
| CatBoost | 10,000 | 0.05 | official default | official default | official default | native | native policy | off | AutoGluon adaptive; use-best with eval set |

## Primary out-of-box defaults

| Contrast | Test RMSE | Test 95% CI | Validation | Train | Infer | Incremental memory | Raw RSS | Dataset wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D/M | 1.012523 | [1.008196, 1.017260] | 1.0118 | 0.9093 | 1.4332 | 0.5660 | 1.0015 | 5/13 |
| D/C | 1.053834 | [1.051124, 1.056798] | 1.0557 | 0.3729 | 1.2561 | 0.1526 | 1.0089 | 1/13 |
| M/C | 1.040800 | [1.037356, 1.044255] | 1.0434 | 0.4101 | 0.8765 | 0.2696 | 1.0074 | 1/13 |

### D/M: darkofit_vs_chimeraboost

Coordinate wins/losses/ties: 16/23/0. Dataset wins/losses/ties: 5/8/0.

| Dataset | Test | Validation | Train | Infer | Incremental memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Another-Dataset-on-used-Fiat-500 | 0.999544 | 0.994605 | 2.2855 | 1.2293 | 0.5848 |
| Food_Delivery_Time | 0.998745 | 0.997883 | 0.6540 | 0.9525 | 0.1086 |
| QSAR-TID-11 | 1.009443 | 1.004524 | 0.5977 | 1.0678 | 0.3578 |
| QSAR_fish_toxicity | 1.000275 | 0.998111 | 2.2728 | 1.3723 | 1.1447 |
| airfoil_self_noise | 1.064752 | 1.044645 | 1.4659 | 1.4873 | 0.7179 |
| concrete_compressive_strength | 0.971329 | 1.000069 | 1.9111 | 1.9498 | 1.4567 |
| diamonds | 0.982015 | 1.000386 | 0.6773 | 0.9215 | 1.1189 |
| healthcare_insurance_expenses | 0.998495 | 0.994602 | 2.0764 | 0.9813 | 15.9713 |
| houses | 1.023751 | 1.013389 | 0.5574 | 2.8627 | 0.0410 |
| miami_housing | 1.005531 | 1.010697 | 0.7683 | 3.7661 | 0.3333 |
| physiochemical_protein | 1.076735 | 1.059047 | 0.2157 | 0.3276 | 0.3763 |
| superconductivity | 1.027917 | 1.029184 | 0.4587 | 2.3121 | 0.3091 |
| wine_quality | 1.009421 | 1.009005 | 0.8569 | 2.9289 | 0.7914 |

### D/C: darkofit_vs_catboost

Coordinate wins/losses/ties: 3/36/0. Dataset wins/losses/ties: 1/12/0.

| Dataset | Test | Validation | Train | Infer | Incremental memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Another-Dataset-on-used-Fiat-500 | 1.012125 | 1.019523 | 1.0982 | 0.8723 | 0.0109 |
| Food_Delivery_Time | 1.003239 | 1.003138 | 0.1973 | 0.8030 | 0.0061 |
| QSAR-TID-11 | 1.020594 | 1.017587 | 0.3404 | 1.2044 | 0.1109 |
| QSAR_fish_toxicity | 1.009130 | 1.013733 | 3.8482 | 1.4466 | 1.7784 |
| airfoil_self_noise | 1.173841 | 1.124995 | 0.1973 | 0.7155 | 0.0371 |
| concrete_compressive_strength | 0.994701 | 1.042799 | 0.4779 | 1.0811 | 0.0795 |
| diamonds | 1.335620 | 1.381211 | 0.1390 | 0.9608 | 0.0296 |
| healthcare_insurance_expenses | 1.009560 | 1.008634 | 2.0858 | 1.0192 | 9.0480 |
| houses | 1.031222 | 1.029414 | 0.1666 | 2.5068 | 0.3008 |
| miami_housing | 1.026883 | 1.027454 | 0.3574 | 3.0260 | 0.2178 |
| physiochemical_protein | 1.070690 | 1.057916 | 0.1193 | 0.5450 | 0.3980 |
| superconductivity | 1.033867 | 1.030371 | 0.1837 | 3.1429 | 1.3594 |
| wine_quality | 1.024431 | 1.018217 | 0.2660 | 1.6139 | 0.0664 |

### M/C: chimeraboost_vs_catboost

Coordinate wins/losses/ties: 4/35/0. Dataset wins/losses/ties: 1/12/0.

| Dataset | Test | Validation | Train | Infer | Incremental memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Another-Dataset-on-used-Fiat-500 | 1.012587 | 1.025053 | 0.4805 | 0.7096 | 0.0186 |
| Food_Delivery_Time | 1.004499 | 1.005266 | 0.3017 | 0.8431 | 0.0561 |
| QSAR-TID-11 | 1.011047 | 1.013004 | 0.5694 | 1.1279 | 0.3100 |
| QSAR_fish_toxicity | 1.008853 | 1.015651 | 1.6932 | 1.0541 | 1.5536 |
| airfoil_self_noise | 1.102455 | 1.076916 | 0.1346 | 0.4811 | 0.0517 |
| concrete_compressive_strength | 1.024062 | 1.042727 | 0.2501 | 0.5545 | 0.0546 |
| diamonds | 1.360081 | 1.380678 | 0.2052 | 1.0426 | 0.0264 |
| healthcare_insurance_expenses | 1.011082 | 1.014108 | 1.0045 | 1.0386 | 0.5665 |
| houses | 1.007298 | 1.015813 | 0.2988 | 0.8757 | 7.3429 |
| miami_housing | 1.021234 | 1.016579 | 0.4652 | 0.8035 | 0.6533 |
| physiochemical_protein | 0.994386 | 0.998932 | 0.5533 | 1.6635 | 1.0576 |
| superconductivity | 1.005789 | 1.001154 | 0.4005 | 1.3593 | 4.3985 |
| wine_quality | 1.014870 | 1.009130 | 0.3104 | 0.5510 | 0.0839 |

## Safe-ordinal diagnostic — separate evidence lane

### Cross-engine comparison under identical safe ordinal inputs

| Contrast | Test RMSE | Test 95% CI | Validation | Train | Infer | Incremental memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dord/Mord | 1.043430 | [1.027496, 1.057976] | 1.0257 | 0.8406 | 1.1561 | 0.2924 |
| Dord/Cord | 1.100748 | [1.095323, 1.107751] | 1.0679 | 0.3704 | 0.9151 | 0.0177 |
| Mord/Cord | 1.054933 | [1.036120, 1.077734] | 1.0411 | 0.4406 | 0.7916 | 0.0606 |

### Within-engine safe ordinal / native uplift

These rows match each diagnostic job to that engine's primary job on the same dataset, repeat, fold, and child-fold structure.

| Engine contrast | Test RMSE | Test 95% CI | Validation |
| --- | ---: | ---: | ---: |
| Dord/Dnative | 0.811049 | [0.798810, 0.821936] | 0.8011 |
| Mord/Mnative | 0.794817 | [0.780872, 0.808299] | 0.7984 |
| Cord/Cnative | 0.922582 | [0.921650, 0.923545] | 0.9352 |

Cross-lane ordinal/native training time, inference time, and memory are intentionally omitted because separate lane execution makes those ratios order- and process-history-confounded.

## Fitted child telemetry

- ChimeraBoost selected lanes: `{"constant": 235, "linear": 125}`.
- ChimeraBoost retained rounds: median 396.0, p90 1221.0, max 2319.
- Known failures/imputations/deadlines/time limits: 0/0/0/0.
- Competitor children with unresolved stop reason: 494 (reported, never silently classified as early stopping).
- A null competitor stop reason can include an unexposed time or memory callback outcome; this qualifies the descriptive comparison and is not evidence that every competitor child avoided truncation.

## Measurement and integrity

- Training and inference timings are same-machine measurements from the same frozen campaign.
- Incremental memory (`peak_mem_cpu - min_mem_cpu`) is the primary memory comparison; raw process peak RSS is secondary.
- Zero incremental-memory observations affecting a ratio: 0. Affected log-ratio aggregates are marked unavailable without an epsilon or pseudocount; raw bytes remain reported.
- All 135 raw result files were verified as opaque bytes. This analyzer never unpickled them.
- Exact source commits/wheel bytes, adapters, runtime, hardware, configuration, feature schemas, order, 135 outer rows, and 1,080 child rows matched the completion attestation.

## Provenance

- DarkoFit Git commit: `6c2dc19d2e50a57f5dd835c916e12adb69264706`.
- ChimeraBoost Git commit: `9c9ea6e704a9fe2bfe6d6c284b22de73914be048`.
- Frozen protocol semantic SHA-256: `fc9583d33affe774d56cbbec30b4e7450d95e87b688e1fb79a88b65b33dce7d3`.
- Ordered-grid SHA-256: `d8c5589f135f167a2de74a5fe80d6af191c3e483175ba36b435bf49a7fd8a1ec`.
- Manifest SHA-256: `2869acaaa4bcc8319d9ba03744a4a9ca8602ed349553a031c3d84ab537de72ee`.
- Completion attestation SHA-256: `213f462aa06103e97864ecd786b75e8fd8e11743c77f556262fa39bdb3e1b7d9`.
- Safe analysis payload SHA-256: `29010ff23ff6714092ae64cfc9c1b171f7523c943247c91107cdf0c86b0fd2e7`.

## Decision boundary

**Descriptive comparison only. No default or ordinal policy is advanced by this analysis.**
