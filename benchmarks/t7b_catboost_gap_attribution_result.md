# T7b CatBoost-gap attribution

**Decision: `development_attribution_only`.** This is spent development evidence and
does not authorize a default change.

## Evidence bindings

- Frozen source commit: `ce252439845b18e8a255ce034f01e7885b3ebf4b`
- Original run-time runner SHA-256:
  `e429a2096a51239639405a66930d1f2457cddaf31616f2758512bba932bd966b`
- Original frozen analyzer SHA-256:
  `08f2b268433d5d266a511f459357a28845e502b388698f2e844530b1d2fbd5e3`
- Current hardened runner SHA-256:
  `e429a2096a51239639405a66930d1f2457cddaf31616f2758512bba932bd966b`
- Current hardened analyzer SHA-256:
  `dd2f5798e027a7f9b1daff4c23b5193fd05f02bcd54a29e73db9975da83407c1`
- Frozen raw file SHA-256: `29055ecac0bf920820ede2735f61df5420ecb1b71c2303ddd12e30f445e1be06`
- Frozen raw canonical SHA-256: `c7dba3b5d21f7d64e71526c78092101124f4051c33382e44f746b85189554fca`

The original hashes identify the source bytes frozen for the run and its
analysis. The current hashes identify the later hardened copies used to
revalidate and publish those same outcomes; no benchmark was rerun.

Historical DarkoFit / CatBoost default RMSE ratio:
`1.029345`.

The seed-4 bridge reports the fraction of that historical numerical gap erased
by a CatBoost perturbation. It is not a causal fraction explained. The
three-seed value is a separate sensitivity estimate.

Multiplicity uses 100,000 deterministic bootstrap draws,
familywise alpha 0.050, and
14 directional hypotheses. The Bonferroni
quantiles are 0.003571 and
0.996429. Seeds are fixed repeat blocks and are averaged,
not independently resampled within each fold.

| Arm | Test ratio | Validation ratio | Bonferroni lower | Bonferroni upper | Seed-4 gap erased | Three-seed sensitivity | Worst task | Contributor LOO | Promising LOO | Label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random_strength_0 | 1.007043 | 1.004935 | 0.990275 | 1.031791 | 41.3% | 24.3% | 1.038335 | 1.002650 | 1.009419 | `not_attributed` |
| bootstrap_no | 1.005317 | 1.005178 | 0.993365 | 1.020164 | 30.6% | 18.3% | 1.027678 | 1.002163 | 1.007572 | `not_attributed` |
| no_split_noise_or_row_sampling | 1.001142 | 1.003760 | 0.976579 | 1.033953 | 4.6% | 3.9% | 1.038102 | 0.995970 | 1.006582 | `not_attributed` |
| l2_leaf_reg_1 | 0.953308 | 0.963543 | 0.882462 | 0.999548 | -201.4% | -165.3% | 1.010590 | 0.945395 | 0.969883 | `promising_config` |
| one_hot_max_size_0 | 1.016574 | 1.009226 | 1.000000 | 1.053207 | 87.8% | 56.8% | 1.068660 | 1.009344 | 1.018964 | `not_attributed` |
| one_hot_max_size_255 | 0.847819 | 0.852375 | 0.610154 | 1.016406 | -600.9% | -570.8% | 1.021808 | 0.825510 | 0.914498 | `not_attributed` |
| leaf10_any_improvement | 0.907834 | 0.936950 | 0.784225 | 1.002382 | -312.9% | -334.3% | 1.033081 | 0.891227 | 0.936518 | `not_attributed` |

## Seed blocks

| Arm | Seed 4 | Seed 17 | Seed 29 |
|---|---:|---:|---:|
| random_strength_0 | 1.012007 | 0.997518 | 1.011671 |
| bootstrap_no | 1.008904 | 0.987582 | 1.019733 |
| no_split_noise_or_row_sampling | 1.001322 | 0.990141 | 1.012083 |
| l2_leaf_reg_1 | 0.943422 | 0.943577 | 0.973233 |
| one_hot_max_size_0 | 1.025720 | 1.017567 | 1.006527 |
| one_hot_max_size_255 | 0.840478 | 0.845804 | 0.857260 |
| leaf10_any_improvement | 0.913471 | 0.907643 | 0.902422 |

## Gate evidence

| Arm | Gate | Outcome |
|---|---|---|
| random_strength_0 | equal_dataset_test_ratio_gt_1 | pass |
| random_strength_0 | equal_dataset_validation_ratio_gt_1 | pass |
| random_strength_0 | bonferroni_lower_gt_1 | fail |
| random_strength_0 | every_leave_one_task_out_ratio_gt_1 | pass |
| random_strength_0 | every_seed_block_ratio_gt_1 | fail |
| random_strength_0 | equal_dataset_test_ratio_lte_0_995 | fail |
| random_strength_0 | equal_dataset_validation_ratio_lte_1_005 | pass |
| random_strength_0 | bonferroni_upper_lt_1 | fail |
| random_strength_0 | worst_task_test_ratio_lte_1_02 | fail |
| random_strength_0 | every_leave_one_task_out_ratio_lte_1 | fail |
| random_strength_0 | every_seed_block_ratio_lte_1_005 | fail |
| bootstrap_no | equal_dataset_test_ratio_gt_1 | pass |
| bootstrap_no | equal_dataset_validation_ratio_gt_1 | pass |
| bootstrap_no | bonferroni_lower_gt_1 | fail |
| bootstrap_no | every_leave_one_task_out_ratio_gt_1 | pass |
| bootstrap_no | every_seed_block_ratio_gt_1 | fail |
| bootstrap_no | equal_dataset_test_ratio_lte_0_995 | fail |
| bootstrap_no | equal_dataset_validation_ratio_lte_1_005 | fail |
| bootstrap_no | bonferroni_upper_lt_1 | fail |
| bootstrap_no | worst_task_test_ratio_lte_1_02 | fail |
| bootstrap_no | every_leave_one_task_out_ratio_lte_1 | fail |
| bootstrap_no | every_seed_block_ratio_lte_1_005 | fail |
| no_split_noise_or_row_sampling | equal_dataset_test_ratio_gt_1 | pass |
| no_split_noise_or_row_sampling | equal_dataset_validation_ratio_gt_1 | pass |
| no_split_noise_or_row_sampling | bonferroni_lower_gt_1 | fail |
| no_split_noise_or_row_sampling | every_leave_one_task_out_ratio_gt_1 | fail |
| no_split_noise_or_row_sampling | every_seed_block_ratio_gt_1 | fail |
| no_split_noise_or_row_sampling | equal_dataset_test_ratio_lte_0_995 | fail |
| no_split_noise_or_row_sampling | equal_dataset_validation_ratio_lte_1_005 | pass |
| no_split_noise_or_row_sampling | bonferroni_upper_lt_1 | fail |
| no_split_noise_or_row_sampling | worst_task_test_ratio_lte_1_02 | fail |
| no_split_noise_or_row_sampling | every_leave_one_task_out_ratio_lte_1 | fail |
| no_split_noise_or_row_sampling | every_seed_block_ratio_lte_1_005 | fail |
| l2_leaf_reg_1 | equal_dataset_test_ratio_gt_1 | fail |
| l2_leaf_reg_1 | equal_dataset_validation_ratio_gt_1 | fail |
| l2_leaf_reg_1 | bonferroni_lower_gt_1 | fail |
| l2_leaf_reg_1 | every_leave_one_task_out_ratio_gt_1 | fail |
| l2_leaf_reg_1 | every_seed_block_ratio_gt_1 | fail |
| l2_leaf_reg_1 | equal_dataset_test_ratio_lte_0_995 | pass |
| l2_leaf_reg_1 | equal_dataset_validation_ratio_lte_1_005 | pass |
| l2_leaf_reg_1 | bonferroni_upper_lt_1 | pass |
| l2_leaf_reg_1 | worst_task_test_ratio_lte_1_02 | pass |
| l2_leaf_reg_1 | every_leave_one_task_out_ratio_lte_1 | pass |
| l2_leaf_reg_1 | every_seed_block_ratio_lte_1_005 | pass |
| one_hot_max_size_0 | equal_dataset_test_ratio_gt_1 | pass |
| one_hot_max_size_0 | equal_dataset_validation_ratio_gt_1 | pass |
| one_hot_max_size_0 | bonferroni_lower_gt_1 | fail |
| one_hot_max_size_0 | every_leave_one_task_out_ratio_gt_1 | pass |
| one_hot_max_size_0 | every_seed_block_ratio_gt_1 | pass |
| one_hot_max_size_0 | equal_dataset_test_ratio_lte_0_995 | fail |
| one_hot_max_size_0 | equal_dataset_validation_ratio_lte_1_005 | fail |
| one_hot_max_size_0 | bonferroni_upper_lt_1 | fail |
| one_hot_max_size_0 | worst_task_test_ratio_lte_1_02 | fail |
| one_hot_max_size_0 | every_leave_one_task_out_ratio_lte_1 | fail |
| one_hot_max_size_0 | every_seed_block_ratio_lte_1_005 | fail |
| one_hot_max_size_255 | equal_dataset_test_ratio_gt_1 | fail |
| one_hot_max_size_255 | equal_dataset_validation_ratio_gt_1 | fail |
| one_hot_max_size_255 | bonferroni_lower_gt_1 | fail |
| one_hot_max_size_255 | every_leave_one_task_out_ratio_gt_1 | fail |
| one_hot_max_size_255 | every_seed_block_ratio_gt_1 | fail |
| one_hot_max_size_255 | equal_dataset_test_ratio_lte_0_995 | pass |
| one_hot_max_size_255 | equal_dataset_validation_ratio_lte_1_005 | pass |
| one_hot_max_size_255 | bonferroni_upper_lt_1 | fail |
| one_hot_max_size_255 | worst_task_test_ratio_lte_1_02 | fail |
| one_hot_max_size_255 | every_leave_one_task_out_ratio_lte_1 | pass |
| one_hot_max_size_255 | every_seed_block_ratio_lte_1_005 | pass |
| leaf10_any_improvement | equal_dataset_test_ratio_gt_1 | fail |
| leaf10_any_improvement | equal_dataset_validation_ratio_gt_1 | fail |
| leaf10_any_improvement | bonferroni_lower_gt_1 | fail |
| leaf10_any_improvement | every_leave_one_task_out_ratio_gt_1 | fail |
| leaf10_any_improvement | every_seed_block_ratio_gt_1 | fail |
| leaf10_any_improvement | equal_dataset_test_ratio_lte_0_995 | pass |
| leaf10_any_improvement | equal_dataset_validation_ratio_lte_1_005 | pass |
| leaf10_any_improvement | bonferroni_upper_lt_1 | fail |
| leaf10_any_improvement | worst_task_test_ratio_lte_1_02 | fail |
| leaf10_any_improvement | every_leave_one_task_out_ratio_lte_1 | pass |
| leaf10_any_improvement | every_seed_block_ratio_lte_1_005 | pass |

## Per-task ratios

| Arm | Task | Dataset | Test ratio | Validation ratio |
|---|---:|---|---:|---:|
| random_strength_0 | 361236 | auction_verification | 1.010633 | 0.987517 |
| random_strength_0 | 361252 | video_transcoding | 0.990564 | 0.984908 |
| random_strength_0 | 361268 | fps_benchmark | 1.004509 | 1.004618 |
| random_strength_0 | 361622 | cars | 1.013372 | 1.019097 |
| random_strength_0 | 363372 | bookprice_prediction | 1.000172 | 1.003369 |
| random_strength_0 | 363375 | ae_price_prediction | 1.001241 | 1.003244 |
| random_strength_0 | 363471 | munich-rent-index-1999 | 1.038335 | 1.035002 |
| random_strength_0 | 363631 | diamonds | 0.998236 | 1.002628 |
| bootstrap_no | 361236 | auction_verification | 0.989673 | 1.022803 |
| bootstrap_no | 361252 | video_transcoding | 0.998161 | 0.997499 |
| bootstrap_no | 361268 | fps_benchmark | 1.019296 | 1.020995 |
| bootstrap_no | 361622 | cars | 1.003808 | 0.995858 |
| bootstrap_no | 363372 | bookprice_prediction | 1.002299 | 1.002464 |
| bootstrap_no | 363375 | ae_price_prediction | 1.002207 | 1.004935 |
| bootstrap_no | 363471 | munich-rent-index-1999 | 1.027678 | 0.995443 |
| bootstrap_no | 363631 | diamonds | 0.999931 | 1.001835 |
| no_split_noise_or_row_sampling | 361236 | auction_verification | 0.963874 | 0.967265 |
| no_split_noise_or_row_sampling | 361252 | video_transcoding | 0.989814 | 0.984055 |
| no_split_noise_or_row_sampling | 361268 | fps_benchmark | 1.000341 | 1.002030 |
| no_split_noise_or_row_sampling | 361622 | cars | 1.015403 | 1.030094 |
| no_split_noise_or_row_sampling | 363372 | bookprice_prediction | 1.002817 | 1.005362 |
| no_split_noise_or_row_sampling | 363375 | ae_price_prediction | 1.001526 | 1.004598 |
| no_split_noise_or_row_sampling | 363471 | munich-rent-index-1999 | 1.038102 | 1.033959 |
| no_split_noise_or_row_sampling | 363631 | diamonds | 0.998804 | 1.004383 |
| l2_leaf_reg_1 | 361236 | auction_verification | 0.934089 | 0.913374 |
| l2_leaf_reg_1 | 361252 | video_transcoding | 0.945274 | 0.938864 |
| l2_leaf_reg_1 | 361268 | fps_benchmark | 0.909142 | 0.911843 |
| l2_leaf_reg_1 | 361622 | cars | 1.010590 | 0.999038 |
| l2_leaf_reg_1 | 363372 | bookprice_prediction | 1.000902 | 1.002767 |
| l2_leaf_reg_1 | 363375 | ae_price_prediction | 0.996886 | 0.998552 |
| l2_leaf_reg_1 | 363471 | munich-rent-index-1999 | 0.844951 | 0.948279 |
| l2_leaf_reg_1 | 363631 | diamonds | 0.997346 | 1.001629 |
| one_hot_max_size_0 | 361236 | auction_verification | 1.000000 | 1.000000 |
| one_hot_max_size_0 | 361252 | video_transcoding | 1.000000 | 1.000000 |
| one_hot_max_size_0 | 361268 | fps_benchmark | 1.057521 | 1.065781 |
| one_hot_max_size_0 | 361622 | cars | 1.000000 | 1.000000 |
| one_hot_max_size_0 | 363372 | bookprice_prediction | 1.000000 | 1.000000 |
| one_hot_max_size_0 | 363375 | ae_price_prediction | 1.009217 | 1.009297 |
| one_hot_max_size_0 | 363471 | munich-rent-index-1999 | 1.068660 | 1.000507 |
| one_hot_max_size_0 | 363631 | diamonds | 1.000000 | 1.000000 |
| one_hot_max_size_255 | 361236 | auction_verification | 0.499049 | 0.518383 |
| one_hot_max_size_255 | 361252 | video_transcoding | 0.971239 | 0.970600 |
| one_hot_max_size_255 | 361268 | fps_benchmark | 0.523746 | 0.520364 |
| one_hot_max_size_255 | 361622 | cars | 1.000000 | 1.000000 |
| one_hot_max_size_255 | 363372 | bookprice_prediction | 0.998779 | 1.003035 |
| one_hot_max_size_255 | 363375 | ae_price_prediction | 1.009034 | 1.016368 |
| one_hot_max_size_255 | 363471 | munich-rent-index-1999 | 1.021808 | 1.013351 |
| one_hot_max_size_255 | 363631 | diamonds | 1.021149 | 1.030200 |
| leaf10_any_improvement | 361236 | auction_verification | 0.844066 | 0.837189 |
| leaf10_any_improvement | 361252 | video_transcoding | 0.870753 | 0.890881 |
| leaf10_any_improvement | 361268 | fps_benchmark | 0.835337 | 0.841048 |
| leaf10_any_improvement | 361622 | cars | 1.033081 | 1.028748 |
| leaf10_any_improvement | 363372 | bookprice_prediction | 1.004709 | 1.005287 |
| leaf10_any_improvement | 363375 | ae_price_prediction | 0.991472 | 0.994451 |
| leaf10_any_improvement | 363471 | munich-rent-index-1999 | 0.730194 | 0.918903 |
| leaf10_any_improvement | 363631 | diamonds | 1.000056 | 1.001881 |

## Noise/sampling incremental comparison

This comparison is descriptive and does not authorize an interaction claim.
Status: `not_supported`. Paired additive log departure:
`-0.011180`.

| Combined over component | Test ratio | Validation ratio |
|---|---:|---:|
| random_strength_0 | 0.994140 | 0.998830 |
| bootstrap_no | 0.995846 | 0.998589 |

## Descriptive runtime

These measurements were collected under concurrent execution and are not
inferential gates.

| Arm | Median fit seconds | Fit / baseline | Median predict seconds | Predict / baseline |
|---|---:|---:|---:|---:|
| baseline | 3.049826 | 1.000 | 0.002574 | 1.000 |
| random_strength_0 | 3.020812 | 0.990 | 0.002543 | 0.988 |
| bootstrap_no | 3.102670 | 1.017 | 0.002390 | 0.928 |
| no_split_noise_or_row_sampling | 3.108804 | 1.019 | 0.002334 | 0.907 |
| l2_leaf_reg_1 | 3.023087 | 0.991 | 0.002474 | 0.961 |
| one_hot_max_size_0 | 3.001217 | 0.984 | 0.002498 | 0.970 |
| one_hot_max_size_255 | 1.497553 | 0.491 | 0.002237 | 0.869 |
| leaf10_any_improvement | 4.932023 | 1.617 | 0.002684 | 1.043 |

Execution peak RSS: median 614875136 bytes; maximum
1060012032 bytes.

Contributors: none.
Promising configurations: ['l2_leaf_reg_1'].

The explanatory direction requires an ablation to worsen CatBoost; the
promising-configuration direction requires an improvement. No confirmation
or lockbox data was opened.

Raw file SHA-256: `29055ecac0bf920820ede2735f61df5420ecb1b71c2303ddd12e30f445e1be06`.
Raw canonical SHA-256: `c7dba3b5d21f7d64e71526c78092101124f4051c33382e44f746b85189554fca`.
