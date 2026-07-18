# Engineering measurements

These are descriptive Tier-E measurements, not universal performance certifications. Each table is generated from immutable same-machine artifacts whose hashes are checked before rendering. Ratios below 1.0 favor DarkoFit.

## Matched large-n fitting

DarkoFit / ChimeraBoost 0.15.0 on the matched numeric core.

| Training rows | Fit ratio | Paired Q1–Q3 | IQR / median | Speedup | RMSE ratio | Peak RSS ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 500,000 | 0.8038× | 0.7927–0.8234× | 0.03821 | 1.2441× | 0.99998× | 0.8432× |
| 1,000,000 | 0.7601× | 0.7532–0.7631× | 0.01294 | 1.3155× | 1.00085× | 0.7129× |

The equal-size geometric mean was 0.7817× (1.2793× faster). The immutable campaign's old 1.30× binary certification remains closed; the measured number is reported without rounding it into a pass.

## Public prediction throughput

Seconds-integrated public prediction, DarkoFit / ChimeraBoost 0.15.0.

| Input | Rows | Median ratio | Paired Q1–Q3 | IQR / median |
|---|---:|---:|---:|---:|
| basketball numeric | 8,192 | 0.805× | 0.747–0.832× | 0.10556 |
| basketball numeric | 65,536 | 0.958× | 0.945–0.976× | 0.03263 |
| basketball numeric | 524,288 | 0.918× | 0.833–0.925× | 0.10032 |
| basketball numeric | 2,000,000 | 0.869× | 0.860–0.911× | 0.05872 |
| synthetic mixed | 8,192 | 0.885× | 0.868–0.899× | 0.03529 |
| synthetic mixed | 65,536 | 0.947× | 0.928–0.972× | 0.04656 |
| synthetic mixed | 524,288 | 0.987× | 0.974–0.988× | 0.01459 |
| synthetic mixed | 2,000,000 | 0.984× | 0.980–0.992× | 0.01259 |

All 8 medians were no slower; 6 also met the old campaign's stability threshold. The two facts are stated separately so a noisy ratio is not mislabeled as certified.

## Sampled fused training

Scope: 5,241-row x 15-feature basketball workload; selected rows, features, and both; unit and variable Hessians; 18 threads.

| Measure | Geometric-mean ratio | Cell-median range | Maximum paired IQR / median |
|---|---:|---:|---:|
| Fit | 0.5348× | 0.5110–0.5639× | 0.2085 |
| Tree build | 0.5265× | 0.5029–0.5521× | 0.2060 |

All eight reference/candidate cells had identical predictions, behavior fingerprints, and canonical serialized model state. The sampled dispatch ships as exact internal engine work. These workload measurements do not imply the same speedup on every dataset or machine.

## Evidence policy

See the [shipping policy](https://github.com/kmedved/darkofit/blob/main/benchmarks/SHIPPING_POLICY.md). Defaults and automatic modeling policies remain Tier-D and require fresh, power-checked, preregistered confirmation with uncertainty, concentration, harm, and cost controls.
