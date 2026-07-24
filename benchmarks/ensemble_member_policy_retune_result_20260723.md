# Ensemble-v3 member-policy retune

Spent general-development evidence; no holdout was consulted.

- Source: `fb3dd4a8621328b20be9ab7af67995b791790d38`
- Cells: `60`
- Selected recipe: `current`
- Public policy changed: `false`

| Recipe | Quality/current | Worst dataset | Worst LOO | Quality/single | Fit/current | Predict/current | Archive/current |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy_auto | 1.008768 | 1.065336 | 1.010200 | 0.951680 | 2.111264 | 2.014894 | 1.637199 |
| intermediate | 1.001809 | 1.030807 | 1.002928 | 0.945114 | 1.198133 | 1.120046 | 1.086640 |
| current | 1.000000 | 1.000000 | 1.000000 | 0.943408 | 1.000000 | 1.000000 | 1.000000 |

The slice contains deterministic synthetic and resampled sklearn
development datasets. It can choose an ensemble opt-in recipe; it does
not establish a new automatic default or unseen-data claim.

Raw SHA-256: `cf6ca60c3f03c67a3b473918139b344048a63bdee9b54b00efe3205cb368ed7e`
