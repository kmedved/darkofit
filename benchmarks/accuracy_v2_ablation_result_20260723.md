# Accuracy-v2 component ablation

Spent M6 regression development evidence; no holdout was consulted.

- Source: `2def9d7662ec5d00592e18e518777665ab341ae7`
- Cells: `24`
- A10 cross engagements: `0`
- Selected profile: `accuracy`

| Contrast | Quality ratio | Worst dataset | Worst LOO | Fit ratio | Predict ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| horizon | 0.999914 | 1.000000 | 1.000000 | 0.954112 | 0.890471 |
| cross_at_1k | 1.000000 | 1.000000 | 1.000000 | 2.071968 | 1.000000 |
| cross_at_10k | 1.000000 | 1.000000 | 1.000000 | 2.128355 | 1.000000 |
| combined | 0.999914 | 1.000000 | 1.000000 | 2.030688 | 0.890471 |

A declined cross arm is prediction-exact to its uncrossed fallback.
This development slice may select an explicit accuracy profile; it
cannot establish a new default or unseen-data claim.

Raw SHA-256: `0a2f7378aa9bcada93ef6a910d557fc8e2727852a0ab21d3cf96c43e88d2279c`
