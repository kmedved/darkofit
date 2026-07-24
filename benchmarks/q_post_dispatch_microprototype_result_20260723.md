# Q post-dispatch packed-histogram microprototype result

Normal development evidence; no holdout ship-check or TabArena data was consulted.

- Source: `832e36d3784642404a210a41855f5985def6566b`
- Integrity: `true`
- Q1 funded: `false`
- Disposition: `close_q_at_microprototype`

| Rows | Control dispatch | Fit ratio | IQR / median | RMSE ratio | Predict ratio | RSS ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 500,000 | fused | 0.835290 | 0.117662 | 1.000000 | 1.006433 | 1.021816 |
| 1,000,000 | unfused | 0.819009 | 0.021810 | 1.000000 | 1.114178 | 1.000413 |

Equal-size geomean fit ratio: `0.827110` (funding bar `<= 0.90`).

The candidate is benchmark-local and changes split selection through stochastic gradient quantization. Leaf values remain float64. This result cannot ship an option or default.

Raw SHA-256: `63f0660ebbb80dc7248f52038fd6d64837bda8fed2cd0ea1425527fec99e48eb`
