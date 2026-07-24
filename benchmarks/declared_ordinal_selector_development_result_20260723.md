# Declared-order selector development result

This is spent development evidence on the two historical declared-order
domains. Inner validation selects the representation; disjoint outer
test rows score native, forced ordinal, and automatic selector arms.

- Source: `8b87e562ffbf9321805e67309df7fd507abd7628`
- Equal-dataset selector/native RMSE ratio: `0.815447`
- Worst coordinate selector/native RMSE ratio: `0.974516`
- Engagements: `6/6`
- Final selected/native refits prediction-exact: `true`

| Dataset | Selector/native | Forced/native | Engaged | Worst |
| --- | ---: | ---: | ---: | ---: |
| airfoil_self_noise | 0.870938 | 0.870938 | 3/3 | 0.974516 |
| diamonds | 0.763492 | 0.763492 | 3/3 | 0.807154 |

The panel is intentionally narrow: both datasets were used by the
historical safe-ordinal campaign. It tests selector behavior and
transfer to held-out rows, not generalization to new datasets.
