# v0.11 private ensemble evidence result

This is Tier-E descriptive evidence for the private eight-member release candidate. It is not public exposure, M2/M4, a default change, or release authority.

## Reproduction and quality

- Immutable M3b reproduction: **PASS** at absolute ratio tolerance `1.0e-10`.
- Pooled v3/single primary-loss ratio: `0.965513`.
- Sports ratio: `0.961077`; season-cluster bootstrap 95% descriptive interval `[0.958861, 0.962867]`.
- General ratio: `0.975569`; case-bootstrap 95% descriptive interval `[0.963303, 0.987718]`.
- The sports unit is three season clusters; the four general cells are fixed seeded cases. The 13 cells are not presented as independent datasets.

## Cost telemetry

| Comparison | Fit | Peak RSS | RSS delta | Archive bytes |
| --- | ---: | ---: | ---: | ---: |
| v3_vs_single | 5.030x | 1.090x | 3.539x | 6.181x |
| v3_vs_existing_bootstrap8 | 0.578x | 0.999x | 0.935x | 0.706x |

Ratios are telemetry beside absolute per-case values in the JSON result. The historical archive-size gate remains retracted.

## Prediction throughput

Seconds ratios below one favor the numerator. Each aggregate is the equal-coordinate geometric mean of three-block median paired ratios.

| Comparison | Seconds ratio | Faster / 16 coordinates |
| --- | ---: | ---: |
| darkofit_single_vs_chimeraboost_single | 0.478x | 16 |
| darkofit_single_vs_catboost_single | 0.871x | 9 |
| chimeraboost_single_vs_catboost_single | 1.827x | 4 |
| darkofit_v3_vs_darkofit_single | 6.251x | 0 |
| darkofit_v3_vs_chimeraboost_ensemble8 | 0.126x | 16 |

## Disposition

No correctness or reproduction stop condition is present. This result does not itself authorize public exposure; all performance, cost, and dispersion findings are disclosures rather than gates.
