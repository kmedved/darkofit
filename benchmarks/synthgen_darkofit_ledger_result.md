# DarkoFit SynthGen df1 ledger result

**Verdict: NOT ADOPTED.**

Mechanism-probe adoption only; this artifact cannot promote a DarkoFit parameter, preset, policy, default, or release claim.

Agreement: **6/9**. Raw coordinates: **1464**.

| # | Decision | Aggregate ratio | W-L-T | Agreement |
|---:|---|---:|---:|:---:|
| 1 | Student-t location versus RMSE | 1.080207 | 18-28-0 | yes |
| 2 | MAE versus RMSE | 1.055499 | 11-35-0 | yes |
| 3 | random_strength=0.5 versus 0.0 | 0.999844 | 1-7-0 | no |
| 4 | random_strength=1.0 versus 0.5 | 0.999477 | 5-3-0 | yes |
| 5 | local linear leaves versus constant leaves | 1.004043 | 2-5-4 | no |
| 6 | global linear residual versus local linear leaves | 0.979073 | 7-4-0 | no |
| 7 | ts_permutations=4 versus 1 | 0.987819 | 10-7-0 | yes |
| 8 | forced ordered boosting versus scalar default | 1.134988 | 3-43-0 | yes |
| 9 | frozen speed-oriented core profile | 1.029495 | 9-37-0 | yes |

## Canary and adoption gates

Categorical-canary equal-dataset mean Brier delta (`TS=4 - control`): `-0.000779117109536`.

- PASS — `complete_raw_boundary`
- PASS — `all_slices_have_at_least_8_datasets`
- PASS — `freeze_floor_passes`
- PASS — `canary_no_variance_passes`
- FAIL — `at_least_7_of_9_decisions_agree`
- PASS — `protected_outcome_sources_not_accessed`

## Provenance

- Source commit: `c05d621e32a3125529eccaa70727788c6cad8e1f`
- Run fingerprint: `c894dcd47fbb4e02191f0d5ddbb0911f0659c148206296847cfdd5816c4c6854`
- Raw SHA-256: `fd8f93ec4c0e1cbd6889200d0d79f235e7e880732f0597e09a2bd025f09af7eb`
- Analyzer SHA-256: `65669eb564417a47919753879ef3e5988965180fc0bf4639bf9be35ed988abc7`
