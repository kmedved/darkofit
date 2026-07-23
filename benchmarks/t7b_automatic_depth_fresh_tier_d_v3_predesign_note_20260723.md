# P1-v3 fillability-first pre-design registry

_Dated resource-enumeration note, 2026-07-23. This names the concrete
as-built panel available for prospective power design. It is not a frozen
confirmation panel, quality evidence, or authorization to run a model._

Enumeration:
`t7b-automatic-depth-fresh-tier-d-v3-enumeration-v2-20260723`.

Create-only artifact:
`t7b_automatic_depth_fresh_tier_d_v3_enumeration_v2_20260723.json`,
SHA-256
`c7c76259823d6ee4d3ce6202b127d4bed984493e6153775dfb0f05a105b22851`.

## As-built result

Thirty-two of the 40 previously declared concrete OpenML identities passed
all resource checks:

- 17 low-density `depth_4` lineages: 9 numeric and 8
  categorical-or-grouped;
- 15 high-density `depth_8` lineages: 5 numeric and 10
  categorical-or-grouped;
- three group-safe lineages; and
- three verified split/weight coordinates per lineage.

Every eligible identity loaded successfully in `darko311` on
macOS arm64 with Python 3.11, NumPy 2.2.6, pandas 2.2.3, sklearn 1.7.1,
Numba 0.61.2, and OpenML 0.15.1. Each passed exact OpenML binding,
repository-history and exact/near-fingerprint contamination checks, target
validity, feature-family checks, deterministic split/group checks, and its
declared automatic-depth branch.

| Lineage | OpenML task | Dataset | Stratum | Branch | Split |
| --- | ---: | ---: | --- | --- | --- |
| `airlines_departure_delay_10m` | 359929 | 42728 | high categorical/grouped | depth 8 | row |
| `bangladesh_station_rainfall` | 189935 | 41539 | high categorical/grouped | depth 8 | row |
| `beijing_pm25` | 362091 | 46285 | high categorical/grouped | depth 8 | row |
| `bng_auto_horse_price` | 7319 | 1192 | high categorical/grouped | depth 8 | row |
| `candy_crush_level` | 362369 | 43471 | high categorical/grouped | depth 8 | row |
| `iot_room_temperature` | 362353 | 43351 | high categorical/grouped | depth 8 | row |
| `kaggle_30_days_tabular` | 362332 | 43090 | high categorical/grouped | depth 8 | row |
| `reddit_climate_opinion` | 363604 | 46896 | high categorical/grouped | depth 8 | row |
| `rossmann_store_sales` | 361924 | 45647 | high categorical/grouped | depth 8 | row |
| `video_game_fps_hardware` | 362129 | 42737 | high categorical/grouped | depth 8 | row |
| `automl_yolanda` | 317614 | 42705 | high numeric | depth 8 | row |
| `bng_pwlinear` | 7325 | 1203 | high numeric | depth 8 | row |
| `french_rte_load` | 362125 | 46337 | high numeric | depth 8 | row |
| `mtpl_claim_count` | 363083 | 45106 | high numeric | depth 8 | row |
| `twitter_buzz_annotation` | 233213 | 4549 | high numeric | depth 8 | row |
| `aids_joint_model_survival_time` | 363196 | 46130 | low categorical/grouped | depth 4 | row |
| `asp_potassco_runtime` | 189936 | 41704 | low categorical/grouped | depth 4 | group |
| `chscase_foot` | 5012 | 703 | low categorical/grouped | depth 4 | row |
| `cpmp_2015_runtime` | 189940 | 41700 | low categorical/grouped | depth 4 | group |
| `mip_2016_par10` | 362319 | 43070 | low categorical/grouped | depth 4 | group |
| `optical_interconnection_network` | 211855 | 42365 | low categorical/grouped | depth 4 | row |
| `sat11_hand_runtime` | 359948 | 41980 | low categorical/grouped | depth 4 | row |
| `uci_auto_mpg` | 2287 | 196 | low categorical/grouped | depth 4 | row |
| `ankara_weather` | 211858 | 42368 | low numeric | depth 4 | row |
| `carpenter_fda_survival_time` | 363215 | 46159 | low numeric | depth 4 | row |
| `chscase_census5` | 4987 | 670 | low numeric | depth 4 | row |
| `friedman_c4_500_100` | 4929 | 610 | low numeric | depth 4 | row |
| `galaxy_velocity` | 5001 | 690 | low numeric | depth 4 | row |
| `los_angeles_mortality_timeseries` | 4984 | 666 | low numeric | depth 4 | row |
| `qsar_aquatic_toxicity` | 362095 | 46295 | low numeric | depth 4 | row |
| `uci_liver_disorders` | 52948 | 8 | low numeric | depth 4 | row |
| `usa_treasury_rates` | 211857 | 42367 | low numeric | depth 4 | row |

## Rejected identities

| Lineage | Stratum | Resource finding |
| --- | --- | --- |
| `california_environmental_conditions` | high numeric | categorical inputs violate numeric role |
| `cern_dielectron_mass` | high numeric | non-finite target |
| `comet_monte_carlo` | high numeric | OpenML task target drift |
| `fao_rice_methane_emissions` | high numeric | categorical inputs violate numeric role |
| `synthetic_workers_compensation` | high numeric | categorical inputs violate numeric role |
| `kdd_el_nino_buoys` | low categorical/grouped | non-finite target |
| `urban_water_treatment` | low categorical/grouped | non-finite target |
| `uhpc_strength_tmp1` | low numeric | categorical inputs violate numeric role |

These are resource findings, not unfavorable model outcomes. They cannot be
renamed, reassigned, imputed, or replaced after a future freeze.

## Boundary and next decision

No target statistics, target values, model fits, candidate/control outcomes,
TabArena, CTR23 execution, or lockbox data were published or inspected. The
fresh confirmation inspection remains unspent.

The next authorized step is a prospective power simulation on exactly these
32 verified identities and their 17/15 branch composition. If both simulated
power and its one-sided Wilson lower bound are at least 80%, the exact
registry, power design, and execution contract may be prepared together for
owner freeze review. The fresh run itself remains separately gated.
