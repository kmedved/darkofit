# B3-v2 activation-gated parallel ensemble result

The activation-gated successor ran on 2026-07-23 from clean source
`b35c092bbdfef45f2ac4d5b0cc16eaaf1c89bf55`. It reused the spent B3-v1
four-case timing grid, fixed 14-CPU budget, three paired blocks, and cold- and
steady-executor measurements. No holdout, sports, or fresh dataset was used.

## What changed

The candidate computes a static pre-fit work score:

`sampled rows × input features × planned iterations × output width`

Output width is one for regression and binary classification and the class
count for multiclass. Work at or above `80,000,000` uses seven process workers
with two model threads each. Smaller work stays on the existing sequential
path. On this grid, numeric binary and categorical multiclass engaged;
Friedman regression and categorical regression fell back.

## Result

All declared checks passed and the disposition is `ready_to_productize`.
Predictions, probabilities, member seeds, sampling identities, OOB identities,
and best iterations were exact between paired arms. The expected route and
full fitted thread schedule matched on every row. RSS sampling was clean.

Ratios are candidate/control fit wall time; lower is better.

| Route | Cold geomean | Cold worst | Steady geomean | Steady worst |
| --- | ---: | ---: | ---: | ---: |
| Parallel engaged | `0.487217` | `0.491280` | `0.235154` | `0.283799` |
| Sequential fallback | `1.015415` | `1.032163` | `1.019149` | `1.049417` |

The engaged route was about 2.05× faster cold and 4.25× faster steady.
Fallback variation stayed inside the declared 5% bound; the candidate and
control execute the same sequential implementation there.

Maximum candidate process-tree RSS was `2,452,717,568` bytes (about
2.28 GiB), below the 6 GiB ceiling. Parallel RSS ratios exceeded 5×, while
paired absolute deltas remained below the 2 GiB allowance; therefore the
standing conjunctive ratio-plus-delta harm rule passed. This is still material
memory telemetry and must remain disclosed.

## Artifacts

- [`b3_parallel_ensemble_v2_launch_20260723.json`](b3_parallel_ensemble_v2_launch_20260723.json),
  SHA-256 `707a4fc3d7283023721ed61417b20e52254d4a7b417e35696e6fe052fbf040a3`;
- [`b3_parallel_ensemble_v2_raw_20260723.json`](b3_parallel_ensemble_v2_raw_20260723.json),
  SHA-256 `1d48276bcde51e9fedd778d35ba521a954ac40f6e927626442c627e0a52b7be1`;
- [`b3_parallel_ensemble_v2_result_20260723.json`](b3_parallel_ensemble_v2_result_20260723.json),
  SHA-256 `f2e34bcb695f28ceea8309177a86e239ae170f53b8c66da7b9d29b55006f7c9c`;
  and
- [`run_b3_parallel_ensemble_v2.py`](run_b3_parallel_ensemble_v2.py),
  SHA-256 `f90bc7d5d0e4d3aaa1429d65c91fc0e67e06d3d91e79a02607b55136c1d88495`.

## Decision

Productize deterministic activation for public eight-member ensemble-v3 fits.
Keep the sequential path below the threshold, preserve a fixed total CPU
budget, persist the resolved route in fitted metadata, and retain a rollback
flag. This result is behavior-exact engineering evidence, not a quality,
default-quality, rival, holdout, or release claim.
