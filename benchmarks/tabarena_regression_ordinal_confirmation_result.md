# TabArena safe-ordinal mechanism confirmation result

_Executed and analyzed on 2026-07-13 from clean DarkoFit commit `96a03d2`.
The source-frozen design is in
[`tabarena_regression_ordinal_confirmation_protocol.md`](tabarena_regression_ordinal_confirmation_protocol.md)._

## Decision

**Do not advance the safe-ordinal policy.** The primary causal `O / B`
contrast passed every frozen accuracy, uncertainty, validation, completeness,
training-time, and memory gate, but its equal-dataset inference-time ratio was
**1.265169**, above the predeclared **1.25** ceiling. The protocol required
both the primary `O / B` and deployment `O / P` gates to pass in full, so the
formal decision is `do_not_advance` even though `O / P` passed.

The accuracy signal is real and strong. Relative to the identical fixed model
policy on the native categorical representation, source-declared safe ordinal
encoding reduced equal-dataset test RMSE by **17.291%** and validation RMSE by
**19.153%**, won all 13 repeat blocks, and won all 33 coordinates. It remains
useful mechanism evidence and a candidate for targeted inference optimization;
it is not an accepted product policy under the frozen resource gate.

## Frozen contrasts

Ratios below one favor the numerator. `O` is the fixed
1,000-round/L2=3/128-bin/LR=0.1 policy with the source-declared safe ordinal
representation; `B` is the same policy with the native categorical path; `P`
is the actual product-default native path with an empty manual model config.

| Contrast | Role | Test RMSE | Upper 95% | Validation RMSE | Train time | Infer time | Peak RSS | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `O / B` | Primary causal | **0.827085 (-17.291%)** | **0.840093** | **0.808471 (-19.153%)** | 1.218356 (+21.8%) | **1.265169 (+26.5%)** | 0.998969 (-0.1%) | **fail: inference** |
| `O / P` | Deployment | **0.815761 (-18.424%)** | **0.830700** | **0.800106 (-19.989%)** | 0.920249 (-8.0%) | 1.022200 (+2.2%) | 1.000195 (+0.0%) | pass |
| `B / P` | Attribution only | 0.986309 (-1.369%) | 0.990251 | 0.989652 (-1.035%) | 0.755320 (-24.5%) | 0.807956 (-19.2%) | 1.001228 (+0.1%) | report only |

The `B / P` contrast explains why deployment inference remained acceptable:
the fixed-base native arm was 19.2% faster than the product default, while the
ordinal representation was 26.5% slower than that faster fixed-base arm.
`B / P` was predeclared as attribution-only and cannot rescue or reject a
policy.

## Primary causal result

| Dataset | Coordinates | Repeats | Test RMSE change | Validation RMSE change | Coordinate wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Airfoil self noise | 27 | 10 | **-12.172%** | **-13.076%** | 27/27 |
| Diamonds | 6 | 3 | **-22.112%** | **-24.805%** | 6/6 |

- Hierarchical test-RMSE ratio interval: **[0.810202, 0.842593]**.
- One-sided 95% upper ratio: **0.840093**.
- Repeat-block wins/losses/ties: **13/0/0**; exact one-sided sign-test
  `p = 0.00012207`.
- Coordinate wins/losses/ties: **33/0/0**.
- Worst coordinate was Airfoil `r9f0`, still a win at ratio **0.917822**.
- Training time and peak RSS passed their 1.50x and 1.25x ceilings. Inference
  time alone failed, by an absolute ratio margin of **0.015169**.

## Evidence boundary

This is **mechanism replication, not independent dataset generalization**.
The campaign excluded the six Airfoil and Diamonds coordinates used to select
the ordinal mechanism and evaluated the remaining 33 mechanism-unused
coordinates. However, the earlier cap-horizon campaign had already evaluated
those coordinates, so they are not globally unseen.

The transforms are also dataset-specific. Airfoil restores physical attack
angle values, while Diamonds uses the published semantic orders for cut,
color, and clarity. The transforms were source-frozen, target-free, and
fail-closed; all 264 ordinal children recorded the declared transform with
zero unknown validation values. This result does not justify inferring order
for arbitrary categorical columns or changing generic preprocessing defaults.

Any future reconsideration should first remove or materially reduce the
measured ordinal inference overhead, then use a newly frozen gate. A broader
claim would additionally require genuinely unseen datasets with externally
declared ordinal semantics.

## Integrity

- **99/99** outer jobs and **792/792** child fits completed.
- All **99** predeclared comparisons were complete (33 primary, 33 deployment,
  and 33 attribution); the exact source-frozen job order and balanced position
  schedule matched.
- Stop reasons were 498 early stops and 294 iteration-limit stops. There were
  zero failures, imputations, missing rows, duplicate rows, deadline hits, or
  `time_limit` stops.
- Every raw-result byte hash and size matched the completion attestation. The
  manifest, normalized safe payload, warmup record, dependency/runtime state,
  hardware identity, Git trees, ordered schemas, representations, and bound
  source files also matched.
- The standalone analyzer treated all raw result files as opaque byte
  artifacts and never unpickled them.

## Retained evidence

The repository retains the analyzer's machine-readable
[`summary`](tabarena_regression_ordinal_confirmation_summary.json),
[`paired splits`](tabarena_regression_ordinal_confirmation_paired_splits.csv),
[`per-repeat estimates`](tabarena_regression_ordinal_confirmation_per_repeat.csv),
[`paired child metadata`](tabarena_regression_ordinal_confirmation_paired_children.csv),
[`run manifest`](tabarena_regression_ordinal_confirmation_run_manifest.json),
[`completion attestation`](tabarena_regression_ordinal_confirmation_completion_attestation.json),
and [`warmup record`](tabarena_regression_ordinal_confirmation_warmup_history.json).
The 2.9 MB normalized analysis payload and 21.0 MB of raw result pickles remain
in the hash-addressed local campaign directory; the committed attestation
binds their hashes and sizes. No pickle or model artifact is committed.

## Provenance

- DarkoFit commit: `96a03d2568c97799c367802ce7e6d0c85a409c5d`;
  Git tree: `2628d297be7f7996fe78bee71ba927890a3e48da`.
- Python: 3.12.13; AutoGluon: 1.5.1b20260712; TabArena commit:
  `4cd1d2526874962daae048a6f2dcf34aa272f3fa`.
- Run manifest SHA-256:
  `67161636305ecb0d1e782ae19e9ffebdc1cecca66d03a7a79460120dd7353191`.
- Completion attestation SHA-256:
  `922e1235afdbfa31b4cfac8097c8fa7afba9ef7ab11d2af8b097106575bf3010`.
- Safe analysis payload SHA-256:
  `612a02ea0bdb5871604fc4fe7498556f5e5e7ffcbdc52bb775493c7cfde4f8e9`.
- Analyzer summary SHA-256:
  `494bd0a1eeb5a604d3c83077d14481cf3d45c581c0a1c08d2a2fcca7d7a41d0e`.
- Paired-split table SHA-256:
  `2fa48579c4c2273b1bb1e1ef2590578edff83e3f26da8fe09260b9d490c4c27b`.
- Paired-child table SHA-256:
  `bd879cbb8b06ea1c5cff7c5e0165361eacc829ad907b1a442b3c48dfe13491ef`.
- Frozen protocol semantic digest:
  `5a077d07ddd96fa86134ad0f4b83aaf3d36534c351cd846ea4c6eef3c4e29222`.

The analyzer revalidated the complete attested campaign immediately before
and after atomically publishing the decision artifacts.
