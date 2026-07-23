# B3 parallel ensemble-members v1 terminal result

Inspection 1 ran once on 2026-07-22/23 under frozen contract
`b3-parallel-ensemble-members-v1-20260723`. The clean published harness was
`5a236e4f37d429fa55c40a6ebc65dc9b2b6d00f5`, the sequential control was
`c4dae58fcf7a8d456533ba2d9b469f039adc453c`, and the exact private candidate
was `5116470e21675f8a869ee7a84145eb2a663ed809`.

## Pre-timing evidence

The private candidate passed its 18 focused process-parallel tests, including
failure injection, categorical and classification paths, deterministic safe
NPZ serialization, and caller thread-mask restoration. The affected frozen
invariant suite passed all 215 tests. Candidate, control, and harness were
clean and published before the create-only launch manifest spent inspection
1. The exclusive-machine audit found no conflicting benchmark process.

The fitted outputs were behavior-exact across the formal `1x14` sequential
control and `7x2` process-parallel candidate: prediction/probability hashes,
member order, seeds, sampled/OOB indices, best iterations, archive bytes, and
recorded thread counts matched. All 24 case/block/arm workers completed and
the parent ambient Numba mask was restored.

## Frozen speed and resource result

Ratios below are candidate/control fit wall time. Lower is better.

| View | Equal-case geomean | Worst case median | Worst LOO | Gate |
| --- | ---: | ---: | ---: | --- |
| Cold executor | `0.684187` | `1.075049` | `0.770631` | **fail** |
| Steady executor | `0.260379` | `0.362653` | `0.286485` | pass |

The cold case medians were:

| Case | Candidate/control |
| --- | ---: |
| Categorical multiclass | `0.497480` |
| Categorical regression | `0.867852` |
| Friedman numeric | `1.075049` |
| Numeric binary | `0.462523` |

Thus the process-parallel route was about 31.6% faster in the equal-case cold
aggregate and faster in three of four cases, but it was 7.5% slower on the
small Friedman case. The frozen contract required every case median to be
`<=1.0`; a large win elsewhere could not hide a workload slowdown.

The steady-executor result was strongly positive in all four cases, with case
medians from `0.198204` to `0.362653` and an equal-case geomean of `0.260379`
(about 3.84x faster). This is evidence that member-level process parallelism
can pay once workers are available, not authority to waive the cold-start
failure or ship this topology.

The hybrid process-tree RSS gate passed. Maximum candidate peak RSS was
`2,399,551,488` bytes (about 2.24 GiB), below the 6 GiB hard ceiling. Candidate
RSS ratios were above the 5x allowance, but the paired absolute deltas stayed
within the 2 GiB allowance, so the contract's conjunctive ratio-plus-delta
harm rule did not fail. Prediction timing is non-gating telemetry in the raw
artifact; predictions themselves were exact.

## Create-only artifacts

- [`b3_parallel_ensemble_v1_invariants_20260723.json`](b3_parallel_ensemble_v1_invariants_20260723.json),
  SHA-256 `9797ebc23bbc790835c2f88428129746c0cbb7744adb158d80f763db7c62e9db`;
- [`b3_parallel_ensemble_v1_inspection1_launch_manifest_20260723.json`](b3_parallel_ensemble_v1_inspection1_launch_manifest_20260723.json),
  SHA-256 `cdf93e46af80c560d7e809f51bb97d053981738b69cc75d9d55ac014f68ee5dd`;
- [`b3_parallel_ensemble_v1_inspection1_raw_20260723.json`](b3_parallel_ensemble_v1_inspection1_raw_20260723.json),
  SHA-256 `7ba73e1d113d8cf412318201268ecc768cfc0102e61ed66696fd473112d344cc`;
- [`b3_parallel_ensemble_v1_inspection1_result_20260723.json`](b3_parallel_ensemble_v1_inspection1_result_20260723.json),
  SHA-256 `9d1e97e23e1bec0ae4449e4c0a9c842bddaf87d45adc2fea6a8e827791d7bb35`;
  and
- [`b3_parallel_ensemble_v1_inspection1_terminal_attestation_20260723.json`](b3_parallel_ensemble_v1_inspection1_terminal_attestation_20260723.json),
  SHA-256 `2b6a43dbf71435c87dab16ba48b77dfb606fb5f343f5ce4f966498a04921025e`.

## Decision

The frozen disposition is `kill`. Inspection 1 is spent and no rerun is
authorized. The exact `7x2` topology and implementation are closed; candidate
`5116470e` remains private and unmerged.

A warm-worker lifecycle or a deterministic activation rule that avoids
parallel startup on short fits would be a distinct future mechanism requiring
a new identity, contract, and owner authority. This result does not authorize
that successor, a candidate merge, a public API/default, a release, fresh
data, sports evidence, TabArena, lockbox access, or a rival-comparison claim.
