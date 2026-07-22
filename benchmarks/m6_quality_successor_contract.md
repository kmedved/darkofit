# M6 quality successor contract

_Frozen before historical backtest execution on 2026-07-21._

## Identity, scope, and authority

Contract identity: `m6-quality-successor-v1`.

This is the quality-only successor authorized by `NEXT_STEPS.md` §4.8 and §6.
It restores a cheap general development rung for the next quality mechanism
(T7b) without reopening terminal M6 v3. It may rank or kill a private quality
mechanism after its backtest is complete. It can never authorize shipping, a
default, public API changes, fresh confirmation, M2/M4, or lockbox access.

The first Git commit containing this file, `m6_quality_successor.py`, its
backtest runner, and their tests is the frozen implementation checkpoint.
Thresholds, grid, historical subset, and artifact hashes below may not change
after that checkpoint. Backtest completion is bound later by a create-only
result hash; that binding may change only eligibility state, not the contract.

## General development slice

The successor uses the ten generic deterministic builders already exercised by
M6 v3: four regression, three binary, and three multiclass datasets, including
numeric and categorical paths. It deliberately drops the noisy 2,500-row
draft cells and freezes only the 10,000-row `medium` cells. Pinned naturally
smaller real datasets retain their natural row count.

Each dataset runs seeds 0, 1, and 2 with unweighted and deterministic
stress-weighted fits: 60 paired cells and 120 fresh-worker rows. Both arms use
public defaults, four threads, three timing repeats, and alternating arm order.
Execution is through `bench_compare_revisions.py` under
`paired-evidence-v1`, which binds the clean source paths, exact data/split/
weight hashes, fitted/runtime thread state, implementation paths, model
metadata, probabilities, and prediction fingerprints before publishing the
create-only CSV.

The M6 analyzer then requires the exact grid and matched primary metric in each
pair. It computes candidate/control primary-loss ratios per cell, the equal-cell
geometric mean, strict win count, worst cell, per-dataset geometric means, and
leave-one-dataset-out sensitivity. Sports data, release anchors, TabArena, and
fresh/lockbox data are not part of this slice.

## Development decision rule

A candidate advances from this cheap rung only if all three quality gates pass:

- equal-cell geometric-mean primary-loss ratio at most `0.98`;
- strict wins in at least `60%` of cells; and
- no cell above `1.02`.

The two-percent aggregate band is the minimum quality movement judged worth a
new mechanism's maintenance and later evidence cost at this cheap stage. The
60% breadth rule prevents a narrow subgroup from carrying the average. The
two-percent cell harm bound prevents that breadth rule from hiding a material
local regression. These are development-ranking rules, not shipping gates;
uncertainty, power, sports evidence, and Tier-D requirements remain downstream.

Every analysis records a stable mechanism id and a positive one-based
inspection index. The index is consumed by any material look at a full result,
including failed launches, and must be repeated in `TESTING_LOG.md`. Reuse,
skips, or resets invalidate the mechanism's M6 audit. Result and adjacent
manifest are create-only and include contract, CSV, analyzer, source, and tree
hashes. Repeated inspection spends this panel for that mechanism.

## Predeclared backtest v2 subset

The backtest replays the analyzer on immutable stored ratios, not on current
hardware timing. This tests the ranking rule itself and makes every replay
executable on the current 14-thread machine.

| Mechanism | Known verdict | Frozen artifact | Replay view |
| --- | --- | --- | --- |
| Combined B1+B2 ensemble-v3 | Advance | `m3b_ensemble_v3_r3_vs_single_readout_20260721.json`, SHA-256 `99d693063c46a0708eb45a704af0b46611fa8ed89dbe4d6469b47c7cd4a27c64` | The 13 stored `b1_b2_combined` per-case primary ratios. |
| 3% linear-leaf selector | Kill | `fresh_selector_confirmation.json`, SHA-256 `4dc158ec4fd11cf29a5822dc2a09aa76715ce9446773673fa9a2828da1b71a7d` | The 14 stored smooth/process selector-vs-default lineage ratios. |

The positive is the quality mechanism subsequently reopened by the owner after
its sole archive-size gate was retracted; the negative is the prospectively
closed selector. The backtest does not reinterpret either campaign or treat
their cases as independent new evidence. It asks only whether the frozen M6
rule returns `advance` and `kill`, respectively.

Backtest passes only if both artifacts match their hashes and schemas, both
replays agree, a clean committed harness executes the frozen runner, and the
create-only result binds all inputs and code. Before that result is hash-bound,
M6 remains non-ranking. A disagreement is terminal for this contract identity;
no threshold relaxation or same-identity rerun is allowed.

## Exact future execution command

After backtest completion, a mechanism first produces its strict raw CSV:

```bash
python benchmarks/bench_compare_revisions.py \
  --policy-suite standing-slice \
  --control /clean/pinned/control \
  --candidate /clean/pinned/candidate \
  --datasets all --sizes medium --seeds 3 --repeat 3 --threads 4 \
  --weight-modes none stress \
  --models control_default candidate_default \
  --evidence-contract paired-evidence-v1 \
  --csv /create-only/path/mechanism.csv
```

Then `m6_quality_successor.py` validates and analyzes it with the declared
mechanism id and inspection index. No result from this contract may be called a
canary, confirmation, certificate, or release result.
