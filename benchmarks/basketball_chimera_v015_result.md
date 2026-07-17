# Basketball versus ChimeraBoost 0.15.0: tree-fit parity achieved

## Decision

Stop low-level optimization of the default constant-leaf tree-fit path as the
explanation for the remaining product-default basketball gap. The frozen
recommendation is `stop_low_level_default_tree_optimization`.

At the same 1,000-tree budget and common core parameters, DarkoFit and
ChimeraBoost produced byte-identical predictions on every creator fold and
player guardrail. DarkoFit's median fit and wall time were about 2.5% lower.
The current tree-fit engines are at parity on this workload.

The current product defaults answer a different question. DarkoFit takes
1.312× as long because ChimeraBoost's early-stopping/selection policy retains
64–163 final trees while DarkoFit intentionally trains 1,000. Overall R² is
essentially tied, and DarkoFit has a descriptive lead on the single
genuinely-cold-player holdout. This characterization authorizes no default
change or population-level superiority claim.

## Current product defaults

| Measurement | DarkoFit 0.9.0 | ChimeraBoost 0.15.0 | Darko / Chimera |
|---|---:|---:|---:|
| Mean 10-fold R² | 0.526749518388 | **0.526981194352** | −0.000232 R² |
| Median summed fit | 12.084s | **9.223s** | 1.310 |
| Median steady wall | 12.136s | **9.251s** | 1.312 |
| Diagnostic prediction total | 26.852ms | **9.227ms** | 2.910 |
| Median fresh-worker RSS | **258.1MB** | 265.6MB | 0.972 |

ChimeraBoost won six folds and DarkoFit won four. The mean difference is only
0.000232 R² and both arms were stable across three reciprocal blocks.

The player views show the trade-off hidden by the mean:

| Guardrail | DarkoFit R² | ChimeraBoost R² | Darko − Chimera |
|---|---:|---:|---:|
| Overlap-exposed held-team | 0.531269 | **0.534423** | −0.003154 |
| Seen-player subset | 0.530247 | **0.537525** | −0.007278 |
| Cold-player subset (585 rows) | **0.500434** | 0.490881 | **+0.009552** |

ChimeraBoost selected linear leaves in all ten creator folds and cross features
in four. Its final models retained 64–163 trees. DarkoFit retained exactly
1,000 trees in every fit. The speed difference is therefore a product-policy
result, not evidence that ChimeraBoost's underlying tree builder is 31% faster.

The descriptive cold-player lead also reinforces the existing sports policy:
do not make automatic early stopping or validation-selected linear leaves a
DarkoFit default from aggregate R² alone. Any policy candidate must
independently preserve the ordinary folds, held-team view, and cold-player
subset.

## Matched constant-leaf engine

Both libraries used 1,000 trees, learning rate 0.1, depth 6, L2 1, 128 bins,
full rows/features, minimum child weight 1, plain boosting, no early stopping,
and no linear leaves, cross features, or categorical combinations.

| Measurement | DarkoFit | ChimeraBoost | Darko / Chimera |
|---|---:|---:|---:|
| Mean 10-fold R² | 0.514540708182 | 0.514540708182 | exact |
| Median summed fit | **12.204s** | 12.517s | **0.975** |
| Median steady wall | **12.256s** | 12.562s | **0.976** |
| Diagnostic prediction total | 27.147ms | **14.865ms** | 1.826 |
| Median fresh-worker RSS | 257.9MB | **255.8MB** | 1.008 |

Every fold and guardrail prediction hash matched exactly. Both arms retained
exactly 1,000 trees throughout. Held-team R² was 0.516309892 and cold-player
R² was 0.499691017 in both libraries. Timing max/min was 1.012 for DarkoFit
and 1.065 for ChimeraBoost, well inside the frozen 1.20 stability bound.

The matched lane passed every preregistered engine-parity gate: exact
predictions, exact tree counts, fit and wall ratios below 1.10, stable timing,
and RSS below 1.10. Further default-tree training work lacks a measured
basketball hotspot with an opportunity score of at least 2.0.

## What remains

Prediction remains slower: 1.83× in the matched lane and 2.91× under product
defaults, although the absolute ten-fold totals are only 12–18ms apart on
these small test sets. Packed prediction is therefore the next engine-shaped
candidate, but it needs a separate throughput protocol with materially sized
batches; the tiny basketball folds alone cannot justify it.

For basketball product quality/speed, the remaining work is policy and feature
selection—not tree construction. The next candidate should remain isolated
and basketball-first. ChimeraBoost's current policy is not a drop-in answer
because it loses 0.00955 cold-player R².

## Evidence boundary

The campaign used clean DarkoFit commit
`e4d7a51` and clean ChimeraBoost commit
`851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d` (`v0.15.0`), 18 threads,
three reciprocal fresh-worker blocks per lane, and one full-fold warmup outside
every timer. Dataset loading/imports and the held-team fit were outside steady
timing. Behavior fingerprints were identical across repeats. The runner bound
its frozen protocol and a normalized content manifest of every tracked
DarkoFit file, rechecked both repositories between workers, and published the
artifact atomically without overwrite. No CTR23 data was used.

## Artifacts

- `basketball_chimera_v015.json`: canonical predictions, fitted metadata,
  reciprocal timings, resource values, and the formal decision.
- `basketball_chimera_v015_protocol.md`: frozen pre-run questions and gates.
- `run_basketball_chimera_v015.py`: source-attested two-lane runner.
- `test_basketball_chimera_v015.py`: estimator, exactness, timing, source, and
  authorization-boundary tests.
