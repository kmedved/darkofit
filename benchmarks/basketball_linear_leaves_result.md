# Basketball linear leaves: retain as opt-in research, do not advance

## Decision

Do not promote per-leaf linear models or the validation selector as a DarkoFit
default. The selector failed four of the five frozen basketball quality gates,
including both the corrected overlap-exposed team holdout and the stricter
cold-player subset. The formal recommendation is `advance_none`.

Retain `linear_leaves=True` only as an explicit, default-off research
mechanism. Its disabled and fallback paths are prediction-exact, the active
model has safe format-v4 persistence and packed prediction, and the feature is
useful for separately designed smooth-data research. This basketball result
does not authorize the 243-coordinate development panel, a public automatic
selector, or any lockbox use.

## Quality

| Arm | Mean 10-fold R² | Fold wins | Overlap-exposed team R² | Cold-player R² |
|---|---:|---:|---:|---:|
| Current default | **0.526750** | — | **0.531269** | **0.500434** |
| Validation-selected linear leaves | 0.526143 | 6 / 10 | 0.518557 | 0.488385 |
| Candidate delta | -0.000607 | — | -0.012712 | -0.012049 |

Only fold breadth passed. The candidate lost the mean-quality gate, and its
leave-one-fold-out mean deltas ranged from -0.002740 to +0.002712 rather than
remaining nonnegative. Four folds regressed; the worst fold lost 0.030476 R².
Both player guardrails also regressed materially.

## Why the selector did not protect basketball

The selector chose linear leaves on all ten external folds and on the team
holdout fit. On its internal validation splits, linear validation RMSE beat
constant validation RMSE by 0.0301–0.1763 (mean 0.1049), yet those choices did
not generalize to the external folds or player guardrails. This is direct
evidence that the small random internal validation split is not a sufficient
guard against the linear model's tail behavior on this noisy sports dataset.

The result therefore rejects both direct default use and this particular
validation-selection policy. It does not show that the implementation failed
to activate: every final candidate fit retained all 1,000 linear-leaf trees,
and the artifact records both complete validation curves and fitted metadata.

## Runtime and resource disposition

The protocol stopped after the clean canonical block because quality failure
was fatal. The three-block reciprocal timing confirmation was intentionally
skipped, so the following single-block observations are directional rather
than formal timing claims:

| Measurement | Current default | Candidate | Candidate / default |
|---|---:|---:|---:|
| Steady wall time | 26.71s | 56.79s | 2.13× |
| Summed fit time | 26.55s | 55.92s | 2.11× |
| Summed prediction time | 0.0226s | 0.0643s | 2.84× |
| Peak RSS | 256.3 MB | 269.5 MB | 1.05× |
| Mean serialized model size | 0.377 MB | 1.621 MB | 4.30× |

The candidate includes two early-stopped selection fits plus one full-data
refit, explaining much of the fit-time ratio. Its mean model-size ratio also
exceeded the frozen 3.0× budget in this canonical block. No additional runs
were warranted after the quality rejection.

## Verification and provenance

The evidence run used clean committed source at
`4bb6c2b7e8024d8e71697765d280aedd61122369`, 18 threads per fit, the creator's
unchanged 10-fold basketball protocol, and the corrected 585-row cold-player
subset. No lockbox data was used. Before the evidence run, the complete suite
passed with 1,405 tests and 23 skips, and both wheel and source distributions
were verified to include `LICENSE` and the Apache-2.0 attribution `NOTICE`.

## Next step

Use basketball as the recurring first fatal gate for future quality or speed
candidates. Do not tune this selector on the now-spent basketball outcomes.
A materially different mechanism needs a new frozen basketball protocol and
must pass mean folds, leave-one-fold-out stability, team holdout, and
cold-player quality before any larger tabular campaign.

## Artifacts

- `basketball_linear_leaves.json`: clean-source predictions, validation curves,
  hashes, fitted metadata, resource observations, and all decision gates.
- `basketball_linear_leaves_protocol.md`: frozen pre-run rules and advance path.
- `run_basketball_linear_leaves.py`: clean-source, quality-fail-fast runner.
- `test_basketball_linear_leaves.py`: split, selector, metadata, gate, warmup,
  and runtime-policy tests.
