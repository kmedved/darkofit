# Ensemble-v3 release-candidate characterization

## Scope

Tier-E characterization on spent/frozen evidence and one pinned current-source
performance grid. This is not M2, M4, a shipping certificate, a public API
change, or authority to release v0.11.

## Quality

The historical combined recipe beat single on 13/13 cases. 
Equal-case ratios were 0.966x overall, 
0.961x on the player-disjoint cold-player sports view, 
and 0.976x on the four fixed general cases.

The three-season clustered descriptive interval was 0.959x--0.963x; 
the four-case general descriptive interval was 0.963x--0.988x. 
With only three seasons and four fixed general cases, neither interval is a population-generalization claim.

## Current fit, memory, and archive telemetry

| Metric | Ensemble-v3 / DarkoFit single |
| --- | ---: |
| Fit wall time, equal-case geomean | 6.142x |
| Process-tree peak RSS, equal-case geomean | 1.136x |
| Safe-NPZ bytes, equal-case geomean | 8.125x |

RSS is worker-plus-recursive-child peak during formal fit. Absolute peak-minus-start deltas are retained in the JSON; the ratio is not used as a gate.

## Prediction throughput

| Comparison | Equal-coordinate time ratio | At/below parity | Worst coordinate |
| --- | ---: | ---: | ---: |
| DarkoFit single / ChimeraBoost 0.18 single | 0.485x | 16/16 | 0.742x |
| DarkoFit ensemble-v3 / ChimeraBoost 0.18 single | 3.014x | 0/16 | 4.507x |
| DarkoFit ensemble-v3 / DarkoFit single | 6.208x | 0/16 | 8.367x |

Minimum integrated interval: 0.75s. Short intervals: 9. Raw paired series and IQR/median are retained for every coordinate.

## Interpretation boundary

Eight members are the only evaluated recipe, not an optimized default. These measurements may inform the later owner ship decision, but they do not expose the API, change defaults, establish sports safety outside the spent panel, or certify prediction performance.

## Evidence

- Raw artifact SHA-256: `005c50a89a06e100aa95cb6a776dd7f67026786de6f261470e808a39f9310a9b`.
- Result JSON SHA-256: `5cfd7b40382187aebed43798715017e1e2867744c5c40f66a00e935f6acefeed`.
- Contract SHA-256: `f8f7b780c6dc915926a33262e24545696754221ef310d76c01da6f9df3b00103`.
- DarkoFit source: `c5e66ef7e6bdcf5665b55b81c6b870f42d76237b`.
- ChimeraBoost source: `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`.
- Fresh/lockbox data: none.
