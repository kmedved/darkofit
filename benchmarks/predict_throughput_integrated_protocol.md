# Seconds-integrated matched prediction throughput

## Question

When each timed interval is long enough to dominate scheduler and timer noise,
does DarkoFit's public matched prediction lane meet the `<=1.30x`
ChimeraBoost 0.15 target with stable paired ratios?

This is a new certification protocol motivated by, but not a rerun of, the
final contiguous-layout campaign. That artifact remains failed and immutable.
No model mechanism changes in this campaign.

## Frozen boundary

- DarkoFit source before protocol implementation: clean `main` at `e7f2e32`.
- ChimeraBoost: clean 0.15.0 at `851ab7f`.
- Models, training sets, parameters, numeric/mixed input construction, batch
  sizes, 18-thread execution, and reciprocal three-block arm order are
  unchanged from `predict_throughput_protocol.md`.
- Model fitting, input construction/hashing, forest-cache construction, and
  one complete warm public prediction are outside timing.
- Only the complete public `predict` boundary is measured. The packed core and
  component diagnosis are already settled by the two predecessor artifacts.

## Seconds-integrated workload

Each timed interval performs a fixed number of complete public calls on the
same immutable input:

| Rows | Calls per interval |
|---:|---:|
| 8,192 | 256 |
| 65,536 | 32 |
| 524,288 | 4 |
| 2,000,000 | 2 |

The timer surrounds the entire call loop. The paired statistic is elapsed
seconds divided by the fixed call count. Every interval must last at least
`0.75` seconds; otherwise the campaign fails rather than silently returning to
a millisecond-scale gate.

The last prediction must be array-identical to the pre-timing warm prediction.
Input and prediction hashes, fitted metadata, behavior fingerprints, peak RSS,
thread environment, and raw elapsed times are retained.

## Gates

The matched lane is certified only if all conditions hold:

1. every measured interval is at least `0.75` seconds;
2. warm and integrated-loop predictions are array-identical;
3. each library has one behavior fingerprint across all three blocks;
4. every case has
   `IQR(Darko/Chimera per-call ratio) / median ratio <= 0.10`;
5. every median Darko/Chimera per-call ratio is `<=1.30`; and
6. paired peak-RSS median ratio is `<=1.05`.

Ratios `<=1.00` are counted as the stretch result. Passing closes P2's
matched-public target; failing leaves it open without authorizing a post-hoc
rerun, threshold change, or packed-core replacement.
