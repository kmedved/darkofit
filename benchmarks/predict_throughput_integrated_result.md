# Seconds-integrated prediction-throughput result

_Run 2026-07-17 from clean DarkoFit `c66279b` and clean ChimeraBoost
0.15.0 `851ab7f`, under the frozen
[`predict_throughput_integrated_protocol.md`](predict_throughput_integrated_protocol.md)._

## Decision

Close P2 development with the contiguous-layout optimization retained, but
without the formal `P2 target certified` claim.

DarkoFit's public median was faster than ChimeraBoost in all eight cases, six
paired-ratio series were stable, exactness passed, behavior was deterministic,
and RSS passed. The all-case protocol nevertheless failed two stability gates
and the minimum-interval gate. This artifact is final. A third protocol or
post-hoc rerun would be gate-shopping.

## Results

| Input | Rows | Darko/Chimera public ratio | IQR/median | Stable |
|---|---:|---:|---:|:---:|
| Basketball numeric | 8,192 | 0.805x | 0.10556 | No |
| Basketball numeric | 65,536 | 0.958x | 0.03263 | Yes |
| Basketball numeric | 524,288 | 0.918x | 0.10032 | No |
| Basketball numeric | 2,000,000 | 0.869x | 0.05872 | Yes |
| Synthetic mixed | 8,192 | 0.885x | 0.03529 | Yes |
| Synthetic mixed | 65,536 | 0.947x | 0.04656 | Yes |
| Synthetic mixed | 524,288 | 0.987x | 0.01459 | Yes |
| Synthetic mixed | 2,000,000 | 0.984x | 0.01259 | Yes |

All eight medians meet both the `<=1.30x` target and the `<=1.0x` stretch
target. The two unstable cases miss the `0.10` ceiling by `0.00032` and
`0.00556`, respectively, but the protocol is conjunctive and has no rounding
exception.

The immutable JSON field
`stretch_public_cases_at_or_below_chimera` has a historically ambiguous name:
its runner counted cases that were both stable and no slower, so its stored
value is 6. The eight stored median ratios establish the separate 8/8
no-slower statement above. Release status tooling reports both counts without
rewriting this frozen artifact.

The canonical DarkoFit numeric-524k interval completed in `0.690` seconds,
below the preregistered `0.75`-second minimum; its other two blocks were
`0.685` and `0.681` seconds. ChimeraBoost's canonical interval for that case
was `0.751` seconds, although another block was `0.735`. Paired peak RSS was
stable at `0.992x`. All warm versus integrated-loop predictions were
array-identical and each library retained one behavior fingerprint across the
three blocks.

## Product interpretation

The engineering conclusion is favorable: after categorical validation and
contiguous-layout work, DarkoFit is no longer behind ChimeraBoost on the
matched public prediction lane represented here. The formal program conclusion
is narrower: the declared all-case stability proof did not pass.

No further packed-core or binning optimization is justified by these data.
Future release tables may report the raw ratios and failed-gate status, but may
not label the P2 target certified. A materially different production workload
may establish its own preregistered service-level benchmark later; it is not a
continuation or repair of this campaign.

## Evidence

- Raw artifact:
  [`predict_throughput_integrated.json`](predict_throughput_integrated.json),
  SHA-256 `5ec81511e3026f5efadd8623228920da8d154a2f99719ca8f4116cd2c5b3653b`.
- Protocol SHA-256:
  `445232e00fe8e236a17a6cde1716a6412ecf1d791c576fcd4e4e19e5f0797243`.
- Runner SHA-256:
  `e52530324a9845fa0df36314b2da13bcdc1fd68cc7c93dcd322e404bfa0591cf`.
- No default, quality coordinate, CTR23 task, or lockbox task was used.
