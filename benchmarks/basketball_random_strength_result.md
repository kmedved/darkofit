# Basketball random-strength result

_Run 2026-07-17 from clean `main` at `a7519f2`, under the frozen
[`basketball_random_strength_protocol.md`](basketball_random_strength_protocol.md)._

## Decision

Keep `random_strength` as an opt-in public parameter. Do not deprecate it and
do not change the default.

`random_strength=0.5` passed the preregistered basketball quality gate and
advances to a future fresh sports confirmation suite. `random_strength=1.0`
did not pass.

## Quality

| Arm | Mean R² | Δ mean R² | Fold wins | Min LOFO Δ | Held-team Δ | Cold-player Δ | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Default (`0.0`) | 0.526750 | — | — | — | — | — | Control |
| `0.5` | 0.528873 | +0.002124 | 7/10 | +0.001173 | +0.006087 | +0.007300 | **Pass** |
| `1.0` | 0.527373 | +0.000624 | 6/10 | −0.000378 | +0.003000 | +0.004429 | Fail |

The passing arm also improved the diagnostic seen-player subset by `+0.005840`
R². Its gain is broad enough to survive removing any one creator fold.

## Cost and limits

The fresh workers recorded 10-fold wall times of 10.61 seconds for the
control, 15.88 seconds for `0.5`, and 16.00 seconds for `1.0`. Timing was
preregistered as descriptive rather than a gate, was not repeated, and
therefore supports no stable speed ratio. It does show that split-score noise
is not a free default.

This was one dataset and two declared nonzero values. It justifies preserving
the capability, not promoting `0.5`, widening the tuning range, or making a
general sports-quality claim. Those require the fresh multi-season sports
suite in Track S4.

## Evidence

- Raw artifact:
  [`basketball_random_strength.json`](basketball_random_strength.json),
  SHA-256 `e8f98c47191c19fa1a20d5133ed2c071c6c36511e7d4b6380ec0cf998a94e906`.
- Protocol SHA-256:
  `9ba730293781e480a4bd07f3504fc3717e24552970342538c419114aa36fc9cd`.
- Runner SHA-256:
  `dc8f0faa02f41c77c97ad56e2aa61510c3d2cc2e57a90c86a1e45ae8b944b55f`.
- The artifact records the clean source revision, environment, data and fold
  fingerprints, fitted metadata, exact predictions, and behavior hashes.
