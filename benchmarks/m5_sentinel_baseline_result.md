# M5 diversity-sentinel baseline

Run 2026-07-20 from clean harness commit `682dddf`, against the exact
post-H1 control source `726e5d8`.

## Result

M5 v1 is established and frozen as a non-ranking drift sentinel.

All 38 fresh-worker rows completed across 19 paired cells and nine domains.
The control and candidate package trees were identical at
`e1fe956f32df0440e321805511ae2d96e383735c`; every paired behavior
fingerprint matched, every save/load prediction was exact, and worker stderr
was empty. All task-normalized losses were within the predeclared `1.10`
range.

The earned known-floor checks passed:

| Canary | Mean excess Brier | Worst seed | Limits |
| --- | ---: | ---: | --- |
| SynthGen df1/647 binary | 0.00314483 | 0.00343480 | mean <=0.005; worst <=0.01 |
| SynthGen df1/077 multiclass | 0.00009772 | 0.00013069 | mean <=0.005; worst <=0.01 |

Raw artifact:
[`m5_sentinel_baseline.json`](m5_sentinel_baseline.json), SHA-256
`0971e06d4ed307d352d75e1e6400b849c0001b5e11f40243173d7080b6c5859d`.

## Resource baseline

Candidate/control median paired ratios were `0.998818` for fit,
`1.001768` for prediction, and `0.995723` for peak RSS. These ratios validate
the null baseline and establish same-machine reference telemetry. They are
not portable performance ranges, an acceptance score, or evidence that a
future candidate improves quality.

Future M5 checks must bind to this artifact. A crash, invalid output,
serialization mismatch, or canary-floor failure is hard failure. Other
fingerprint or quality drift blocks advancement for investigation but does
not rank the mechanism.
