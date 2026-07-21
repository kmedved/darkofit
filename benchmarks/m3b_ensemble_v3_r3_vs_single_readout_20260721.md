# M3b r3 arm-vs-single development readout — 2026-07-21

This is a dated, hash-bound post-hoc readout from the immutable r3 quality artifact. It does not amend the frozen M3b result or gates.

| Arm | Pooled | Sports cold-player | General | Worst | Wins |
|---|---:|---:|---:|---:|---:|
| control (existing group8) | 0.985581 | 0.983233 | 0.990886 | 1.049062 | 8/13 |
| b1_sampling | 0.973502 | 0.967966 | 0.986072 | 1.013693 | 12/13 |
| b2_member_policy | 0.982026 | 0.980822 | 0.984741 | 1.029100 | 9/13 |
| b1_b2_combined | 0.965513 | 0.961077 | 0.975569 | 0.991888 | 13/13 |

The combined arm beat the matched single on all 13 development cases. Its quality payload is therefore promising development evidence, but it did not survive the prospectively frozen campaign: median archive size was 5.534767x single against the unchanged <= 4.0x gate.

The nine sports primaries are player-disjoint cold-player rows within the frozen held-team view. The four general cases use the seeded 75/25 development split.

Frozen disposition: `close_b1_b2_preserve_existing_opt_in`. No serializer, public/default surface, fresh confirmation, or lockbox access is authorized by this readout.

Bound artifacts:

- quality: `5fec218cbc0ec97ef4b3fec10f65a89131a377cf026dbb80da809d6396ead6c3`
- frozen result: `3e6d0750e772c156b6c4daed948eb6baa640564ce87fe1ffee7414b3fe03c8bc`
