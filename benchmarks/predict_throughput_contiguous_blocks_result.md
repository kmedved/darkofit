# Contiguous-layout prediction-throughput result

_Run 2026-07-17 from clean DarkoFit `4caa8f2` and clean ChimeraBoost
0.15.0 `851ab7f`, under the frozen
[`predict_throughput_contiguous_blocks_protocol.md`](predict_throughput_contiguous_blocks_protocol.md)._

## Decision

Keep the contiguous-layout optimization, but do not claim that P2's formal
matched-prediction target is closed.

The mechanism passed exactness, isolated performance, public ratio, and memory
checks. The campaign failed only the preregistered paired-ratio stability gate
in four of eight cases. The artifact is final and will not be rerun or
reinterpreted.

## Results

| Input | Rows | Public ratio | Stable | Binning ratio | Packed-core ratio |
|---|---:|---:|:---:|---:|---:|
| Basketball numeric | 8,192 | 0.855x | No | 0.987x | 0.917x |
| Basketball numeric | 65,536 | 0.897x | Yes | 0.850x | 0.843x |
| Basketball numeric | 524,288 | 0.793x | Yes | 0.792x | 0.908x |
| Basketball numeric | 2,000,000 | 0.796x | No | 0.833x | 0.821x |
| Synthetic mixed | 8,192 | 0.712x | No | 0.790x | 0.769x |
| Synthetic mixed | 65,536 | 0.935x | No | 1.175x | 1.042x |
| Synthetic mixed | 524,288 | 1.015x | Yes | 1.326x | 0.938x |
| Synthetic mixed | 2,000,000 | 0.909x | Yes | 1.261x | 0.901x |

All ratios are DarkoFit / ChimeraBoost medians paired in three reciprocal
fresh-worker blocks. Every public median is below the `1.30x` target and the
worst is `1.015x`; three of the four stable cases also meet the `<=1.0x`
stretch target.

The failed public stability values were:

- numeric 8k: `0.218`;
- numeric 2M: `0.110`;
- mixed 8k: `0.258`; and
- mixed 65k: `0.104`.

The other four public cases were stable. Exact predictions and each library's
behavior fingerprint were stable. Paired peak RSS was stable at `0.969x`.

## Interpretation

The copy removal is a clear mechanism success. On the same 524k-row matrices,
the isolated binner ran at `0.586x` its old time for numeric input and `0.547x`
for mixed input with identical bin bytes. In the full matched campaign,
numeric preprocessing/binning is now at or faster than ChimeraBoost at every
size.

The mixed 65k–2M component remains `1.18–1.33x` because the phase includes
public input validation and categorical mapping/encoding, not just the now
faster bin kernel. The packed forest remains generally faster and should not
be replaced.

The certification problem is measurement duration: several public cases are
short enough that three blocks still produce unstable ratios, even when the
median advantage is large. The next protocol must accumulate each timed case
to a seconds-scale workload before computing paired ratios. That is a new
preregistered protocol, not a rerun. In parallel, the mixed validation phase
is the remaining implementation target.

## Evidence

- Raw artifact:
  [`predict_throughput_contiguous_blocks.json`](predict_throughput_contiguous_blocks.json),
  SHA-256 `430341e101194b8bc3fbb98014b568d4bd460517686acea6188d88958619ad61`.
- Protocol SHA-256:
  `70364c2114fe8ed773efa88d4df31e4562d461c8d1cbcc1a940238401592463a`.
- Successor runner SHA-256:
  `32c910eae69961f928235c4ad93e42ada7a8f1ded57f7f7a34e8630314c5b376`.
- The unchanged predecessor worker runner is hash-bound inside the artifact.
- No default, quality coordinate, CTR23 task, or lockbox task was used.
