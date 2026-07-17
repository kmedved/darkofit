# Matched prediction-throughput result

_Run 2026-07-17 from clean DarkoFit `27ff54e` and clean ChimeraBoost 0.15.0
`851ab7f`. The create-only
[`predict_throughput.json`](predict_throughput.json) artifact has SHA-256
`cf25311f3364f4f939cf0324eb97f08641e92b68a0e82eac54c390d0b64e71c9`
and binds the frozen
[`predict_throughput_protocol.md`](predict_throughput_protocol.md)._

## Decision

Start P2 with preprocessing/binning. Do not optimize or replace the packed
forest core.

DarkoFit's median public prediction time was already below ChimeraBoost's in
every case and below the `1.30x` ceiling target. The full formal claim did not
pass only because three small-batch paired ratios exceeded the preregistered
stability threshold.

| Input | Rows | Public ratio | Stable | Binning ratio | Packed-core ratio |
| --- | ---: | ---: | :---: | ---: | ---: |
| Basketball numeric | 8,192 | 0.926x | No | 1.381x | 0.900x |
| Basketball numeric | 65,536 | 0.918x | No | 1.346x | 1.124x |
| Basketball numeric | 524,288 | 0.884x | Yes | 0.984x | 0.779x |
| Basketball numeric | 2,000,000 | 0.840x | Yes | 1.331x | 0.918x |
| Synthetic mixed | 8,192 | 1.025x | No | 1.359x | 1.074x |
| Synthetic mixed | 65,536 | 0.926x | Yes | 1.322x | 0.878x |
| Synthetic mixed | 524,288 | 1.020x | Yes | 1.375x | 0.840x |
| Synthetic mixed | 2,000,000 | 0.964x | Yes | 1.325x | 0.884x |

Ratios are DarkoFit / ChimeraBoost medians paired within three reciprocal
fresh-worker blocks. Stability requires paired-ratio IQR / median at most
0.10. Peak RSS was 0.969x and stable.

## Interpretation

The earlier 1.83x matched-lane prediction gap does not survive the current
large-batch protocol after the packed prediction and validation work. At
512k–2M rows, DarkoFit is 3.6–16.0% faster publicly while using slightly less
peak memory.

The component breakdown still exposes a durable opportunity: DarkoFit's
adaptive-`uint8` packed core is generally faster than ChimeraBoost's
fixed-`uint16` core, but preprocessing often gives back roughly one third.
The next behavior-preserving mechanism should therefore consume the binner's
row-major output with less conversion/copy overhead. It must retain the
current packed forest and rerun this protocol with a newly bound artifact.

No model default, promotion evidence, CTR23 coordinate, or lockbox task was
used.
