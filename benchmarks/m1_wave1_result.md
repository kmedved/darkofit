# Wave 1 M1 large-n result

_Executed 2026-07-20. Tier-E matched-capacity product-path characterization._

## Outcome

DarkoFit is the fastest current arm on this 14-logical-CPU machine. Against
pinned quantized ChimeraBoost 0.18, its equal-size geometric-mean fit ratio is
`0.844722`: 15.53% lower wall time, or a `1.1838x` speedup. Against the
current ChimeraBoost float lane, the ratio is `0.762876`: 23.71% lower wall
time, or a `1.3108x` speedup.

The causal ChimeraBoost quantized/float comparison improved equal-size
geometric-mean fit time by 9.64% (`0.903595x`). That misses the frozen
material-donor threshold of `0.90` by 0.003595 ratio points. The result is
stable and quality-neutral, but the preregistered verdict is therefore
**no material quantization donor signal**.

Every integrity check passed:

- all 36 fresh workers completed without stderr;
- all arms retained exactly 300 trees, depth 6, learning rate 0.1, and the
  fixed 14-thread budget;
- arm-specific quantization, selector, float-histogram, and border-sample
  metadata matched the contract;
- DarkoFit's fused kernel engaged in every worker;
- behavior fingerprints were stable within every arm and size; and
- every paired timing contrast was well inside the declared
  `IQR / median <= 0.10` stability limit.

M1's provisional Q disposition is **do not fund the quantization prototype**.
Q0 found a real local hotspot and passed its projection screen, but Gate G-M
requires both a material M1 opportunity and a credible Q0 hotspot. M1 did not
meet the first condition, and DarkoFit already occupies the faster side of
the current comparison. G-M should close Q unless the owner explicitly
creates a new re-entry condition; M3a does not supply quantization evidence.

## Fit and quality

| Train rows | DarkoFit fit series, s | Chimera quantized, s | Chimera float, s | Darko/quantized | Darko/float | Quantized/float |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 500,000 | 3.3563, 3.3696, 3.4725, 3.5478, 3.4141, 3.4289 | 3.9145, 3.9644, 3.9888, 3.9801, 3.9436, 4.0061 | 4.2931, 4.3706, 4.4188, 4.3911, 4.3615, 4.4283 | 0.861553 | 0.782276 | 0.905519 |
| 1,000,000 | 6.3046, 6.4901, 6.6016, 6.5680, 6.3577, 6.4750 | 7.7890, 7.7698, 7.9249, 7.7516, 7.8800, 7.8635 | 8.6604, 8.8462, 8.5973, 8.6430, 8.5503, 8.6989 | 0.828220 | 0.743957 | 0.901676 |
| Equal-size geometric mean | — | — | — | **0.844722** | **0.762876** | **0.903595** |

The paired-ratio `IQR / median` range was `0.0029–0.0267`, far below 0.10.

| Train rows | DarkoFit RMSE | Chimera quantized RMSE | Chimera float RMSE | Darko/quantized | Quantized/float |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500,000 | 0.509638 | 0.509858 | 0.509646 | 0.999569 | 1.000416 |
| 1,000,000 | 0.507569 | 0.507421 | 0.507139 | 1.000292 | 1.000555 |

Quantized/float RMSE stayed below the frozen `1.002` ceiling at both sizes.
DarkoFit and quantized ChimeraBoost traded a negligible quality edge by size.

## Other cost

DarkoFit used less peak RSS than quantized ChimeraBoost: paired median ratios
were `0.8518` at 500,000 rows and `0.7936` at 1,000,000 rows. Common
pre-predict pickle sizes were 309,666 bytes for DarkoFit and 332,242 bytes for
both ChimeraBoost arms. Native DarkoFit safe-NPZ sizes and all raw RSS,
prediction, phase, serialization, and metadata rows remain in the artifact.

Prediction medians were effectively tied. DarkoFit/quantized ratios were
`1.0021` at 500,000 rows and `1.0472` at 1,000,000 rows; prediction was
report-only and no cross-size prediction claim is made.

## Provenance

- DarkoFit package source:
  `726e5d8e6131c580bce948db833a5007d0692dca`.
- ChimeraBoost package source:
  `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`.
- Harness source: `c39c15e26ea545e19c822505ff0fbc345815aec2`.
- Machine: arm64 macOS 26.5.2, 14 logical CPUs.
- Runtime: Anaconda Python 3.12.7, NumPy 1.26.4, Numba 0.60.0,
  scikit-learn 1.5.1.
- Command:
  `python benchmarks/run_m1_q0_wave1.py --campaign m1 --darkofit-source /private/tmp/darkofit-wave1-source-726e5d8 --chimeraboost-source /Users/konstantinmedvedovsky/code/chimeraboost`.
- Raw artifact:
  `m1_wave1.json`, SHA-256
  `74fd4c9c85948a4c19664a57534e19be3efb0483c78c13767c2521194626eb7a`.
- Frozen protocol SHA-256:
  `7b25851753f83916c8dd542d8dd0f8d569c5b871b9ef38cb8e933f0f46ff2a34`.
- Executed runner/analyzer SHA-256:
  `83690fa0873f017512e9d9c82f42a6be464547832b935786f627debbbb6ab2ab`.

Deterministic reanalysis reproduces the stored analysis exactly.

## Scope and terminal handling

This compares matched capacity while preserving each product's public border
construction; preprocessing is not byte-identical above 200,000 rows. The
thread budget and machine differ from the historical 18-core measurement, and
the optional ChimeraBoost 0.15 arm was not run. The result therefore
characterizes current within-machine arms but cannot attribute movement from
the older result to a particular ChimeraBoost release or mechanism.

M1 is published once and will not be rerun to cross the 0.90 donor threshold.
Its terminal donor verdict is
`no_material_quantization_donor_signal`. No certification, prototype, public
option, default change, or lockbox access is authorized.
