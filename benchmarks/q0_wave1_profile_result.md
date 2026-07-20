# Wave 1 Q0 scalar profile result

_Executed 2026-07-20. Tier-E engineering profile; spent descriptive evidence._

## Outcome

The profile is valid and clears its predeclared funding screen. DarkoFit's
current fused histogram-plus-split kernel accounted for a median 52.37% of
end-to-end fit wall time at 500,000 rows and 62.61% at 1,000,000 rows. Under
the frozen conservative `1.30x` eligible-kernel prior, the equal-size
geometric-mean end-to-end ratio is `0.867242`, a projected 13.28% reduction
against the required 10%.

Q is therefore **eligible for the G-M quantization funding decision**. This
does not authorize a prototype, public option, or default change. M1 must
still establish a material current-source quantized-versus-float donor signal,
and G-M must choose the portfolio slot.

All integrity checks passed:

- all 12 fresh workers completed with no stderr;
- all fits retained 40 trees at the exact 14-thread budget;
- production engaged the fused kernel and the reference avoided it;
- the reference separately exercised histogram construction and split search;
- sibling subtraction had zero calls;
- component accounting stayed within the whole-tree timer; and
- predictions, RMSE, and fitted behavior fingerprints were exact across
  production and unfused-reference paths at both sizes.

## Measurements

| Train rows | Production fit series, s | Production median, s | Eligible fused share | Projected ratio at 1.30x | Infinite-speed ratio |
| ---: | --- | ---: | ---: | ---: | ---: |
| 500,000 | 0.745491, 0.681700, 0.701439 | 0.701439 | 0.523739 | 0.879137 | 0.476261 |
| 1,000,000 | 1.129679, 1.052035, 1.065592 | 1.065592 | 0.626129 | 0.855509 | 0.373871 |

The production medians decompose as follows:

| Train rows | Preprocess, s | Grad/Hess, s | Tree build, s | Fused hist+split, s | Leaf values, s | Leaf routing, s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500,000 | 0.218534 | 0.006841 | 0.442177 | 0.363231 | 0.031141 | 0.029552 |
| 1,000,000 | 0.228292 | 0.013268 | 0.792730 | 0.671068 | 0.060815 | 0.032956 |

The forced-unfused diagnostic was behavior-exact and unexpectedly faster on
this 14-logical-CPU machine: reference/production paired median fit ratios
were `0.901011` at 500,000 rows and `0.981264` at 1,000,000 rows, both stable
under the declared `IQR / median <= 0.10` rule. Its separate median
histogram/split times were `0.325009 / 0.014631` seconds at 500,000 rows and
`0.619496 / 0.019686` seconds at 1,000,000 rows.

That result contradicts the protocol text's incidental expectation that the
reference would be slower. The protocol's binding rule is unchanged: the
reference contributes no production share to the quantization projection.
The direction is nevertheless an important current-hardware diagnostic and
should enter mechanism ranking after G-M; a simpler hardware-aware fused-lane
dispatch may compete with arithmetic quantization.

## Provenance and execution

- DarkoFit package source:
  `726e5d8e6131c580bce948db833a5007d0692dca`.
- ChimeraBoost source recorded but not executed in Q0:
  `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`.
- Harness source: `18bc48c7778eed0980efa430ad6fa722310919bb`.
- Machine: arm64 macOS 26.5.2, 14 logical CPUs.
- Runtime: Anaconda Python 3.12.7, NumPy 1.26.4, Numba 0.60.0,
  scikit-learn 1.5.1.
- Command:
  `python benchmarks/run_m1_q0_wave1.py --campaign q0 --darkofit-source /private/tmp/darkofit-wave1-source-726e5d8 --chimeraboost-source /Users/konstantinmedvedovsky/code/chimeraboost`.
- Raw artifact:
  `q0_wave1_profile.json`, SHA-256
  `9111f14ae4d0d89e122f541b53f85c76c6bd5e76f4fa781c69039c1020c04e1c`.
- Frozen protocol SHA-256:
  `7b25851753f83916c8dd542d8dd0f8d569c5b871b9ef38cb8e933f0f46ff2a34`.
- Executed runner/analyzer SHA-256:
  `793f764c7287a3007b20d83dc452917fd1ed56339195d508db71a5544ab8f179`.

All workers and the artifact write completed. The parent then exited nonzero
while printing the already-computed disposition because a presentation-only
`dict.get` default eagerly accessed M1's absent `g_m_input` key. The artifact
was not replaced or rerun. Deterministic reanalysis reproduced its stored
analysis exactly. Commit `bb40018` fixes only that post-write print path and
adds a regression test; it does not alter Q0 analysis or this evidence.

## Limitations and terminal disposition

The 40-tree Q0 fits measure attribution, not 300-tree M1 throughput. The
`1.30x` kernel prior is a preregistered screening assumption, not a measured
DarkoFit prototype. Python timing shims add small call-boundary overhead,
although component accounting remained valid. The results apply to this
numeric scalar CatBoost-mode lane and current hardware; they do not establish
classification, categorical, leafwise, or broad product gains.

Terminal Q0-profile disposition:
`eligible_for_g_m_quantization_funding_decision`. No implementation is
authorized before M1 and G-M.
