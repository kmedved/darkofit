# Q post-dispatch microprototype: SHIP_RULES disposition

Date: 2026-07-24

This note preserves, but supersedes for roadmap sequencing, the generated
`close_q_at_microprototype` / `q1_funded: false` disposition in
[`q_post_dispatch_microprototype_result_20260723.md`](q_post_dispatch_microprototype_result_20260723.md).
That generated disposition correctly applied the benchmark's inherited
`IQR / median <= 0.10` timing-dispersion diagnostic. It is not rewritten.

Under the governing [`../SHIP_RULES.md`](../SHIP_RULES.md), that diagnostic
miss does not outweigh the implementation signal:

- the packed candidate cleared the declared fit-time funding bar
  (`0.827110x` versus `<= 0.90`);
- it was faster in all six paired comparisons and at both measured shapes;
- all integrity, arithmetic-bound, deterministic-seed, fitted-state, and
  prediction checks passed on the measured grid; and
- the only failed check was `0.117662` dispersion at 500k rows, driven by the
  first control timing, while the 1M series passed at `0.021810`.

**Owner disposition:** Q1 is funded as the next post-v0.12 speed-mechanism
slot. This is implementation authorization, not shipping evidence and not a
claim that quantization is behavior-exact in general.

Q1 must remain narrowly scoped. It must provide an explicit packed lane or
opt-in, a byte-identical float fallback outside the supported envelope,
seed-exact repeatability, arithmetic-overflow guards, and broad quality and
resource characterization. Because stochastic gradient quantization can
change split selection even though this microprototype reproduced the control
at both measured shapes, it is a quality-affecting mechanism rather than
behavior-exact engineering. The `1.114178x` prediction-time reading at 1M rows
must also be measured directly and explained before any shipping decision.

No holdout, sports ship-check, TabArena data, or external-comparator result was
consulted for this disposition.
