# Minimal CTR23 regression confirmation — reviewed publication

The campaign stops without establishing confirmation. The reviewed publication
status is **`confirmation_not_established_protocol_deviation_stop`**.

| Measure | Point | Registered interval statistic | Gate |
| --- | ---: | ---: | --- |
| A10 / ChimeraBoost | 0.942029 (-5.797%) | one-sided 95% upper 1.012091 | **FAIL** |
| A10 / product default | 0.866066 (-13.393%) | simultaneous max-regret 95% upper 1.046121 | **FAIL** |
| A10 / CatBoost, r0f0 only | 1.055904 (+5.590%) | descriptive 95% interval [0.951335, 1.186517] | descriptive only |

The point estimate favors A10 over ChimeraBoost, but the registered uncertainty
bound crosses 1. A10 won 4 of 9 tasks and 11 of 27 splits, so superiority was
not confirmed. The product-default guardrail also failed. CatBoost is
single-fold descriptive context and does not support a parity claim.

An independent audit found one operational protocol deviation: the frozen
protocol required allowed swap-in to be measured and retained, but the runner
persisted only swap-out telemetry. Zero swap-out, the complete grid, resource
limits, and all quality calculations are verified; however, full protocol
compliance is false. Timing and memory-performance evidence was inadmissible
from the outset.

For integrity authentication, the frozen analyzer also read four operational
JSON artifacts beyond the protocol's literal input allowlist and read each raw
result as opaque bytes to recompute its SHA-256. It never imported `pickle` or
deserialized result contents. This was consistent with hash authentication but
violated the literal allowlist and instruction not to open/read raw result
pickles. The reviewed summary therefore records two protocol deviations and
keeps full compliance false.

The immutable frozen-analyzer outputs are preserved in
[`tabarena_ctr23_minimal_confirmation_analyzer_result.md`](tabarena_ctr23_minimal_confirmation_analyzer_result.md)
and
[`tabarena_ctr23_minimal_confirmation_analyzer_summary.json`](tabarena_ctr23_minimal_confirmation_analyzer_summary.json).
The corrected machine-readable disposition is in
[`tabarena_ctr23_minimal_confirmation_summary.json`](tabarena_ctr23_minimal_confirmation_summary.json),
and the full independent audit is in
[`tabarena_ctr23_minimal_confirmation_independent_review.md`](tabarena_ctr23_minimal_confirmation_independent_review.md).

Stop here: no extra folds, post-outcome tuning, lockbox access, or default/preset
change is authorized.
