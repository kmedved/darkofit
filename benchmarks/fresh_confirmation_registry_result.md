# Fresh confirmation registry result

_Built 2026-07-17 from clean `main` at `8664f7c`, against clean
ChimeraBoost 0.15.0 at `851ab7f`, under the frozen
[`fresh_confirmation_registry_protocol.md`](fresh_confirmation_registry_protocol.md)._

## Decision

Authorize the exact 3% smooth linear-leaf selector profile for one fresh
confirmation run. Do not open the CTR23 lockbox and do not promote the
selector.

Registry construction fit no model and computed no target statistic or
candidate benchmark score. All 20 declared primary tasks passed the
contamination checks.

## Frozen panel

| Stratum | Independent lineages | Coordinates |
|---|---:|---:|
| Smooth numeric, primary | 14 | 42 |
| Categorical guardrail | 3 | 9 |
| Noisy-tabular guardrail | 3 | 9 |
| **Total** | **20** | **60** |

Each lineage uses OpenML repeat 0, folds 0–2, sample 0. Related same-source
tasks are declared but do not receive independent votes.

The audit binds the pre-registry DarkoFit commit, ChimeraBoost's OpenML,
Grinsztajn, PMLB, high-cardinality and TabArena-name universes, and every
semantic source fingerprint in CTR23-v3. No exact ID/name/fingerprint,
repository reference, conservative name-containment, canonicalization, or
near-lineage alarm fired.

## Power

The frozen 200,000-draw empirical bootstrap passed in 199,993 simulations:
**99.9965%** conditional pass probability versus the required 80%.

Each draw sampled 14 lineage effects from the 21 already-spent
selector/default split ratios. Passing required:

- equal-lineage geometric-mean RMSE ratio at most 0.98;
- at least 9 of 14 lineage wins; and
- no lineage ratio above 1.02.

This is design adequacy under the spent three-lineage effect distribution, not
generalization evidence. The fresh run exists to test that assumption.

## Scope

The registry records `confirmation_data_scored=false`,
`selector_promotion_authorized=false`, and `lockbox_run_authorized=false`.
Only a passed fresh primary and both guardrail strata can justify a separate,
observed-effect lockbox power freeze.

The first registry invocation falsely matched numeric task IDs inside unrelated
JSON decimals and failed before writing an artifact. The
[`invalid-attempt record`](fresh_confirmation_registry_invalid_attempt.md)
documents the correction; no candidate or gate changed.

## Evidence

- Registry artifact:
  [`fresh_confirmation_registry.json`](fresh_confirmation_registry.json),
  file SHA-256
  `37799ed0b788af3c1d69c8f0f7cf37a656fde998ce2d54b5c4a2196c369df4c3`;
  canonical registry SHA-256
  `2d1f232e998d9f815a97f80735cbfebe5587c8b36f3fe246b26fbf355c4b5f64`.
- Protocol SHA-256:
  `65275c673b0a6fe9927e1846f67a4a089a34234df483061f07c24bda4d236166`.
- Declarations SHA-256:
  `8fe7b5a6111b7f58c180fe0a37c05b0839793a408b653aab60499cbb8f080e70`.
- Builder SHA-256:
  `ae20dbe5b02c71c1ab689a144b8cc330d4b19a94271114068542e5371561487e`.
