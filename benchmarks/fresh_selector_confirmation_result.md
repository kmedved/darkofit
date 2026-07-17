# Fresh smooth/process selector confirmation result

_Run 2026-07-17 from clean DarkoFit `main` at `29bd30c`, with clean
ChimeraBoost 0.15.0 at `851ab7f`, under the frozen
[`fresh_selector_confirmation_protocol.md`](fresh_selector_confirmation_protocol.md)._

## Decision

Close the 3% validation-margin local-linear selector. Do not calculate
lockbox power, expose the selector, or change defaults. The CTR23 lockbox
remains sealed.

The candidate passed every categorical and noisy-tabular safety gate, but
failed three independent primary gates:

| Primary smooth/process gate | Required | Observed | Result |
|---|---:|---:|---:|
| Selector / default equal-lineage RMSE | ≤0.9800× | **0.9893×** | Fail |
| Lineage wins | ≥9 / 14 | **2 / 14** | Fail |
| Worst selector / default lineage | ≤1.0200× | 1.0000× | Pass |
| Selector / ChimeraBoost product | ≤1.0000× | **1.1196×** | Fail |

The selector improved DarkoFit by 1.07% without regressing any lineage, but
engaged on only 5 of 42 primary coordinates across two lineages. That is too
sparse to support the intended smooth-data policy.

## Primary detail

| Primary lineage | Selector / default | Selected folds |
|---|---:|---:|
| Coffee distribution | **0.8766×** | 3 / 3 |
| Debutanizer | **0.9807×** | 2 / 3 |
| Remaining 12 lineages | 1.0000× each | 0 / 36 |

Fixed linear leaves were not a general answer on this fresh panel:
`1.0069×` default in aggregate, with 3 wins, 8 losses, 3 ties, and a
`1.2641×` worst lineage. The selector correctly avoided those regressions,
but its internal validation margin did not identify enough transferable
benefit.

The `1.1196×` ChimeraBoost ratio is heavily affected by
3D RSSI localization, where DarkoFit default/selector was `6.1163×`
ChimeraBoost. Removing or reweighting that frozen lineage is not allowed.
More importantly, the candidate still independently failed its DarkoFit
improvement and win-count gates, so the closure does not depend on that one
comparator outlier.

## Guardrails and report-only comparators

| Stratum | Selector / default | Lineage record | Worst lineage |
|---|---:|---:|---:|
| Categorical | **0.8248×** | 1 win, 2 ties | 1.0000× |
| Noisy tabular | **0.9775×** | 1 win, 2 ties | 1.0000× |

Across all 60 coordinates the selector engaged 11 times and declined 49.
It took about `2.00×` the summed default worker wall time, consistent with
two selection fits plus a final fit; timing was report-only by protocol.

CatBoost was also report-only. Its aggregate comparisons are not used to
rescue or reject this selector, and the anomalous CatBoost result on 3D RSSI
must not be promoted as a product claim without a dedicated diagnosis.

## Validity

All 100 task/arm workers completed: 20 tasks × five arms, each with the three
frozen folds. Every live OpenML split matched its frozen train/test index
hashes. No task was dropped or imputed, both source trees stayed clean, and
no worker emitted stderr.

The first launch failed before artifact creation because CatBoost product
defaults legitimately report no best iteration without an eval set. The
[`invalid-attempt record`](fresh_selector_confirmation_invalid_attempt.md)
documents the report-only metadata correction. No outcome from that launch
was inspected and the complete unchanged campaign was rerun.

## Evidence

- Raw artifact:
  [`fresh_selector_confirmation.json`](fresh_selector_confirmation.json),
  SHA-256
  `4dc158ec4fd11cf29a5822dc2a09aa76715ce9446773673fa9a2828da1b71a7d`.
- Protocol SHA-256:
  `5eaee9c5c32ab29049dd7baad539dd4d7badf6ec26cd50fa41bc6f91dcfdce28`.
- Runner SHA-256:
  `ae127d7011283c90716a8ac2fb6cf36265bfb21c31ead15b42656ace70c87a67`.
- Registry v1/v2 file SHA-256:
  `37799ed0b788af3c1d69c8f0f7cf37a656fde998ce2d54b5c4a2196c369df4c3`
  and
  `0d878d690e32f6781a170fa3e5c232eef13d20d51d25b352c96a20ddc87e3970`.
