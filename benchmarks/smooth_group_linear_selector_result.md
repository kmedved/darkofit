# Smooth group-safe linear selector result

_Run 2026-07-17 from clean `main` at `a81e874`, with clean ChimeraBoost
0.15.0 at `851ab7f`, under the frozen
[`smooth_group_linear_selector_protocol.md`](smooth_group_linear_selector_protocol.md)._

## Decision

Advance the exact 3% validation-margin selector to fresh-confirmation design.
Do not expose a public selector, change defaults, or touch the CTR23 lockbox.

The selector passed all nine frozen development gates. It selected local
linear leaves on 20 of 21 spent coordinates and declined the one
grid-stability fold whose internal validation gain was only 1.53%.

## Primary result

| Contrast | Equal-task RMSE ratio | Difference | Dataset record | Split record |
|---|---:|---:|---:|---:|
| Selector / DarkoFit default | **0.9233×** | **7.67% better** | 3–0 | 20–0, 1 tie |
| Fixed linear / DarkoFit default | 0.9203× | 7.97% better | 3–0 | 21–0 |
| Selector / fixed linear | 1.0033× | 0.33% worse | 0–1, 2 ties | 0–1, 20 ties |
| Selector / ChimeraBoost product | 1.0196× | 1.96% worse | 1–2 | 5–16 |

The selector retained 96.17% of fixed linear leaves' aggregate improvement,
above the frozen 90% gate. Its worst dataset ratio versus fixed linear was
`1.009985` on grid stability, inside the 1% budget. Every internal split used
the declared weighted target-stratification policy.

## Per-dataset result

| Dataset | Selector / default | Selector / ChimeraBoost product | Selected |
|---|---:|---:|---:|
| Grid stability | **0.9497×** | 1.0307× | 6/7 |
| kin8nm | **0.8958×** | 1.0379× | 7/7 |
| space_ga | **0.9253×** | **0.9910×** | 7/7 |

The selector solves the policy-safety problem demonstrated by noisy
basketball data, but not the remaining ChimeraBoost product gap. That 1.96%
gap remains consistent with ChimeraBoost's separately selected cross-feature
mechanism and must not be described as local-linear leaf inferiority.

## Scope

All 21 coordinates were already spent development data. The campaign
authorizes only construction and power analysis of a genuinely fresh,
contamination-screened confirmation panel for this frozen selector. The
lockbox remains sealed.

Three pre-result launches failed closed because of a missing optional OpenML
dependency and then an over-strict stop-reason assertion; the
[`invalid-attempt record`](smooth_group_linear_selector_invalid_attempt.md)
documents them. No invalid launch wrote an artifact or supported a decision.

## Evidence

- Raw artifact:
  [`smooth_group_linear_selector.json`](smooth_group_linear_selector.json),
  SHA-256
  `13fe1d232843b728388e35585c0e9c9f2322e0e854d896a941ad77db44bade8d`.
- Protocol SHA-256:
  `07aa7e9386273aff226d7ea4de966af9179cb61e8bd103802b628c9e6120a776`.
- Runner SHA-256:
  `474ef45cebc6d4c19f5d0f080986b3caf9cfbe7962faceaab5c7273c5be2068f`.
