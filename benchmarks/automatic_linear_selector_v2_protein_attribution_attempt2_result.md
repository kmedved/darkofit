# Automatic linear-selector v2 Protein attempt-2 result

_Run once on 2026-07-22/23 under the frozen attempt-2 protocol, from clean
published harness `4cb9deb`, against private candidate `a53d4bf`. The
historical replay was then computed without fitting any model from its own
published analyzer `f72b549`._

## Decision

**Kill this selector identity.** Do not rerun, merge the candidate, design a
fresh campaign for it, change a default, or make a product claim.

Protein quality was promising: automatic over constant-leaf RMSE was
`0.968638×` across the three coordinates, and neither the aggregate nor any
coordinate breached the standing `1.02×` harm bound. The candidate failed the
separate binding behavior rule, however. It had to select linear leaves and
match the explicit-linear final model on every coordinate. On coordinate 1,
the internal validation margin was `0.025179`, below the frozen `0.03`
threshold, so automatic mode correctly followed its code contract and chose
constant leaves while explicit linear leaves improved test RMSE from
`3.816238` to `3.611996`. Prediction and normalized core state therefore did
not match. The gate is false and cannot be relaxed after inspection.

| Coordinate | Selector margin | Auto / constant RMSE | Auto choice | Exact to explicit linear |
| ---: | ---: | ---: | --- | --- |
| 0 | `0.032388` | `0.951434×` | Linear | Yes |
| 1 | `0.025179` | `1.000000×` | Constant | **No** |
| 2 | `0.037364` | `0.955225×` | Linear | Yes |

The automatic arm's descriptive geometric-mean costs were `9.7597 s` fit,
`0.01442 s` per prediction call, and `340.65 MB` peak RSS. Corresponding
constant-leaf values were `2.2863 s`, `0.00872 s`, and `323.95 MB`. These are
three spent coordinates, not performance claims.

## Historical guardrail replay

The development contract's final consistency check was completed from
immutable old artifacts only. It executed no candidate code and fitted no
model. Results depend on previously inspected outcomes and are not fresh or
independent evidence.

| Historical stratum | Lineages | Selector / default | Worst lineage | Worst split | Worst LOO |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smooth/process | 14 | `0.989264×` | `1.000000×` | `1.000000×` | `0.998504×` |
| Categorical | 3 | `0.824781×` | `1.000000×` | `1.000000×` | `1.000000×` |
| Noisy tabular | 3 | `0.977456×` | `1.000000×` | `1.000000×` | `1.000000×` |
| Group-safe sports | 1 | `1.000000×` exact | `1.000000×` | `1.000000×` | Not defined |
| **Combined** | **21** | **`0.962739×`** | **`1.000000×`** | **`1.000000×`** | **`0.989084×`** |

This confirms that the historical selector was harm-free on those spent
lineages. It does not rescue the terminal Protein failure: the frozen Protein
rule tests whether the new automatic policy recognizes all three places where
linear leaves help, and it did not.

## Execution lineage

Attempt 1 remains terminal and spent after its frozen worker environment could
not import `autogluon.common`; no fit completed. Attempt 2 used a new R1-
authorized execution identity with the same candidate, grid, arms, thresholds,
and invariants. Its direct OpenML 0.15.1 loader reproduced all three immutable
release-ladder split fingerprints before manifest creation. All nine attempt-2
cells then completed exactly once. No rerun is authorized.

## Evidence

- Attempt-2 protocol:
  `automatic_linear_selector_v2_protein_attribution_attempt2_protocol.md`.
- Launch manifest SHA-256:
  `d6cbee2249046bc8eca05080eea38457c21d0d130076a59839b313c52c8b54b7`.
- Raw artifact SHA-256:
  `0caaa2f97fd527976233f6511267c3df2b6487bc8d5a665d87c9fad2c3b11be7`.
- Result artifact SHA-256:
  `4b75f4ae048e926ec07bf3a17c4a9e9356b52a7adfd869409594cc3878f7e61c`.
- Historical replay SHA-256:
  `1d0ac7eedbcc86dd83b47f826e77efb70071381d88a27a68c6dc61d31e707122`.
- Terminal attestation SHA-256:
  `e35b6907ce1872a9f01ce5359d2b49064ef2bcd112b5952fcdd53db6a166387a`.

The next authorized mechanism slot is the separately governed categorical-
crosses campaign. This close authorizes no catcross outcome inspection by
itself.
