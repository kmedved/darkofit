# Automatic linear-selector v3 Protein development result

_Run 2026-07-23 from clean source and harness commit `0df50be`, with the
automatic 2-SE selector fixed before these Protein outcomes were inspected._

## Result

The automatic selector is **ready for the SHIP_RULES holdout ship-check**.

- Equal-coordinate geometric-mean automatic/constant RMSE: **`0.951040x`**
- Worst coordinate ratio: **`0.955225x`**
- Coordinates improved: **3/3**
- Automatic selections: **linear leaves on 3/3**
- Final models exact to the selected explicit arm: **3/3**

The three coordinate ratios were `0.951434x`, `0.946481x`, and `0.955225x`.
Their paired validation-gain z scores were `6.578`, `2.958`, and `8.625`.
The middle coordinate is the important selector result: its observed
validation improvement was `2.52%`, below the retired fixed `3%` cutoff, but
well above the new 2-SE noise guard.

## Interpretation

This is spent development evidence. It shows that the noise-based selector
recovers the known smooth-data benefit without changing the fitted result
relative to an explicit linear-leaf model. The preceding non-Protein
calibration engaged zero of 18 eligible cells at 2 SE, so the observed value
remains concentrated in the intended smooth-data regime.

It is not holdout evidence and does not authorize a default. Under
`SHIP_RULES.md`, the next step is the fixed holdout ship-check. The selector
may become automatic only if it is not worse there and remains revertible
through `linear_leaves=False`.

## Artifact

- Result JSON:
  `automatic_linear_selector_v3_protein_20260723.json`
  (`b14cdde34d2a938c845ee31fb900ad16202dee7ce9c2617e753258374d859a72`)

