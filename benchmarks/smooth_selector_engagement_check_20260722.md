# Smooth-selector engagement check

_Create-only Phase B-1 record, 2026-07-22. This inspects spent artifacts; it
is not new quality evidence._

## Verdict

The proposed two-dataset explanation is **half right**. ChimeraBoost's
linear-leaf selector is a persistent signature on `physiochemical_protein`,
but it did not engage on `airfoil_self_noise`. The automatic-selector campaign
therefore remains funded for Protein and the generic smooth/process class; it
may not use Airfoil as evidence for the selector's causal claim.

This narrowing does not leave a current Airfoil deficit unexplained. Against
the current ChimeraBoost v0.20 default, DarkoFit v0.11 already wins Airfoil at
`0.953347x` RMSE. Protein remains worse at `1.067903x` RMSE while fitting in
`0.233154x` the time, leaving room to spend compute on safe automatic
selection.

## What the fitted metadata says

| Spent evidence | Airfoil | Protein |
|---|---:|---:|
| M2 / ChimeraBoost `f14be60` outer child fits | 24 | 24 |
| Linear lane selected | 0 / 24 | **24 / 24** |
| Linear audition performed | 0 / 24 | **24 / 24** |
| Cross features resolved | 0 / 24 | 0 / 24 |
| Categorical combinations enabled | 0 / 24 | 0 / 24 |
| Current v0.20 default members reporting linear selection | 0 / 3 (`null`) | **3 / 3** |
| Current v0.20 default members reporting cross selection | 0 / 3 (`null`) | 1 / 3 |

The v0.18 evidence isolates linear leaves on Protein: all 24 child models chose
the linear lane, while neither cross features nor categorical combinations
were enabled. The current v0.20 evidence reproduces linear selection on all
three direct default fits; one also selected 30 cross pairs. Airfoil reports no
selector engagement in either generation. Its earlier gap belongs to the
already-spent configuration/representation history, not this campaign.

## Binding and decision

The machine-readable record is
[`smooth_selector_engagement_check_20260722.json`](smooth_selector_engagement_check_20260722.json),
SHA-256
`878ffdc0bfb615714b5acd0ea0c1d09f63604d4d423d57ab0898f9bd377ab3d1`.
It was produced by
[`extract_smooth_selector_engagement.py`](extract_smooth_selector_engagement.py),
SHA-256
`368764dcf102d79a37b0cb16156a1fd56192de899f5d1ec7e001d688389876cf`,
after verifying every inspected M2 pickle against completion attestation
`1fbd09e...b96c4b5c` and binding the current raw release ladder at
`96f594da...0636f79`.

The exact command was:

```console
python benchmarks/extract_smooth_selector_engagement.py \
  --m2-cache <v011-m2-v3-cache> \
  --output benchmarks/smooth_selector_engagement_check_20260722.json
```

**Next action:** open a new-identity automatic-selector development campaign
on spent data, scoped to the verified Protein/smooth-process signature. Airfoil
is excluded from the selector causal claim. No manual product switch, default
change, fresh confirmation, TabArena, or lockbox access is authorized by this
record; any default-on policy still needs a separately authorized,
prospectively frozen Tier-D confirmation with design-time power at least 80%.
