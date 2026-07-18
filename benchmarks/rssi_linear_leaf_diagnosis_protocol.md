# RSSI linear-leaf parity diagnosis

## Purpose and claim boundary

This development-only diagnostic explains the large historical DarkoFit versus
ChimeraBoost gap on the 3D RSSI task. It asks three narrow questions:

1. are the two libraries' constant- and linear-leaf engines behavior-exact
   when data, validation rows, and model parameters are identical;
2. how much of the observed product gap comes from validation/default policy
   rather than the leaf implementation; and
3. does ChimeraBoost's 100-round linear-leaf audition choose the same lane as
   a full-budget comparison?

This is **not** a Tier-D confirmation campaign and cannot promote a default.
There is no win-count, quality, timing, or shipping gate. The output is a
mechanism diagnosis used to design the later frozen T5 composite candidate.

## Spent-data boundary

- OpenML task `363132`, dataset
  `3D_Estimation_using_RSSI_of_WLAN_dataset`, target `Receiver_Height`.
- Official repeat `0`, fold `0`, sample `0`.
- This coordinate was already scored by
  `benchmarks/fresh_selector_confirmation.json`; its use here spends no fresh
  evidence.
- No CTR23 lockbox task or unscored fresh-panel coordinate may be loaded.
- The task must remain numeric, complete, 5,760 rows by 7 features, with one
  repeat and ten folds.

The official outer test fold is untouched by all model selection. A shared
inner validation split is made from the official training rows with
`ShuffleSplit(test_size=0.20, random_state=4)`.

## Source boundary

- DarkoFit is the clean committed branch running this protocol.
- ChimeraBoost is clean commit
  `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d` (`v0.15.0`).
- Both source heads, source cleanliness, protocol hash, prior-artifact hash,
  data hashes, and split hashes are recorded.
- Source heads are rechecked after all fits and before the output is written.

## Frozen arms

All matched arms use seed 4, six threads, depth 6, L2 1, 128 bins, learning
rate 0.1, 1,000 rounds, minimum child hessian 1, early stopping, and the same
explicit 20% validation rows.

| Arm | Purpose |
|---|---|
| `darko_default` | Reproduce the historical DarkoFit product baseline. |
| `darko_matched_auto10_linear` | Matched parameters with DarkoFit's 10% automatic validation split. |
| `darko_matched_auto20_linear` | Same parameters with a 20% automatic split; must equal the explicit shared split. |
| `darko_shared_constant` | DarkoFit forced constant leaves on the shared split. |
| `darko_shared_linear` | DarkoFit forced linear leaves on the shared split. |
| `chimera_shared_constant` | ChimeraBoost forced constant leaves on the shared split. |
| `chimera_shared_linear` | ChimeraBoost forced linear leaves on the shared split. |
| `chimera_full_selector` | Full-budget constant-versus-linear selection, cross features disabled. |
| `chimera_capped_selector` | The same race with the product's 100-round audition. |
| `chimera_full_product` | Full-budget linear and cross-feature selection. |
| `chimera_product` | Unmodified ChimeraBoost 0.15 product defaults. |

Fit seconds are recorded only to make the artifact auditable. They are not
warmed, paired, or interpreted as performance evidence.

## Binding diagnostic checks

The runner fails unless:

1. the matched DarkoFit and ChimeraBoost constant arms have identical bin
   borders, validation histories, serialized normalized tree fingerprints,
   prediction bytes, retained tree counts, validation minima, and test RMSE;
2. the matched linear arms satisfy the same exactness checks;
3. DarkoFit's automatic 20% linear arm equals its explicit shared-split arm;
4. the full selector's recorded lane agrees with the lower of the two forced
   full-budget validation minima; and
5. all predictions are finite and have the official test-fold shape.

Whether the capped audition agrees with the full race, and whether cross
features are selected, are outcomes to report—not preregistered pass/fail
conditions.

