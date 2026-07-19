# Product Offense: calibrating the evidence machine and going on offense

*Drafted 2026-07-17 at `main` = `248c6ab` (0.10.0 prepared), after independent
verification of the closed BEYOND_PARITY program. Adopted for execution from
base `a288db9`. The governing constitution is
[`benchmarks/SHIPPING_POLICY.md`](benchmarks/SHIPPING_POLICY.md); this file is
the agenda. Companion to the execution ledger in `BEYOND_PARITY_PLAN.md`.*

## 0. Execution ledger

| Item | State | Binding evidence |
|---|---|---|
| G1–G4 shipping governance | Complete | `benchmarks/SHIPPING_POLICY.md` |
| Tier-E subset fusion | Complete | exactness suite + immutable prior measurements |
| Tier-E measurements page | Complete | hash-bound generated page from immutable artifacts |
| T1 accuracy preset | Complete | code, equivalence, persistence, metadata, docs |
| T2 ensemble v2 | Complete | row/group OOB, safe preprocessing, SHAP, persistence, docs |
| T3 feature recipes | Complete | docs tied to immutable evidence |
| T4 capped selection | Complete | final-refit exactness + metadata |
| T6a RSSI diagnosis | Complete | `benchmarks/rssi_linear_leaf_diagnosis_result.md` |
| T6b smooth cross features | Complete | exact mechanism + 5% T5 nominee |
| T5 composite confirmation | Closed | fail-closed on two non-finite-target tasks before candidate wave |
| T7 CatBoost attribution | Complete | one frozen `(n,p)` depth-policy research candidate |
| T7b CatBoost attribution | Complete | fixed-LR, three-seed follow-on; no gap contributor identified |
| T8 distributional flagship | Complete | 75/75; conformal coverage/width reported jointly |
| T9 SynthGen corrected ledger | Complete | 8/9; probe-tier direction finder only |
| T10 sports panel #2 | Closed | OOB-5 lost 8/9 fresh player-disjoint lineages |
| Panel 3 lockbox | Preparing | exact-policy spent-data power decision required before fresh access |

Status in this table is operational only. It cannot amend a frozen protocol or
promote a Tier-D candidate.

## 1. The diagnosis: our gates are calibrated ~3–4× stricter than the competitor's shipping bar

The program did not fail to find effects. It found them and then held them to
bars the competitor has never met.

**Exhibit A — linear leaves.** Our fresh-panel selector: 0.9893× RMSE
(−1.07%), zero regressing lineages, worst lineage 1.0000×. We required
≤0.9800× and ≥9/14 wins → closed. ChimeraBoost shipped `linear_leaves=None`
as their *default* on: −0.58% mean, 20W/9T/7L with real prior casualties
(their 0.14.1 changelog). **Our rejected selector is comparable-or-better
evidence than what they shipped on.** Their whole product edge is an
accumulation of +0.3…+1.5% selected wins, each of which would have died at
our bars.

**Exhibit B — engineering claims.** Large-n fit: measured 1.2793× faster than
their engine (1.3155× at 1M rows, RSS 0.71–0.84×) — no claim, because the
frozen bar said 1.30. Predict: medians 0.805–0.987× theirs in all eight
cases — no certification, because one paired-stability series read 0.1056
against a 0.10 limit. These are true facts we are forbidding ourselves from
stating.

**Exhibit C — opt-in surfaces held to default-grade gates.** ChimeraBoost
ships `n_ensembles`, `ordered_boosting`, `linear_lambda`, etc. as documented
opt-ins with *no panel gates* — only defaults get panels. We killed an
ensemble **API** twice on timing-noise gates while its quality gates passed
(+0.0039 mean, +0.0193 cold-player). We have never shipped the measured
−2.44% accuracy configuration (A10) even as an opt-in preset.

**What stays sacred** (this is calibration, not abandonment): the lockbox,
contamination registries, no-rerun rules for *confirmation* evidence,
bit-exactness discipline for engine work, and honest docs. The changes below
touch claim tiers and bars, not integrity.

## 2. Governance changes

The complete binding policy is `benchmarks/SHIPPING_POLICY.md`.

- **G1. Two-tier claims.**
  *Tier-D (defaults and automatic policies):* frozen preregistered gates,
  power-checked, no-rerun.
  *Tier-E (opt-in APIs, presets, recipes, engineering facts):* ship on
  exactness/correctness tests plus honest measurement with uncertainty.
- **G2. Effect-size-honest Tier-D bars.** Future default gates derive their
  bars from plausible competitive effects and include a design-time power
  analysis. Win counts are abolished.
- **G3. Profile-scoped ladders.** The product surfaces are `default`,
  `sports`, and `tabular`. A profile's named confirmation panel is fatal;
  irrelevant panels are exactness/no-harm screens, not universal vetoes.
- **G4. Descriptive engineering reporting.** Fit, prediction, and RSS facts
  are reported with dispersion as measurements, never binary certifications.

## 3. Immediate ships (Tier-E: no new panels required)

- **T0. Subset fused kernels.** Promote the already-implemented,
  behavior-exact selected-row/selected-feature dispatch. Existing immutable
  evidence reports 0.5348× fit and 0.5265× tree-build geometric-mean ratios;
  its old rejection concerned variance in the size of the speedup, not
  behavior or a crossed-1.0 regression.
- **T1. `preset="accuracy"`** — the frozen A10 profile (`tree_mode="auto"`,
  10k rounds under ES, L2=3, 128 bins, LR 0.1): measured −2.44% versus live
  ChimeraBoost on the spent 13-task development panel, −3.64% versus our
  product default, at the already-recorded inference cost. It remains opt-in
  and the concentration caveat must be adjacent to the headline.
- **T2. `n_ensembles` API, v2 shape** — configurable bagging
  (`bootstrap="rows" | "groups"`), per-member OOB early stopping, safe shared
  preprocessing, soft-vote/mean aggregation, and SHAP averaging. The API is
  opt-in; closed campaigns continue to bar default promotion.
- **T3. Documented recipes** for already-shipped robust heads,
  `random_strength`, `linear_leaves`, and `ordinal_features`, printing wins,
  nulls, and failure boundaries together.
- **T4. `selection_rounds`** — capped audition fits for internal selection
  races, followed by a fresh full-budget fit of the selected lane. The final
  fit must be exact to an explicit full-budget fit of that lane.
- **T4a. Measurements page.** Generate or deterministically derive a page
  from immutable artifacts, including 1.2793× matched large-n fit and the
  public prediction ratios. Every number carries workload, version,
  dispersion, and evidence-scope labels.

## 4. The offense campaigns (Tier-D where they touch defaults)

- **T5. Composite tabular policy.** Build one candidate combining size-gated
  early stopping, LR 0.1 under ES, validation-selected linear leaves, scoped
  cross features and declared ordinals, and capped auditions. Test the unit
  against ChimeraBoost and CatBoost on a fresh, contamination-screened,
  approximately 25-dataset panel. Freeze the exact policy, power simulation,
  uncertainty, leave-one-out concentration, harm, and cost rules before
  outcomes. The 25-lineage campaign was frozen and powered, but its first
  control wave found two tasks with non-finite targets. Because control
  outcomes already existed and the protocol forbade task dropping or
  imputation, the campaign closed without running the candidate or changing a
  default. The failure record is
  `benchmarks/t5_composite_confirmation_failure.md`; every lineage is spent
  for confirmation.
- **T6. Smooth/geometry cross features.** Develop diff/product features on
  spent smooth tasks, beginning with a 3D-RSSI diagnosis. The closed sports
  application remains closed. Confirmation belongs inside the prospective
  Panel 3 dual-candidate campaign. T6a
  found byte-exact matched linear-leaf engines: the RSSI gap was validation
  and selection policy, the 100-round linear audition chose the wrong
  full-budget lane, and cross features were not selected. T6b therefore uses
  other spent smooth tasks for cross-feature development rather than
  reimplementing linear leaves. T6b then reproduced ChimeraBoost's native
  full-budget cross path exactly on 21 coordinates. Its raw selector had a
  1.0708× worst split; a development-derived 5% validation margin retained a
  0.9591× equal-dataset ratio with exact declines and no observed split harm.
  That guarded mechanism is the second, separately adjudicated Panel 3
  candidate, not a promoted default.
- **T7. CatBoost attribution.** On development data, isolate ordered
  boosting, border count, leaf estimation/backtracking, combinations, and
  depth policy by `(n, p)`. The output is an attribution table and at most
  three frozen candidates, not a post-hoc default change. The complete
  216-fit attribution rejected Ordered boosting, 128 borders, and CTR
  complexity 2 as automatic explanations. Its leaf-estimation arm also changed
  learning rate, so that direction remained unresolved until T7b. The
  predeclared samples-per-feature depth policy improved CatBoost's
  equal-dataset RMSE by 3.78% with three wins, no losses, and five exact
  defaults; it is frozen as the sole research candidate. See
  `benchmarks/t7_catboost_attribution_result.md`.
- **T7b. CatBoost attribution follow-on.** Freeze the per-coordinate default
  learning rate and use three model seeds while testing stochastic
  regularization, sampling, L2, one-hot thresholds, and extra leaf-estimation
  steps. The completed campaign attributed none of the gap. `l2_leaf_reg=1`
  was the sole promising configuration. One-hot 255 produced a large mean
  improvement but failed the Bonferroni upper and worst-task gates, so it does
  not support a broad policy. See
  `benchmarks/t7b_catboost_gap_attribution_result.md`.
- **T8. Distributional flagship.** Benchmark CRPS, coverage, width, and NLL
  against NGBoost, CatBoost uncertainty, and quantile LightGBM. Add
  `predict_interval(..., calibrate="conformal")` as an opt-in, evaluated
  coverage-first without hiding interval width. The frozen 75-coordinate
  campaign completed without skips. Across five equally weighted datasets,
  conformal DarkoFit had the smallest mean absolute 90%-coverage error
  (`0.0110`) while its geometric-mean width was `0.9831×` the parametric
  DarkoFit interval. The result is descriptive Tier-E evidence, not a default
  change or conditional-coverage claim; all per-dataset NLL, CRPS, coverage,
  width, and timing values remain adjacent in the
  [`result`](benchmarks/t8_distributional_flagship_result.md).
- **T9. SynthGen corrected-ledger re-backtest.** Score the instrument against
  the corrected fresh-panel outcome ledger. It may become a probe-tier
  direction finder only if it meets its frozen adoption rule; it never
  substitutes for confirmation. The immutable synthetic artifact reproduced
  the original 6/9 scorecard, then reached 8/9 when exactly two later
  confirmation outcomes superseded their development labels:
  `random_strength=0.5` failed the fresh sports panel and fixed local linear
  leaves regressed on the fresh smooth/process panel. SynthGen is therefore
  adopted only to rank cheap development probes. The explicitly retrospective
  [`result`](benchmarks/t9_synthgen_corrected_ledger_result.md) cannot kill,
  confirm, promote, or justify a product decision.
- **T10. Sports panel #2.** Build a fresh multi-season/multi-target
  confirmation bed for the sports profile. It is the only route by which
  ensemble/random-strength composites can become a sports automatic policy.
  The source-frozen 2014–2016 panel used nine equally weighted target-season
  lineages, player-disjoint primary folds, and held-team/cold-player
  guardrails. The row-OOB five-member ensemble passed every declared cost gate
  (`2.92×` fit, `2.68×` prediction, `1.06×` RSS) but failed every primary
  quality gate: `1.0231×` aggregate RMSE, `1.0401×` bootstrap upper bound,
  eight losses, and a `1.0645×` worst lineage. Cold-player RMSE was
  `1.0287×`. The sports automatic-policy path is closed without retuning.
  Current DarkoFit control was `0.9719×` ChimeraBoost 0.15.0 but `1.0526×`
  CatBoost 1.2.10 on the same player-disjoint coordinates. See the frozen
  [`result`](benchmarks/basketball_sports_panel_v2_result.md).

## 5. Sequencing

1. **Immediate Tier-E wave:** G1–G4 · subset fusion · measurements page · T1
   preset · T2 ensemble API · T3 recipes · T4 selection rounds.
2. **Development wave:** T6a RSSI diagnosis · T5 panel construction and
   composite · T6 cross features · T7/T7b CatBoost attribution · T9 SynthGen
   re-backtest.
3. **Ceiling wave:** T8 distributional flagship · T10 sports panel #2.
4. **Panel 3 decision:** run the exact-policy calibration only on spent
   coordinates; publish its preregistered power go/no-go before any fresh
   target access; open the lockbox only if at least one candidate survives.
5. **Closeout:** independent review, complete test partitions, packaging,
   documentation build, multi-Python CI, canonical commit and push.

## 6. What failure would actually look like

Failure is not a preregistered gate rejecting a candidate; that is evidence.
Failure is leaving correct opt-in surfaces hidden, never assembling the
composite because each ingredient once failed alone, or turning measurements
into binary marketing claims. The governing policy is designed so small,
honest wins can accumulate while default changes still require uncertainty,
concentration, harm, cost, contamination, and fresh-data discipline.
