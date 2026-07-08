# Kalman-Readiness Plan for Gaussian σ̂ (WNBA DARKO Observation Noise)

**Date:** 2026-07-07 (rev 2 — incorporates two Oracle reviews of the shipped implementation)
**Follows:** `DISTRIBUTIONAL_REVIEW.md`, `benchmarks/wnba_realdata_distributional_summary.md`
**Question this document answers:** the real-data check passed overall (NLL 1.435 → 0.423 vs constant-σ, coverage 0.893, std-resid RMS 1.014), but σ-bin calibration is imperfect and two external reviews flagged metric-consistency defects. What, concretely, makes σ̂ production-ready as the Kalman observation variance `R_t`, and in what order?

> **Rev-2 note.** The two Oracle reviews **contradict each other** on the central issue: review A wants the calibrator moved onto the clipped objective the scorers use; review B wants the clip removed from the scorers (and then, inconsistently, also wants the calibrator clipped). §3/W0 adjudicates this with verified numbers rather than adopting either literally. Everything else material from both reviews is folded into the workstreams; the new-heads plans are condensed in the Appendix.

> **Implementation note.** W0, W1, W2, W3, and W7/M0-M4 are implemented in the current branch. The change was made because the WNBA scalar-calibrated bins showed variance-scale compression (`std-resid RMS 0.824 → 1.127` from low to high σ̂), while the old clipped validation NLL/CRPS could under-penalize large standardized residuals. Validation NLL now uses a `|z| <= 1000` overflow guard, CRPS uses the true closed form without clipping, calibration uses the same overflow guard plus an influence diagnostic, Gaussian `predict()` returns a copied mean buffer, and distributional load rejects `n_outputs`/loss mismatches. The new `dist_calibration="affine"` lane fits `σ' = exp(a + b log σ)` with profiled `a` and golden-section search over `b`; `dist_calibration="per_metric_affine"` fits the same map per `metric_code`/group with a global affine fallback; the deprecated `sigma_calibration` alias still works for one release. WNBA diagnostics now include rolling origins, richer causal dispersion features, ρ-head LR, per-head L2, and a source/tuning sweep. The latest per-metric affine WNBA split scores NLL/CRPS `0.404/0.391`, coverage `0.901`, and pooled sigma-bin RMS `1.002/0.934/1.002/1.035/0.989`, but per-metric slices still fail strict G1 at `pf_100`/`pts_100` edges. A scalar game-metric Kalman shadow replay now exists in `benchmarks/bench_wnba_kalman_replay.py`: Chimera `predict_variance()` improves normalized-innovation calibration in 2 of 3 seasons, but loses replay NLL to the incumbent `sigma2 / sample_weight` heuristic, so it is not a production replacement yet.

---

## 1. Diagnosis: what the WNBA bins actually say

Calibrated Gaussian on held-out 2024–2026, equal-count bins by predicted σ̂:

| σ̂ bin (low → high) | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 90% coverage (want 0.90) | 0.957 | 0.929 | 0.901 | 0.861 | 0.856 |
| std-resid RMS (want 1.00) | 0.824 | 0.856 | 0.990 | 1.111 | 1.127 |

Two facts fall out of this table, and they drive everything below:

1. **This is variance-scale miscalibration, not tail-shape miscalibration.** The per-bin second moment (RMS) and the per-bin coverage deviate *together and in the same direction*. If the problem were heavy tails, RMS would sit near 1.0 while coverage fell short. So the fix is a variance correction, and a Student-t head is **not** indicated yet (tripwire in W5).
2. **The specific shape is σ̂ dynamic-range compression.** σ̂ too *large* at the low end, too *small* at the high end — classic shrinkage from regularized trees plus L2 on the ρ head. A single scalar mathematically cannot fix this (it moves all bins together). A monotone *stretch* can: from the bin ratios, the needed log-space slope is roughly **b ≈ 1.1–1.2** — mild and well-conditioned.

Framing fact that simplifies everything: **the Kalman filter consumes only second moments.** Optimal-linear filtering needs `E[(y−μ)²|x] ≈ σ̂²` per slice; Gaussian shape matters only for likelihood evaluation and outlier gating. The binding criterion is per-slice RMS ≈ 1; coverage is a secondary shape check.

## 2. Definition of done — acceptance gates

- **G1 (variance, binding):** std-resid RMS ∈ **[0.95, 1.05]** in every σ̂ quintile **and** every per-metric slice (all 6 metrics).
- **G2 (shape, secondary):** 90% coverage ∈ [0.87, 0.93] per σ̂ quintile; PIT histogram passes a KS/eyeball check.
- **G3 (independence):** per-player standardized residuals show no material lag-1 autocorrelation (correlated noise ⇒ DGP needs a player effect, not a bigger R).
- **G4 (replay, the real gate):** injecting σ̂² as `R_t` in the WNBA DARKO filter beats the incumbent heuristic R on normalized-innovation-squared ≈ 1 per season and next-observation predictive accuracy, across ≥ 2 held-out seasons.
- **G5 (stability):** G1–G2 hold under rolling-origin evaluation (≥ 3 origins), not just the single 2021/2023 split.

## 3. W0 — Metric & calibration objective consistency *(new; do before W1 — the affine calibrator must be fit against the corrected objective)*

### The adjudication, with verified numbers

Before W0, the eval/scoring kernels clipped the standardized residual at |z| ≤ 10 (`_gaussian_nll_eval`, `_gaussian_crps`, tuner `_neg_gaussian_nll`), while the scalar calibrator used unclipped z². Review B's degeneracy claim **verifies numerically**: for an outlier row (y−μ = 100), clipped eval NLL *falls monotonically* — 52.9 at σ=e², 50.9 at σ=1, 45.9 at σ=e⁻⁵, 36.9 at σ=e⁻¹⁴ — while true NLL explodes to 7×10¹⁵. The clipped metric literally rewards overconfidence on tail rows, and early stopping / best-model selection / tuner ranking all consumed it. Clipped CRPS was similarly distorted: at that outlier it reported 9.4 where true Gaussian CRPS is 99.4 — a 10× underestimate of the tail penalty — despite true CRPS being *linear* in |y−μ| and therefore needing no clip at all.

Review A's opposite fix (clip the calibrator at |z| ≤ 10 for consistency) is rejected for a Kalman-specific reason: winsorizing z² at 100 can only shrink the fitted scale, i.e. push σ̂ *down* in the tail — **the exact direction of our observed high-bin under-coverage**, and a bias against the honest second moments the filter needs. Review B's own secondary suggestion (unclip eval but clip the calibrator) fails the same test and contradicts its primary fix.

### The position

1. **W0.1 — Unclip CRPS entirely.** True Gaussian CRPS is a strictly proper *and* naturally robust (linear-tail) scoring rule with no overflow risk (`erf` saturates; `σ·z → |y−μ|`). This single change makes `eval_metric="crps"` the recommended selection metric for heavy-tailed data — proper and robust at once — and makes the README's "closed-form Gaussian CRPS" claim true (review A's docs point resolves itself).
2. **W0.2 — Demote the NLL eval clip to a pure overflow guard.** Replace |z| ≤ 10 with |z| ≤ 1e3 (z² ≤ 1e6) in `_gaussian_nll_eval` and the tuner's `_neg_gaussian_nll`. The degenerate region moves from "reachable by ordinary outliers" to "astronomically far away," while a single pathological row still can't turn the validation mean into `inf`. Document the guard.
3. **W0.3 — Training kernel clip at |z| ≤ 10 stays.** It is an intentional Huber-like gradient-robustness device. Note (review A's hidden assumption, correct): the implemented ρ-gradient `1 − z_clip²` is a *straight-through* clip, not the derivative of the clipped objective — that is deliberate recovery behavior (it keeps pushing σ up on outliers) and must be documented, not "fixed."
4. **W0.4 — Calibrator optimizes the same objective as selection.** `_fit_scalar_sigma_scale` keeps unclipped z² (the closed-form MLE of the now-near-proper validation NLL), winsorized only at the W0.2 overflow guard, plus an **influence diagnostic**: warn when the top-k validation residuals contribute > X% of Σwz² (review B's explosion scenario becomes a visible warning instead of a silent bias in either direction).
5. **Tests** (from both reviews): outlier row shows eval NLL now increasing as σ→0; calibration on a validation fold with one extreme positive-weight residual leaves `sigma_scale_` finite and does not worsen validation NLL/CRPS; CRPS at |z|≫10 matches the closed form.
6. **Re-verification:** re-run both benchmark suites after W0. Expect negligible movement on synthetic (no |z|>10 mass) and small movement on WNBA (z-RMS ≈ 1.01 ⇒ thin tail mass), but the committed tables must be regenerated before being cited as promotion evidence.

## 4. Workstreams

### W1 — Affine log-σ calibration (`sigma_calibration="affine"`) — *evidence-backed by §1; fit against the W0 objective*

- Map: `σ' = exp(a + b·ρ̂)`, `ρ̂ = log σ̂`. **Profile out `a` in closed form** — for fixed `b`, `exp(2a) = Σ wᵢ rᵢ² e^(−2b·ρ̂ᵢ) / Σ wᵢ` — then 1-D search over `b ∈ [0.5, 2.0]` on the weighted W0 objective (a few Newton/golden steps; no scipy needed, though review B's L-BFGS route is fine too).
- Guards: fall back to `"scalar"` when the calibration fold is small (existing threshold) or `b` pins to a bound; record `(a, b)` + fallback reason in `auto_params_["sigma_calibration"]`.
- Plumbing: extend `_normalize_sigma_calibration`; persist `(a, b)` through the existing wrapper-state path; apply in `_predict_dist_checked` only; refit freezes `(a, b)` from the selection fold.
- **SearchCV caveat (review A, correct):** fold *scales* pool exactly (the existing mass-weighted trick), but affine `(a, b)` coefficients do **not** pool from per-fold scales. Store per-fold `(a, b, weighted_nll)` metadata and either weighted-average with a diagnostic or mark calibration "not pooled" until the OOF lane (W1b) exists.
- W1b *(small-data option, review A)*: `calibration_split="nested"` / OOF calibration lane for `cv ≤ 3` or validation effective-n < 200. Default stays fold-shared for speed.
- **Gate:** rerun the WNBA benchmark with an affine lane → per-bin RMS flattens into [0.95, 1.05]. Expected fit: b ≈ 1.1–1.2.

### W2 — Diagnostics + benchmark integrity — *parallel with W0/W1*

Diagnostics (WNBA script): per-metric calibration slices (6 metrics × σ̂ terciles — pooled bins conflate "high σ̂" with "which metric"); PIT histogram + per-bin E[z²] vs coverage side-by-side; per-player lag-1 z autocorrelation (G3); rolling-origin mode `--origins 2021,2022,2023` (G5).

Benchmark integrity (synthetic script — both items block re-citing the promotion table):

- **Fix the LightGBM twin's log-χ² bias (review A, verified math).** The twin regresses `log((y−μ_oof)²+eps)` and predicts `sqrt(exp(·))`; since `E[log ε²] = ψ(½)+log 2 ≈ −1.27036`, it systematically estimates ≈ 0.53·σ. Add the `+1.27036` correction **and** a validation-calibrated twin lane. The twin's reported 0.62 coverage is largely this bias — today's table flatters us against the one baseline practitioners would actually build.
- **Add weight modes** (`--weight-modes none uniform stress`) and pass train/val/test weights through every lane that supports them, reporting weighted NLL/CRPS/coverage. The promotion contract requires weighted evidence; the synthetic Gaussian lanes currently fit unweighted (the WNBA run is weighted, so real-data evidence already exists).

### W3 — Improve σ̂ at the source — *only if W1 leaves per-metric slices out of gate*

1. **Tune it**: point the Gaussian tuner lane at the WNBA data (current params are hand-picked; lower `l2_leaf_reg` and deeper structure plausibly reduce σ̂ shrinkage directly).
2. **Feature work**: longer-window within-player dispersion features, metric interactions.
3. **ρ-head learning-rate multiplier** *(both reviews propose; mechanism per review A)*: scale `tree.values[:, 1]` by `rho_learning_rate_multiplier ∈ {0.25, 0.5, 0.75}` between `build_leafwise_multiclass_tree()` and `add_multiclass_leaf_values_inplace()` — no kernel changes. Evidence gate: better validation NLL/coverage at equal μ RMSE. Addresses the "ρ overfits shrinking train residuals faster than μ" failure mode (review B), which is consistent with our observed σ-overfit-past-the-NLL-optimum behavior.
4. **Per-head L2** *(review B variant)*: `l2_leaf_reg=[l2_mu, l2_rho]` routed through the vector kernels. More invasive (touches split gain + leaf values); only after 1–3 demonstrably fail.

### W4 — Kalman replay harness (G4) — *the real gate; lives in `wnba_darko`; start once W0+W1 numbers land*

Shadow mode: run the filter over 2024–2026 with incumbent heuristic `R` vs `R_t = clip(σ̂'², floor, ceil)` (floor ≈ 0.1², ceil ≈ 3.0² on the standardized scale; log clip events). Compare per-season normalized-innovation-squared mean ≈ 1, innovation whiteness, next-observation predictive log-lik / projection RMSE. Adopt if replay wins ≥ 2 of 3 held-out seasons without degrading projections; keep the heuristic as an automatic fallback path.

**Current status:** partially run, not passed. `benchmarks/bench_wnba_kalman_replay.py`
performs a scalar per-metric shadow replay on the WNBA DARKO game-metric
observation artifact. It injects per-metric-affine Chimera `predict_variance()`
as row-level `R_t` and compares against a validation-tuned incumbent
`sigma2 / sample_weight` heuristic. Held-out 2024–2026 results:
Chimera NIS `0.922`, coverage `0.910`, NLL `0.186`; incumbent NIS `1.088`,
coverage `0.889`, NLL `0.117`. Chimera is closer to NIS=1 in 2 of 3 seasons,
but loses NLL in 3 of 3 seasons, so the adoption gate fails. The next replay
must either improve the `R_t` model or wire this same contract into the full
production player DARKO filter while retaining the incumbent fallback.

### W5 — Student-t head — *implemented as a distributional head; Kalman tripwire unchanged*

The fixed-ν K=2 Student-t head is implemented for general distributional use (`dist_params={"nu": ...}`, tuner categorical over {3, 4, 6, 10, 30}). For the Kalman-readiness path, the tripwire is unchanged: prefer Gaussian variance calibration unless, after affine calibration, some slice shows per-bin RMS ≈ 1 but coverage < 0.87 (variance right, tails fat). The rejected alternative remains review B's learned-ν K=3 (`log(ν−2)` head): learning ν needs polygamma (absent from numba) and a third head for marginal benefit. Details in Appendix.

### W6 — Production ops — *before first deployment*

Retrain model+calibrator before each season; recalibrate (calibrator only — closed-form/1-D) monthly in-season. Weekly rolling per-metric E[z²] against [0.9, 1.1] alarm bounds; alarm ⇒ auto-fallback to constant-σ R. Pin model artifact hashes in the DARKO pipeline config.

### W7 — Distribution-protocol generalization — *prerequisite for any new head; both reviews converge here*

Before W7, the public surface was Gaussian-shaped in four places: core `predict_dist()` hardcoded `GaussianNLL.mean_and_sigma`; wrapper `predict()` branched on `self.loss == "Gaussian"` and sliced `raw[:, 0]`; `_require_gaussian()` guarded public methods; interval/sample assumed Normal. The implemented W7 work replaced those seams as follows:

- Add the vector-loss protocol on `VECTOR_LOSSES` classes: `mean_from_raw(raw)`, `params_from_raw(raw)`, `interval_from_raw(raw, alpha)` (may raise `NotImplementedError` per head), `sample_from_raw(raw, rng, n)`, plus `distribution_name` / `default_eval_metric` / `supported_eval_metrics`. Core and wrapper delegate; `_require_gaussian` becomes `_require_distributional(method, capability)`.
- **Rename `sigma_calibration` → `dist_calibration`** (deprecation alias for one release) — "sigma" is Gaussian-specific; Poisson calibrates the mean, NB the dispersion.
- **Serialization hardening (review A):** on load of `DistributionalBoosting`, require `header["n_outputs"] == booster.loss_.n_outputs` (a corrupted archive can currently pass tree-width validation with inconsistent semantics). Add `header["loss_state"]` for fitted loss state (NB global dispersion, future calibrated-distribution state).

## 5. Small fixes (fold into the next commit)

1. `predict()` / staged path: return the distributional predictive mean and apply validation calibration when that calibration changes the mean (Poisson/NB mean calibration, LogNormal scale calibration). `predict_raw()` remains the uncalibrated score surface.
2. Until W0.1 lands, README/metadata should say "clipped Gaussian CRPS" (review A). After W0.1, revert to "closed-form."
3. Serialization `n_outputs` cross-check (from W7 — can land independently as pure hardening + the mutate-header-to-`n_outputs=3` rejection test).

## 6. Sequence and decision tree

```
W0 metric consistency  ──→  re-run both benchmarks (tables regenerate)
        │
W1 affine (+W1b OOF lane)  +  W2 diagnostics & benchmark integrity   [parallel]
        │
        ├─ pooled + per-metric RMS in [0.95, 1.05]?
        │        ├─ yes → W4 replay ──→ wins? → W6 ops → DONE
        │        │                       └─ no → G3 innovations check
        │        │                              (correlated ⇒ DGP issue, not R)
        │        └─ no  → W3.1 tune → W3.2 features → W3.3 ρ-LR multiplier
        │                      └─ still failing → W3.4 per-head L2
        └─ RMS flat but coverage low in a slice → W7 protocol → W5 Student-t
```

W0 is a half-day (three constants, one calibrator edit, tests); it goes first because W1's affine fit must target the corrected objective, and because every downstream gate (early stopping, tuner, replay selection) reads these metrics.

## 7. Repo hygiene before the PR

- Commit the WNBA artifacts (script + CSV + summary + notes). The parquet is a private Dropbox path — the script must fail with a clear message without it.
- Resolve dirty `chimeraboost/tree.py` (`_count_leaf_rows` scalar-builder change): commit separately with an `ab_compare` bit-identity check, or revert. Unreviewed core-tree drift must not ride along.
- Same for the dtype-campaign edits in `benchmarks/bench_feature_modes.py`.

---

## Appendix — new-heads plans

The full, Codex-executable implementation spec for the distribution-head protocol (W7) and the Student-t, Poisson, Negative Binomial, and LogNormal heads lives in **`DISTRIBUTIONAL_HEADS_SPEC.md`** — parametrizations, kernel code, gradients/Fisher with bounds, target validation, calibration formulas (including which pool exactly across SearchCV folds), serialization (`loss_state`), tuner integration, per-head test suites, and milestone order (M0 protocol → M1 LogNormal → M2 StudentT → M3 Poisson → M4 NB-global → M5 NB-hetero, gated).

Standing invariants every head must preserve (enforced in that spec's kernels and tests): strictly positive per-head Fisher/hessian mass for positive-weight rows (split legality flows through summed hessians in the vector path); zero-weight rows skipped before any parameter arithmetic; raw-score clips before every `exp`; W0 eval policy — true NLL with overflow guards only, no robustness clips in eval; `variance_from_raw` implemented on every head (the Kalman consumer — for Student-t it is `scale²·ν/(ν−2)`, not `scale²`).

Adjudications recorded there (do not relitigate): Student-t is fixed-ν K=2, not learned-ν K=3; NB ships global-dispersion K=1 first with heterodispersion behind a real-data evidence gate; scipy allowed only in wrapper-level quantile code (guaranteed transitive dep of scikit-learn); `sigma_calibration` renames to `dist_calibration` with a one-release alias; Gamma is deferred behind a recorded trigger.
