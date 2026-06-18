# Changelog

## Unreleased

* Replace the previous automatic learning-rate heuristic with a transparent
  CatBoost-form selector keyed to loss, resolved iteration budget, eval-set
  presence, and Kish effective sample size.
* Add an LR-only ChimeraBoost correction for materially weighted RMSE fits in
  CatBoost/oblivious-tree mode; unweighted and all-ones-weight fits keep the
  raw CatBoost-form learning rate.
* Damp unweighted LightGBM-mode automatic learning rates as a provisional
  tree-mode-specific correction; weighted projection fits stay on the hotter
  effective-sample-size corridor.
* Add a bounded high-dimensional automatic learning-rate shrinkage based on the
  post-preprocessing model feature count relative to Kish effective sample size;
  the multiplier is recorded in `auto_params_["learning_rate"]`.
* Add `use_best_model=True` default behavior for fits with validation data,
  keeping the best validation prefix even when early-stopping patience does not
  fire. Set `use_best_model=False` to keep every fitted tree.
* Resolve default early-stopping patience from the fitted learning rate when
  `early_stopping=True` and `early_stopping_rounds` is left unset, using
  `ceil(5 / lr)` clipped to `20..200`.
* Add `early_stopping_min_delta`, preserving the legacy `1e-9` tolerance by
  default while allowing explicit numeric tolerances or opt-in
  `early_stopping_min_delta="auto"` resolution from baseline validation loss.
  Min-delta controls patience resets; best-prefix selection uses the true
  validation argmin.
* Add opt-in `validation_fraction="auto"` and regression
  `validation_strategy="weighted_stratified"` for automatic validation splits
  that account for effective sample size and weighted target quantiles, with
  feasible-strata caps for small default validation fractions. The realized
  split policy is recorded separately from the requested strategy.
* Raise the default boosting budget to `iterations=1000` and the default
  numeric bin budget to `max_bins=254`.
* Expose `auto_params_` on fitted boosters and preserve it through `.npz`
  serialization. The metadata records resolved learning rate, effective sample
  size, feature counts, tree sizing, regularization, binning, early stopping,
  sampling, validation split policy, target statistics, and threading context
  from the last fit.
* Preserve scalar refit wrapper metadata through `.npz` save/load and mark the
  fold-selection model as intentionally non-persistent with
  `selection_model_persisted_=False` on loaded wrappers.
* Add `auto_params_["diagnostics"]` for low effective sample size and automatic
  learning-rate clipping, with throttled runtime warnings controlled by
  `diagnostic_warnings={"once","always","never"}`. Diagnostics also record
  weighted-binning activation, observed bin counts, feature expansion, and the
  best-prefix policy.
* Add opt-in auto structure defaults (`depth`, `num_leaves`, `l2_leaf_reg`,
  `min_child_samples`, `min_child_weight`, and `cat_smoothing`) with resolved
  values recorded under `auto_params_["auto_structure"]`; refit helpers freeze
  these resolved values along with the resolved learning rate.
* Add opt-in sklearn-wrapper learning-rate probing with
  `auto_learning_rate_probe=True`, recording candidate scores, the selected
  explicit learning rate, and the final-budget automatic base rate under
  `auto_params_["learning_rate_probe"]`.
* Add opt-in CatBoost-like stochastic regularization: Bayesian bootstrap
  (`bootstrap_type="bayesian"` / `bagging_temperature`), MVS row sampling
  (`sampling="mvs"` / `mvs_reg`), and deterministic split-score noise
  (`random_strength`). The resolved settings and per-fit sampling diagnostics
  are recorded under `auto_params_["stochastic_regularization"]` and persisted
  through `.npz` save/load.
* Add `sampling="weighted_goss"` as an opt-in sample-weight-aware GOSS variant;
  existing `sampling="goss"` behavior is unchanged.
* Make numeric bin construction respect non-uniform `sample_weight` through
  weighted quantile borders. `sample_weight=None` and all-ones weights keep the
  previous unweighted behavior.
* Add sklearn-wrapper refit helpers: `get_refit_params()`, selected-round and
  resolved-learning-rate properties, and optional `refit=True` full-data refits
  after early-stopping selection.
