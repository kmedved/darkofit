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
* Raise the default boosting budget to `iterations=1000` and the default
  numeric bin budget to `max_bins=254`.
* Expose `auto_params_` on fitted boosters and preserve it through `.npz`
  serialization. The metadata records resolved learning rate, effective sample
  size, feature counts, tree sizing, regularization, binning, early stopping,
  sampling, and threading context from the last fit.
* Make numeric bin construction respect non-uniform `sample_weight` through
  weighted quantile borders. `sample_weight=None` and all-ones weights keep the
  previous unweighted behavior.
* Add sklearn-wrapper refit helpers: `get_refit_params()`, selected-round and
  resolved-learning-rate properties, and optional `refit=True` full-data refits
  after early-stopping selection.
