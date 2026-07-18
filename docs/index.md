# DarkoFit documentation

DarkoFit is a NumPy/Numba gradient-boosting library for tabular data with
scikit-learn-style estimators. It supports numeric and categorical predictors,
sample weights, early stopping, safe model serialization, distributional
regression, opt-in row/group bootstrap ensembles, local linear leaves, and
exact interventional TreeSHAP for supported oblivious-tree models.

Start with [Getting started](getting-started.md), then use:

- [Parameters](parameters.md) for the main model controls.
- [Core concepts](concepts.md) for trees, categoricals, validation, and
  serialization.
- [Distributional regression](uncertainty.md) for predictive distributions,
  intervals, calibration, and sampling.
- [TreeSHAP](shap.md) for supported explanation paths.
- [Benchmarks](benchmarks.md) for the evidence boundary and current claims.
- [Development](development.md) for local setup and test commands.
- [FAQ](faq.md) for compatibility and troubleshooting.

The root [README](https://github.com/kmedved/darkofit/blob/main/README.md)
remains the compact package overview. Historical
implementation plans and handoffs are indexed under [Archive](archive/README.md);
they are evidence records, not current API documentation.
