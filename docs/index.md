# ChimeraBoost

Gradient boosting on oblivious (symmetric) decision trees, written in Python with a
[numba](https://numba.pydata.org/) backend. It depends only on NumPy, scikit-learn,
SciPy, and pandas — no C++ extensions and no build step, so you can read and modify
every line.

> *What if CatBoost was slightly worse, 12× faster, and all in Python?*

```bash
pip install chimeraboost
```

```python
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

clf = ChimeraBoostClassifier(random_state=0)
clf.fit(X_train, y_train, cat_features=[0, 1])
proba = clf.predict_proba(X_test)
```

New here? Start with [Getting started](getting-started.md) — a runnable walkthrough with
real output.

## What it does

- Regression (squared error, absolute error, quantile), binary and multiclass classification.
- Categorical features via ordered target statistics — pass column indices, no manual encoding.
- Missing values handled directly (NaN routes to its own bin), no imputation.
- Sample weights, bagging (`n_ensembles`), and grouped validation splits.
- Calibrated probabilities (`predict_proba` is temperature-scaled on the validation split).
- Exact SHAP attributions ([`shap_values`](shap.md)), including the per-leaf linear models.
- A scikit-learn estimator API that drops into `Pipeline`, `GridSearchCV`, and `cross_val_score`.

## Benchmarks

[![TabArena-Lite Elo vs training time](https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/tabarena_pareto.png){ width="560" }](https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/tabarena_pareto.png)

On TabArena-Lite, ChimeraBoost sits on the accuracy-vs-training-time Pareto frontier:
ahead of XGBoost and LightGBM defaults on both axes, and within reach of CatBoost's
accuracy at a fraction of its training time.

## Documentation

- [Getting started](getting-started.md) — install and a runnable end-to-end example.
- [Recipes](recipes.md) — worked examples for every task.
- [How it works](concepts.md) — the design behind the defaults.
- [Parameters](parameters.md) — what each option does and when to change it.
- [SHAP](shap.md) — exact feature attributions.
- [API reference](api.md) — classes, methods, and signatures.
- [FAQ](faq.md)
