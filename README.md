# chimeraboost
Full-featured gradient boosting library with a Python/numba backend, inspired by CatBoost.

<img width="500" height="500" alt="ChatGPT Image May 26, 2026, 05_12_17 PM" src="https://github.com/user-attachments/assets/ee98a4e2-9fa7-4ef1-9e64-e398f398966c" />

* **What?**
    * GBDT library that only depends on numpy, numba, and scikit-learn
    * Near equivalent performance to CatBoost on CPU (~97% R^2 / F1 in benchmarks), at 20x the speed
    * Supports sample weights and automatic early stopping

* **Why?**
    * I want to be able to modify my GBDT library at will
    * I know Python and I don't know C

* **How?**

```
# pip install chimeraboost

from chimeraboost import ChimeraBoostClassifier
clf = ChimeraBoostClassifier(early_stopping=True)
clf.fit(X, y, sample_weight=w)
```

Models can be saved without pickle:

```
clf.save_model("model.npz")
clf2 = ChimeraBoostClassifier.load_model("model.npz")
```

After fitting, `model_.auto_params_` records the resolved training context:
the actual learning rate, effective sample size, feature counts, tree sizing,
regularization, binning policy, early-stopping policy, sampling policy, and
threading choice used by that fit.

```
clf.model_.auto_params_["learning_rate"]
clf.model_.auto_params_["binning"]
```

When `learning_rate=None` or `"auto"`, ChimeraBoost uses a transparent
CatBoost-form rule fitted as `log(lr) ~ log(n) + log(iterations)`, with Kish
effective sample size replacing raw row count when `sample_weight` is supplied.
For materially weighted RMSE fits in CatBoost/oblivious-tree mode, the selector
applies an LR-only ChimeraBoost correction; unweighted and all-ones-weight fits
keep the raw CatBoost-form value.
LightGBM-mode unweighted fits apply an additional provisional dampening factor
because the CatBoost coefficients were fitted on symmetric-tree regularization,
not ChimeraBoost's leaf-wise stack.
The default boosting budget is `iterations=1000`, and the default numeric bin
budget is `max_bins=254`.

Tree builders are selectable:

```
ChimeraBoostClassifier(tree_mode="catboost")  # symmetric/oblivious default
ChimeraBoostClassifier(tree_mode="lightgbm")  # leaf-wise, non-oblivious
```

Tree modes:

* `tree_mode="catboost"` builds symmetric / oblivious trees and supports
  ordered boosting.
* `tree_mode="lightgbm"` builds ChimeraBoost's LightGBM-like histogram trees:
  non-oblivious, leaf-wise, best-first CART-style trees. This is not model or
  prediction compatibility with Microsoft LightGBM.

In LightGBM mode, `num_leaves` is the main tree-size control and `depth` is a
maximum path-depth cap. `ordered_boosting` defaults to off for this mode; setting
`ordered_boosting=True` with `tree_mode="lightgbm"` raises a `ValueError`.
Categorical features still use ChimeraBoost's ordered target-stat preprocessing,
not native LightGBM category-partition splits.

Row sampling is selectable with `sampling="uniform"` (default) or
`sampling="goss"` plus `top_rate` / `other_rate`.

Large-fit preprocessing samples up to 200,000 rows when learning numeric bin
borders, similar to LightGBM's `bin_construct_sample_cnt`. Set
`bin_sample_count=None` to recover exact full-data border learning.
When `sample_weight` is supplied, those numeric borders use weighted quantiles;
`sample_weight=None` and all-ones weights preserve the unweighted border path.

Early-stopping selection can optionally be followed by a full-data refit:

```
clf = ChimeraBoostClassifier(
    early_stopping=True,
    refit=True,
    refit_strategy="exact",
)
clf.fit(X, y)
```

`get_refit_params()` returns the frozen parameters for a manual full-data refit:
it disables early stopping, uses the selected round count, and freezes the
resolved learning rate. Strategies `"sqrt"` and `"linear"` scale the selected
round count by the automatic validation split ratio; `"scaled"` aliases
`"linear"`.

Training loss is evaluated every round by default for diagnostics. Set
`eval_train_loss=False` to skip that pass when you only care about the fitted
model or validation-set early stopping; validation loss and early stopping are
unchanged.

`histogram_parallelism="row"` enables an experimental row-parallel histogram
builder. The default `"auto"` keeps the measured-best feature-parallel path on
the current benchmark machine; use the row-parallel lane only when profiling
shows it helps your hardware.

Not implemented in `tree_mode="lightgbm"`: native LightGBM categorical splits,
DART, GPU training, sparse optimization, monotone constraints, ranking, custom
objectives, custom eval metrics, or LightGBM model import/export.

Benchmark notes and fair comparison recipes live in
[BENCHMARK_NOTES.md](BENCHMARK_NOTES.md).



* **To Do:**
    * Update multi-class classification loss scheme
