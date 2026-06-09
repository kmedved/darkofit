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

Not implemented in `tree_mode="lightgbm"`: native LightGBM categorical splits,
GOSS, DART, GPU training, sparse optimization, monotone constraints, ranking,
custom objectives, custom eval metrics, or LightGBM model import/export.



* **To Do:**
    * Update multi-class classification loss scheme
