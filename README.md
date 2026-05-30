# chimeraboost
### What if CatBoost, but 30x faster, slightly worse, and all in Python?

> ⚠️ **Project is in active development:** breaking changes should be expected.

<center>
<img width="500" height="500" alt="chimeraboost logo" src="https://github.com/user-attachments/assets/ee98a4e2-9fa7-4ef1-9e64-e398f398966c" />
</center>

* **Installation**

```
pip install chimeraboost
```

* **Sample code:**

```python
from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

# classification
clf = ChimeraBoostClassifier(early_stopping=True)
clf.fit(X, y, cat_features=[0, 1], sample_weight=w)
proba = clf.predict_proba(X_test)

# regression (RMSE, MAE, or Quantile)
reg = ChimeraBoostRegressor(loss="Quantile", alpha=0.9, early_stopping=True)
reg.fit(X, y)
```

<p><a href="https://github.com/bbstats/chimeraboost/blob/main/images/summary.png"><img src="https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/summary.png" width="500" alt="Benchmark summary" /></a></p>
<p><a href="https://github.com/bbstats/chimeraboost/blob/main/images/slowdown_hist.png"><img src="https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/slowdown_hist.png" width="500" alt="Slowdown distribution" /></a></p>

<sub><i><code>ChimeraBoostEns10</code> is ChimeraBoost bagged with 10 base gradient boosters.</i></sub>

* **Reproduce the benchmark**

```
python benchmarks/run_benchmarks.py --openml --seeds 5 --save --models ChimeraBoost ChimeraBoostEns10 sklearn_HGB CatBoost LightGBM
```

* **What?**
    * Exceedingly opinionated GBDT library that only depends on common Python libraries
        * Accepts categorical features, with catboost-like feature processing
        * Bagging as a first-class feature
        * Automatic early stopping, with automatic grouped splitting for the validation set available
    * Supports regression, quantile regression, binary and multiclass classification.
    * Categorical features, sample weights, and automatic early stopping
    * Within ~3% F1 / ~5% RMSE of CatBoost on a 34-dataset OpenML benchmark, at ~30× the speed

* **Why?**
    * I want to be able to modify my GBDT library at will
    * I know Python and I don't know C