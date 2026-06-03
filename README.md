# chimeraboost
### What if CatBoost was slightly worse, 12× faster, and all in Python?

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
clf = ChimeraBoostClassifier(early_stopping=True, n_ensembles=2)
clf.fit(X, y, cat_features=[0, 1], sample_weight=w)
proba = clf.predict_proba(X_test)

# regression (RMSE, MAE, or Quantile)
reg = ChimeraBoostRegressor(loss="Quantile", alpha=0.9, early_stopping=True, n_ensembles=10)
reg.fit(X, y)
```

<p><a href="https://github.com/bbstats/chimeraboost/blob/main/images/tabarena_pareto.png"><img src="https://raw.githubusercontent.com/bbstats/chimeraboost/main/images/tabarena_pareto.png" width="500" alt="TabArena-Lite Elo vs speed Pareto" /></a></p>

* **What?**
    * Exceedingly opinionated GBDT library that only depends on common Python libraries
        * Categorical features (catboost-like processing) and sample weights
        * Bagging as a first-class feature (`n_ensembles`)
        * Automatic early stopping, with optional grouped splitting for the validation set
    * Supports regression, quantile regression, binary and multiclass classification
    * On **TabArena-Lite** (default configs, 51 tasks): beats XGBoost and LightGBM on **both** Elo and speed, and trails only CatBoost (−~10% Elo) at **~12× its speed** — on the strength-vs-speed Pareto frontier (chart above)

* **Tuning tips**
    * Interaction-heavy regression: raise `depth` to 8–10 (default 6 is conservative to protect small data).

* **Inspirations / Citations**
    * **CatBoost** — Prokhorenkova et al., *NeurIPS* 2018 — ordered boosting, ordered target statistics, oblivious trees
    * **Gradient boosting** — Friedman, *Annals of Statistics* 2001 — the GBM framework
    * **XGBoost** — Chen & Guestrin, *KDD* 2016 — regularized objective, Newton leaf estimation, column subsampling
    * **LightGBM** — Ke et al., *NeurIPS* 2017 — histogram-based split finding
    * **TabArena** — Erickson et al., *NeurIPS* 2025 (arXiv:2506.16791) — the benchmark

* **Why?**
    * I want to be able to modify my GBDT library at will
    * I know Python and I don't know C
