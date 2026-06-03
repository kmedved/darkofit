"""Bagged-Grinsztajn proxy: does the linear-leaf gain SURVIVE outer bagging?

TabArena scores ChimeraBoost wrapped in AutoGluon's outer bag (~K CV-fold copies
averaged). A bias reduction (linear leaves add capacity) should pass through the
bag; a variance reduction would be absorbed by it. This compares BAGGED constant
vs BAGGED linear leaves on Grinsztajn -- if linear still helps when BOTH are
bagged, the gain is bias and will likely move TabArena, without touching the
sealed holdout. (Uses our own n_ensembles as the proxy for the outer bag.)
"""
import sys
import numpy as np
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.model_selection import train_test_split

sys.argv = ["x"]
try:
    import benchmarks.run_benchmarks as rb
except ModuleNotFoundError:
    import run_benchmarks as rb
from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier

CASES = [("gr:clf_num/electricity", "binary"),
         ("gr:clf_num/MiniBooNE", "binary"),
         ("gr:clf_num/covertype", "binary"),
         ("gr:reg_num/pol", "regression"),
         ("gr:reg_num/abalone", "regression")]
SEEDS = [0, 1]
N_BAG = 5
LAM = 1.0


def _eval(key, task, ll):
    rng = np.random.default_rng(0)
    out = []
    for s in SEEDS:
        X, y, cat, tt = rb.DATASETS[key](True, rng)
        X = np.asarray(X, dtype=float)
        strat = y if task == "binary" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                              random_state=s, stratify=strat)
        Est = ChimeraBoostClassifier if task == "binary" else ChimeraBoostRegressor
        kw = dict(n_ensembles=N_BAG, ensemble_n_jobs=N_BAG, random_state=s)
        if ll:
            kw.update(linear_leaves=True, linear_lambda=LAM)
        m = Est(**kw).fit(Xtr, ytr)
        if task == "binary":
            yb = (np.asarray(yte) == m.classes_[1]).astype(int)
            out.append(brier_score_loss(yb, m.predict_proba(Xte)[:, 1]))
        else:
            out.append(mean_squared_error(yte, m.predict(Xte)) ** 0.5)
    return float(np.mean(out))


def main():
    rb._add_grinsztajn_datasets()
    print(f"=== BAGGED (n={N_BAG}) constant vs linear leaves ===")
    print(f"{'dataset':>14} {'metric':>6} {'bag_const':>10} {'bag_linear':>11} {'delta%':>8}")
    deltas = []
    for key, task in CASES:
        metric = "Brier" if task == "binary" else "RMSE"
        c = _eval(key, task, False)
        l = _eval(key, task, True)
        d = 100.0 * (c - l) / c
        deltas.append(d)
        print(f"{key.split('/')[-1]:>14} {metric:>6} {c:>10.5f} {l:>11.5f} {d:>+8.2f}")
    wins = sum(1 for d in deltas if d > 0.1)
    print(f"\n  -> survives-the-bag wins {wins}/{len(CASES)}  mean delta {np.mean(deltas):+.2f}%")
    print("  (positive = linear leaves STILL help after bagging = bias = TabArena-relevant)")


if __name__ == "__main__":
    main()
