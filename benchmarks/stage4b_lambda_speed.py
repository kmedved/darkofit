"""linear_lambda robustness + reference-impl speed cost for linear leaves.

Confirms the Tier-2 gain isn't a fluke at lambda=1, and measures the fit-time
multiple of the (unoptimized, reference) linear-leaf path vs constant leaves --
so we know how much a fused predictor + faster leaf solve must claw back before
linear leaves could be a default.
"""
import sys
import time
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
         ("gr:reg_num/pol", "regression"),
         ("gr:reg_num/abalone", "regression")]
LAMBDAS = [0.3, 1.0, 3.0, 10.0]
SEEDS = [0, 1]


def _fit_eval(key, task, ll, lam, seed):
    rng = np.random.default_rng(0)
    X, y, cat, tt = rb.DATASETS[key](True, rng)
    X = np.asarray(X, dtype=float)
    strat = y if task == "binary" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=seed, stratify=strat)
    Est = ChimeraBoostClassifier if task == "binary" else ChimeraBoostRegressor
    kw = dict(random_state=seed, thread_count=4)
    if ll:
        kw.update(linear_leaves=True, linear_lambda=lam)
    t0 = time.time()
    m = Est(**kw).fit(Xtr, ytr)
    secs = time.time() - t0
    if task == "binary":
        yb = (np.asarray(yte) == m.classes_[1]).astype(int)
        score = brier_score_loss(yb, m.predict_proba(Xte)[:, 1])
    else:
        score = mean_squared_error(yte, m.predict(Xte)) ** 0.5
    return score, secs


def main():
    rb._add_grinsztajn_datasets()
    for key, task in CASES:
        metric = "Brier" if task == "binary" else "RMSE"
        print(f"\n=== {key.split('/')[-1]} ({metric}) ===")
        base = np.mean([_fit_eval(key, task, False, None, s) for s in SEEDS], axis=0)
        print(f"  const           {metric} {base[0]:.5f}   fit {base[1]:.2f}s")
        for lam in LAMBDAS:
            res = np.mean([_fit_eval(key, task, True, lam, s) for s in SEEDS], axis=0)
            d = 100.0 * (base[0] - res[0]) / base[0]
            print(f"  linear lam={lam:<4} {metric} {res[0]:.5f}   fit {res[1]:.2f}s "
                  f"({res[1]/base[1]:.1f}x)   delta {d:+.2f}%")


if __name__ == "__main__":
    main()
