"""Tier-2 dev panel for LOCAL linear-leaf models (Option B), reference impl.

Tests the real hypothesis (per-leaf regularized linear models help where step
leaves underfit, with the constant-fallback protecting irregular data) on real
datasets, single-model. Regression RMSE on a smooth/irregular mix, and Brier on
the high-signal binary cluster (our actual weak leg). Reports per-dataset
delta vs constant leaves + a win/loss count. Directional gate before any fast build.
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

REG = ["gr:reg_num/houses", "gr:reg_num/diamonds", "gr:reg_num/medical_charges",
       "gr:reg_num/abalone", "gr:reg_num/elevators",          # smooth-ish
       "gr:reg_num/pol", "gr:reg_num/Brazilian_houses"]       # irregular controls
BIN = ["gr:clf_num/electricity", "gr:clf_num/pol", "gr:clf_num/covertype",
       "gr:clf_num/MiniBooNE", "gr:clf_num/credit"]
SEEDS = [0, 1, 2]
LAMBDAS = [1.0]   # linear_lambda sweep (extend if signal is promising)


def _eval(key, task, hl):
    rng = np.random.default_rng(0)
    rows = []
    for s in SEEDS:
        X, y, cat, tt = rb.DATASETS[key](True, rng)
        X = np.asarray(X, dtype=float)
        strat = y if task == "binary" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                              random_state=s, stratify=strat)
        if task == "binary":
            base = ChimeraBoostClassifier(random_state=s, thread_count=4).fit(Xtr, ytr)
            lin = ChimeraBoostClassifier(linear_leaves=True, linear_lambda=hl,
                                         random_state=s, thread_count=4).fit(Xtr, ytr)
            yb = (np.asarray(yte) == base.classes_[1]).astype(int)
            b = brier_score_loss(yb, base.predict_proba(Xte)[:, 1])
            l = brier_score_loss(yb, lin.predict_proba(Xte)[:, 1])
        else:
            base = ChimeraBoostRegressor(random_state=s, thread_count=4).fit(Xtr, ytr)
            lin = ChimeraBoostRegressor(linear_leaves=True, linear_lambda=hl,
                                        random_state=s, thread_count=4).fit(Xtr, ytr)
            b = mean_squared_error(yte, base.predict(Xte)) ** 0.5
            l = mean_squared_error(yte, lin.predict(Xte)) ** 0.5
        rows.append((b, l))
    arr = np.array(rows)
    return arr[:, 0].mean(), arr[:, 1].mean()


def _panel(keys, task, metric, lam):
    print(f"\n=== {metric}: constant vs linear leaves (linear_lambda={lam}) ===")
    print(f"{'dataset':>18} {'const':>10} {'linear':>10} {'delta%':>8}")
    deltas = []
    for key in keys:
        c, l = _eval(key, task, lam)
        d = 100.0 * (c - l) / c          # positive = linear better
        deltas.append(d)
        print(f"{key.split('/')[-1]:>18} {c:>10.5f} {l:>10.5f} {d:>+8.2f}")
    wins = sum(1 for d in deltas if d > 0.1)
    losses = sum(1 for d in deltas if d < -0.1)
    print(f"  -> wins {wins}  losses {losses}  mean delta {np.mean(deltas):+.2f}%")


def main():
    rb._add_grinsztajn_datasets()
    for lam in LAMBDAS:
        _panel(REG, "regression", "RMSE", lam)
        _panel(BIN, "binary", "Brier", lam)


if __name__ == "__main__":
    main()
