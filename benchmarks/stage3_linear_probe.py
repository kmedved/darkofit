"""Derisking probe for linear leaf models (Option B): does a linear component
help our UNDERFIT regression datasets at all?

Linear leaves are a big build (touches the forest predictor). Before that, this
checks the weaker, strictly-easier hypothesis with ZERO core changes: boost off
a regularized linear init (Ridge) instead of the mean. If even a single global
linear term moves the tree-wall / smooth-underfit cluster (pol, sulfur, ...),
per-leaf linear models (more capacity, local slopes) should help more, and the
build is justified. If a linear component does nothing here, reconsider B.

Pure-Python: sklearn Ridge is already a ChimeraBoost dependency.
"""
import sys
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.argv = ["x"]
try:
    import benchmarks.run_benchmarks as rb
except ModuleNotFoundError:
    import run_benchmarks as rb
from chimeraboost import ChimeraBoostRegressor

# underfit / tree-wall cluster + two controls (cpu_act, houses) where linear
# structure is weak so we expect ~no help.
PANEL = ["gr:reg_num/pol", "gr:reg_num/sulfur", "gr:reg_num/superconduct",
         "gr:reg_num/Brazilian_houses", "gr:reg_num/cpu_act", "gr:reg_num/houses"]
SEEDS = [0, 1, 2]


def _ridge_init(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(Xtr), ytr)
    return r.predict(sc.transform(Xtr)), r.predict(sc.transform(Xte))


def main():
    rb._add_grinsztajn_datasets()
    rng = np.random.default_rng(0)
    print(f"{'dataset':>16} {'base_RMSE':>10} {'lininit':>10} {'ridge':>10} {'delta%':>8}")
    agg = {"base": [], "lin": []}
    for key in PANEL:
        b_list, l_list = [], []
        for s in SEEDS:
            X, y, cat, tt = rb.DATASETS[key](True, rng)
            X = np.asarray(X, dtype=float)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                                  random_state=s)
            base = ChimeraBoostRegressor(random_state=s, thread_count=4)
            base.fit(Xtr, ytr)
            b = mean_squared_error(yte, base.predict(Xte)) ** 0.5

            ptr, pte = _ridge_init(Xtr, ytr, Xte)
            cb = ChimeraBoostRegressor(random_state=s, thread_count=4)
            cb.fit(Xtr, ytr - ptr)
            lin = mean_squared_error(yte, pte + cb.predict(Xte)) ** 0.5
            ridge_only = mean_squared_error(yte, pte) ** 0.5

            b_list.append(b); l_list.append(lin)
        bm, lm = np.mean(b_list), np.mean(l_list)
        agg["base"].append(bm); agg["lin"].append(lm)
        delta = 100.0 * (bm - lm) / bm   # positive = linear init better
        print(f"{key.split('/')[-1]:>16} {bm:>10.4f} {lm:>10.4f} "
              f"{ridge_only:>10.4f} {delta:>+8.2f}")

    wins = sum(1 for b, l in zip(agg["base"], agg["lin"]) if l < b - 1e-9)
    print(f"\nlinear-init wins {wins}/{len(PANEL)} on RMSE "
          f"(positive delta% = linear init helps)")


if __name__ == "__main__":
    main()
