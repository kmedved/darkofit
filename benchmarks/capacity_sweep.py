"""Can more per-tree capacity close the BIAS gap to CatBoost?

bias_variance.py showed the numeric gap is bias: DarkoFit fits TRAINING
worse than CatBoost with equal trees. Container probes then found the mechanical
cause -- min_child_weight=1.0 truncates oblivious trees well below the requested
depth (depth-6 trees averaging ~3-4 achieved levels), capping capacity. This
sweep tests whether loosening that floor (and the secondary knobs, depth and l2)
recovers test performance toward CatBoost on the real datasets.

Each variant is paired against the baseline per seed (cancels seed noise). For
every config we report TRAIN fit and ACHIEVED mean depth as well as test, so you
can confirm a test gain is the bias closing (train fit + depth rising), not
noise. CatBoost's train/test is printed as the target line.

    python benchmarks/capacity_sweep.py --datasets oml:wine_quality oml:kc1 oml:phoneme
    python benchmarks/capacity_sweep.py --datasets oml:bank-marketing --seeds 8

CAUTION: this is a capacity experiment, not a new default. min_child_weight=1.0
exists to stop deeper-tree overfitting in the variance regime; loosening it may
hurt small/noisy datasets. Decide any default change on the full sweep
(run_benchmarks.py) and per-dataset, not on these few sets -- the whole point of
the bias finding is that one regime's good default is another's bad one.
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import run_benchmarks as B  # noqa: E402

from sklearn.model_selection import train_test_split  # noqa: E402
from darkofit import DarkoRegressor, DarkoClassifier  # noqa: E402


# (label, depth, l2, min_child_weight, max_bins, lever)
CONFIGS = [
    ("baseline d6 l2=3 mcw=1 b128", 6, 3.0, 1.0, 128, "baseline"),
    ("mcw=0.1",                     6, 3.0, 0.1, 128, "mcw"),
    ("mcw=0.0",                     6, 3.0, 0.0, 128, "mcw"),
    ("depth=8",                     8, 3.0, 1.0, 128, "depth"),
    ("depth=8 mcw=0.1",             8, 3.0, 0.1, 128, "depth+mcw"),
    ("l2=1",                        6, 1.0, 1.0, 128, "l2"),
    ("bins=254",                    6, 3.0, 1.0, 254, "bins"),
    ("bins=254 depth=8",            8, 3.0, 1.0, 254, "bins+depth"),
]


def _mean_depth(est):
    """Mean achieved (not requested) tree depth across the fitted ensemble."""
    trees = est.model_.trees_
    if not trees:
        return 0.0
    if isinstance(trees[0], list):                     # multiclass: rounds of K
        ds = [t.depth for r in trees for t in r]
    else:
        ds = [t.depth for t in trees]
    return float(np.mean(ds))


def _fit_darkofit(task, Xf, yf, Xv, yv, cat, threads, depth, l2, mcw, max_bins):
    Est = DarkoRegressor if task == "regression" else DarkoClassifier
    m = Est(iterations=B.MAX_ITERS, early_stopping_rounds=B.PATIENCE,
            depth=depth, l2_leaf_reg=l2, min_child_weight=mcw, max_bins=max_bins,
            ordered_boosting=True, thread_count=threads, random_state=0)
    m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
    return m


def _fit_catboost(task, Xf, yf, Xv, yv, cat, threads):
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
    except Exception:
        return None
    Est = CatBoostRegressor if task == "regression" else CatBoostClassifier
    m = Est(iterations=B.MAX_ITERS, early_stopping_rounds=B.PATIENCE,
            thread_count=threads or -1, verbose=False, random_seed=0)
    m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
    return m


def _paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = d.size
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else (abs(d[0]) if n else 0.0)
    return float(d.mean()), float(se)


def _disp(task, hib):
    return -hib if task == "regression" else hib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--no-catboost", dest="catboost", action="store_false",
                    default=True)
    ap.add_argument("--no-warmup", action="store_true",
                    help="include first-call DarkoFit Numba compile time")
    args = ap.parse_args()

    requested = args.datasets
    if (requested is None) or any(d.startswith("oml:") for d in requested):
        B._add_openml_datasets()
    if requested is None:
        requested = [k for k in B.DATASETS if k.startswith("oml:")]
    unknown = [d for d in requested if d not in B.DATASETS]
    if unknown:
        ap.error(f"unknown datasets: {unknown}\navailable: {list(B.DATASETS)}")

    if not args.no_warmup:
        print("Warming up DarkoFit Numba kernels...")
        B._warmup_darkofit(args.threads)

    have_cb = args.catboost and (
        args.no_warmup or B._has_competitor("catboost")
    )

    print(f"seeds={args.seeds}  threads={args.threads or 'all'}  "
          f"max_iter={B.MAX_ITERS}  patience={B.PATIENCE}\n"
          "train/test in natural units; depth = achieved mean tree depth; "
          "delta = paired test vs baseline (+ = better)\n")

    for ds in requested:
        builder = B.DATASETS[ds]
        # per-config per-seed accumulators
        tr = {c[0]: [] for c in CONFIGS}
        te = {c[0]: [] for c in CONFIGS}
        dp = {c[0]: [] for c in CONFIGS}
        cb_tr, cb_te = [], []
        try:
            for s in range(args.seeds):
                rng = np.random.default_rng(1000 + s)
                X, y, cat, task = builder(1.0, rng)
                strat = y if task != "regression" else None
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.25, random_state=s, stratify=strat)
                Xf, Xv, yf, yv = B._val_split(Xtr, ytr, task, 0)
                for label, depth, l2, mcw, max_bins, _lever in CONFIGS:
                    m = _fit_darkofit(task, Xf, yf, Xv, yv, cat, args.threads,
                                     depth, l2, mcw, max_bins)
                    tr[label].append(B._score(task, yf, m, Xf))
                    te[label].append(B._score(task, yte, m, Xte))
                    dp[label].append(_mean_depth(m))
                if have_cb:
                    mc = _fit_catboost(task, Xf, yf, Xv, yv, cat, args.threads)
                    if mc is not None:
                        cb_tr.append(B._score(task, yf, mc, Xf))
                        cb_te.append(B._score(task, yte, mc, Xte))
        except Exception as e:
            print(f"### {ds}: SKIPPED ({e})\n")
            continue

        print(f"### {ds}  [{task}]")
        base = CONFIGS[0][0]
        best_label, best_delta, best_se = None, -np.inf, 0.0
        for label, depth, l2, mcw, max_bins, lever in CONFIGS:
            trm = _disp(task, np.mean(tr[label]))
            tem = _disp(task, np.mean(te[label]))
            dpm = np.mean(dp[label])
            if lever == "baseline":
                print(f"  {label:24s} train {trm:8.4f}  test {tem:8.4f}  "
                      f"depth {dpm:.2f}")
            else:
                d, se = _paired(te[label], te[base])
                flag = "  <-- recovers" if d > 2 * se else ""
                print(f"  {label:24s} train {trm:8.4f}  test {tem:8.4f}  "
                      f"depth {dpm:.2f}   delta {d:+.4f} +/- {se:.4f}{flag}")
                if d > best_delta:
                    best_label, best_delta, best_se = label, d, se
        if have_cb and cb_te:
            print(f"  {'CatBoost (target)':24s} train {_disp(task, np.mean(cb_tr)):8.4f}"
                  f"  test {_disp(task, np.mean(cb_te)):8.4f}")
            # how much of the baseline->CatBoost test gap does the best variant close?
            base_te = np.mean(te[base]); cb = np.mean(cb_te)
            gap = cb - base_te  # higher-is-better space; >0 means CatBoost ahead
            if best_label is not None and abs(gap) > 1e-9:
                closed = (np.mean(te[best_label]) - base_te) / gap
                print(f"  -> best variant: {best_label} "
                      f"(closes {100*closed:.0f}% of the CatBoost test gap)")
        print()


if __name__ == "__main__":
    main()
