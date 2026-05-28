"""Is the residual CatBoost gap bias or variance?

The lr/bins diagnostic showed the gap is NOT tuning and NOT categorical -- the
clean gaps (wine_quality, kc1, phoneme) are numeric-only datasets where both
models build oblivious trees and ChimeraBoost is already converged. That leaves
two explanations, with opposite fixes:

  variance  ChimeraBoost overfits more per tree (similar train fit, worse test).
            -> a regularizer (real permutation ordered boosting) could close it.
  bias      ChimeraBoost extracts less signal per fit (lower train AND test).
            -> ordered boosting won't help; the lever is split/leaf quality.

This fits each model on the SAME per-seed split (same internal val carve as the
harness), then reports train fit (on the fitted portion Xf), test, and the
train-test gap for each model. The gap is in higher-is-better space, so a
positive gap = train better than test = overfitting. We paired-test the
DIFFERENCE in gaps between the two models.

    python benchmarks/bias_variance.py --datasets oml:wine_quality oml:kc1 oml:phoneme
    python benchmarks/bias_variance.py --datasets oml:bank-marketing --seeds 8
    python benchmarks/bias_variance.py --self-check     # validate the instrument

Run the failing NUMERIC datasets first -- they are where the gap is real and
where the bias/variance answer is least confounded.
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
from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier  # noqa: E402


def _hib(task, y_true, model, X):
    """Score in higher-is-better space: -RMSE for regression, macro-F1 else.
    Identical convention to B._score, just reused for train data too."""
    return B._score(task, y_true, model, X)


def _fit_chimera(task, Xf, yf, Xv, yv, cat, threads, mcw=1.0):
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    t = time.time()
    m = Est(iterations=B.MAX_ITERS, early_stopping_rounds=B.PATIENCE,
            min_child_weight=mcw, ordered_boosting=True,
            thread_count=threads, random_state=0)
    m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
    return m, time.time() - t


def _fit_catboost(task, Xf, yf, Xv, yv, cat, threads):
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
    except Exception:
        return None, 0.0
    Est = CatBoostRegressor if task == "regression" else CatBoostClassifier
    t = time.time()
    m = Est(iterations=B.MAX_ITERS, early_stopping_rounds=B.PATIENCE,
            thread_count=threads or -1, verbose=False, random_seed=0)
    m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
    return m, time.time() - t


def _paired(a, b):
    """mean and standard error of the paired difference a - b."""
    d = np.asarray(a) - np.asarray(b)
    n = d.size
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else (abs(d[0]) if n else 0.0)
    return float(d.mean()), float(se)


def _disp(task, hib_mean):
    """Show a higher-is-better mean in natural units (RMSE positive)."""
    return -hib_mean if task == "regression" else hib_mean


def _collect(builder, seeds, threads, have_cb, mcw_pair=None):
    """Return per-seed train/test arrays for each model on shared splits.

    mcw_pair: optional (mcw_a, mcw_b) to run TWO ChimeraBoost configs instead of
    ChimeraBoost vs CatBoost -- used by --self-check to prove the gap readout
    distinguishes an overfit model from a regularized one.
    """
    out = {}

    def add(model_key, tr, te):
        out.setdefault(model_key, {"train": [], "test": []})
        out[model_key]["train"].append(tr)
        out[model_key]["test"].append(te)

    task_seen = None
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        X, y, cat, task = builder(1.0, rng)
        task_seen = task
        strat = y if task != "regression" else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=s, stratify=strat)
        Xf, Xv, yf, yv = B._val_split(Xtr, ytr, task, 0)

        if mcw_pair is not None:
            for tag, mcw in zip(("mcw=1 (loose)", f"mcw={mcw_pair[1]} (tight)"),
                                mcw_pair):
                m, _ = _fit_chimera(task, Xf, yf, Xv, yv, cat, threads, mcw=mcw)
                add(tag, _hib(task, yf, m, Xf), _hib(task, yte, m, Xte))
        else:
            m, _ = _fit_chimera(task, Xf, yf, Xv, yv, cat, threads)
            add("ChimeraBoost", _hib(task, yf, m, Xf), _hib(task, yte, m, Xte))
            if have_cb:
                mc, _ = _fit_catboost(task, Xf, yf, Xv, yv, cat, threads)
                if mc is not None:
                    add("CatBoost", _hib(task, yf, mc, Xf), _hib(task, yte, mc, Xte))
    return task_seen, out


def _report(name, task, data):
    print(f"### {name}  [{task}]")
    rows = {}
    for model, d in data.items():
        tr = np.array(d["train"]); te = np.array(d["test"])
        gap = tr - te                         # higher-is-better: +gap = overfit
        rows[model] = (tr, te, gap)
        print(f"  {model:22s} train {_disp(task, tr.mean()):8.4f}   "
              f"test {_disp(task, te.mean()):8.4f}   "
              f"gap {gap.mean():+.4f} +/- {gap.std():.4f}")
    return rows


def _verdict_two_models(task, rows, a, b):
    """Compare model a to baseline b. Returns a one-line diagnosis."""
    tr_a, te_a, gap_a = rows[a]
    tr_b, te_b, gap_b = rows[b]
    d_test, se_test = _paired(te_a, te_b)        # a test - b test
    d_train, se_train = _paired(tr_a, tr_b)      # a train - b train
    d_gap, se_gap = _paired(gap_a, gap_b)        # a gap - b gap

    if d_test > -2 * se_test:
        return (f"{a} matches/beats {b} on test "
                f"(delta {d_test:+.4f} +/- {se_test:.4f}) -- no gap to explain")

    # a is behind on test. Bias or variance?
    train_behind = d_train < -2 * se_train       # a fits training notably worse
    overfits_more = d_gap > 2 * se_gap           # a's train-test gap notably wider

    if train_behind and not overfits_more:
        return (f"BIAS: {a} fits training worse too "
                f"(train delta {d_train:+.4f} +/- {se_train:.4f}); a regularizer "
                f"like ordered boosting won't help -- look at split/leaf quality")
    if overfits_more and not train_behind:
        return (f"VARIANCE: {a} fits training as well but generalizes worse "
                f"(gap delta {d_gap:+.4f} +/- {se_gap:.4f}); a regularizer "
                f"(real ordered boosting) is the candidate -- validate on the "
                f"prediction-shift synthetic before building it")
    if overfits_more and train_behind:
        return (f"MIXED: {a} both fits less and overfits more "
                f"(train {d_train:+.4f}, gap {d_gap:+.4f}) -- inconclusive, "
                f"raise --seeds")
    return (f"UNCLEAR: {a} behind on test ({d_test:+.4f}) but neither train fit "
            f"nor gap separates at 2 SE -- raise --seeds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="dataset keys (default: the OpenML suite)")
    ap.add_argument("--no-catboost", dest="catboost", action="store_false",
                    default=True)
    ap.add_argument("--self-check", action="store_true",
                    help="validate the instrument: loose vs tight min_child_weight "
                         "on a noisy synthetic; the loose model MUST show the "
                         "wider train-test gap")
    ap.add_argument("--no-warmup", action="store_true",
                    help="include first-call ChimeraBoost Numba compile time")
    args = ap.parse_args()

    if not args.no_warmup:
        print("Warming up ChimeraBoost Numba kernels...")
        B._warmup_chimera(args.threads)

    if args.self_check:
        # A noisy regression where an unconstrained tree can memorize.
        from sklearn.datasets import make_regression

        def builder(scale, rng):
            X, y = make_regression(n_samples=1500, n_features=20, n_informative=8,
                                   noise=40.0, random_state=int(rng.integers(1e9)))
            return X, y, None, "regression"

        task, data = _collect(builder, args.seeds, args.threads, have_cb=False,
                               mcw_pair=(1.0, 30.0))
        rows = _report("self-check (loose vs tight regularization)", task, data)
        loose = "mcw=1 (loose)"; tight = "mcw=30.0 (tight)"
        d_gap, se_gap = _paired(rows[loose][2], rows[tight][2])
        ok = d_gap > 2 * se_gap
        print(f"\n  loose-minus-tight gap difference: {d_gap:+.4f} +/- {se_gap:.4f}")
        print(f"  INSTRUMENT {'OK' if ok else 'INCONCLUSIVE'}: the looser model "
              f"{'shows the wider train-test gap as expected' if ok else 'did not separate -- raise --seeds'}")
        return

    requested = args.datasets
    need_openml = (requested is None) or any(d.startswith("oml:") for d in requested)
    if need_openml:
        B._add_openml_datasets()
    if requested is None:
        requested = [k for k in B.DATASETS if k.startswith("oml:")]
    unknown = [d for d in requested if d not in B.DATASETS]
    if unknown:
        ap.error(f"unknown datasets: {unknown}\navailable: {list(B.DATASETS)}")

    have_cb = args.catboost and (
        args.no_warmup or B._has_competitor("catboost")
    )
    if not have_cb:
        print("NOTE: CatBoost not available -- reporting ChimeraBoost train/test "
              "gap only (no comparison verdict).\n")
    print(f"seeds={args.seeds}  threads={args.threads or 'all'}  "
          f"gap = train - test in higher-is-better space (+ = overfit)\n")

    for ds in requested:
        builder = B.DATASETS[ds]
        try:
            task, data = _collect(builder, args.seeds, args.threads, have_cb)
        except Exception as e:
            print(f"### {ds}: SKIPPED ({e})\n")
            continue
        rows = _report(ds, task, data)
        if have_cb and "CatBoost" in rows:
            print("  verdict: " +
                  _verdict_two_models(task, rows, "ChimeraBoost", "CatBoost"))
        print()


if __name__ == "__main__":
    main()
