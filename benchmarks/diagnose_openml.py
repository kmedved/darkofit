"""Is the CatBoost gap a tuning artifact or a capability gap?

For each dataset, run ChimeraBoost at its default config and at a few cheap
tuning variants (lower learning rate -> more trees; finer bins), using the EXACT
protocol from run_benchmarks.py so the numbers are comparable. Variants are
compared to the baseline with PAIRED per-seed deltas (same train/test split per
seed for baseline and variant), which cancels the large seed-to-seed swing on
small datasets -- the thing that makes single-split gaps look real when they are
not. If CatBoost is installed it is run on the same splits for the gap.

Verdict per lever:
  flat      -> the best variant does not beat baseline by more than paired noise
  responds  -> some variant beats baseline by > 2 standard errors (paired)

If both levers are flat but a CatBoost gap remains, that gap is capability, not
tuning: smaller steps / finer bins won't close it, so spend effort on the
algorithm (feature combinations, true ordered boosting) instead.

Place this file in benchmarks/ next to run_benchmarks.py and run from anywhere:
    python benchmarks/diagnose_openml.py                 # full OpenML suite, 5 seeds
    python benchmarks/diagnose_openml.py --seeds 3
    python benchmarks/diagnose_openml.py --datasets oml:bank-marketing oml:adult
    python benchmarks/diagnose_openml.py --quick         # one lr variant, faster
    python benchmarks/diagnose_openml.py --no-catboost
"""

import argparse
import os
import sys
import time

import numpy as np

# Import the harness and the chimeraboost package regardless of CWD: put both
# the benchmarks dir (for run_benchmarks) and the repo root (for chimeraboost)
# on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import run_benchmarks as B  # noqa: E402

from sklearn.model_selection import train_test_split  # noqa: E402
from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier  # noqa: E402


# --------------------------------------------------------------------------
# A ChimeraBoost runner that mirrors B._run_chimera but also controls max_bins.
# Same internal val split (seed 0), same iterations/patience, same model seed.
# --------------------------------------------------------------------------
def _run_chimera_cfg(task, Xtr, ytr, Xte, yte, cat, threads, lr, max_bins):
    Xf, Xv, yf, yv = B._val_split(Xtr, ytr, task, 0)
    Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    t = time.time()
    m = Est(iterations=B.MAX_ITERS, early_stopping_rounds=B.PATIENCE,
            learning_rate=lr, max_bins=max_bins, ordered_boosting=True,
            thread_count=threads, random_state=0)
    m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
    return B._score(task, yte, m, Xte), time.time() - t, m.best_iteration_


# Config grid. Each entry: (label, lr, max_bins, lever). lr=None -> auto (0.1).
def _build_configs(quick):
    cfgs = [("baseline (lr~0.10, 128b)", None, 128, "baseline")]
    cfgs.append(("lr=0.05", 0.05, 128, "lr"))
    if not quick:
        cfgs.append(("lr=0.03", 0.03, 128, "lr"))
    cfgs.append(("bins=254", None, 254, "bins"))
    return cfgs


def _paired_stats(variant_scores, baseline_scores):
    """Paired per-seed delta in higher-is-better space (B._score is already
    higher-is-better: -RMSE for regression, macro-F1 for classification).
    Returns (mean_delta, std_error_of_mean)."""
    d = np.asarray(variant_scores) - np.asarray(baseline_scores)
    n = d.size
    if n == 0:
        return 0.0, 0.0
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else abs(d[0])
    return float(d.mean()), float(se)


def _lever_verdict(best_delta, best_se):
    """flat unless the best variant for this lever beats baseline by > 2 SE."""
    if best_delta > 0 and best_delta > 2.0 * best_se:
        return "RESPONDS"
    return "flat"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="one lr variant instead of two (faster)")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="dataset keys to run (default: the OpenML suite). "
                         "Accepts synthetic keys too, e.g. cat_binary.")
    ap.add_argument("--no-catboost", dest="catboost", action="store_false",
                    default=True, help="skip the CatBoost gap column")
    ap.add_argument("--no-warmup", action="store_true",
                    help="include first-call ChimeraBoost Numba compile time")
    args = ap.parse_args()

    # Register oml:* datasets if needed.
    requested = args.datasets
    need_openml = (requested is None) or any(d.startswith("oml:") for d in requested)
    if need_openml:
        B._add_openml_datasets()
    if requested is None:
        requested = [k for k in B.DATASETS if k.startswith("oml:")]

    unknown = [d for d in requested if d not in B.DATASETS]
    if unknown:
        ap.error(f"unknown datasets: {unknown}\navailable: {list(B.DATASETS)}")

    if not args.no_warmup:
        print("Warming up ChimeraBoost Numba kernels...")
        B._warmup_chimera(args.threads)

    have_cb = args.catboost and B._has_competitor("catboost")
    configs = _build_configs(args.quick)

    print(f"seeds={args.seeds}  threads={args.threads or 'all'}  "
          f"max_iter={B.MAX_ITERS}  patience={B.PATIENCE}  "
          f"catboost={'yes' if have_cb else 'no'}")
    print("paired deltas vs baseline; positive delta = better (lower RMSE / "
          "higher F1); RESPONDS = best variant beats baseline by > 2 standard "
          "errors\n")

    suite_verdict = []

    for ds in requested:
        builder = B.DATASETS[ds]
        # peek task
        try:
            _, _, _, task = builder(1.0, np.random.default_rng(0))
        except Exception as e:
            print(f"### {ds}: SKIPPED (load failed: {e})\n")
            continue

        # Collect per-seed scores for every config (and CatBoost), on shared splits.
        scores = {c[0]: [] for c in configs}
        iters = {c[0]: [] for c in configs}
        cb_scores, cb_iters = [], []

        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            X, y, cat, task = builder(1.0, rng)
            strat = y if task != "regression" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=s, stratify=strat)
            for label, lr, mb, _lever in configs:
                sc, _secs, it = _run_chimera_cfg(
                    task, Xtr, ytr, Xte, yte, cat, args.threads, lr, mb)
                scores[label].append(sc)
                iters[label].append(it)
            if have_cb:
                out = B._run_catboost(task, Xtr, ytr, Xte, yte, cat, args.threads)
                if out is not None:
                    cb_scores.append(out[0])
                    cb_iters.append(out[2])

        # ---- report ----
        metric = {"regression": "RMSE (lower better)"}.get(task, "F1 macro (higher better)")
        print(f"### {ds}  [{task}]  metric={metric}")
        base_label = configs[0][0]
        base = scores[base_label]

        def disp(v):  # show RMSE as positive
            return -np.mean(v) if task == "regression" else np.mean(v)

        for label, lr, mb, lever in configs:
            mu = disp(scores[label])
            sd = (np.std(scores[label]))
            it = int(np.mean(iters[label]))
            if lever == "baseline":
                print(f"  {label:24s} {mu:9.4f} +/- {sd:.4f}  trees~{it}")
            else:
                d, se = _paired_stats(scores[label], base)
                # delta sign is in higher-is-better space already
                arrow = "+" if d > 0 else ""
                print(f"  {label:24s} {mu:9.4f} +/- {sd:.4f}  trees~{it}"
                      f"   delta {arrow}{d:.4f} +/- {se:.4f}")

        # per-lever verdicts: best paired delta among that lever's variants
        lever_best = {}
        for label, lr, mb, lever in configs:
            if lever == "baseline":
                continue
            d, se = _paired_stats(scores[label], base)
            cur = lever_best.get(lever)
            if cur is None or d > cur[0]:
                lever_best[lever] = (d, se)
        lever_strs = []
        for lever in ("lr", "bins"):
            if lever in lever_best:
                d, se = lever_best[lever]
                lever_strs.append(f"{lever}={_lever_verdict(d, se)}")

        # CatBoost gap (paired against best ChimeraBoost variant)
        gap_str = ""
        if have_cb and cb_scores:
            cb_mu = disp(cb_scores)
            print(f"  {'CatBoost':24s} {cb_mu:9.4f} +/- {np.std(cb_scores):.4f}"
                  f"  trees~{int(np.mean(cb_iters))}")
            # best chimera config by mean score (higher-is-better space)
            best_label = max(scores, key=lambda L: np.mean(scores[L]))
            d_cb, se_cb = _paired_stats(scores[best_label], cb_scores)
            # d_cb > 0 means our best beats CatBoost
            if d_cb > -2.0 * se_cb:
                gap_str = "matched/ahead of CatBoost (within paired noise)"
            else:
                # express residual gap in the natural metric for readability
                resid = abs(np.mean(scores[best_label]) - np.mean(cb_scores))
                tuned = best_label != base_label
                gap_str = (f"CatBoost ahead by {resid:.4f} "
                           f"({'best variant still behind' if tuned else 'untuned'})")

        verdict_line = "  verdict: " + ", ".join(lever_strs)
        if gap_str:
            verdict_line += f"  ->  {gap_str}"
        print(verdict_line + "\n")
        suite_verdict.append((ds, dict(lever_best), gap_str))

    # ---- suite summary ----
    print("=" * 70)
    print("SUITE SUMMARY")
    print("=" * 70)
    for ds, lb, gap in suite_verdict:
        levers = ", ".join(f"{k}={_lever_verdict(*v)}" for k, v in lb.items())
        print(f"  {ds:22s} {levers:24s} {gap}")
    print()
    n_flat = sum(
        all(_lever_verdict(*v) == "flat" for v in lb.values())
        for _ds, lb, _g in suite_verdict)
    print(f"{n_flat}/{len(suite_verdict)} datasets: both levers flat "
          f"(remaining gap is capability, not tuning).")


if __name__ == "__main__":
    main()
