"""
Stage-1 capacity GATE for Alternative A (discrete logical-AND splits).

AND splits are functionally "depth-2d interaction capacity at depth-d leaf count".
So before writing the split-search code we test the cheaper question directly:
does raw extra DEPTH close the gap on the targets? If yes -> interaction capacity
is the bind and AND splits (which buy it efficiently) are justified. If extra
depth does NOTHING, capacity isn't the lever and A won't help -> stop.

Targets:
  (A) pol  (gr:reg_num/pol)  — our one genuine remaining regression underfit (§6;
      memory: depth-10+mcw=0 took it 79%->111% of sklearn). RMSE, lower=better.
  (B) Family A v2 (multi-pocket, 8 informative feats > depth-6) — controlled.
      excess Brier over Bayes, lower=better.

Report-only Stage 1. Nothing ships off this; it only green-lights/kills building A.

Usage:  python benchmarks/stage1_capacity_gate.py [--pol-seeds 5] [--syn-seeds 8]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benchmarks as rb  # noqa: E402
from synthetic import family_a_multi_pocket, _bayes_brier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

DEPTHS = [6, 8, 10, 12]


def _mean_sd(xs):
    return float(np.nanmean(xs)), float(np.nanstd(xs))


def gate_pol(seeds):
    rb._add_grinsztajn_datasets()
    X, y, cat, task = rb.DATASETS["gr:reg_num/pol"](1.0, np.random.default_rng(0))
    assert task == "regression"
    print(f"\n[A] pol regression (n={len(y)}) — RMSE (lower=better). "
          f"Does extra depth close the leaf-wise gap?\n")

    # Chimera depth sweep (default mcw) + capacity-ceiling probes (mcw=0) + refs.
    chimera = {f"Chimera d{d}": dict(depth=d) for d in DEPTHS}
    chimera["Chimera d10 mcw=0"] = dict(depth=10, mcw=0.0)
    chimera["Chimera d12 mcw=0"] = dict(depth=12, mcw=0.0)
    refs = ["CatBoost", "LightGBM", "sklearn_HGB"]

    res = {nm: [] for nm in list(chimera) + refs}
    for s in range(seeds):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
        for nm, kw in chimera.items():
            try:
                m, _t, _it = rb._run_chimera(task, Xtr, ytr, Xte, yte, cat, None, **kw)
                res[nm].append(m["rmse"])
            except Exception as e:
                print(f"  ! {nm} seed={s}: {e}"); res[nm].append(np.nan)
        for nm in refs:
            try:
                m, _t, _it = rb.RUNNERS[nm](task, Xtr, ytr, Xte, yte, cat, None)
                res[nm].append(m["rmse"])
            except Exception as e:
                print(f"  ! {nm} seed={s}: {e}"); res[nm].append(np.nan)

    best_ref = min(np.nanmean(res[r]) for r in refs)
    for nm in list(chimera) + refs:
        mu, sd = _mean_sd(res[nm])
        tag = "  <-- best leaf-wise/ref" if abs(mu - best_ref) < 1e-9 else ""
        pct = 100.0 * best_ref / mu if mu else float("nan")
        print(f"  {nm:22s} RMSE {mu:8.4f} +/- {sd:6.4f}   ({pct:5.1f}% of best ref){tag}")
    return res


def gate_family_a(seeds, n=8000):
    print(f"\n[B] Family A v2 (n={n}) — excess Brier (lower=better). "
          f"Does extra depth close the CatBoost gap?\n")
    chimera = {f"Chimera d{d}": dict(depth=d) for d in DEPTHS}
    refs = ["CatBoost", "LightGBM"]
    res = {nm: [] for nm in list(chimera) + refs}
    for s in range(seeds):
        X, y, tp = family_a_multi_pocket(n, seed=3000 + s)
        Xtr, Xte, ytr, yte, _p, pte = train_test_split(
            X, y, tp, test_size=0.2, random_state=3000 + s, stratify=y)
        bayes = _bayes_brier(yte, pte)
        for nm, kw in chimera.items():
            try:
                m, _t, _it = rb._run_chimera("binary", Xtr, ytr, Xte, yte, None, None, **kw)
                res[nm].append(max(0.0, m["brier"] - bayes))
            except Exception as e:
                print(f"  ! {nm} seed={s}: {e}"); res[nm].append(np.nan)
        for nm in refs:
            try:
                m, _t, _it = rb.RUNNERS[nm]("binary", Xtr, ytr, Xte, yte, None, None)
                res[nm].append(max(0.0, m["brier"] - bayes))
            except Exception as e:
                print(f"  ! {nm} seed={s}: {e}"); res[nm].append(np.nan)
    for nm in sorted(list(chimera) + refs, key=lambda k: np.nanmean(res[k])):
        mu, sd = _mean_sd(res[nm])
        print(f"  {nm:14s} excessBrier {mu:.4f} +/- {sd:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol-seeds", type=int, default=5)
    ap.add_argument("--syn-seeds", type=int, default=8)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"Stage-1 capacity gate | depths={DEPTHS}")
    gate_pol(args.pol_seeds)
    gate_family_a(args.syn_seeds)
    print("\nGate done. Interpretation: if higher depth (or depth+mcw=0) "
          "materially closes the gap on a target, interaction capacity is the "
          "lever -> build Alternative A. If flat, A won't help -> stop.")


if __name__ == "__main__":
    main()
