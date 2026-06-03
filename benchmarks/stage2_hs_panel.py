"""Tier-2 dev panel for hierarchical shrinkage on REAL binary datasets.

Targets the high-signal Brier cluster we trail on (electricity/pol/covertype/
MiniBooNE) plus a couple where overfitting/variance could let an adaptive
shrinker help (credit/heloc). Reports test Brier per dataset across an
hs_lambda sweep and a win/loss count vs the hs=0 baseline. Directional only.
"""
import sys
import time
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

sys.argv = ["x"]
try:
    import benchmarks.run_benchmarks as rb       # run as module from repo root
except ModuleNotFoundError:
    import run_benchmarks as rb                  # run as a file inside benchmarks/
from chimeraboost import ChimeraBoostClassifier

PANEL = ["gr:clf_num/electricity", "gr:clf_num/pol", "gr:clf_num/covertype",
         "gr:clf_num/MiniBooNE", "gr:clf_num/credit", "gr:clf_num/heloc"]
SWEEP = [0.0, 1.0, 4.0]
SEEDS = [0, 1, 2]


def main():
    rb._add_grinsztajn_datasets()
    rng = np.random.default_rng(0)
    # results[ds][hs] = mean test Brier over seeds
    results = {}
    for key in PANEL:
        results[key] = {}
        for hl in SWEEP:
            briers = []
            for s in SEEDS:
                X, y, cat, tt = rb.DATASETS[key](True, rng)
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.25, random_state=s, stratify=y)
                m = ChimeraBoostClassifier(hs_lambda=hl, random_state=s,
                                           thread_count=4)
                m.fit(Xtr, ytr)
                p = m.predict_proba(Xte)[:, 1]
                # classes_ may be strings; map positive class to classes_[1]
                yb = (np.asarray(yte) == m.classes_[1]).astype(int)
                briers.append(brier_score_loss(yb, p))
            results[key][hl] = float(np.mean(briers))

    name_w = max(len(k.split("/")[-1]) for k in PANEL)
    header = f"{'dataset':>{name_w}} " + " ".join(f"{f'hs={h}':>10}" for h in SWEEP)
    print("\n=== Tier-2 HS dev panel: test Brier (lower=better) ===")
    print(header)
    for key in PANEL:
        row = results[key]
        cells = " ".join(f"{row[h]:>10.5f}" for h in SWEEP)
        print(f"{key.split('/')[-1]:>{name_w}} {cells}")

    # Win/loss vs hs=0 per nonzero hs.
    print("\n=== delta vs hs=0 (negative = HS better) + sign count ===")
    for hl in SWEEP[1:]:
        deltas = {k: results[k][hl] - results[k][0.0] for k in PANEL}
        wins = sum(1 for d in deltas.values() if d < -1e-5)
        losses = sum(1 for d in deltas.values() if d > 1e-5)
        print(f"hs={hl}:  wins {wins}  losses {losses}  "
              f"mean delta {np.mean(list(deltas.values())):+.5f}")
        for k, d in deltas.items():
            print(f"   {k.split('/')[-1]:>{name_w}} {d:+.5f}")


if __name__ == "__main__":
    main()
