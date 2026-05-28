"""
Generate a Pareto frontier plot of Training Time vs. Model Quality.

This script traces the performance curve of ChimeraBoost against CatBoost, 
LightGBM, and XGBoost by sweeping the learning rate. We set a massive tree limit 
(10,000) and rely entirely on early_stopping (patience=50). 
High LR = fast/rough; Low LR = slow/accurate.

Usage:
    pip install matplotlib pandas
    python benchmarks/plot_frontier.py
"""

import argparse
import time
import os
import sys
import importlib.util
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import run_benchmarks as B
from sklearn.model_selection import train_test_split

# Sweeping Learning Rate. 
# Trees are practically unlimited; early stopping decides the time spent.
LEARNING_RATES = [0.2, 0.15, 0.10, 0.05, 0.035, 0.02]
MAX_TREES = 10000
PATIENCE = 50

MODELS = ["ChimeraBoost", "CatBoost"]#, "LightGBM", "XGBoost"]
COLORS = {
    "ChimeraBoost": "#2b83ba",  # Blue
    "CatBoost": "#fdae61",      # Orange
    "LightGBM": "#abdda4",      # Green
    "XGBoost": "#d7191c"        # Red
}
    # "credit-g":      dict(data_id=31,    task="binary",     cats="auto"),
    # "adult":         dict(data_id=1590,  task="binary",     cats="auto"),
    # "bank-marketing":dict(data_id=1461,  task="binary",     cats="auto"),
    # "kc1":           dict(data_id=1067,  task="binary",     cats=None),
    # "phoneme":       dict(data_id=1489,  task="binary",     cats=None),
DEFAULT_DATASETS = [
    "diabetes", "oml:boston", "oml:cpu_act", "oml:wine_quality",  # Regression
    "oml:kc1", "oml:phoneme", "oml:credit-g", "oml:adult"     # Classification
]

def _is_installed(name):
    if name == "ChimeraBoost": return True
    return importlib.util.find_spec(name.lower()) is not None

def _fit_evaluate(name, task, Xtr, ytr, Xte, yte, cat, threads, lr):
    Xf, Xv, yf, yv = B._val_split(Xtr, ytr, task, 0)
    t0 = time.time()
    
    if name == "ChimeraBoost":
        from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier
        Est = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
        m = Est(iterations=MAX_TREES, learning_rate=lr, early_stopping_rounds=PATIENCE, 
                thread_count=threads, random_state=0, ordered_boosting=True)
        m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
        fit_time = time.time() - t0
        score = B._score(task, yte, m, Xte)
        
    elif name == "CatBoost":
        from catboost import CatBoostRegressor, CatBoostClassifier
        Est = CatBoostRegressor if task == "regression" else CatBoostClassifier
        m = Est(iterations=MAX_TREES, learning_rate=lr, early_stopping_rounds=PATIENCE,
                thread_count=threads or -1, verbose=False, random_seed=0)
        m.fit(Xf, yf, cat_features=cat, eval_set=(Xv, yv))
        fit_time = time.time() - t0
        score = B._score(task, yte, m, Xte)
        
    elif name == "LightGBM":
        from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
        import pandas as pd
        import warnings
        warnings.filterwarnings("ignore")
                
        Xf_df = pd.DataFrame(Xf).copy()
        Xv_df = pd.DataFrame(Xv).copy()
        Xte_df = pd.DataFrame(Xte).copy()

        # Convert all object columns to categorical
        for col in Xf_df.columns:
            if Xf_df[col].dtype == "object":
                Xf_df[col] = Xf_df[col].astype("category")
                Xv_df[col] = Xv_df[col].astype("category")
                Xte_df[col] = Xte_df[col].astype("category")
                
        Est = LGBMRegressor if task == "regression" else LGBMClassifier
        m = Est(n_estimators=MAX_TREES, learning_rate=lr, n_jobs=threads or -1, random_state=0, verbose=-1)
        m.fit(Xf_df, yf, eval_set=[(Xv_df, yv)], callbacks=[early_stopping(PATIENCE, verbose=False)])
        fit_time = time.time() - t0
        score = B._score(task, yte, m, Xte_df)
        
    elif name == "XGBoost":
        from xgboost import XGBRegressor, XGBClassifier
        import pandas as pd
        Xf_df = pd.DataFrame(Xf).copy()
        Xv_df = pd.DataFrame(Xv).copy()
        Xte_df = pd.DataFrame(Xte).copy()

        # Convert all object columns to categorical
        for col in Xf_df.columns:
            if Xf_df[col].dtype == "object":
                Xf_df[col] = Xf_df[col].astype("category")
                Xv_df[col] = Xv_df[col].astype("category")
                Xte_df[col] = Xte_df[col].astype("category")
                
        Est = XGBRegressor if task == "regression" else XGBClassifier
        m = Est(n_estimators=MAX_TREES, learning_rate=lr, early_stopping_rounds=PATIENCE, 
                n_jobs=threads or -1, random_state=0, enable_categorical=True, tree_method="hist")
        m.fit(Xf_df, yf, eval_set=[(Xv_df, yv)], verbose=False)
        fit_time = time.time() - t0
        score = B._score(task, yte, m, Xte_df)
        
    return fit_time, score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    B._add_openml_datasets()
    available_models = [m for m in MODELS if _is_installed(m)]
    print(f"Running models: {available_models}")
    print(f"Sweeping Learning Rates: {LEARNING_RATES} (Max Trees: {MAX_TREES}, Patience: {PATIENCE})\n")

    # raw_results[dataset][model][lr_idx] = {'time': t, 'score': s}
    raw_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    tasks = {}

    for ds in DEFAULT_DATASETS:
        print(f"--- Processing {ds} ---")
        try:
            X, y, cat, task = B.DATASETS[ds](1.0, np.random.default_rng(42))
        except Exception as e:
            print(f"Skipping {ds} - Failed to load: {e}\n")
            continue
            
        tasks[ds] = task
        strat = y if task != "regression" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)
        
        for model in available_models:
            for lr_idx, lr in enumerate(LEARNING_RATES):
                try:
                    fit_time, score = _fit_evaluate(model, task, Xtr, ytr, Xte, yte, cat, args.threads, lr)
                    raw_results[ds][model][lr_idx] = {'time': fit_time, 'score': score}
                    print(f"  {model:12s} | LR={lr:<4.3f} | Time: {fit_time:6.2f}s | Score: {score:.4f}")
                except Exception as e:
                    print(f"  {model} failed on {ds} (LR={lr}): {e}")
        print()

# ==========================================
    # Aggregation & Ranking
    # ==========================================
    agg_times = {"regression": defaultdict(list), "classification": defaultdict(list)}
    agg_ranks = {"regression": defaultdict(list), "classification": defaultdict(list)}

    for ds in tasks:
        task = "regression" if tasks[ds] == "regression" else "classification"
        
        # 1. Collect all scores for this dataset to determine ranks
        results_list = []
        for model in available_models:
            for lr_idx in range(len(LEARNING_RATES)):
                if 'score' in raw_results[ds][model][lr_idx]:
                    s = raw_results[ds][model][lr_idx]['score']
                    results_list.append((s, model, lr_idx))
                    
        if not results_list: continue
        
        # 2. Sort results to assign ranks. B._score is higher-is-better for
        # every task: classification is F1, regression is -RMSE.
        results_list.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Assign ranks (1 is the best)
        ranks_dict = {}
        for rank, (s, model, lr_idx) in enumerate(results_list, start=1):
            ranks_dict[(model, lr_idx)] = rank

        # 4. Store the ranks and times
        for model in available_models:
            times, ranks = [], []
            for lr_idx in range(len(LEARNING_RATES)):
                if 'score' in raw_results[ds][model][lr_idx]:
                    t = raw_results[ds][model][lr_idx]['time']
                    r = ranks_dict[(model, lr_idx)]
                    times.append(t)
                    ranks.append(r)
                else:
                    times.append(np.nan)
                    ranks.append(np.nan)
                    
            agg_times[task][model].append(times)
            agg_ranks[task][model].append(ranks)

    # ==========================================
    # Plotting
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Time vs. Quality Pareto Frontier\nAverage Rank across Datasets (Unlimited Trees, Patience=50)', fontsize=14, fontweight='bold')

    for task, ax, title in [
        ("classification", ax1, "Classification"),
        ("regression", ax2, "Regression")
    ]:
        for model in available_models:
            if not agg_times[task][model]: continue
            
            # Average across datasets (ignoring NaNs from failed runs)
            avg_t = np.nanmean(agg_times[task][model], axis=0)
            avg_r = np.nanmean(agg_ranks[task][model], axis=0)
            
            # Sort by time to make the line continuous (low time -> high time)
            valid_mask = ~np.isnan(avg_t) & ~np.isnan(avg_r)
            avg_t, avg_r = avg_t[valid_mask], avg_r[valid_mask]
            
            sort_idx = np.argsort(avg_t)
            avg_t, avg_r = avg_t[sort_idx], avg_r[sort_idx]
            
            ax.plot(avg_t, avg_r, marker='o', linewidth=2.5, markersize=8, 
                    label=model, color=COLORS.get(model, "#333333"))

        ax.set_xscale('log')
        ax.set_xlabel('Average Training Time (Seconds) - Log Scale', fontsize=12)
        ax.set_ylabel("Average Rank (1 = Best)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        # Invert the Y-axis so Rank 1 is at the top. 
        # This keeps the optimal models in the top-left corner.
        if not ax.yaxis_inverted():
            ax.invert_yaxis()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    filename = "pareto_frontier_ranked.png"
    plt.savefig(filename, dpi=300)
    print(f"\n✅ Beautiful frontier plot saved to '{filename}'!")

if __name__ == "__main__":
    main()
