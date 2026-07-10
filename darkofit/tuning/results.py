"""Result extraction helpers for Optuna studies."""

from __future__ import annotations

import numpy as np


def build_cv_results(trials):
    rows = [t for t in trials if t.user_attrs.get("params_full") is not None]
    return {
        "trial_number": [t.number for t in rows],
        "params": [
            t.user_attrs.get("params_model") or t.user_attrs.get("params_full")
            for t in rows
        ],
        "trial_params": [t.user_attrs.get("params_full") for t in rows],
        "phase": [t.user_attrs.get("phase") for t in rows],
        "tree_mode": [t.user_attrs.get("tree_mode_lane") for t in rows],
        "mean_test_score": [t.user_attrs.get("mean_score") for t in rows],
        "std_test_score": [t.user_attrs.get("std_score") for t in rows],
        "mean_test_loss": [t.user_attrs.get("mean_loss") for t in rows],
        "std_test_loss": [t.user_attrs.get("std_loss") for t in rows],
        "mean_best_iteration": [t.user_attrs.get("mean_best_iteration") for t in rows],
        "median_best_iteration": [t.user_attrs.get("median_best_iteration") for t in rows],
        "mean_learning_rate": [t.user_attrs.get("mean_learning_rate") for t in rows],
        "fit_time": [t.user_attrs.get("fit_time") for t in rows],
        "status": [
            t.user_attrs.get("status") or str(t.state).split(".")[-1]
            for t in rows
        ],
        "error": [t.user_attrs.get("error") for t in rows],
    }


def phase_summary(trials):
    out = {}
    for trial in trials:
        state_name = getattr(
            trial.state, "name", str(trial.state).split(".")[-1]
        )
        if state_name != "COMPLETE":
            continue
        if trial.user_attrs.get("status") == "ERROR_SCORE":
            continue
        phase = trial.user_attrs.get("phase")
        lane = trial.user_attrs.get("tree_mode_lane")
        if phase is None or lane is None or trial.value is None:
            continue
        key = (phase, lane)
        current = out.get(key)
        if current is None or trial.value < current["best_loss"]:
            out[key] = {
                "phase": phase,
                "tree_mode": lane,
                "best_loss": float(trial.value),
                "best_score": trial.user_attrs.get("mean_score"),
                "best_trial_number": trial.number,
            }
    return list(out.values())


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0.0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))
