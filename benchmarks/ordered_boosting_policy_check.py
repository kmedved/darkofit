"""Reproducible guardrail matrix for the ordered-boosting auto policy.

The comparison uses shared train/validation/test splits for the old always-on
CatBoost-mode behavior, the new task-aware ``"auto"`` policy, and explicit
plain boosting. Optional CatBoost and LightGBM installations provide strong
CPU reference points on the same splits.

Run from the repository root:

    python benchmarks/ordered_boosting_policy_check.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_diabetes,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darkofit import DarkoRegressor  # noqa: E402


def _normalize_openml_frame(X):
    """Return a copy with every nonnumeric column normalized as categorical."""
    from pandas.api.types import is_numeric_dtype

    X = X.copy()
    cat_features = []
    for index, column in enumerate(X.columns):
        if is_numeric_dtype(X[column].dtype):
            continue
        cat_features.append(index)
        values = X[column].astype(object)
        X[column] = values.where(values.notna(), "__MISSING__")
    return X, cat_features


def _datasets():
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
    california_X, california_y = fetch_california_housing(return_X_y=True)
    abalone_X, abalone_y = fetch_openml(
        name="abalone", version=1, as_frame=True, return_X_y=True
    )
    abalone_y = np.asarray(abalone_y, dtype=np.float64)
    abalone_X, abalone_cat = _normalize_openml_frame(abalone_X)
    house_X, house_y = fetch_openml(
        name="house_prices", as_frame=True, return_X_y=True
    )
    house_X, house_cat = _normalize_openml_frame(house_X)
    house_y = np.asarray(house_y, dtype=np.float64)
    return (
        ("diabetes", diabetes_X, diabetes_y, None, False),
        ("diabetes_weighted", diabetes_X, diabetes_y, None, True),
        ("california", california_X, california_y, None, False),
        ("california_weighted", california_X, california_y, None, True),
        ("abalone_categorical", abalone_X, abalone_y, abalone_cat, False),
        ("house_prices_categorical", house_X, house_y, house_cat, False),
    )


def _weights(y):
    distance = np.abs(y - np.median(y))
    return 0.5 + 2.5 * distance / max(float(np.max(distance)), 1e-12)


def _rmse(y_true, y_pred, sample_weight=None):
    return float(np.sqrt(mean_squared_error(
        y_true, y_pred, sample_weight=sample_weight
    )))


def _split(X, y, weighted, seed):
    indices = np.arange(len(y))
    train_valid, test = train_test_split(
        indices, test_size=0.25, random_state=seed
    )
    train, valid = train_test_split(
        train_valid, test_size=0.2, random_state=10_000 + seed
    )
    w = _weights(y) if weighted else None

    def take(values, rows):
        iloc = getattr(values, "iloc", None)
        return iloc[rows] if iloc is not None else values[rows]

    return {
        "X_train": take(X, train),
        "X_valid": take(X, valid),
        "X_test": take(X, test),
        "y_train": y[train],
        "y_valid": y[valid],
        "y_test": y[test],
        "w_train": None if w is None else w[train],
        "w_valid": None if w is None else w[valid],
        "w_test": None if w is None else w[test],
    }


def _darkofit(data, cat_features, ordered_boosting, seed, threads):
    model = DarkoRegressor(
        iterations=500,
        early_stopping=True,
        early_stopping_rounds=50,
        ordered_boosting=ordered_boosting,
        random_state=seed,
        thread_count=threads,
    )
    started = time.perf_counter()
    model.fit(
        data["X_train"],
        data["y_train"],
        cat_features=cat_features,
        eval_set=(data["X_valid"], data["y_valid"]),
        sample_weight=data["w_train"],
        eval_sample_weight=data["w_valid"],
    )
    elapsed = time.perf_counter() - started
    prediction = model.predict(data["X_test"])
    score = _rmse(data["y_test"], prediction, data["w_test"])
    return float(score), float(elapsed), prediction


def _catboost(data, cat_features, seed, threads):
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError:
        return None
    model = CatBoostRegressor(
        iterations=500,
        loss_function="RMSE",
        random_seed=seed,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )
    train_pool = Pool(
        data["X_train"], data["y_train"],
        cat_features=cat_features,
        weight=data["w_train"],
    )
    valid_pool = Pool(
        data["X_valid"], data["y_valid"],
        cat_features=cat_features,
        weight=data["w_valid"],
    )
    started = time.perf_counter()
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=50,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    prediction = model.predict(data["X_test"])
    score = _rmse(data["y_test"], prediction, data["w_test"])
    return float(score), float(elapsed)


def _lightgbm(data, cat_features, seed, threads):
    if cat_features:
        return None
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    model = lgb.LGBMRegressor(
        n_estimators=500,
        random_state=seed,
        n_jobs=threads,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(
        data["X_train"],
        data["y_train"],
        sample_weight=data["w_train"],
        eval_set=[(data["X_valid"], data["y_valid"])],
        eval_sample_weight=(
            None if data["w_valid"] is None else [data["w_valid"]]
        ),
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    elapsed = time.perf_counter() - started
    prediction = model.predict(data["X_test"])
    score = _rmse(data["y_test"], prediction, data["w_test"])
    return float(score), float(elapsed)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    rows = []
    for dataset, X, y, cat_features, weighted in _datasets():
        for seed in range(args.seeds):
            data = _split(X, np.asarray(y, dtype=np.float64), weighted, seed)
            predictions = {}
            for label, policy in (
                ("old_ordered_on", True),
                ("new_auto", "auto"),
                ("plain_off", False),
            ):
                rmse, seconds, prediction = _darkofit(
                    data, cat_features, policy, seed, args.threads
                )
                predictions[label] = prediction
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "model": label,
                    "rmse": rmse,
                    "seconds": seconds,
                })
            expected = "plain_off"
            if not np.array_equal(predictions["new_auto"], predictions[expected]):
                raise AssertionError(
                    f"{dataset} seed {seed}: auto predictions differ from {expected}"
                )
            for label, runner in (("catboost", _catboost), ("lightgbm", _lightgbm)):
                result = runner(data, cat_features, seed, args.threads)
                if result is not None:
                    rmse, seconds = result
                    rows.append({
                        "dataset": dataset,
                        "seed": seed,
                        "model": label,
                        "rmse": rmse,
                        "seconds": seconds,
                    })

    print(json.dumps(rows, indent=2, sort_keys=True))
    print("\nSUMMARY")
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        by_model = {}
        for row in dataset_rows:
            by_model.setdefault(row["model"], []).append(row["rmse"])
        old = float(np.mean(by_model["old_ordered_on"]))
        auto = float(np.mean(by_model["new_auto"]))
        delta = 100.0 * (old - auto) / old
        summary = " ".join(
            f"{model}={np.mean(values):.6g}"
            for model, values in sorted(by_model.items())
        )
        print(f"{dataset}: {summary} auto_improvement_vs_old={delta:+.2f}%")


if __name__ == "__main__":
    main()
