#!/usr/bin/env python3
"""Ablate the accuracy horizon and guarded numeric crosses on spent data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_adapters import (  # noqa: E402
    build_dataset,
    make_sample_weight,
    split_case,
)
from benchmarks.m6_quality_rule_v3 import (  # noqa: E402
    DATASETS as M6_DATASETS,
    SEEDS,
    THREADS,
    WEIGHT_MODES,
    quality_decision,
)
from benchmarks.run_smooth_cross_features import candidate_pairs  # noqa: E402
from benchmarks.weighted_metrics import metric_bundle  # noqa: E402


DATASETS = tuple(
    name
    for name in M6_DATASETS
    if name in {
        "diabetes_resampled",
        "friedman_numeric",
        "wide_numeric_reg",
        "categorical_reg",
    }
)
SIZE = "medium"
HORIZONS = (1_000, 10_000)
CROSS_GUARD_RATIO = 0.95
ARMS = ("a1", "a1_cross", "a10", "a10_cross")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def source_state(expected_sha: str) -> dict[str, object]:
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if head != expected_sha:
        raise RuntimeError(f"source SHA differs: expected {expected_sha}, got {head}")
    if status:
        raise RuntimeError(f"accuracy-v2 benchmark requires a clean tree: {status}")
    return {"head": head, "tree": _git("rev-parse", "HEAD^{tree}"), "clean": True}


def expected_cell_keys() -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (dataset, int(seed), weight_mode)
        for dataset in DATASETS
        for seed in SEEDS
        for weight_mode in WEIGHT_MODES
    )


def augment_crosses(X, pairs):
    array = np.asarray(X)
    if array.ndim != 2:
        raise RuntimeError("accuracy-v2 input must be two-dimensional")
    columns = [array]
    for left, right, operation in pairs:
        try:
            left_values = np.asarray(array[:, left], dtype=np.float64)
            right_values = np.asarray(array[:, right], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("declared accuracy-v2 cross must be numeric") from exc
        with np.errstate(over="ignore", invalid="ignore"):
            values = (
                left_values - right_values
                if operation == "diff"
                else left_values * right_values
            )
        values = np.asarray(values, dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        columns.append(values.reshape(-1, 1))
    return np.column_stack(columns)


def _model(horizon: int, *, seed: int):
    from darkofit import DarkoRegressor

    common = {
        "random_state": int(seed),
        "thread_count": THREADS,
        "diagnostic_warnings": "never",
        "linear_leaves": "auto",
    }
    if horizon == 10_000:
        return DarkoRegressor(preset="accuracy", **common)
    if horizon != 1_000:
        raise ValueError("accuracy-v2 horizon must be 1,000 or 10,000")
    return DarkoRegressor(
        iterations=1_000,
        tree_mode="auto",
        l2_leaf_reg=3.0,
        max_bins=128,
        learning_rate=0.1,
        ts_permutations=1,
        linear_residual=False,
        early_stopping=True,
        use_best_model=True,
        **common,
    )


def _fit_selection(model, X, y, cat_features, X_val, y_val, w, w_val):
    kwargs = {
        "cat_features": cat_features,
        "eval_set": (X_val, y_val),
    }
    if w is not None:
        kwargs["sample_weight"] = w
        kwargs["eval_sample_weight"] = w_val
    started = time.perf_counter()
    model.fit(X, y, **kwargs)
    return model, time.perf_counter() - started


def _fit_final(selection_model, X, y, cat_features, w):
    from darkofit import DarkoRegressor

    params = selection_model.get_refit_params()
    params["diagnostic_warnings"] = "never"
    model = DarkoRegressor(**params)
    kwargs = {"cat_features": cat_features}
    if w is not None:
        kwargs["sample_weight"] = w
    started = time.perf_counter()
    model.fit(X, y, **kwargs)
    return model, time.perf_counter() - started


def _prediction_record(model, X_test, y_test, w_test, *, fit_seconds):
    started = time.perf_counter()
    prediction = model.predict(X_test)
    predict_seconds = time.perf_counter() - started
    metrics = metric_bundle(
        "regression",
        y_test,
        prediction,
        sample_weight=w_test,
    )
    return {
        "prediction": np.asarray(prediction, dtype=np.float64),
        "primary_metric": metrics["primary_metric"],
        "primary_value": metrics["primary_value"],
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "tree_count": int(len(model.model_.trees_)),
        "selected_tree_mode": str(model.model_.tree_mode_),
        "selected_linear_leaves": bool(
            getattr(model.model_, "linear_leaves_active_", False)
        ),
    }


def run_horizon(
    split,
    cat_features,
    *,
    horizon: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    X_fit = split["X_fit"]
    y_fit = split["y_fit"]
    X_val = split["X_val"]
    y_val = split["y_val"]
    w_fit = split["w_fit"]
    w_val = split["w_val"]
    X_train = np.concatenate([X_fit, X_val], axis=0)
    y_train = np.concatenate([y_fit, y_val], axis=0)
    w_train = (
        None
        if w_fit is None
        else np.concatenate([w_fit, w_val], axis=0)
    )

    base, base_selection_seconds = _fit_selection(
        _model(horizon, seed=seed),
        X_fit,
        y_fit,
        cat_features,
        X_val,
        y_val,
        w_fit,
        w_val,
    )
    pairs = candidate_pairs(
        np.asarray(base.feature_importances_, dtype=np.float64),
        cat_features,
        np.asarray(X_fit).shape[1],
    )
    crossed = None
    crossed_selection_seconds = 0.0
    crossed_score = None
    if pairs:
        crossed, crossed_selection_seconds = _fit_selection(
            _model(horizon, seed=seed),
            augment_crosses(X_fit, pairs),
            y_fit,
            cat_features,
            augment_crosses(X_val, pairs),
            y_val,
            w_fit,
            w_val,
        )
        crossed_score = float(crossed.best_score_)
    base_score = float(base.best_score_)
    engaged = (
        crossed is not None
        and crossed_score <= CROSS_GUARD_RATIO * base_score
    )

    base_final, base_final_seconds = _fit_final(
        base, X_train, y_train, cat_features, w_train
    )
    base_record = _prediction_record(
        base_final,
        split["X_test"],
        split["y_test"],
        split["w_test"],
        fit_seconds=base_selection_seconds + base_final_seconds,
    )
    if engaged:
        crossed_train = augment_crosses(X_train, pairs)
        crossed_test = augment_crosses(split["X_test"], pairs)
        crossed_final, crossed_final_seconds = _fit_final(
            crossed, crossed_train, y_train, cat_features, w_train
        )
        guarded_record = _prediction_record(
            crossed_final,
            crossed_test,
            split["y_test"],
            split["w_test"],
            fit_seconds=(
                base_selection_seconds
                + crossed_selection_seconds
                + crossed_final_seconds
            ),
        )
    else:
        guarded_record = {
            **base_record,
            "prediction": base_record["prediction"].copy(),
            "fit_seconds": float(
                base_record["fit_seconds"] + crossed_selection_seconds
            ),
        }
    guarded_record.update({
        "engaged": bool(engaged),
        "pair_count": len(pairs),
        "validation_ratio": (
            None if crossed_score is None else crossed_score / base_score
        ),
        "fallback_prediction_exact": (
            None
            if engaged
            else bool(np.array_equal(
                guarded_record["prediction"], base_record["prediction"]
            ))
        ),
    })
    base_record.update({
        "engaged": False,
        "pair_count": 0,
        "validation_ratio": None,
        "fallback_prediction_exact": True,
    })
    return {"base": base_record, "guarded": guarded_record}


def _geomean(values) -> float:
    values = np.asarray(tuple(values), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise RuntimeError("accuracy-v2 ratios must be positive and finite")
    return float(np.exp(np.mean(np.log(values))))


def analyze_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    expected = set(expected_cell_keys())
    cells: dict[tuple[str, int, str], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["seed"]), str(row["weight_mode"]))
        arm = str(row["arm"])
        if key not in expected or arm not in ARMS:
            raise RuntimeError("accuracy-v2 rows contain an unexpected identity")
        if arm in cells.setdefault(key, {}):
            raise RuntimeError("accuracy-v2 rows contain a duplicate arm")
        cells[key][arm] = row
    if set(cells) != expected or any(set(cell) != set(ARMS) for cell in cells.values()):
        raise RuntimeError("accuracy-v2 rows do not cover the exact grid")

    comparisons = {}
    for name, numerator, denominator in (
        ("horizon", "a10", "a1"),
        ("cross_at_1k", "a1_cross", "a1"),
        ("cross_at_10k", "a10_cross", "a10"),
        ("combined", "a10_cross", "a1"),
    ):
        ratios = {}
        groups = {}
        fit_ratios = []
        predict_ratios = []
        for dataset, seed, weight_mode in expected_cell_keys():
            key = (dataset, seed, weight_mode)
            top = cells[key][numerator]
            bottom = cells[key][denominator]
            if top["primary_metric"] != bottom["primary_metric"]:
                raise RuntimeError("accuracy-v2 paired metrics differ")
            coordinate = f"{dataset}/{seed}/{weight_mode}"
            ratios[coordinate] = float(top["primary_value"]) / float(
                bottom["primary_value"]
            )
            groups[coordinate] = dataset
            fit_ratios.append(
                float(top["fit_seconds"]) / float(bottom["fit_seconds"])
            )
            predict_ratios.append(
                float(top["predict_seconds"]) / float(bottom["predict_seconds"])
            )
        comparisons[name] = {
            "quality": quality_decision(ratios, groups=groups),
            "fit_seconds_geometric_mean_ratio": _geomean(fit_ratios),
            "predict_seconds_geometric_mean_ratio": _geomean(predict_ratios),
        }

    cross = comparisons["cross_at_10k"]["quality"]
    engagements = sum(bool(row["engaged"]) for row in rows if row["arm"] == "a10_cross")
    select_v2 = (
        engagements > 0
        and cross["disposition"] == "advance"
        and cross["geometric_mean_ratio"] < 1.0
    )
    return {
        "cell_count": len(expected),
        "comparisons": comparisons,
        "a10_cross_engagements": engagements,
        "selected_accuracy_profile": "accuracy_v2" if select_v2 else "accuracy",
        "public_profile_changed": bool(select_v2),
    }


def _json_record(record):
    return {
        key: value
        for key, value in record.items()
        if key != "prediction"
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(result):
    summary = result["summary"]
    lines = [
        "# Accuracy-v2 component ablation",
        "",
        "Spent M6 regression development evidence; no holdout was consulted.",
        "",
        f"- Source: `{result['source']['head']}`",
        f"- Cells: `{summary['cell_count']}`",
        f"- A10 cross engagements: `{summary['a10_cross_engagements']}`",
        f"- Selected profile: `{summary['selected_accuracy_profile']}`",
        "",
        "| Contrast | Quality ratio | Worst dataset | Worst LOO | Fit ratio | Predict ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("horizon", "cross_at_1k", "cross_at_10k", "combined"):
        comparison = summary["comparisons"][name]
        quality = comparison["quality"]
        lines.append(
            f"| {name} | {quality['geometric_mean_ratio']:.6f} | "
            f"{quality['worst_group_ratio']:.6f} | "
            f"{quality['worst_loo_ratio']:.6f} | "
            f"{comparison['fit_seconds_geometric_mean_ratio']:.6f} | "
            f"{comparison['predict_seconds_geometric_mean_ratio']:.6f} |"
        )
    lines.extend([
        "",
        "A declined cross arm is prediction-exact to its uncrossed fallback.",
        "This development slice may select an explicit accuracy profile; it",
        "cannot establish a new default or unseen-data claim.",
        "",
    ])
    return "\n".join(lines)


def run(args) -> int:
    for path in (args.raw_output, args.result_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"accuracy-v2 output is create-only: {path}")
    source = source_state(args.expected_source_sha)
    rows = []
    for dataset, seed, weight_mode in expected_cell_keys():
        spec, X, y, cat_features = build_dataset(dataset, SIZE, seed)
        weights = make_sample_weight(y, spec.task, weight_mode)
        split = split_case(X, y, spec.task, seed, weights)
        for horizon in HORIZONS:
            records = run_horizon(
                split,
                cat_features,
                horizon=horizon,
                seed=seed,
            )
            prefix = "a1" if horizon == 1_000 else "a10"
            for suffix, record in (("", records["base"]), ("_cross", records["guarded"])):
                rows.append({
                    "dataset": dataset,
                    "size": SIZE,
                    "seed": seed,
                    "weight_mode": weight_mode,
                    "arm": prefix + suffix,
                    "train_rows": int(split["n_train"] + split["n_val"]),
                    "test_rows": int(split["n_test"]),
                    "feature_count": int(split["n_features"]),
                    **_json_record(record),
                })
        print(f"completed {dataset} seed={seed} weights={weight_mode}", flush=True)
        gc.collect()
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "spent_general_regression_development_slice",
        "source": source,
        "grid": {
            "datasets": list(DATASETS),
            "size": SIZE,
            "seeds": list(SEEDS),
            "weight_modes": list(WEIGHT_MODES),
            "threads": THREADS,
            "horizons": list(HORIZONS),
            "cross_guard_ratio": CROSS_GUARD_RATIO,
            "arms": list(ARMS),
        },
        "rows": rows,
        "summary": analyze_rows(rows),
    }
    args.raw_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.result_output.write_text(
        _markdown(result) + f"\nRaw SHA-256: `{_sha256(args.raw_output)}`\n"
    )
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
