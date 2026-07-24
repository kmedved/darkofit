#!/usr/bin/env python3
"""Development panel for the declared-order native/ordinal selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from darkofit import DarkoRegressor


SEEDS = (4, 17, 29)
AIRFOIL_SHA256 = (
    "5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c"
)
DIAMONDS_SHA256 = (
    "974c2ce1c1ce245508bd357ca11a7fba2b37813ecf0f1158808a9249ebff67a1"
)
DIAMOND_ORDERS = {
    "cut": ("Fair", "Good", "Very Good", "Premium", "Ideal"),
    "color": ("J", "I", "H", "G", "F", "E", "D"),
    "clarity": ("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"),
}
MODEL_PARAMS = {
    "iterations": 1_000,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "max_bins": 128,
    "tree_mode": "catboost",
    "early_stopping": True,
    "use_best_model": True,
    "linear_leaves": False,
    "ts_permutations": 1,
    "thread_count": 14,
    "diagnostic_warnings": "never",
}


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _index_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_airfoil(path: Path):
    if _sha256(path) != AIRFOIL_SHA256:
        raise RuntimeError("Airfoil archive hash does not match the pinned source")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != ["airfoil_self_noise.dat"]:
            raise RuntimeError(f"unexpected Airfoil archive members: {names!r}")
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle, sep=r"\s+", header=None)
    if frame.shape != (1_503, 6):
        raise RuntimeError(f"unexpected Airfoil shape: {frame.shape!r}")
    frame.columns = [
        "frequency",
        "attack_angle",
        "chord_length",
        "free_stream_velocity",
        "suction_side_displacement_thickness",
        "sound_pressure",
    ]
    categories = tuple(sorted(frame["attack_angle"].unique().tolist()))
    X = frame.drop(columns=["sound_pressure"]).copy()
    X["attack_angle"] = pd.Categorical(
        X["attack_angle"],
        categories=categories,
        ordered=True,
    )
    y = frame["sound_pressure"].to_numpy(dtype=np.float64)
    return X, y, {"attack_angle": categories}, ["attack_angle"]


def _load_diamonds(path: Path):
    if _sha256(path) != DIAMONDS_SHA256:
        raise RuntimeError("Diamonds CSV hash does not match the pinned source")
    frame = pd.read_csv(path)
    if frame.shape != (53_940, 11) or frame.columns[0] != "rownames":
        raise RuntimeError(f"unexpected Diamonds schema: {frame.shape!r}")
    frame = frame.drop(columns=["rownames"])
    for name, categories in DIAMOND_ORDERS.items():
        observed = set(frame[name].dropna().unique().tolist())
        if observed != set(categories):
            raise RuntimeError(
                f"Diamonds {name!r} domain changed: {sorted(observed)!r}"
            )
        frame[name] = pd.Categorical(
            frame[name],
            categories=categories,
            ordered=True,
        )
    X = frame.drop(columns=["price"])
    y = frame["price"].to_numpy(dtype=np.float64)
    return X, y, dict(DIAMOND_ORDERS), list(DIAMOND_ORDERS)


def _split_positions(n_rows: int, seed: int):
    all_positions = np.arange(n_rows, dtype=np.int64)
    development, test = train_test_split(
        all_positions,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    train, validation = train_test_split(
        development,
        test_size=0.2,
        random_state=seed + 10_000,
        shuffle=True,
    )
    train = np.sort(train)
    validation = np.sort(validation)
    test = np.sort(test)
    if (
        np.intersect1d(train, validation).size
        or np.intersect1d(train, test).size
        or np.intersect1d(validation, test).size
        or len(train) + len(validation) + len(test) != n_rows
    ):
        raise RuntimeError("development split positions are not a partition")
    return train, validation, test


def _rmse(y_true, prediction) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != y_true.shape or not np.isfinite(prediction).all():
        raise RuntimeError("development prediction is invalid")
    return float(np.sqrt(np.mean(np.square(y_true - prediction))))


def _geometric_mean(values) -> float:
    values = [float(value) for value in values]
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _fit_arm(
    name,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    *,
    seed,
    cat_features,
    ordinal_features,
):
    model = DarkoRegressor(**MODEL_PARAMS, random_state=seed)
    started = time.perf_counter_ns()
    if name == "selector":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_validation, y_validation),
            ordinal_features="select",
        )
    else:
        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            eval_set=(X_validation, y_validation),
            ordinal_features=ordinal_features,
        )
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = model.predict(X_test)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    selector = getattr(model, "automatic_ordinal_selector_", None)
    return {
        "name": name,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "prediction": prediction,
        "prediction_sha256": _array_sha256(prediction),
        "best_n_estimators": int(model.best_n_estimators_),
        "ordinal_active": bool(model.ordinal_features_),
        "selector": selector,
    }


def _run_dataset(name, X, y, orders, cat_features):
    rows = []
    for seed in SEEDS:
        train, validation, test = _split_positions(len(y), seed)
        arms = {}
        for arm_name, ordinal_features in (
            ("native", None),
            ("forced_ordinal", orders),
            ("selector", None),
        ):
            arms[arm_name] = _fit_arm(
                arm_name,
                X.iloc[train],
                y[train],
                X.iloc[validation],
                y[validation],
                X.iloc[test],
                seed=seed,
                cat_features=cat_features,
                ordinal_features=ordinal_features,
            )
        native = arms["native"]
        forced = arms["forced_ordinal"]
        selector = arms["selector"]
        selected = bool(selector["selector"]["selected"])
        expected = forced if selected else native
        if not np.array_equal(selector["prediction"], expected["prediction"]):
            raise RuntimeError(
                f"{name} seed {seed}: selector final refit is not exact"
            )
        scores = {
            arm_name: _rmse(y[test], arm["prediction"])
            for arm_name, arm in arms.items()
        }
        rows.append(
            {
                "dataset": name,
                "seed": seed,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_positions_sha256": _index_sha256(train),
                "validation_positions_sha256": _index_sha256(validation),
                "test_positions_sha256": _index_sha256(test),
                "native_rmse": scores["native"],
                "forced_ordinal_rmse": scores["forced_ordinal"],
                "selector_rmse": scores["selector"],
                "selector_over_native_ratio": (
                    scores["selector"] / scores["native"]
                ),
                "forced_over_native_ratio": (
                    scores["forced_ordinal"] / scores["native"]
                ),
                "selector_over_forced_ratio": (
                    scores["selector"] / scores["forced_ordinal"]
                ),
                "selector_selected": selected,
                "selector_reason": selector["selector"]["reason"],
                "selector_gain_z": selector["selector"]["paired_mse_gain_z"],
                "selector_final_exact": True,
                "arms": {
                    arm_name: {
                        key: value
                        for key, value in arm.items()
                        if key != "prediction"
                    }
                    for arm_name, arm in arms.items()
                },
            }
        )
    return rows


def _summarize(rows):
    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "selector_over_native_ratio": _geometric_mean(
                row["selector_over_native_ratio"] for row in dataset_rows
            ),
            "forced_over_native_ratio": _geometric_mean(
                row["forced_over_native_ratio"] for row in dataset_rows
            ),
            "engagements": sum(
                bool(row["selector_selected"]) for row in dataset_rows
            ),
            "coordinates": len(dataset_rows),
            "worst_selector_over_native_ratio": max(
                row["selector_over_native_ratio"] for row in dataset_rows
            ),
        }
    dataset_ratios = [
        result["selector_over_native_ratio"]
        for result in by_dataset.values()
    ]
    return {
        "equal_dataset_selector_over_native_ratio": _geometric_mean(
            dataset_ratios
        ),
        "worst_coordinate_selector_over_native_ratio": max(
            row["selector_over_native_ratio"] for row in rows
        ),
        "engagements": sum(bool(row["selector_selected"]) for row in rows),
        "coordinates": len(rows),
        "all_final_refits_exact": all(
            bool(row["selector_final_exact"]) for row in rows
        ),
        "by_dataset": by_dataset,
    }


def _render(result):
    summary = result["summary"]
    lines = [
        "# Declared-order selector development result",
        "",
        "This is spent development evidence on the two historical declared-order",
        "domains. Inner validation selects the representation; disjoint outer",
        "test rows score native, forced ordinal, and automatic selector arms.",
        "",
        f"- Source: `{result['source_sha']}`",
        (
            "- Equal-dataset selector/native RMSE ratio: "
            f"`{summary['equal_dataset_selector_over_native_ratio']:.6f}`"
        ),
        (
            "- Worst coordinate selector/native RMSE ratio: "
            f"`{summary['worst_coordinate_selector_over_native_ratio']:.6f}`"
        ),
        (
            f"- Engagements: `{summary['engagements']}/"
            f"{summary['coordinates']}`"
        ),
        (
            "- Final selected/native refits prediction-exact: "
            f"`{str(summary['all_final_refits_exact']).lower()}`"
        ),
        "",
        "| Dataset | Selector/native | Forced/native | Engaged | Worst |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, row in summary["by_dataset"].items():
        lines.append(
            f"| {dataset} | {row['selector_over_native_ratio']:.6f} | "
            f"{row['forced_over_native_ratio']:.6f} | "
            f"{row['engagements']}/{row['coordinates']} | "
            f"{row['worst_selector_over_native_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The panel is intentionally narrow: both datasets were used by the",
            "historical safe-ordinal campaign. It tests selector behavior and",
            "transfer to held-out rows, not generalization to new datasets.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--airfoil-archive", type=Path, required=True)
    parser.add_argument("--diamonds-csv", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
    source_sha = _git("rev-parse", "HEAD")
    if source_sha != args.expected_source_sha:
        raise RuntimeError(
            f"source SHA changed: expected {args.expected_source_sha}, got {source_sha}"
        )
    if _git("status", "--porcelain"):
        raise RuntimeError("selector development requires a clean worktree")
    for path in (args.raw_output, args.result_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    airfoil = _load_airfoil(args.airfoil_archive)
    diamonds = _load_diamonds(args.diamonds_csv)
    rows = [
        *_run_dataset("airfoil_self_noise", *airfoil),
        *_run_dataset("diamonds", *diamonds),
    ]
    result = {
        "schema_version": 1,
        "source_sha": source_sha,
        "seeds": list(SEEDS),
        "model_params": MODEL_PARAMS,
        "data_sources": {
            "airfoil": {
                "path": str(args.airfoil_archive),
                "sha256": AIRFOIL_SHA256,
                "url": (
                    "https://archive.ics.uci.edu/static/public/291/"
                    "airfoil+self+noise.zip"
                ),
            },
            "diamonds": {
                "path": str(args.diamonds_csv),
                "sha256": DIAMONDS_SHA256,
                "url": (
                    "https://raw.githubusercontent.com/"
                    "vincentarelbundock/Rdatasets/master/csv/ggplot2/"
                    "diamonds.csv"
                ),
            },
        },
        "rows": rows,
        "summary": _summarize(rows),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }
    raw_bytes = (
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_output.open("xb") as handle:
        handle.write(raw_bytes)
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    with args.result_output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_render(result))


if __name__ == "__main__":
    main()
