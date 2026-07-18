#!/usr/bin/env python3
"""Run the spent smooth-task full-budget cross-feature development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
CHIMERA_ROOT = ROOT.parent / "chimeraboost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CHIMERA_ROOT) not in sys.path:
    sys.path.insert(0, str(CHIMERA_ROOT))

TASKS = {
    361251: "grid_stability",
    361258: "kin8nm",
    361623: "space_ga",
}
FOLDS = tuple(range(3, 10))
RANDOM_STATE = 4
THREADS = 6
TOP_NUMERIC_FEATURES = 6
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
PROTOCOL = ROOT / "benchmarks" / "smooth_cross_features_protocol.md"
PARTITION = ROOT / "benchmarks" / "ctr23_partition.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "smooth_cross_features.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value, dtype=None) -> str:
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_state(path: Path, *, expected_head=None):
    head = _git(path, "rev-parse", "HEAD")
    if _git(path, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(f"source tree is not clean: {path}")
    if expected_head is not None and head != expected_head:
        raise RuntimeError(f"unexpected source head: {head} != {expected_head}")
    return {
        "path": str(path),
        "head": head,
        "branch": _git(path, "branch", "--show-current"),
        "clean": True,
    }


def _partition_boundary():
    partition = json.loads(PARTITION.read_text())
    confirmation = set(partition["confirmation_task_ids"])
    lockbox = set(partition["lockbox_task_ids"])
    task_ids = set(TASKS)
    if not task_ids <= confirmation or task_ids & lockbox:
        raise RuntimeError("smooth cross-feature task boundary changed")
    for task_id in task_ids:
        row = partition["task_allocation_metadata"][str(task_id)]
        if row["has_categorical"] != 0.0 or row["has_missing_features"] != 0.0:
            raise RuntimeError("smooth cross-feature task profile changed")
    return {
        "partition_sha256": _sha256(PARTITION),
        "confirmation_task_ids": sorted(confirmation),
        "lockbox_task_ids": sorted(lockbox),
        "lockbox_data_used": False,
        "default_promotion_authorized": False,
    }


def candidate_pairs(importances, categorical_indices, n_features):
    """Return deterministic top-six numeric diff/product pair declarations."""
    categorical = set(categorical_indices or ())
    numeric = [index for index in range(n_features) if index not in categorical]
    if len(numeric) < 2:
        return []
    importance = np.zeros(n_features, dtype=np.float64)
    supplied = np.asarray(importances, dtype=np.float64)
    importance[: min(n_features, supplied.size)] = supplied[:n_features]
    top = sorted(numeric, key=lambda index: (-importance[index], index))[
        :TOP_NUMERIC_FEATURES
    ]
    return [
        (top[left], top[right], operation)
        for left in range(len(top))
        for right in range(left + 1, len(top))
        for operation in ("diff", "prod")
    ]


def augment_numeric_crosses(X, pairs):
    """Append declared numeric crosses without consulting the target."""
    array = np.asarray(X)
    numeric = np.asarray(X, dtype=np.float64)
    columns = [array]
    for left, right, operation in pairs:
        a = numeric[:, left]
        b = numeric[:, right]
        value = a - b if operation == "diff" else a * b
        columns.append(value[:, None])
    return np.hstack(columns)


def _load_task(task_id):
    import openml

    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical, _names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    if any(bool(value) for value in categorical):
        raise RuntimeError(f"{TASKS[task_id]} unexpectedly has categoricals")
    X_array = np.asarray(X, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(X_array)) or not np.all(np.isfinite(y_array)):
        raise RuntimeError(f"{TASKS[task_id]} unexpectedly has nonfinite data")
    if task.get_split_dimensions() != (1, 10, 1):
        raise RuntimeError(f"{TASKS[task_id]} split dimensions changed")
    return task, X, y, {
        "task_id": int(task_id),
        "dataset_id": int(dataset.dataset_id),
        "dataset_name": str(dataset.name),
        "target_name": str(task.target_name),
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "X_sha256": _array_sha256(X_array, "<f8"),
        "y_sha256": _array_sha256(y_array, "<f8"),
    }


def _update_array_hash(digest, label, value):
    digest.update(label.encode("utf-8"))
    if value is None:
        digest.update(b"<none>")
        return
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _model_sha256(core, tree_limit=None):
    digest = hashlib.sha256()
    _update_array_hash(digest, "init", np.asarray([core.init_], dtype="<f8"))
    trees = core.trees_ if tree_limit is None else core.trees_[:tree_limit]
    for index, tree in enumerate(trees):
        digest.update(str(index).encode("ascii"))
        _update_array_hash(digest, "splits_feat", tree.splits_feat)
        _update_array_hash(digest, "splits_thr", tree.splits_thr)
        _update_array_hash(digest, "values", tree.values)
        _update_array_hash(
            digest,
            "linear_features",
            getattr(tree, "linear_features", getattr(tree, "lin_feats", None)),
        )
        _update_array_hash(
            digest,
            "linear_coefficients",
            getattr(
                tree, "linear_coefficients", getattr(tree, "lin_coef", None)
            ),
        )
    return digest.hexdigest()


def _fingerprint(model, prediction, y_test, *, tree_limit=None):
    core = model.model_
    history = np.asarray(core.valid_history_, dtype=np.float64)
    best_rounds = int(np.argmin(history)) + 1
    if tree_limit is None:
        tree_limit = len(core.trees_)
    borders = np.concatenate(
        [
            np.asarray(border, dtype=np.float64)
            for border in core.prep_.binner_.borders_
        ]
    )
    prediction = np.asarray(prediction, dtype=np.float64)
    return {
        "actual_retained_tree_count": int(len(core.trees_)),
        "best_prefix_tree_count": best_rounds,
        "fingerprinted_tree_count": int(tree_limit),
        "best_validation_rmse": float(np.min(history)),
        "test_rmse": float(
            mean_squared_error(np.asarray(y_test, dtype=np.float64), prediction)
            ** 0.5
        ),
        "prediction_sha256": _array_sha256(prediction, "<f8"),
        "validation_history_sha256": _array_sha256(history, "<f8"),
        "borders_sha256": _array_sha256(borders, "<f8"),
        "model_sha256": _model_sha256(core, tree_limit=tree_limit),
    }


def _staged_prediction_at(model, X, rounds):
    prediction = None
    for index, staged in enumerate(model.staged_predict(X), start=1):
        if index == rounds:
            prediction = np.asarray(staged, dtype=np.float64)
            break
    if prediction is None:
        raise RuntimeError(f"model did not yield staged round {rounds}")
    return prediction


def _darko_model(*, linear_leaves):
    from darkofit import DarkoRegressor

    return DarkoRegressor(
        iterations=2000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=1.0,
        max_bins=128,
        min_child_weight=1.0,
        linear_leaves=bool(linear_leaves),
        early_stopping=True,
        use_best_model=True,
        refit=False,
        random_state=RANDOM_STATE,
        thread_count=THREADS,
    )


def _fit(model, X_train, y_train, X_validation, y_validation):
    started = time.perf_counter_ns()
    model.fit(
        X_train,
        y_train,
        eval_set=(X_validation, y_validation),
    )
    return (time.perf_counter_ns() - started) / 1e9


def _evaluate_fold(task, X, y, task_id, fold):
    outer_train, outer_test = task.get_train_test_split_indices(
        repeat=0, fold=fold, sample=0
    )
    subtrain, validation = next(
        ShuffleSplit(
            n_splits=1, test_size=0.20, random_state=RANDOM_STATE
        ).split(X.iloc[outer_train])
    )
    fit_indices = np.asarray(outer_train)[subtrain]
    validation_indices = np.asarray(outer_train)[validation]
    eval_X = X.iloc[validation_indices]
    eval_y = y.iloc[validation_indices]

    base_candidates = []
    darko_fit_seconds = 0.0
    for linear in (False, True):
        model = _darko_model(linear_leaves=linear)
        darko_fit_seconds += _fit(
            model,
            X.iloc[fit_indices],
            y.iloc[fit_indices],
            eval_X,
            eval_y,
        )
        base_candidates.append(model)
    base = min(
        base_candidates,
        key=lambda model: (
            float(model.best_score_),
            bool(model.linear_leaves),
        ),
    )
    base_linear = bool(base.linear_leaves)
    pairs = candidate_pairs(base.feature_importances_, (), X.shape[1])
    augmented = augment_numeric_crosses(X, pairs)
    crossed = _darko_model(linear_leaves=base_linear)
    darko_fit_seconds += _fit(
        crossed,
        augmented[fit_indices],
        y.iloc[fit_indices],
        augmented[validation_indices],
        eval_y,
    )
    cross_selected = float(crossed.best_score_) < float(base.best_score_)
    selected = crossed if cross_selected else base
    selected_prediction = selected.predict(
        augmented[outer_test] if cross_selected else X.iloc[outer_test]
    )
    base_prediction = base.predict(X.iloc[outer_test])

    from chimeraboost import ChimeraBoostRegressor

    chimera = ChimeraBoostRegressor(
        n_estimators=2000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=1.0,
        max_bins=128,
        min_child_weight=1.0,
        linear_leaves=None,
        cross_features=None,
        selection_rounds=None,
        early_stopping=True,
        random_state=RANDOM_STATE,
        thread_count=THREADS,
    )
    chimera_fit_seconds = _fit(
        chimera,
        X.iloc[fit_indices],
        y.iloc[fit_indices],
        eval_X,
        eval_y,
    )
    chimera_actual_prediction = chimera.predict(X.iloc[outer_test])
    chimera_best_rounds = int(np.argmin(chimera.model_.valid_history_)) + 1
    chimera_prediction = _staged_prediction_at(
        chimera, X.iloc[outer_test], chimera_best_rounds
    )

    y_test = y.iloc[outer_test]
    base_fingerprint = _fingerprint(base, base_prediction, y_test)
    selected_fingerprint = _fingerprint(
        selected, selected_prediction, y_test
    )
    chimera_fingerprint = _fingerprint(
        chimera,
        chimera_prediction,
        y_test,
        tree_limit=chimera_best_rounds,
    )
    chimera_actual_fingerprint = _fingerprint(
        chimera,
        chimera_actual_prediction,
        y_test,
        tree_limit=len(chimera.model_.trees_),
    )
    exact_fields = (
        "best_prefix_tree_count",
        "fingerprinted_tree_count",
        "best_validation_rmse",
        "test_rmse",
        "prediction_sha256",
        "validation_history_sha256",
        "borders_sha256",
        "model_sha256",
    )
    mismatches = [
        field
        for field in exact_fields
        if selected_fingerprint[field] != chimera_fingerprint[field]
    ]
    chimera_pairs = list(chimera.cross_pairs_ or ())
    selected_pairs = pairs if cross_selected else []
    if (
        bool(chimera.linear_leaves_selected_) != base_linear
        or bool(chimera.cross_features_selected_) != cross_selected
        or chimera_pairs != selected_pairs
        or mismatches
    ):
        raise RuntimeError(
            f"external/native cross mismatch on {task_id}/{fold}: "
            f"fields={mismatches}"
        )
    return {
        "task_id": int(task_id),
        "dataset_name": TASKS[task_id],
        "fold": int(fold),
        "outer_train_index_sha256": _array_sha256(outer_train, "<i8"),
        "outer_test_index_sha256": _array_sha256(outer_test, "<i8"),
        "fit_index_sha256": _array_sha256(fit_indices, "<i8"),
        "validation_index_sha256": _array_sha256(
            validation_indices, "<i8"
        ),
        "base_linear_selected": base_linear,
        "cross_selected": cross_selected,
        "candidate_cross_pairs": [list(pair) for pair in pairs],
        "selected_cross_pairs": [list(pair) for pair in selected_pairs],
        "base": base_fingerprint,
        "selected": selected_fingerprint,
        "chimera": chimera_fingerprint,
        "chimera_actual": chimera_actual_fingerprint,
        "external_native_exact": True,
        "darko_total_fit_seconds": float(darko_fit_seconds),
        "chimera_total_fit_seconds": float(chimera_fit_seconds),
    }


def _geomean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.any(values <= 0.0):
        raise RuntimeError("geomean requires positive values")
    return float(np.exp(np.mean(np.log(values))))


def analyze(rows):
    expected = {(task_id, fold) for task_id in TASKS for fold in FOLDS}
    actual = {(int(row["task_id"]), int(row["fold"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("smooth cross-feature coordinate set is incomplete")
    if not all(row["external_native_exact"] for row in rows):
        raise RuntimeError("external/native cross parity is not exact")

    datasets = {}
    for task_id, name in TASKS.items():
        task_rows = [row for row in rows if row["task_id"] == task_id]
        ratios = [
            row["selected"]["test_rmse"] / row["base"]["test_rmse"]
            for row in task_rows
        ]
        datasets[name] = {
            "task_id": task_id,
            "geomean_ratio": _geomean(ratios),
            "worst_split_ratio": float(max(ratios)),
            "cross_selected_count": int(
                sum(row["cross_selected"] for row in task_rows)
            ),
            "linear_selected_count": int(
                sum(row["base_linear_selected"] for row in task_rows)
            ),
        }
    dataset_ratios = [
        record["geomean_ratio"] for record in datasets.values()
    ]
    leave_one_out = {
        name: _geomean(
            [
                record["geomean_ratio"]
                for other, record in datasets.items()
                if other != name
            ]
        )
        for name in datasets
    }
    return {
        "claim_tier": "development_diagnostic_only",
        "fresh_claim_eligible": False,
        "external_native_exact": True,
        "coordinate_count": len(rows),
        "equal_dataset_geomean_ratio": _geomean(dataset_ratios),
        "worst_dataset_ratio": float(max(dataset_ratios)),
        "worst_split_ratio": float(
            max(
                row["selected"]["test_rmse"] / row["base"]["test_rmse"]
                for row in rows
            )
        ),
        "leave_one_out_equal_dataset_ratios": leave_one_out,
        "datasets": datasets,
        "cross_selected_coordinates": int(
            sum(row["cross_selected"] for row in rows)
        ),
        "linear_selected_coordinates": int(
            sum(row["base_linear_selected"] for row in rows)
        ),
        "summed_darko_fit_seconds": float(
            sum(row["darko_total_fit_seconds"] for row in rows)
        ),
        "summed_chimera_fit_seconds": float(
            sum(row["chimera_total_fit_seconds"] for row in rows)
        ),
        "timing_claim_eligible": False,
    }


def run(output):
    darko_source = _source_state(ROOT)
    chimera_source = _source_state(
        CHIMERA_ROOT, expected_head=EXPECTED_CHIMERA_HEAD
    )
    partition = _partition_boundary()
    rows = []
    tasks = {}
    for task_id in TASKS:
        task, X, y, metadata = _load_task(task_id)
        tasks[str(task_id)] = metadata
        for fold in FOLDS:
            rows.append(_evaluate_fold(task, X, y, task_id, fold))
    analysis = analyze(rows)
    if _git(ROOT, "rev-parse", "HEAD") != darko_source["head"]:
        raise RuntimeError("DarkoFit source changed during campaign")
    if _git(CHIMERA_ROOT, "rev-parse", "HEAD") != chimera_source["head"]:
        raise RuntimeError("ChimeraBoost source changed during campaign")
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "smooth_cross_features",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "tasks": TASKS,
            "folds": list(FOLDS),
        },
        "sources": {
            "darkofit": darko_source,
            "chimeraboost": chimera_source,
        },
        "partition_boundary": partition,
        "tasks": tasks,
        "results": rows,
        "analysis": analysis,
    }
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = run(args.output.resolve())
    print(json.dumps(artifact["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
