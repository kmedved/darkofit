#!/usr/bin/env python3
"""Run the spent-data RSSI linear-leaf parity diagnosis."""

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

TASK_ID = 363132
DATASET_ID = 45718
TASK_NAME = "3D_Estimation_using_RSSI_of_WLAN_dataset"
TARGET_NAME = "Receiver_Height"
OUTER_REPEAT = 0
OUTER_FOLD = 0
OUTER_SAMPLE = 0
RANDOM_STATE = 4
THREADS = 6
EXPECTED_SHAPE = (5760, 7)
EXPECTED_SPLIT_DIMENSIONS = (1, 10, 1)
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
PROTOCOL = ROOT / "benchmarks" / "rssi_linear_leaf_diagnosis_protocol.md"
REGISTRY = ROOT / "benchmarks" / "fresh_confirmation_registry.json"
PRIOR_RESULT = ROOT / "benchmarks" / "fresh_selector_confirmation.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "rssi_linear_leaf_diagnosis.json"

ARMS = (
    "darko_default",
    "darko_matched_auto10_linear",
    "darko_matched_auto20_linear",
    "darko_shared_constant",
    "darko_shared_linear",
    "chimera_shared_constant",
    "chimera_shared_linear",
    "chimera_full_selector",
    "chimera_capped_selector",
    "chimera_full_product",
    "chimera_product",
)

EXACT_FIELDS = (
    "borders_sha256",
    "validation_history_sha256",
    "model_sha256",
    "prediction_sha256",
    "fitted_tree_count",
    "best_validation_rmse",
    "test_rmse",
)


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


def _source_state(path: Path, *, expected_head: str | None = None) -> dict[str, Any]:
    head = _git(path, "rev-parse", "HEAD")
    status = _git(path, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"source tree is not clean: {path}")
    if expected_head is not None and head != expected_head:
        raise RuntimeError(
            f"unexpected source head for {path}: {head} != {expected_head}"
        )
    return {
        "path": str(path),
        "head": head,
        "branch": _git(path, "branch", "--show-current"),
        "clean": True,
    }


def _verify_spent_boundary() -> dict[str, Any]:
    registry = json.loads(REGISTRY.read_text())
    coordinates = {
        (
            int(row["task_id"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["sample"]),
        )
        for row in registry["coordinates"]
    }
    coordinate = (TASK_ID, OUTER_REPEAT, OUTER_FOLD, OUTER_SAMPLE)
    if coordinate not in coordinates:
        raise RuntimeError("RSSI diagnostic coordinate is not declared spent")

    prior = json.loads(PRIOR_RESULT.read_text())
    scored = False
    for result in prior["results"]:
        if int(result["task_id"]) != TASK_ID:
            continue
        if any(
            int(row["fold"]) == OUTER_FOLD
            and int(row.get("repeat", 0)) == OUTER_REPEAT
            and int(row.get("sample", 0)) == OUTER_SAMPLE
            for row in result["folds"]
        ):
            scored = True
            break
    if not scored:
        raise RuntimeError("RSSI diagnostic coordinate lacks a prior outcome")
    return {
        "coordinate": {
            "task_id": TASK_ID,
            "repeat": OUTER_REPEAT,
            "fold": OUTER_FOLD,
            "sample": OUTER_SAMPLE,
        },
        "registry_sha256": _sha256(REGISTRY),
        "prior_result_sha256": _sha256(PRIOR_RESULT),
        "prior_outcome_exists": True,
        "fresh_claim_eligible": False,
    }


def _load_data():
    import openml

    task = openml.tasks.get_task(TASK_ID, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical, _names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    if int(dataset.dataset_id) != DATASET_ID or str(dataset.name) != TASK_NAME:
        raise RuntimeError("RSSI dataset identity changed")
    if str(task.target_name) != TARGET_NAME:
        raise RuntimeError("RSSI target changed")
    if X.shape != EXPECTED_SHAPE or task.get_split_dimensions() != EXPECTED_SPLIT_DIMENSIONS:
        raise RuntimeError("RSSI task shape or split dimensions changed")
    if any(bool(value) for value in categorical):
        raise RuntimeError("RSSI task unexpectedly contains categoricals")
    X_array = np.asarray(X, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(X_array)) or not np.all(np.isfinite(y_array)):
        raise RuntimeError("RSSI task unexpectedly contains non-finite values")

    outer_train, outer_test = task.get_train_test_split_indices(
        repeat=OUTER_REPEAT,
        fold=OUTER_FOLD,
        sample=OUTER_SAMPLE,
    )
    inner_train, inner_validation = next(
        ShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=RANDOM_STATE,
        ).split(X.iloc[outer_train])
    )
    fit_indices = np.asarray(outer_train)[inner_train]
    validation_indices = np.asarray(outer_train)[inner_validation]
    metadata = {
        "task_id": TASK_ID,
        "dataset_id": DATASET_ID,
        "dataset_name": TASK_NAME,
        "target_name": TARGET_NAME,
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "X_sha256": _array_sha256(X_array, "<f8"),
        "y_sha256": _array_sha256(y_array, "<f8"),
        "outer_train_index_sha256": _array_sha256(outer_train, "<i8"),
        "outer_test_index_sha256": _array_sha256(outer_test, "<i8"),
        "shared_fit_index_sha256": _array_sha256(fit_indices, "<i8"),
        "shared_validation_index_sha256": _array_sha256(
            validation_indices, "<i8"
        ),
        "outer_train_rows": int(len(outer_train)),
        "shared_fit_rows": int(len(fit_indices)),
        "shared_validation_rows": int(len(validation_indices)),
        "outer_test_rows": int(len(outer_test)),
    }
    return (
        X,
        y,
        np.asarray(outer_train),
        np.asarray(outer_test),
        fit_indices,
        validation_indices,
        metadata,
    )


def _update_array_hash(digest, label: str, value) -> None:
    digest.update(label.encode("utf-8"))
    if value is None:
        digest.update(b"<none>")
        return
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _model_sha256(core) -> str:
    digest = hashlib.sha256()
    _update_array_hash(digest, "init", np.asarray([core.init_], dtype="<f8"))
    for index, tree in enumerate(core.trees_):
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


def _record(
    arm: str,
    model,
    prediction,
    y_test,
    fit_seconds: float,
    *,
    library: str,
) -> dict[str, Any]:
    core = model.model_
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != np.asarray(y_test).shape or not np.all(
        np.isfinite(prediction)
    ):
        raise RuntimeError(f"{arm} produced invalid predictions")
    history = np.asarray(getattr(core, "valid_history_", ()), dtype=np.float64)
    borders = np.concatenate(
        [
            np.asarray(border, dtype=np.float64)
            for border in core.prep_.binner_.borders_
        ]
    )
    tree = core.trees_[0]
    linear_features = getattr(
        tree, "linear_features", getattr(tree, "lin_feats", None)
    )
    selected_linear = getattr(model, "linear_leaves_selected_", None)
    selected_cross = getattr(model, "cross_features_selected_", None)
    return {
        "arm": arm,
        "library": library,
        "fit_seconds": float(fit_seconds),
        "fitted_tree_count": int(len(core.trees_)),
        "resolved_learning_rate": float(core.lr_),
        "best_validation_rmse": (
            None if history.size == 0 else float(np.min(history))
        ),
        "test_rmse": float(
            mean_squared_error(
                np.asarray(y_test, dtype=np.float64), prediction
            )
            ** 0.5
        ),
        "prediction_sha256": _array_sha256(prediction, "<f8"),
        "validation_history_sha256": _array_sha256(history, "<f8"),
        "borders_sha256": _array_sha256(borders, "<f8"),
        "model_sha256": _model_sha256(core),
        "first_tree_splits_feat": np.asarray(tree.splits_feat).tolist(),
        "first_tree_splits_thr": np.asarray(tree.splits_thr).tolist(),
        "first_tree_linear_features": (
            None
            if linear_features is None
            else np.asarray(linear_features).tolist()
        ),
        "linear_leaves_selected": (
            None if selected_linear is None else bool(selected_linear)
        ),
        "cross_features_selected": (
            None if selected_cross is None else bool(selected_cross)
        ),
        "cross_pair_count": int(
            len(getattr(model, "cross_pairs_", None) or ())
        ),
    }


def _fit_darko(
    arm: str,
    X,
    y,
    outer_train,
    fit_indices,
    validation_indices,
    outer_test,
):
    from darkofit import DarkoRegressor

    common = {
        "iterations": 1000,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "min_child_weight": 1.0,
        "random_state": RANDOM_STATE,
        "thread_count": THREADS,
    }
    eval_set = None
    train_indices = outer_train
    if arm == "darko_default":
        params = {
            "random_state": RANDOM_STATE,
            "thread_count": THREADS,
        }
    elif arm == "darko_matched_auto10_linear":
        params = dict(
            common,
            linear_leaves=True,
            early_stopping=True,
            validation_fraction=0.10,
            use_best_model=True,
            refit=False,
        )
    elif arm == "darko_matched_auto20_linear":
        params = dict(
            common,
            linear_leaves=True,
            early_stopping=True,
            validation_fraction=0.20,
            use_best_model=True,
            refit=False,
        )
    elif arm in {"darko_shared_constant", "darko_shared_linear"}:
        params = dict(
            common,
            linear_leaves=arm.endswith("linear"),
            early_stopping=True,
            use_best_model=True,
            refit=False,
        )
        train_indices = fit_indices
        eval_set = (X.iloc[validation_indices], y.iloc[validation_indices])
    else:
        raise ValueError(f"unknown DarkoFit arm: {arm}")
    model = DarkoRegressor(**params)
    started = time.perf_counter_ns()
    model.fit(X.iloc[train_indices], y.iloc[train_indices], eval_set=eval_set)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    prediction = model.predict(X.iloc[outer_test])
    return _record(
        arm,
        model,
        prediction,
        y.iloc[outer_test],
        fit_seconds,
        library="darkofit",
    )


def _fit_chimera(
    arm: str,
    X,
    y,
    outer_train,
    fit_indices,
    validation_indices,
    outer_test,
):
    from chimeraboost import ChimeraBoostRegressor

    common = {
        "n_estimators": 1000,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "min_child_weight": 1.0,
        "random_state": RANDOM_STATE,
        "thread_count": THREADS,
        "early_stopping": True,
    }
    train_indices = fit_indices
    eval_set = (X.iloc[validation_indices], y.iloc[validation_indices])
    if arm == "chimera_shared_constant":
        params = dict(
            common,
            linear_leaves=False,
            cross_features=False,
            selection_rounds=None,
        )
    elif arm == "chimera_shared_linear":
        params = dict(
            common,
            linear_leaves=True,
            cross_features=False,
            selection_rounds=None,
        )
    elif arm == "chimera_full_selector":
        params = dict(
            common,
            linear_leaves=None,
            cross_features=False,
            selection_rounds=None,
        )
    elif arm == "chimera_capped_selector":
        params = dict(
            common,
            linear_leaves=None,
            cross_features=False,
            selection_rounds=100,
        )
    elif arm == "chimera_full_product":
        params = dict(
            common,
            linear_leaves=None,
            cross_features=None,
            selection_rounds=None,
        )
    elif arm == "chimera_product":
        params = {
            "random_state": RANDOM_STATE,
            "thread_count": THREADS,
        }
        train_indices = outer_train
        eval_set = None
    else:
        raise ValueError(f"unknown ChimeraBoost arm: {arm}")
    model = ChimeraBoostRegressor(**params)
    started = time.perf_counter_ns()
    model.fit(X.iloc[train_indices], y.iloc[train_indices], eval_set=eval_set)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    prediction = model.predict(X.iloc[outer_test])
    return _record(
        arm,
        model,
        prediction,
        y.iloc[outer_test],
        fit_seconds,
        library="chimeraboost",
    )


def _exact_pair(records: dict[str, dict[str, Any]], left: str, right: str):
    mismatches = [
        field
        for field in EXACT_FIELDS
        if records[left][field] != records[right][field]
    ]
    if mismatches:
        raise RuntimeError(
            f"{left} and {right} differ in exact fields: {mismatches}"
        )
    return {
        "left": left,
        "right": right,
        "exact_fields": list(EXACT_FIELDS),
        "exact": True,
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = {row["arm"]: row for row in rows}
    missing = sorted(set(ARMS) - set(records))
    extra = sorted(set(records) - set(ARMS))
    if missing or extra or len(rows) != len(ARMS):
        raise RuntimeError(f"arm set mismatch: missing={missing}, extra={extra}")

    parity = [
        _exact_pair(records, "darko_shared_constant", "chimera_shared_constant"),
        _exact_pair(records, "darko_shared_linear", "chimera_shared_linear"),
        _exact_pair(
            records, "darko_matched_auto20_linear", "darko_shared_linear"
        ),
    ]
    constant_best = records["chimera_shared_constant"]["best_validation_rmse"]
    linear_best = records["chimera_shared_linear"]["best_validation_rmse"]
    full_winner = "linear" if linear_best < constant_best else "constant"
    full_selected = (
        "linear"
        if records["chimera_full_selector"]["linear_leaves_selected"]
        else "constant"
    )
    if full_selected != full_winner:
        raise RuntimeError("full selector disagrees with forced full-budget race")
    capped_selected = (
        "linear"
        if records["chimera_capped_selector"]["linear_leaves_selected"]
        else "constant"
    )

    product = records["chimera_product"]["test_rmse"]
    default = records["darko_default"]["test_rmse"]
    shared_constant = records["darko_shared_constant"]["test_rmse"]
    shared_linear = records["darko_shared_linear"]["test_rmse"]
    return {
        "claim_tier": "development_diagnostic_only",
        "fresh_claim_eligible": False,
        "parity_checks": parity,
        "forced_full_budget_validation_winner": full_winner,
        "full_selector_winner": full_selected,
        "capped_selector_winner": capped_selected,
        "capped_selector_disagrees_with_full": capped_selected != full_winner,
        "chimera_product_linear_selected": records["chimera_product"][
            "linear_leaves_selected"
        ],
        "chimera_product_cross_selected": records["chimera_product"][
            "cross_features_selected"
        ],
        "chimera_full_product_cross_selected": records[
            "chimera_full_product"
        ]["cross_features_selected"],
        "test_rmse_ratios": {
            "darko_default_over_chimera_product": float(default / product),
            "darko_shared_constant_over_chimera_product": float(
                shared_constant / product
            ),
            "darko_shared_linear_over_chimera_product": float(
                shared_linear / product
            ),
            "shared_linear_over_shared_constant": float(
                shared_linear / shared_constant
            ),
            "darko_auto10_linear_over_shared20_linear": float(
                records["darko_matched_auto10_linear"]["test_rmse"]
                / shared_linear
            ),
        },
        "diagnosis": [
            "matched_constant_engine_parity",
            "matched_linear_engine_parity",
            (
                "capped_linear_selection_misselection"
                if capped_selected != full_winner
                else "capped_linear_selection_agrees"
            ),
            (
                "product_cross_features_selected"
                if records["chimera_product"]["cross_features_selected"]
                else "product_cross_features_not_selected"
            ),
        ],
    }


def run(output: Path) -> dict[str, Any]:
    darko_source = _source_state(ROOT)
    chimera_source = _source_state(
        CHIMERA_ROOT, expected_head=EXPECTED_CHIMERA_HEAD
    )
    spent = _verify_spent_boundary()
    (
        X,
        y,
        outer_train,
        outer_test,
        fit_indices,
        validation_indices,
        data,
    ) = _load_data()

    rows = []
    for arm in ARMS:
        if arm.startswith("darko_"):
            row = _fit_darko(
                arm,
                X,
                y,
                outer_train,
                fit_indices,
                validation_indices,
                outer_test,
            )
        else:
            row = _fit_chimera(
                arm,
                X,
                y,
                outer_train,
                fit_indices,
                validation_indices,
                outer_test,
            )
        rows.append(row)

    analysis = analyze(rows)
    if _git(ROOT, "rev-parse", "HEAD") != darko_source["head"]:
        raise RuntimeError("DarkoFit source head changed during diagnosis")
    if _git(CHIMERA_ROOT, "rev-parse", "HEAD") != chimera_source["head"]:
        raise RuntimeError("ChimeraBoost source head changed during diagnosis")

    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "rssi_linear_leaf_diagnosis",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "arms": list(ARMS),
            "timing_claim_eligible": False,
        },
        "sources": {
            "darkofit": darko_source,
            "chimeraboost": chimera_source,
        },
        "spent_boundary": spent,
        "data": data,
        "results": rows,
        "analysis": analysis,
    }
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = run(args.output.resolve())
    print(json.dumps(artifact["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

