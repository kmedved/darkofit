#!/usr/bin/env python3
"""Run the catcross v1 mixed-feature basketball/cold-player guardrail."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

try:
    from . import basketball_harness as basketball
    from . import run_basketball_creator_benchmark as creator
except ImportError:  # direct script execution
    import basketball_harness as basketball
    import run_basketball_creator_benchmark as creator


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve()
GUARDRAIL_ID = "group-centered-categorical-crosses-v1-sports-guardrail-20260723"
CANDIDATE_HEAD = "c3f2608cd3033cfc00aa0737897a92ed868b5865"
THREADS = 14
ITERATIONS = 1_000
RANDOM_STATE = 4
WORKER_TIMEOUT_SECONDS = 7_200.0
ARMS = ("control", "automatic")
CAT_COLUMNS = ("Pos", "Age_cat", "Tm", "starter")
EXPECTED_RAW_SHA256 = (
    "43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2"
)
EXPECTED_TRAIN_ROWS = 5_241
EXPECTED_HOLDOUT_ROWS = 2_409
EXPECTED_COLD_ROWS = 585


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: Any, *, dtype="<f8") -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _strings_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    for item in np.asarray(value, dtype=object).reshape(-1):
        encoded = str(item).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    if Path(_git(repository, "rev-parse", "--show-toplevel")).resolve() != repository:
        raise RuntimeError("source path must name its Git root")
    status = _git(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    return {
        "path": str(repository),
        "head": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "clean": not status,
        "status": status,
    }


def _mixed_view(cache_path: Path) -> dict[str, Any]:
    import pandas as pd

    dataset = basketball.load_basketball_dataset(cache_path)
    if (
        dataset.raw_metadata.get("sha256") != EXPECTED_RAW_SHA256
        or len(dataset.X) != EXPECTED_TRAIN_ROWS
        or len(dataset.player_guardrail.X_holdout) != EXPECTED_HOLDOUT_ROWS
        or int(np.sum(dataset.player_guardrail.cold_player_mask))
        != EXPECTED_COLD_ROWS
    ):
        raise RuntimeError("basketball guardrail source changed")
    frame = dataset.frame.copy()
    frame["starter"] = np.where(frame["GS"] / frame["G"] >= 0.5, "1", "0")
    frame["Age_cat"] = frame["Age"].astype("Int64").astype(str)
    train_index = dataset.X.index
    holdout_index = dataset.player_guardrail.X_holdout.index

    def combine(numeric, index):
        numeric = numeric.copy()
        categories = frame.loc[index, list(CAT_COLUMNS)].copy()
        categories.index = numeric.index
        for column in CAT_COLUMNS:
            categories[column] = categories[column].astype("category")
        combined = pd.concat([numeric, categories], axis=1)
        if not combined.columns.is_unique or combined.isnull().any().any():
            raise RuntimeError("mixed basketball feature view is invalid")
        return combined

    X_train = combine(dataset.X, train_index)
    X_holdout = combine(
        dataset.player_guardrail.X_holdout,
        holdout_index,
    )
    y_train = dataset.y.to_numpy(dtype=np.float64)
    y_holdout = dataset.player_guardrail.y_holdout.to_numpy(dtype=np.float64)
    train_players = frame.loc[train_index, "Player"].astype(str).to_numpy()
    cold_mask = np.asarray(
        dataset.player_guardrail.cold_player_mask, dtype=np.bool_
    )
    if (
        X_train.shape != (EXPECTED_TRAIN_ROWS, 19)
        or X_holdout.shape != (EXPECTED_HOLDOUT_ROWS, 19)
        or y_train.shape != (EXPECTED_TRAIN_ROWS,)
        or y_holdout.shape != (EXPECTED_HOLDOUT_ROWS,)
        or train_players.shape != (EXPECTED_TRAIN_ROWS,)
        or cold_mask.shape != (EXPECTED_HOLDOUT_ROWS,)
    ):
        raise RuntimeError("mixed basketball view shape changed")
    return {
        "dataset": dataset,
        "X_train": X_train,
        "X_holdout": X_holdout,
        "y_train": y_train,
        "y_holdout": y_holdout,
        "train_players": train_players,
        "cold_mask": cold_mask,
        "metadata": {
            "raw_sha256": dataset.raw_metadata["sha256"],
            "numeric_train_sha256": dataset.processed_metadata["x_train_sha256"],
            "target_sha256": dataset.processed_metadata["y_train_sha256"],
            "fold_sha256": dataset.fold_fingerprint_sha256,
            "mixed_train_numeric_sha256": _array_sha256(
                X_train.iloc[:, :15], dtype="<f8"
            ),
            "mixed_holdout_numeric_sha256": _array_sha256(
                X_holdout.iloc[:, :15], dtype="<f8"
            ),
            "mixed_train_categorical_sha256": _strings_sha256(
                X_train.loc[:, list(CAT_COLUMNS)]
            ),
            "mixed_holdout_categorical_sha256": _strings_sha256(
                X_holdout.loc[:, list(CAT_COLUMNS)]
            ),
            "train_players_sha256": _strings_sha256(train_players),
            "cold_mask_sha256": _array_sha256(cold_mask, dtype=np.uint8),
            "columns": [str(column) for column in X_train.columns],
            "categorical_columns": list(CAT_COLUMNS),
            "train_rows": len(X_train),
            "holdout_rows": len(X_holdout),
            "cold_rows": int(cold_mask.sum()),
        },
    }


def coordinate_specs(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = []
    folds = list(creator.creator_cv().split(view["X_train"], view["y_train"]))
    if len(folds) != 10:
        raise RuntimeError("basketball creator fold count changed")
    for fold, (train_index, test_index) in enumerate(folds):
        specs.append(
            {
                "coordinate": f"fold_{fold}",
                "kind": "fold",
                "fold": fold,
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "train_index_sha256": _array_sha256(train_index, dtype="<i8"),
                "test_index_sha256": _array_sha256(test_index, dtype="<i8"),
            }
        )
    specs.append(
        {
            "coordinate": "held_team",
            "kind": "held_team",
            "fold": None,
            "train_rows": EXPECTED_TRAIN_ROWS,
            "test_rows": EXPECTED_HOLDOUT_ROWS,
            "cold_rows": EXPECTED_COLD_ROWS,
        }
    )
    return specs


def build_manifest(cache_path: Path) -> dict[str, Any]:
    cache_path = cache_path.expanduser().resolve()
    view = _mixed_view(cache_path)
    return {
        "schema_version": 1,
        "guardrail_id": GUARDRAIL_ID,
        "status": "ready",
        "kind": "spent_cold_player_sports_guardrail",
        "candidate_head": CANDIDATE_HEAD,
        "data_cache": {
            "path": str(cache_path),
            "bytes": cache_path.stat().st_size,
            "sha256": file_sha256(cache_path),
        },
        "data": view["metadata"],
        "coordinates": coordinate_specs(view),
        "arms": list(ARMS),
        "model": {
            "iterations": ITERATIONS,
            "loss": "RMSE",
            "tree_mode": "catboost",
            "random_state": RANDOM_STATE,
            "thread_count": THREADS,
            "diagnostic_warnings": "never",
            "groups": "player_identity",
        },
        "checks": {
            "fold_geomean_at_most": 1.0,
            "worst_fold_at_most": 1.02,
            "held_team_at_most": 1.0,
            "cold_player_at_most": 1.0,
            "eligible_all_coordinates": True,
        },
        "interpretation": [
            "The data, creator folds, held-team split, and cold-player mask are spent.",
            "The mixed view adds four natural categorical fields to the established "
            "15 numeric features so numeric-by-category crosses can engage.",
            "This is an opt-in sports guardrail, not holdout or default evidence.",
        ],
    }


def _model():
    from darkofit import DarkoRegressor

    return DarkoRegressor(
        iterations=ITERATIONS,
        loss="RMSE",
        tree_mode="catboost",
        random_state=RANDOM_STATE,
        thread_count=THREADS,
        diagnostic_warnings="never",
    )


def _rmse(truth, prediction) -> float:
    value = float(
        math.sqrt(
            mean_squared_error(
                np.asarray(truth, dtype=np.float64),
                np.asarray(prediction, dtype=np.float64),
            )
        )
    )
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("sports guardrail RMSE is invalid")
    return value


def _prediction_sha256(prediction: Any) -> str:
    return _array_sha256(prediction, dtype="<f8")


def run_worker(
    spec: Mapping[str, Any],
    *,
    source: Path,
    cache_path: Path,
    arm: str,
) -> dict[str, Any]:
    import numba

    source = source.expanduser().resolve()
    state = source_state(source)
    if not state["clean"] or state["head"] != CANDIDATE_HEAD:
        raise RuntimeError("catcross candidate source changed")
    sys.path.insert(0, str(source))
    import darkofit

    if Path(darkofit.__file__).resolve().parents[1] != source:
        raise RuntimeError("sports worker imported DarkoFit from the wrong source")
    if os.cpu_count() != THREADS:
        raise RuntimeError("sports guardrail requires the 14-CPU host")
    view = _mixed_view(cache_path)
    expected_spec = next(
        row
        for row in coordinate_specs(view)
        if row["coordinate"] == spec["coordinate"]
    )
    if dict(spec) != expected_spec:
        raise RuntimeError("sports guardrail coordinate changed")
    if spec["kind"] == "fold":
        train_index, test_index = list(
            creator.creator_cv().split(view["X_train"], view["y_train"])
        )[int(spec["fold"])]
        X_fit = view["X_train"].iloc[train_index]
        y_fit = view["y_train"][train_index]
        groups = view["train_players"][train_index]
        X_test = view["X_train"].iloc[test_index]
        y_test = view["y_train"][test_index]
        masks = {"all": np.ones(len(y_test), dtype=np.bool_)}
    else:
        X_fit = view["X_train"]
        y_fit = view["y_train"]
        groups = view["train_players"]
        X_test = view["X_holdout"]
        y_test = view["y_holdout"]
        masks = {
            "all": np.ones(len(y_test), dtype=np.bool_),
            "seen_player": ~view["cold_mask"],
            "cold_player": view["cold_mask"],
        }
    model = _model()
    if arm == "control":
        model._group_centered_crosses_private_mode = "off"
    elif arm != "automatic":
        raise RuntimeError(f"unknown sports guardrail arm: {arm}")
    ambient = int(numba.get_num_threads())
    started = time.perf_counter()
    model.fit(
        X_fit,
        y_fit,
        cat_features=list(CAT_COLUMNS),
        groups=groups,
    )
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = time.perf_counter() - started
    if (
        prediction.shape != y_test.shape
        or not np.isfinite(prediction).all()
        or int(numba.get_num_threads()) != ambient
    ):
        raise RuntimeError("sports guardrail prediction or thread state is invalid")
    selector = getattr(model, "group_centered_categorical_crosses_", None)
    fitted_pairs = [
        [int(numeric), int(categorical)]
        for numeric, categorical in getattr(
            model.model_.prep_, "group_centered_pairs_", ()
        )
    ]
    if arm == "control":
        if selector is not None or fitted_pairs:
            raise RuntimeError("sports control unexpectedly used catcross")
    elif (
        not isinstance(selector, Mapping)
        or selector.get("eligible") is not True
        or not isinstance(selector.get("selected"), bool)
        or not selector.get("pairs")
        or fitted_pairs
        != (selector["pairs"] if selector["selected"] else [])
    ):
        raise RuntimeError("sports automatic selector provenance is invalid")
    metrics = {}
    for name, mask in masks.items():
        truth = y_test[mask]
        predicted = prediction[mask]
        metrics[name] = {
            "rows": int(mask.sum()),
            "rmse": _rmse(truth, predicted),
            "r2": float(r2_score(truth, predicted)),
            "prediction_sha256": _prediction_sha256(predicted),
        }
    return {
        "schema_version": 1,
        "guardrail_id": GUARDRAIL_ID,
        "status": "ok",
        "coordinate": spec["coordinate"],
        "kind": spec["kind"],
        "fold": spec["fold"],
        "arm": arm,
        "source": state,
        "fit_rows": int(len(y_fit)),
        "test_rows": int(len(y_test)),
        "feature_count": int(X_fit.shape[1]),
        "cat_features": list(CAT_COLUMNS),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "prediction_sha256": _prediction_sha256(prediction),
        "metrics": metrics,
        "selector": selector,
        "fitted_pairs": fitted_pairs,
        "tree_count": int(len(model.model_.trees_)),
        "resolved_threads": int(model.model_.n_threads_),
        "ambient_thread_restored": True,
    }


def _geomean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or not array.size
        or np.any(array <= 0.0)
        or not np.isfinite(array).all()
    ):
        raise RuntimeError("invalid sports guardrail geomean input")
    return float(np.exp(np.mean(np.log(array))))


def _valid_metric(metric: Any) -> bool:
    if not isinstance(metric, Mapping):
        return False
    try:
        rows = int(metric["rows"])
        rmse = float(metric["rmse"])
        r2 = float(metric["r2"])
        prediction_sha256 = str(metric["prediction_sha256"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        set(metric) == {"rows", "rmse", "r2", "prediction_sha256"}
        and rows > 0
        and math.isfinite(rmse)
        and rmse > 0.0
        and math.isfinite(r2)
        and len(prediction_sha256) == 64
        and all(character in "0123456789abcdef" for character in prediction_sha256)
    )


def analyze(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected_coordinates = {
        row["coordinate"]: row for row in manifest["coordinates"]
    }
    if len(rows) != 2 * len(expected_coordinates):
        raise RuntimeError("sports guardrail row census changed")
    indexed = {}
    for row in rows:
        key = (str(row["coordinate"]), str(row["arm"]))
        if (
            key in indexed
            or row.get("schema_version") != 1
            or row.get("guardrail_id") != GUARDRAIL_ID
            or row.get("status") != "ok"
            or key[0] not in expected_coordinates
            or key[1] not in ARMS
            or row.get("kind") != expected_coordinates[key[0]]["kind"]
            or row.get("fold") != expected_coordinates[key[0]]["fold"]
            or not math.isfinite(float(row.get("fit_seconds", math.nan)))
            or float(row["fit_seconds"]) <= 0.0
            or not math.isfinite(float(row.get("predict_seconds", math.nan)))
            or float(row["predict_seconds"]) <= 0.0
            or row.get("resolved_threads") != THREADS
            or row.get("ambient_thread_restored") is not True
            or row.get("feature_count") != 19
            or row.get("cat_features") != list(CAT_COLUMNS)
            or row.get("source", {}).get("head") != CANDIDATE_HEAD
            or row.get("source", {}).get("clean") is not True
            or set(row.get("metrics", {}))
            != (
                {"all"}
                if expected_coordinates[key[0]]["kind"] == "fold"
                else {"all", "seen_player", "cold_player"}
            )
            or any(
                not _valid_metric(metric)
                for metric in row.get("metrics", {}).values()
            )
        ):
            raise RuntimeError("invalid sports guardrail worker row")
        indexed[key] = row
    if set(indexed) != {
        (coordinate, arm)
        for coordinate in expected_coordinates
        for arm in ARMS
    }:
        raise RuntimeError("sports guardrail coordinate grid changed")

    fold_rows = []
    selected = 0
    eligible = 0
    fit_ratios = []
    predict_ratios = []
    for coordinate, spec in expected_coordinates.items():
        control = indexed[(coordinate, "control")]
        automatic = indexed[(coordinate, "automatic")]
        selector = automatic.get("selector")
        if (
            control.get("selector") is not None
            or control.get("fitted_pairs") != []
            or not isinstance(selector, Mapping)
            or selector.get("eligible") is not True
            or not isinstance(selector.get("selected"), bool)
            or not selector.get("pairs")
            or automatic.get("fitted_pairs")
            != (selector["pairs"] if selector["selected"] else [])
        ):
            raise RuntimeError("sports guardrail selector provenance changed")
        eligible += 1
        selected += int(selector["selected"])
        if not selector["selected"] and (
            automatic["prediction_sha256"] != control["prediction_sha256"]
            or automatic["metrics"] != control["metrics"]
        ):
            raise RuntimeError("declined sports selector did not fall back exactly")
        fit_ratios.append(
            float(automatic["fit_seconds"]) / float(control["fit_seconds"])
        )
        predict_ratios.append(
            float(automatic["predict_seconds"])
            / float(control["predict_seconds"])
        )
        if spec["kind"] == "fold":
            control_rmse = float(control["metrics"]["all"]["rmse"])
            automatic_rmse = float(automatic["metrics"]["all"]["rmse"])
            fold_rows.append(
                {
                    "coordinate": coordinate,
                    "fold": int(spec["fold"]),
                    "rmse_ratio": automatic_rmse / control_rmse,
                    "r2_delta": (
                        float(automatic["metrics"]["all"]["r2"])
                        - float(control["metrics"]["all"]["r2"])
                    ),
                    "selected": bool(selector["selected"]),
                }
            )
    full_control = indexed[("held_team", "control")]
    full_automatic = indexed[("held_team", "automatic")]
    views = {}
    for name in ("all", "seen_player", "cold_player"):
        views[name] = {
            "rmse_ratio": (
                float(full_automatic["metrics"][name]["rmse"])
                / float(full_control["metrics"][name]["rmse"])
            ),
            "r2_delta": (
                float(full_automatic["metrics"][name]["r2"])
                - float(full_control["metrics"][name]["r2"])
            ),
            "rows": int(full_control["metrics"][name]["rows"]),
        }
    fold_ratios = [row["rmse_ratio"] for row in fold_rows]
    checks = manifest["checks"]
    gates = {
        "fold_geomean_at_most_1_0": (
            _geomean(fold_ratios) <= float(checks["fold_geomean_at_most"])
        ),
        "worst_fold_at_most_1_02": (
            max(fold_ratios) <= float(checks["worst_fold_at_most"])
        ),
        "held_team_at_most_1_0": (
            views["all"]["rmse_ratio"] <= float(checks["held_team_at_most"])
        ),
        "cold_player_at_most_1_0": (
            views["cold_player"]["rmse_ratio"]
            <= float(checks["cold_player_at_most"])
        ),
        "eligible_all_coordinates": eligible == len(expected_coordinates),
    }
    gates["passes"] = all(gates.values())
    return {
        "schema_version": 1,
        "guardrail_id": GUARDRAIL_ID,
        "kind": "spent_cold_player_sports_guardrail",
        "integrity": {
            "passes": True,
            "workers": len(rows),
            "pairs": len(expected_coordinates),
        },
        "quality": {
            "fold_equal_geomean_rmse_ratio": _geomean(fold_ratios),
            "worst_fold_rmse_ratio": max(fold_ratios),
            "fold_wins_ties_losses": {
                "wins": sum(value < 1.0 for value in fold_ratios),
                "ties": sum(value == 1.0 for value in fold_ratios),
                "losses": sum(value > 1.0 for value in fold_ratios),
            },
            "folds": fold_rows,
            "held_team_views": views,
        },
        "engagement": {
            "eligible_coordinates": eligible,
            "selected_coordinates": selected,
            "total_coordinates": len(expected_coordinates),
            "held_team_selected": bool(
                full_automatic["selector"]["selected"]
            ),
        },
        "cost_telemetry": {
            "fit_equal_coordinate_geomean_ratio": _geomean(fit_ratios),
            "predict_equal_coordinate_geomean_ratio": _geomean(
                predict_ratios
            ),
        },
        "gates": gates,
        "disposition": (
            "sports_guardrail_supports_scoped_opt_in"
            if gates["passes"]
            else "sports_guardrail_does_not_support_opt_in"
        ),
        "limitations": [
            "Spent basketball development data and established split boundaries.",
            "Mixed-view benchmark only; no holdout, default, or release claim.",
            "Fit and prediction times are single-run telemetry.",
        ],
    }


def output_paths(prefix: Path) -> dict[str, Path]:
    prefix = prefix.expanduser().resolve()
    try:
        prefix.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("sports guardrail outputs must be outside the source tree")
    return {
        "launch": Path(str(prefix) + "_launch.json"),
        "raw": Path(str(prefix) + "_raw.json"),
        "result": Path(str(prefix) + "_result.json"),
    }


def exclusive_machine_audit() -> dict[str, Any]:
    import psutil

    own = {os.getpid()}
    parent = psutil.Process().parent()
    while parent is not None:
        own.add(parent.pid)
        try:
            parent = parent.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            break
    markers = (
        "run_group_centered_categorical_crosses_v1_sports_guardrail",
        "run_group_centered_categorical_crosses_v1_attribution",
        "run_basketball_",
        "run_m6_quality_successor",
        "run_v011_",
        "run_m3",
        "run_b3",
    )
    conflicts = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid not in own and any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"another benchmark process is active: {conflicts}")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "conflicting_benchmark_processes": [],
        "load_average": [float(value) for value in os.getloadavg()],
    }


def _worker_env(source: Path, cache: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(source.resolve()),
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": str(THREADS),
            "OMP_THREAD_LIMIT": str(THREADS),
            "OPENBLAS_NUM_THREADS": str(THREADS),
            "MKL_NUM_THREADS": str(THREADS),
            "NUMEXPR_NUM_THREADS": str(THREADS),
            "NUMBA_NUM_THREADS": str(THREADS),
            "VECLIB_MAXIMUM_THREADS": str(THREADS),
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "DARKOFIT_WARMUP": "0",
            "NUMBA_CACHE_DIR": str(cache),
        }
    )
    return environment


def execute(
    *,
    manifest_path: Path,
    source: Path,
    cache_path: Path,
    prefix: Path,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest != build_manifest(cache_path):
        raise RuntimeError("sports guardrail manifest is invalid")
    paths = output_paths(prefix)
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise RuntimeError(f"sports guardrail output collision: {collisions}")
    harness = source_state(ROOT)
    candidate = source_state(source)
    if not harness["clean"]:
        raise RuntimeError("sports guardrail harness must be clean")
    if not candidate["clean"] or candidate["head"] != CANDIDATE_HEAD:
        raise RuntimeError("sports guardrail candidate source changed")
    if os.cpu_count() != THREADS:
        raise RuntimeError("sports guardrail requires the 14-CPU host")
    launch = {
        "schema_version": 1,
        "guardrail_id": GUARDRAIL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {"harness": harness, "candidate": candidate},
        "source_hashes": {
            "manifest": file_sha256(manifest_path),
            "runner": file_sha256(RUNNER),
            "data_cache": file_sha256(cache_path),
        },
        "exclusive_machine_audit": exclusive_machine_audit(),
        "planned_workers": 2 * len(manifest["coordinates"]),
    }
    _write_create_only(paths["launch"], launch)
    rows = []
    with tempfile.TemporaryDirectory(prefix="darkofit-catcross-sports-") as temp:
        temp_path = Path(temp)
        numba_cache = temp_path / "numba-cache"
        numba_cache.mkdir()
        for index, spec in enumerate(manifest["coordinates"]):
            order = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
            spec_path = temp_path / f"coordinate-{index:02d}.json"
            spec_path.write_bytes(canonical_json_bytes(spec))
            for arm in order:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(RUNNER),
                        "worker",
                        "--spec",
                        str(spec_path),
                        "--source",
                        str(source),
                        "--data-cache",
                        str(cache_path),
                        "--arm",
                        arm,
                    ],
                    cwd=ROOT,
                    env=_worker_env(source, numba_cache),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=WORKER_TIMEOUT_SECONDS,
                )
                if completed.returncode:
                    raise RuntimeError(
                        f"sports guardrail worker {index}/{arm} failed: "
                        + completed.stderr[-5000:]
                    )
                lines = [
                    line for line in completed.stdout.splitlines() if line.strip()
                ]
                if not lines:
                    raise RuntimeError("sports guardrail worker returned no row")
                row = json.loads(lines[-1])
                if row.get("status") != "ok":
                    raise RuntimeError("sports guardrail worker integrity failed")
                rows.append(row)
    if source_state(ROOT) != harness:
        raise RuntimeError("sports guardrail harness changed during execution")
    if source_state(source) != candidate:
        raise RuntimeError("sports guardrail candidate changed during execution")
    raw = {
        "schema_version": 1,
        "guardrail_id": GUARDRAIL_ID,
        "complete": True,
        "launch_sha256": file_sha256(paths["launch"]),
        "manifest": manifest,
        "rows": rows,
    }
    _write_create_only(paths["raw"], raw)
    result = analyze(rows, manifest)
    result["source_hashes"] = {
        "raw": file_sha256(paths["raw"]),
        "runner": file_sha256(RUNNER),
    }
    _write_create_only(paths["result"], result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--data-cache", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--spec", type=Path, required=True)
    worker.add_argument("--source", type=Path, required=True)
    worker.add_argument("--data-cache", type=Path, required=True)
    worker.add_argument("--arm", choices=ARMS, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--data-cache", type=Path, required=True)
    run.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "manifest":
        if args.output.exists() or args.output.is_symlink():
            raise RuntimeError(f"refusing existing output: {args.output}")
        manifest = build_manifest(args.data_cache)
        _write_create_only(args.output, manifest)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "workers": 2 * len(manifest["coordinates"]),
                }
            )
        )
        return 0
    if args.command == "worker":
        print(
            json.dumps(
                run_worker(
                    _load_json(args.spec),
                    source=args.source,
                    cache_path=args.data_cache,
                    arm=args.arm,
                ),
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    result = execute(
        manifest_path=args.manifest,
        source=args.source,
        cache_path=args.data_cache,
        prefix=args.output_prefix,
    )
    print(
        json.dumps(
            {
                "fold_ratio": result["quality"][
                    "fold_equal_geomean_rmse_ratio"
                ],
                "cold_ratio": result["quality"]["held_team_views"][
                    "cold_player"
                ]["rmse_ratio"],
                "passes": result["gates"]["passes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
