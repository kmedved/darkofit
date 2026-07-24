#!/usr/bin/env python3
"""Characterize feature-only GPBoost and DarkoFit on spent basketball panel 2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
_ROOT_TEXT = str(ROOT)
if _ROOT_TEXT in sys.path:
    sys.path.remove(_ROOT_TEXT)
sys.path.insert(0, _ROOT_TEXT)
PANEL_MANIFEST = ROOT / "benchmarks" / "basketball_sports_panel_v2_manifest.json"
PANEL_CACHE = ROOT / ".cache" / "basketball-sports-panel-v2" / "panel.csv"
PROTOCOL = ROOT / "benchmarks" / "gpboost_basketball_v1_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "gpboost_basketball_v1_raw_20260723.json"
WORKER_PREFIX = "GPBOOST_BASKETBALL_RESULT="
THREADS = 4
SEED = 4

DARKO_DEFAULT = "darkofit_default"
GPBOOST_DEFAULT = "gpboost_default"
DARKO_MATCHED = "darkofit_matched_budget"
GPBOOST_MATCHED = "gpboost_matched_budget"
ARMS = (DARKO_DEFAULT, GPBOOST_DEFAULT, DARKO_MATCHED, GPBOOST_MATCHED)
LANES = {
    "public_defaults": (DARKO_DEFAULT, GPBOOST_DEFAULT),
    "near_matched_tree_budget": (DARKO_MATCHED, GPBOOST_MATCHED),
}
BLOCK_ORDERS = (
    (DARKO_DEFAULT, GPBOOST_DEFAULT, DARKO_MATCHED, GPBOOST_MATCHED),
    (GPBOOST_MATCHED, DARKO_MATCHED, GPBOOST_DEFAULT, DARKO_DEFAULT),
    (DARKO_MATCHED, GPBOOST_DEFAULT, DARKO_DEFAULT, GPBOOST_MATCHED),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_sha256(path: Path) -> str:
    """Hash a source snapshot without depending on filesystem timestamps."""
    digest = hashlib.sha256()
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file() or file_path.is_symlink():
            continue
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(file_path)))
    return digest.hexdigest()


def _atomic_create(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def _prediction_sha256(values: np.ndarray) -> str:
    data = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _score(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if y_true.shape != prediction.shape or not len(y_true):
        raise RuntimeError("invalid score inputs")
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("model produced non-finite predictions")
    error = y_true - prediction
    return {
        "rows": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "r2": float(r2_score(y_true, prediction)),
        "target_sha256": _prediction_sha256(y_true),
        "prediction_sha256": _prediction_sha256(prediction),
    }


def _load_panel(cache_path: Path, manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    from benchmarks import build_basketball_sports_panel_v2 as builder

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["processed_panel"]
    if (
        manifest.get("name") != "darkofit_basketball_sports_panel_v2"
        or not cache_path.is_file()
        or cache_path.is_symlink()
        or cache_path.stat().st_size != expected["bytes"]
        or _sha256(cache_path) != expected["sha256"]
    ):
        raise RuntimeError("sports panel cache differs from the frozen manifest")
    frame = pd.read_csv(cache_path)
    expected_columns = [
        *builder.IDENTITY_COLUMNS,
        *builder.FEATURE_COLUMNS,
        *builder.TARGET_COLUMNS,
    ]
    if frame.columns.tolist() != expected_columns or len(frame) != expected["rows"]:
        raise RuntimeError("sports panel schema differs from the frozen manifest")
    identities = frame.loc[:, list(builder.IDENTITY_COLUMNS)].values.tolist()
    if _json_sha256(identities) != expected["identities_sha256"]:
        raise RuntimeError("sports panel identities differ from the frozen manifest")
    numeric = frame.loc[
        :, [*builder.FEATURE_COLUMNS, *builder.TARGET_COLUMNS]
    ].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("sports panel contains non-finite values")
    return frame, manifest


def _season_views(
    panel: pd.DataFrame, manifest: dict[str, Any], season: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    from benchmarks import build_basketball_sports_panel_v2 as builder

    split = manifest["split"]["seasons"][str(season)]
    seasonal = panel.loc[panel["year"] == season].reset_index(drop=True)
    held = frozenset(split["held_teams"])
    primary = seasonal.loc[~seasonal["Tm"].isin(held)].reset_index(drop=True)
    holdout = seasonal.loc[seasonal["Tm"].isin(held)].reset_index(drop=True)
    seen = holdout["bref_id"].astype(str).isin(
        frozenset(primary["bref_id"].astype(str))
    ).to_numpy()
    if (
        len(primary) != split["primary_rows"]
        or len(holdout) != split["held_team_rows"]
        or int(np.sum(seen)) != split["seen_player_rows"]
        or int(np.sum(~seen)) != split["cold_player_rows"]
        or _json_sha256(
            primary.loc[:, list(builder.IDENTITY_COLUMNS)].values.tolist()
        )
        != split["primary_identities_sha256"]
        or _json_sha256(
            holdout.loc[:, list(builder.IDENTITY_COLUMNS)].values.tolist()
        )
        != split["held_identities_sha256"]
    ):
        raise RuntimeError(f"sports split differs from manifest for season {season}")
    return primary, holdout, seen, list(split["folds"])


def _import_darkofit(source: Path) -> Any:
    source = source.resolve()
    if not (source / "darkofit" / "__init__.py").is_file():
        raise RuntimeError(f"DarkoFit source snapshot is invalid: {source}")
    source_text = str(source)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    module = importlib.import_module("darkofit")
    module_file = Path(module.__file__).resolve()
    if not module_file.is_relative_to(source):
        raise RuntimeError("DarkoFit imported outside the requested source snapshot")
    return module


def _build_estimator(arm: str, darkofit_source: Path, threads: int) -> Any:
    if arm in (DARKO_DEFAULT, DARKO_MATCHED):
        module = _import_darkofit(darkofit_source)
        params: dict[str, Any] = {
            "random_state": SEED,
            "thread_count": int(threads),
            "diagnostic_warnings": "never",
        }
        if arm == DARKO_MATCHED:
            params.update(
                {
                    "iterations": 1000,
                    "learning_rate": 0.1,
                    "depth": 6,
                    "l2_leaf_reg": 1.0,
                    "max_bins": 128,
                    "subsample": 1.0,
                    "colsample": 1.0,
                    "min_child_samples": 1,
                    "tree_mode": "catboost",
                    "ordered_boosting": False,
                }
            )
        return module.DarkoRegressor(**params)
    if arm in (GPBOOST_DEFAULT, GPBOOST_MATCHED):
        from gpboost import GPBoostRegressor

        params = {"random_state": SEED, "n_jobs": int(threads)}
        if arm == GPBOOST_MATCHED:
            params.update(
                {
                    "n_estimators": 1000,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "num_leaves": 64,
                    "max_bin": 128,
                    "reg_lambda": 1.0,
                    "reg_alpha": 0.0,
                    "min_child_samples": 1,
                    "min_child_weight": 0.001,
                    "min_split_gain": 0.0,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                }
            )
        return GPBoostRegressor(**params)
    raise ValueError(f"unknown arm: {arm}")


def _implementation(model: Any) -> dict[str, Any]:
    package = model.__class__.__module__.split(".", 1)[0]
    module = importlib.import_module(package)
    return {
        "package": package,
        "distribution_version": importlib.metadata.version(package),
        "module_version": getattr(module, "__version__", None),
        "module_file": str(Path(module.__file__).resolve()),
        "estimator_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "parameters": model.get_params(deep=False),
    }


def _fit_metadata(model: Any, arm: str) -> dict[str, Any]:
    if arm in (DARKO_DEFAULT, DARKO_MATCHED):
        core = model.model_
        return {
            "fitted_tree_count": int(model.n_estimators_),
            "best_iteration": int(model.best_n_estimators_),
            "resolved_learning_rate": float(model.learning_rate_),
            "tree_mode": str(core.tree_mode_),
            "thread_count": int(core.n_threads_),
        }
    booster = model.booster_
    return {
        "fitted_tree_count": int(booster.num_trees()),
        "best_iteration": (
            None if model.best_iteration_ is None else int(model.best_iteration_)
        ),
        "configured_n_estimators": int(model.n_estimators),
        "learning_rate": float(model.learning_rate),
        "max_depth": int(model.max_depth),
        "num_leaves": int(model.num_leaves),
        "n_jobs": int(model.n_jobs),
    }


def _fit_predict(
    arm: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    darkofit_source: Path,
    threads: int,
) -> dict[str, Any]:
    model = _build_estimator(arm, darkofit_source, threads)
    implementation = _implementation(model)
    started = time.perf_counter_ns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if arm in (GPBOOST_DEFAULT, GPBOOST_MATCHED):
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    return {
        "prediction": prediction,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "implementation": implementation,
        "fitted": _fit_metadata(model, arm),
    }


def _structural_behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Keep deterministic arm/split/model structure, not floating predictions."""
    cells = []
    for cell in result["cells"]:
        cells.append(
            {
                "season": cell["season"],
                "target": cell["target"],
                "primary": {
                    key: cell["primary"][key] for key in ("rows", "target_sha256")
                },
                "folds": [
                    {
                        "fold": fold["fold"],
                        "train_rows": fold["train_rows"],
                        "test_rows": fold["test_rows"],
                        "score": {
                            key: fold["score"][key]
                            for key in ("rows", "target_sha256")
                        },
                        "implementation": fold["implementation"],
                        "fitted": fold["fitted"],
                    }
                    for fold in cell["folds"]
                ],
                "guardrail": {
                    key: {
                        field: cell["guardrail"][key][field]
                        for field in ("rows", "target_sha256")
                    }
                    for key in ("held_team", "seen_player", "cold_player")
                }
                | {
                    "implementation": cell["guardrail"]["implementation"],
                    "fitted": cell["guardrail"]["fitted"],
                },
            }
        )
    return {
        "arm": result["arm"],
        "thread_environment": result["thread_environment"],
        "cells": cells,
    }


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    from benchmarks import build_basketball_sports_panel_v2 as builder

    panel, manifest = _load_panel(args.panel_cache, args.panel_manifest)
    first, _, _, first_folds = _season_views(panel, manifest, builder.SEASONS[0])
    X_first = first.loc[:, list(builder.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    warmup_fold = first_folds[0]
    warmup_started = time.perf_counter_ns()
    warmup = _fit_predict(
        args.worker_arm,
        X_first[warmup_fold["train_indices"]],
        first[builder.TARGET_COLUMNS[0]].to_numpy(dtype=np.float64)[
            warmup_fold["train_indices"]
        ],
        X_first[warmup_fold["test_indices"]],
        args.darkofit_source,
        args.threads,
    )
    del warmup
    gc.collect()
    warmup_seconds = (time.perf_counter_ns() - warmup_started) / 1e9

    cells: list[dict[str, Any]] = []
    total_fit_seconds = 0.0
    total_predict_seconds = 0.0
    started = time.perf_counter_ns()
    for season in builder.SEASONS:
        primary, holdout, seen, folds = _season_views(panel, manifest, season)
        X_primary = primary.loc[:, list(builder.FEATURE_COLUMNS)].to_numpy(
            dtype=np.float64
        )
        X_holdout = holdout.loc[:, list(builder.FEATURE_COLUMNS)].to_numpy(
            dtype=np.float64
        )
        for target in builder.TARGET_COLUMNS:
            y_primary = primary[target].to_numpy(dtype=np.float64)
            y_holdout = holdout[target].to_numpy(dtype=np.float64)
            oof = np.empty(len(primary), dtype=np.float64)
            fold_rows: list[dict[str, Any]] = []
            for frozen in folds:
                train = np.asarray(frozen["train_indices"], dtype=np.int64)
                test = np.asarray(frozen["test_indices"], dtype=np.int64)
                fitted = _fit_predict(
                    args.worker_arm,
                    X_primary[train],
                    y_primary[train],
                    X_primary[test],
                    args.darkofit_source,
                    args.threads,
                )
                prediction = fitted.pop("prediction")
                oof[test] = prediction
                total_fit_seconds += fitted["fit_seconds"]
                total_predict_seconds += fitted["predict_seconds"]
                fold_rows.append(
                    {
                        "fold": int(frozen["fold"]),
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "score": _score(y_primary[test], prediction),
                        **fitted,
                    }
                )
            guardrail = _fit_predict(
                args.worker_arm,
                X_primary,
                y_primary,
                X_holdout,
                args.darkofit_source,
                args.threads,
            )
            holdout_prediction = guardrail.pop("prediction")
            total_fit_seconds += guardrail["fit_seconds"]
            total_predict_seconds += guardrail["predict_seconds"]
            cells.append(
                {
                    "season": int(season),
                    "target": target,
                    "primary": _score(y_primary, oof),
                    "folds": fold_rows,
                    "guardrail": {
                        "held_team": _score(y_holdout, holdout_prediction),
                        "seen_player": _score(
                            y_holdout[seen], holdout_prediction[seen]
                        ),
                        "cold_player": _score(
                            y_holdout[~seen], holdout_prediction[~seen]
                        ),
                        **guardrail,
                    },
                }
            )
    result = {
        "arm": args.worker_arm,
        "cells": cells,
        "total_fit_seconds": float(total_fit_seconds),
        "total_predict_seconds": float(total_predict_seconds),
        "steady_wall_seconds": float((time.perf_counter_ns() - started) / 1e9),
        "warmup_seconds_outside_timing": float(warmup_seconds),
        "peak_rss_bytes": _peak_rss_bytes(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        },
    }
    result["prediction_fingerprint_sha256"] = _json_sha256(result["cells"])
    result["structural_fingerprint_sha256"] = _json_sha256(
        _structural_behavior_payload(result)
    )
    return result


def _worker_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    thread_value = str(args.threads)
    environment.update(
        {
            "OMP_NUM_THREADS": thread_value,
            "OPENBLAS_NUM_THREADS": thread_value,
            "MKL_NUM_THREADS": thread_value,
            "VECLIB_MAXIMUM_THREADS": thread_value,
            "NUMEXPR_NUM_THREADS": thread_value,
            "NUMBA_NUM_THREADS": thread_value,
            "LOKY_MAX_CPU_COUNT": thread_value,
            "PYTHONHASHSEED": "0",
            "DARKOFIT_WARMUP": "0",
            "MPLCONFIGDIR": str(args.worker_cache.resolve()),
        }
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(args.darkofit_source.resolve()), str(ROOT))
    )
    return environment


def _run_worker_process(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-arm",
        arm,
        "--threads",
        str(args.threads),
        "--panel-cache",
        str(args.panel_cache),
        "--panel-manifest",
        str(args.panel_manifest),
        "--darkofit-source",
        str(args.darkofit_source),
        "--worker-cache",
        str(args.worker_cache),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_worker_environment(args),
        capture_output=True,
        check=False,
        text=True,
    )
    lines = [
        line for line in completed.stdout.splitlines() if line.startswith(WORKER_PREFIX)
    ]
    if completed.returncode or len(lines) != 1:
        raise RuntimeError(
            f"worker {arm} failed with exit code {completed.returncode}"
            f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result = json.loads(lines[0][len(WORKER_PREFIX) :])
    result["worker_stdout"] = "\n".join(
        line for line in completed.stdout.splitlines() if not line.startswith(WORKER_PREFIX)
    ).strip() or None
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def _geomean(values: list[float]) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    if not len(values_array) or np.any(values_array <= 0) or not np.all(
        np.isfinite(values_array)
    ):
        raise RuntimeError("geometric mean received invalid values")
    return float(np.exp(np.mean(np.log(values_array))))


def _metric_view(cell: dict[str, Any], view: str) -> dict[str, Any]:
    if view == "primary":
        return cell["primary"]
    return cell["guardrail"][view]


def _comparison_summary(
    darko: dict[str, Any], gpboost: dict[str, Any], view: str
) -> dict[str, Any]:
    darko_cells = {(cell["season"], cell["target"]): cell for cell in darko["cells"]}
    gpboost_cells = {
        (cell["season"], cell["target"]): cell for cell in gpboost["cells"]
    }
    if set(darko_cells) != set(gpboost_cells):
        raise RuntimeError("arm cells are not comparable")
    ratios = []
    rows = []
    wins = losses = ties = 0
    for key in sorted(darko_cells):
        darko_score = _metric_view(darko_cells[key], view)
        gpboost_score = _metric_view(gpboost_cells[key], view)
        if darko_score["target_sha256"] != gpboost_score["target_sha256"]:
            raise RuntimeError(f"target changed between arms for {key}")
        ratio = float(gpboost_score["rmse"] / darko_score["rmse"])
        ratios.append(ratio)
        if ratio < 1.0 - 1e-12:
            wins += 1
        elif ratio > 1.0 + 1e-12:
            losses += 1
        else:
            ties += 1
        rows.append(
            {
                "season": int(key[0]),
                "target": key[1],
                "darkofit_rmse": darko_score["rmse"],
                "gpboost_rmse": gpboost_score["rmse"],
                "gpboost_over_darkofit_rmse": ratio,
            }
        )
    return {
        "orientation": "gpboost_rmse / darkofit_rmse; lower is better for GPBoost",
        "geometric_mean_ratio": _geomean(ratios),
        "wins_losses_ties": {"gpboost": wins, "darkofit": losses, "ties": ties},
        "lineages": rows,
    }


def _timing_summary(
    repeats: list[dict[str, Any]], darko_arm: str, gpboost_arm: str
) -> dict[str, Any]:
    darko_by_block = {
        row["block"]: row["result"] for row in repeats if row["arm"] == darko_arm
    }
    gpboost_by_block = {
        row["block"]: row["result"]
        for row in repeats
        if row["arm"] == gpboost_arm
    }
    if set(darko_by_block) != set(gpboost_by_block) or len(darko_by_block) != 3:
        raise RuntimeError("timing repeats are incomplete")
    output: dict[str, Any] = {
        "orientation": "gpboost / darkofit; lower is faster or smaller for GPBoost",
        "blocks": sorted(darko_by_block),
    }
    for metric in (
        "total_fit_seconds",
        "total_predict_seconds",
        "steady_wall_seconds",
        "peak_rss_bytes",
    ):
        darko_values = [darko_by_block[block][metric] for block in sorted(darko_by_block)]
        gpboost_values = [
            gpboost_by_block[block][metric] for block in sorted(gpboost_by_block)
        ]
        ratios = [
            gpboost_by_block[block][metric] / darko_by_block[block][metric]
            for block in sorted(darko_by_block)
        ]
        output[metric] = {
            "darkofit_median": float(np.median(darko_values)),
            "gpboost_median": float(np.median(gpboost_values)),
            "ratio_median": float(np.median(ratios)),
            "paired_ratios": [float(value) for value in ratios],
        }
    return output


def _repeat_quality_summary(
    repeats: list[dict[str, Any]], darko_arm: str, gpboost_arm: str, view: str
) -> dict[str, Any]:
    by_block = {
        block: {
            row["arm"]: row["result"]
            for row in repeats
            if row["block"] == block and row["arm"] in (darko_arm, gpboost_arm)
        }
        for block in range(len(BLOCK_ORDERS))
    }
    comparisons = [
        _comparison_summary(values[darko_arm], values[gpboost_arm], view)
        for _, values in sorted(by_block.items())
    ]
    ratios = [row["geometric_mean_ratio"] for row in comparisons]
    return {
        "orientation": comparisons[0]["orientation"],
        "geometric_mean_ratio_by_block": ratios,
        "geometric_mean_ratio_median": float(np.median(ratios)),
        "geometric_mean_ratio_min": float(np.min(ratios)),
        "geometric_mean_ratio_max": float(np.max(ratios)),
        "wins_losses_ties_by_block": [row["wins_losses_ties"] for row in comparisons],
        "lineages_by_block": [row["lineages"] for row in comparisons],
    }


def _summarize(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {arm: [row for row in repeats if row["arm"] == arm] for arm in ARMS}
    structural_fingerprints = {
        arm: sorted({row["result"]["structural_fingerprint_sha256"] for row in rows})
        for arm, rows in by_arm.items()
    }
    changed = {
        arm for arm, values in structural_fingerprints.items() if len(values) != 1
    }
    if changed:
        raise RuntimeError(f"model structure changed across repeats: {sorted(changed)}")
    prediction_fingerprints = {
        arm: [row["result"]["prediction_fingerprint_sha256"] for row in rows]
        for arm, rows in by_arm.items()
    }
    lanes: dict[str, Any] = {}
    for lane, (darko_arm, gpboost_arm) in LANES.items():
        lanes[lane] = {
            "arms": {"darkofit": darko_arm, "gpboost": gpboost_arm},
            "quality": {
                view: _repeat_quality_summary(repeats, darko_arm, gpboost_arm, view)
                for view in ("primary", "held_team", "cold_player")
            },
            "cost": _timing_summary(repeats, darko_arm, gpboost_arm),
            "fitted_tree_counts": {
                "darkofit": sorted(
                    {
                        fold["fitted"]["fitted_tree_count"]
                        for row in by_arm[darko_arm]
                        for cell in row["result"]["cells"]
                        for fold in cell["folds"]
                    }
                ),
                "gpboost": sorted(
                    {
                        fold["fitted"]["fitted_tree_count"]
                        for row in by_arm[gpboost_arm]
                        for cell in row["result"]["cells"]
                        for fold in cell["folds"]
                    }
                ),
            },
        }
    return {
        "structurally_reproducible": True,
        "structural_fingerprints": {
            arm: values[0] for arm, values in structural_fingerprints.items()
        },
        "prediction_fingerprints_by_block": prediction_fingerprints,
        "lanes": lanes,
    }


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_attestation(args: argparse.Namespace) -> dict[str, Any]:
    requested = _git_output(args.darkofit_repo, "rev-parse", args.darkofit_revision)
    if not (args.darkofit_source / "darkofit").is_dir():
        raise RuntimeError("DarkoFit source archive has no darkofit package")
    return {
        "repository": str(args.darkofit_repo.resolve()),
        "requested_revision": args.darkofit_revision,
        "resolved_commit": requested,
        "snapshot": str(args.darkofit_source.resolve()),
        "snapshot_tree_sha256": _tree_sha256(args.darkofit_source),
        "package_tree_sha256": _tree_sha256(args.darkofit_source / "darkofit"),
    }


def _run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {args.output}")
    if args.threads < 1:
        raise ValueError("thread count must be positive")
    if _sha256(args.protocol) != args.protocol_sha256:
        raise RuntimeError("protocol changed after launch")
    panel, manifest = _load_panel(args.panel_cache, args.panel_manifest)
    del panel
    args.worker_cache.mkdir(parents=True, exist_ok=True)
    source = _source_attestation(args)
    repeats = []
    for block, order in enumerate(BLOCK_ORDERS):
        for position, arm in enumerate(order):
            print(
                f"[gpboost-basketball] block={block} position={position} arm={arm}",
                flush=True,
            )
            repeats.append(
                {
                    "block": block,
                    "position": position,
                    "order": list(order),
                    "arm": arm,
                    "result": _run_worker_process(args, arm),
                }
            )
    artifact = {
        "schema_version": 1,
        "name": "gpboost_vs_darkofit_basketball_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "spent_tier_e_descriptive_characterization",
        "non_claims": [
            "No product-policy, default, release, or Pareto-dominance claim.",
            "No fresh or lockbox data access and no retuning authorization.",
            "Feature-only comparison; GPBoost random effects are not evaluated.",
        ],
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": args.protocol_sha256,
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "panel": {
            "manifest_path": str(args.panel_manifest.relative_to(ROOT)),
            "manifest_sha256": _sha256(args.panel_manifest),
            "processed_panel_sha256": manifest["processed_panel"]["sha256"],
            "split_manifest_sha256": manifest["split"]["split_manifest_sha256"],
            "previously_spent": bool(manifest.get("panel_spent")),
        },
        "source": source,
        "execution": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "threads": args.threads,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "worker_count": len(repeats),
            "gpboost_version": importlib.metadata.version("gpboost"),
        },
        "repeats": repeats,
        "summary": _summarize(repeats),
    }
    _atomic_create(args.output, artifact)
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-arm", choices=ARMS)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--panel-cache", type=Path, default=PANEL_CACHE)
    parser.add_argument("--panel-manifest", type=Path, default=PANEL_MANIFEST)
    parser.add_argument("--darkofit-source", type=Path, required=True)
    parser.add_argument("--darkofit-repo", type=Path, default=ROOT)
    parser.add_argument("--darkofit-revision", default="HEAD")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--worker-cache", type=Path, default=Path("/private/tmp/gpboost-basketball-mpl"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    args.panel_cache = args.panel_cache.resolve()
    args.panel_manifest = args.panel_manifest.resolve()
    args.darkofit_source = args.darkofit_source.resolve()
    args.darkofit_repo = args.darkofit_repo.resolve()
    args.protocol = args.protocol.resolve()
    args.worker_cache = args.worker_cache.resolve()
    args.output = args.output.resolve()
    if args.protocol_sha256 is None:
        args.protocol_sha256 = _sha256(args.protocol)
    return args


def main() -> int:
    args = _parse_args()
    if args.worker_arm is not None:
        result = _run_worker(args)
        print(WORKER_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    artifact = _run_parent(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "workers": len(artifact["repeats"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
