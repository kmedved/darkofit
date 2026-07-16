#!/usr/bin/env python3
"""Run the frozen five-arm DarkoFit basketball ablation.

This is a diagnostic extension of ``run_basketball_creator_benchmark.py``.
It preserves the pinned data, features, unshuffled ten folds, seed, and steady
timing controls while adding fitted-model telemetry and the creator-defined
alphabetical team holdout as a guardrail.  It does not replace the frozen
creator benchmark or make a product-default claim.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import math
import os
import pstats
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import run_basketball_creator_benchmark as baseline  # noqa: E402


CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    "a10_numeric": {
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "learning_rate": 0.1,
    },
    "a10_numeric_2000": {
        "iterations": 2_000,
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "learning_rate": 0.1,
    },
    "a10_early_stopping_refit": {
        "iterations": 2_000,
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "learning_rate": 0.1,
        "early_stopping": True,
        "early_stopping_rounds": "auto",
        "validation_fraction": 0.1,
        "use_best_model": True,
        "refit": True,
        "refit_strategy": "exact",
    },
    "linear_residual": {
        "linear_residual": True,
    },
}
CONFIG_ORDER = tuple(CONFIGS)
WORKER_RESULT_PREFIX = "BASKETBALL_ABLATION_RESULT="

CATBOOST_TARGET_R2 = 0.5363082206115677
HISTORICAL_DARKOFIT_STEADY_SECONDS = 28.299252333
HISTORICAL_CHIMERABOOST_STEADY_SECONDS = 9.289367125
MATERIAL_R2_GAIN = 0.002
MIN_FOLD_WINS = 6
RUNTIME_TOLERANCE = 1.0
PROFILE_FOLD = 0


def _prediction_sha256(prediction: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(prediction, dtype="<f8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def prepare_held_team_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, Any]]:
    """Recreate the creator's alphabetical team split, including the holdout."""
    filtered = frame.loc[frame["MP"] > 500].copy()
    filtered["MPG"] = filtered["MP"] / filtered["G"]
    filtered["starter"] = np.where(
        filtered["GS"] / filtered["G"] >= 0.5, 1, 0
    )
    teams = filtered["Tm"].sort_values().drop_duplicates().tolist()
    test_teams = teams[: len(teams) // 3]
    test_team_set = set(test_teams)
    train = filtered.loc[~filtered["Tm"].isin(test_team_set)]
    test = filtered.loc[filtered["Tm"].isin(test_team_set)]
    return (
        train.loc[:, baseline.FEATURES],
        train.loc[:, "MPG"],
        test.loc[:, baseline.FEATURES],
        test.loc[:, "MPG"],
        {
            "train_rows": int(train.shape[0]),
            "test_rows": int(test.shape[0]),
            "train_team_count": len(teams) - len(test_teams),
            "test_team_count": len(test_teams),
            "test_teams": test_teams,
        },
    )


def _phase_timing(core: Any) -> dict[str, float]:
    timing = getattr(core, "timing_", {}) or {}
    result = {str(key): float(value) for key, value in timing.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise RuntimeError("fitted phase timing contains an invalid value")
    return result


def _core_metadata(core: Any) -> dict[str, Any]:
    training = dict(getattr(core, "training_metadata_", {}) or {})
    return {
        "iterations_requested": int(
            training.get("iterations_requested", getattr(core, "iterations_", 0))
        ),
        "iterations_attempted": int(
            training.get(
                "iterations_attempted", getattr(core, "iterations_attempted_", 0)
            )
        ),
        "rounds_completed": int(
            training.get("rounds_completed", getattr(core, "rounds_completed_", 0))
        ),
        "rounds_retained": int(
            training.get("rounds_retained", len(getattr(core, "trees_", ())))
        ),
        "stop_reason": str(
            training.get("stop_reason", getattr(core, "stop_reason_", "unknown"))
        ),
        "phase_seconds": _phase_timing(core),
    }


def extract_fit_metadata(model: Any) -> dict[str, Any]:
    """Extract only public or persisted fitted metadata used by this screen."""
    core = model.model_
    linear_active = bool(getattr(model, "linear_residual_active_", False))
    selection_core = getattr(model, "selection_model_", None)
    selection = None if selection_core is None else _core_metadata(selection_core)
    return {
        "best_iteration": int(model.best_n_estimators_),
        "fitted_tree_count": int(model.n_estimators_),
        "resolved_learning_rate": float(model.learning_rate_),
        "requested_tree_mode": str(model.tree_mode),
        "selected_tree_mode": str(core.tree_mode_),
        "selected_lane": "linear_residual" if linear_active else "boosting",
        "linear_residual_active": linear_active,
        "resolved_thread_count": int(core.n_threads_),
        "refit": bool(getattr(model, "refit_", False)),
        "refit_strategy": getattr(model, "refit_strategy_", None),
        "final_fit": _core_metadata(core),
        "selection_fit": selection,
    }


def _build_model(config_name: str):
    from darkofit import DarkoRegressor

    if config_name not in CONFIGS:
        raise ValueError(f"unknown config: {config_name}")
    return DarkoRegressor(
        random_state=baseline.RANDOM_STATE,
        verbose_timing=True,
        **CONFIGS[config_name],
    )


def _fit_and_predict(
    config_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[Any, np.ndarray, float, float]:
    model = _build_model(config_name)
    started = time.perf_counter_ns()
    model.fit(X_train, y_train)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    if prediction.ndim != 1 or not np.all(np.isfinite(prediction)):
        raise RuntimeError("DarkoFit produced invalid basketball predictions")
    return model, prediction, float(fit_seconds), float(predict_seconds)


def _warmup(config_name: str, X: pd.DataFrame, y: pd.Series) -> float:
    train_indices, test_indices = next(baseline.creator_cv().split(X, y))
    started = time.perf_counter_ns()
    _fit_and_predict(
        config_name,
        X.iloc[train_indices],
        y.iloc[train_indices],
        X.iloc[test_indices],
    )
    return float((time.perf_counter_ns() - started) / 1e9)


def _sum_phase_times(folds: list[dict[str, Any]], fit_name: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for fold in folds:
        fit = fold["fit_metadata"].get(fit_name)
        if fit is None:
            continue
        for phase, seconds in fit["phase_seconds"].items():
            totals[phase] = totals.get(phase, 0.0) + float(seconds)
    return dict(sorted(totals.items()))


def run_config_worker(config_name: str, cache_path: Path) -> dict[str, Any]:
    frame, raw_metadata = baseline.load_raw_data(cache_path)
    X, y, processed_metadata = baseline.prepare_creator_data(frame)
    X_held_train, y_held_train, X_held, y_held, held_metadata = (
        prepare_held_team_data(frame)
    )
    if not X.equals(X_held_train) or not y.equals(y_held_train):
        raise RuntimeError("held-team reconstruction changed creator training rows")
    if held_metadata["test_rows"] != processed_metadata["test_rows"]:
        raise RuntimeError("held-team reconstruction changed creator holdout rows")

    warmup_seconds = _warmup(config_name, X, y)
    folds: list[dict[str, Any]] = []
    steady_started = time.perf_counter_ns()
    for fold, (train_indices, test_indices) in enumerate(
        baseline.creator_cv().split(X, y)
    ):
        model, prediction, fit_seconds, predict_seconds = _fit_and_predict(
            config_name,
            X.iloc[train_indices],
            y.iloc[train_indices],
            X.iloc[test_indices],
        )
        score = float(r2_score(y.iloc[test_indices], prediction))
        folds.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "test_indices": [int(value) for value in test_indices],
                "r2": score,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "prediction_sha256": _prediction_sha256(prediction),
                "predictions": [float(value) for value in prediction],
                "fit_metadata": extract_fit_metadata(model),
            }
        )
    steady_seconds = (time.perf_counter_ns() - steady_started) / 1e9

    held_model, held_prediction, held_fit_seconds, held_predict_seconds = (
        _fit_and_predict(config_name, X_held_train, y_held_train, X_held)
    )
    scores = np.asarray([fold["r2"] for fold in folds], dtype=np.float64)
    return {
        "config": config_name,
        "config_params": baseline._jsonable(CONFIGS[config_name]),
        "mean_r2": float(np.mean(scores)),
        "std_r2": float(np.std(scores)),
        "fold_scores": [float(value) for value in scores],
        "folds": folds,
        "steady_wall_seconds": float(steady_seconds),
        "warmup_seconds_outside_timing": warmup_seconds,
        "summed_fit_seconds": float(sum(fold["fit_seconds"] for fold in folds)),
        "summed_predict_seconds": float(
            sum(fold["predict_seconds"] for fold in folds)
        ),
        "summed_final_phase_seconds": _sum_phase_times(folds, "final_fit"),
        "summed_selection_phase_seconds": _sum_phase_times(
            folds, "selection_fit"
        ),
        "held_team": {
            "r2": float(r2_score(y_held, held_prediction)),
            "rows": int(len(y_held)),
            "fit_seconds": held_fit_seconds,
            "predict_seconds": held_predict_seconds,
            "prediction_sha256": _prediction_sha256(held_prediction),
            "predictions": [float(value) for value in held_prediction],
            "fit_metadata": extract_fit_metadata(held_model),
            **held_metadata,
        },
        "raw_data": raw_metadata,
        "processed_data": processed_metadata,
        "thread_environment": {
            key: os.environ.get(key) for key in baseline.THREAD_ENV_KEYS
        },
    }


def _profile_rows(profile: cProfile.Profile, limit: int = 30) -> list[dict[str, Any]]:
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, function), values in stats.stats.items():
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _ = values
        rows.append(
            {
                "file": str(Path(filename).resolve()) if filename else filename,
                "line": int(line),
                "function": function,
                "primitive_calls": int(primitive_calls),
                "total_calls": int(total_calls),
                "self_seconds": float(self_seconds),
                "cumulative_seconds": float(cumulative_seconds),
            }
        )
    rows.sort(
        key=lambda row: (row["cumulative_seconds"], row["self_seconds"]),
        reverse=True,
    )
    return rows[:limit]


def run_profile_worker(
    config_name: str, cache_path: Path, profile_fold: int
) -> dict[str, Any]:
    frame, _ = baseline.load_raw_data(cache_path)
    X, y, _ = baseline.prepare_creator_data(frame)
    splits = list(baseline.creator_cv().split(X, y))
    if profile_fold < 0 or profile_fold >= len(splits):
        raise ValueError("profile fold is outside the frozen ten-fold split")
    warmup_seconds = _warmup(config_name, X, y)
    train_indices, test_indices = splits[profile_fold]
    profile = cProfile.Profile()
    profile.enable()
    model, prediction, fit_seconds, predict_seconds = _fit_and_predict(
        config_name,
        X.iloc[train_indices],
        y.iloc[train_indices],
        X.iloc[test_indices],
    )
    profile.disable()
    return {
        "config": config_name,
        "fold": int(profile_fold),
        "warmup_seconds_outside_profile": warmup_seconds,
        "profiled_fit_seconds": fit_seconds,
        "profiled_predict_seconds": predict_seconds,
        "r2": float(r2_score(y.iloc[test_indices], prediction)),
        "prediction_sha256": _prediction_sha256(prediction),
        "fit_metadata": extract_fit_metadata(model),
        "top_cumulative_functions": _profile_rows(profile),
    }


def _worker_environment(threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key == "ENABLE_IPC" or key.startswith(baseline.EXECUTION_ENV_PREFIXES):
            environment.pop(key)
    for key in baseline.THREAD_LIMIT_ENV_KEYS:
        environment[key] = str(threads)
    environment.update(
        {
            "CHIMERABOOST_WARMUP": "0",
            "NUMBA_DISABLE_JIT": "0",
            "ENABLE_IPC": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    for key in (
        "NUMBA_CPU_NAME",
        "NUMBA_CPU_FEATURES",
        "NUMBA_THREADING_LAYER",
        "NUMBA_CACHE_DIR",
        "JOBLIB_START_METHOD",
        "JOBLIB_TEMP_FOLDER",
    ):
        environment.pop(key, None)
    return environment


def _run_worker_process(
    args: argparse.Namespace, config_name: str, *, profile: bool = False
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-cache",
        str(args.data_cache),
    ]
    if profile:
        command.extend(
            ["--profile-config", config_name, "--profile-fold", str(PROFILE_FOLD)]
        )
    else:
        command.extend(["--worker-config", config_name])
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_worker_environment(args.threads),
        check=False,
        capture_output=True,
        text=True,
    )
    result_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if completed.returncode or len(result_lines) != 1:
        raise RuntimeError(
            f"ablation worker {config_name!r} failed with exit code "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result = json.loads(result_lines[0][len(WORKER_RESULT_PREFIX) :])
    result["worker_stdout"] = "\n".join(
        line
        for line in completed.stdout.splitlines()
        if not line.startswith(WORKER_RESULT_PREFIX)
    ).strip() or None
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def analyze_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {result["config"]: result for result in results}
    if len(results) != len(CONFIG_ORDER) or set(by_name) != set(CONFIG_ORDER):
        raise RuntimeError("ablation result set does not match the frozen configs")
    default = by_name["default"]
    default_scores = np.asarray(default["fold_scores"], dtype=np.float64)
    if default_scores.shape != (baseline.N_SPLITS,):
        raise RuntimeError("default result does not contain ten fold scores")
    quality_gap = CATBOOST_TARGET_R2 - float(default["mean_r2"])
    candidates = []
    for config_name in CONFIG_ORDER[1:]:
        candidate = by_name[config_name]
        scores = np.asarray(candidate["fold_scores"], dtype=np.float64)
        if scores.shape != default_scores.shape or not np.all(np.isfinite(scores)):
            raise RuntimeError(f"{config_name} has invalid fold scores")
        deltas = scores - default_scores
        jackknife = [float(np.mean(np.delete(deltas, fold))) for fold in range(10)]
        mean_delta = float(candidate["mean_r2"] - default["mean_r2"])
        held_delta = float(
            candidate["held_team"]["r2"] - default["held_team"]["r2"]
        )
        runtime = float(candidate["steady_wall_seconds"])
        gates = {
            "material_r2_gain": mean_delta >= MATERIAL_R2_GAIN,
            "fold_breadth": int(np.count_nonzero(deltas > 0.0)) >= MIN_FOLD_WINS,
            "positive_leave_one_fold_out_gain": min(jackknife) > 0.0,
            "held_team_no_regression": held_delta >= 0.0,
            "historical_runtime_cap": (
                runtime
                <= HISTORICAL_DARKOFIT_STEADY_SECONDS * RUNTIME_TOLERANCE
            ),
        }
        candidates.append(
            {
                "config": config_name,
                "mean_r2": float(candidate["mean_r2"]),
                "mean_r2_delta": mean_delta,
                "catboost_gap_closed_fraction": (
                    mean_delta / quality_gap if quality_gap > 0.0 else None
                ),
                "fold_wins": int(np.count_nonzero(deltas > 0.0)),
                "fold_losses": int(np.count_nonzero(deltas < 0.0)),
                "fold_ties": int(np.count_nonzero(deltas == 0.0)),
                "fold_r2_deltas": [float(value) for value in deltas],
                "leave_one_fold_out_mean_deltas": jackknife,
                "held_team_r2": float(candidate["held_team"]["r2"]),
                "held_team_r2_delta": held_delta,
                "steady_wall_seconds": runtime,
                "runtime_vs_current_default": (
                    runtime / float(default["steady_wall_seconds"])
                ),
                "runtime_vs_historical_chimeraboost": (
                    runtime / HISTORICAL_CHIMERABOOST_STEADY_SECONDS
                ),
                "gates": gates,
                "passes_all_gates": all(gates.values()),
            }
        )
    screen_winner = max(candidates, key=lambda row: row["mean_r2"])
    advancing = [row for row in candidates if row["passes_all_gates"]]
    advancing.sort(key=lambda row: row["mean_r2"], reverse=True)
    return {
        "gate_definition": {
            "material_r2_gain": MATERIAL_R2_GAIN,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "leave_one_fold_out_gain_must_be_positive": True,
            "held_team_r2_delta_minimum": 0.0,
            "historical_runtime_seconds": HISTORICAL_DARKOFIT_STEADY_SECONDS,
            "runtime_tolerance_fraction": RUNTIME_TOLERANCE - 1.0,
            "runtime_cap_seconds": (
                HISTORICAL_DARKOFIT_STEADY_SECONDS * RUNTIME_TOLERANCE
            ),
        },
        "targets": {
            "catboost_mean_r2": CATBOOST_TARGET_R2,
            "historical_darkofit_steady_seconds": (
                HISTORICAL_DARKOFIT_STEADY_SECONDS
            ),
            "historical_chimeraboost_steady_seconds": (
                HISTORICAL_CHIMERABOOST_STEADY_SECONDS
            ),
        },
        "default": {
            "mean_r2": float(default["mean_r2"]),
            "held_team_r2": float(default["held_team"]["r2"]),
            "steady_wall_seconds": float(default["steady_wall_seconds"]),
            "catboost_r2_gap": quality_gap,
        },
        "candidates": candidates,
        "screen_winner": screen_winner["config"],
        "advancing_candidate": advancing[0]["config"] if advancing else None,
        "recommendation": (
            "advance_" + advancing[0]["config"]
            if advancing
            else "advance_none"
        ),
    }


def _validate_source_state(args: argparse.Namespace) -> dict[str, Any]:
    namespace = argparse.Namespace(
        lane="steady",
        allow_dirty_source=args.allow_dirty_source,
        allow_chimeraboost_drift=False,
        chimeraboost_repo=args.chimeraboost_repo,
    )
    return baseline._validate_sources(namespace)


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_symlink():
        raise RuntimeError(f"refusing symlink benchmark output: {args.output}")
    sources = _validate_source_state(args)
    frame, raw_metadata = baseline.load_raw_data(args.data_cache)
    X, y, processed_metadata = baseline.prepare_creator_data(frame)
    fold_digest, fold_test_sizes = baseline.fold_fingerprint(X, y)
    results = []
    for config_name in CONFIG_ORDER:
        baseline._assert_sources_unchanged(
            sources,
            _validate_source_state(args),
            boundary=f"before {config_name}",
        )
        print(f"running {config_name}...", flush=True)
        result = _run_worker_process(args, config_name)
        results.append(result)
        print(
            f"  mean R2={result['mean_r2']:.12f}, "
            f"held={result['held_team']['r2']:.12f}, "
            f"steady={result['steady_wall_seconds']:.2f}s",
            flush=True,
        )

    decision = analyze_results(results)
    profiles = {}
    for config_name in ("default", decision["screen_winner"]):
        baseline._assert_sources_unchanged(
            sources,
            _validate_source_state(args),
            boundary=f"before profiling {config_name}",
        )
        print(f"profiling {config_name} on fold {PROFILE_FOLD}...", flush=True)
        profiles[config_name] = _run_worker_process(
            args, config_name, profile=True
        )
        matching = next(
            result for result in results if result["config"] == config_name
        )["folds"][PROFILE_FOLD]
        if (
            profiles[config_name]["prediction_sha256"]
            != matching["prediction_sha256"]
            or profiles[config_name]["r2"] != matching["r2"]
        ):
            raise RuntimeError(
                f"profile changed {config_name} fold-{PROFILE_FOLD} behavior"
            )

    baseline._assert_sources_unchanged(
        sources,
        _validate_source_state(args),
        boundary="during the ablation",
    )
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "darkofit_basketball_frozen_five_arm_ablation",
            "diagnostic_only": True,
            "parent_protocol": (
                "bbstats_basketball_creator_default_regressor_benchmark"
            ),
            "config_order": list(CONFIG_ORDER),
            "configs": baseline._jsonable(CONFIGS),
            "scoring": "r2",
            "cv": {
                "kind": "KFold",
                "n_splits": baseline.N_SPLITS,
                "shuffle": False,
                "n_jobs": 1,
                "fold_fingerprint_sha256": fold_digest,
                "fold_test_sizes": fold_test_sizes,
            },
            "random_state": baseline.RANDOM_STATE,
            "threads_per_fit": args.threads,
            "warmup": "one full first-fold fit and predict per config outside timing",
            "weights_used": False,
            "held_out_teams_scored": True,
            "profile_fold": PROFILE_FOLD,
        },
        "raw_data": raw_metadata,
        "processed_data": processed_metadata,
        "sources": sources,
        "environment": {
            "machine": baseline._machine_details(),
            "dependencies": baseline._dependency_versions(),
        },
        "results": results,
        "decision": decision,
        "profiles": profiles,
    }
    baseline._atomic_write_bytes(
        args.output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )
    print(f"decision: {decision['recommendation']}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "basketball_darkofit_ablation.json",
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=(
            baseline.DEFAULT_CACHE / "basketball_reference_toy_data.csv"
        ),
    )
    parser.add_argument(
        "--chimeraboost-repo",
        type=Path,
        default=baseline.DEFAULT_CHIMERABOOST_REPO,
    )
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument(
        "--worker-config", choices=CONFIG_ORDER, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--profile-config", choices=CONFIG_ORDER, help=argparse.SUPPRESS
    )
    parser.add_argument("--profile-fold", type=int, default=PROFILE_FOLD)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.output = baseline._absolute_lexical_path(args.output)
    args.data_cache = baseline._absolute_lexical_path(args.data_cache)
    args.chimeraboost_repo = args.chimeraboost_repo.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_config and args.profile_config:
        raise RuntimeError("worker and profile modes are mutually exclusive")
    if args.worker_config:
        result = run_config_worker(args.worker_config, args.data_cache)
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    if args.profile_config:
        result = run_profile_worker(
            args.profile_config, args.data_cache, args.profile_fold
        )
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
