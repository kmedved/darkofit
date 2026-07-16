#!/usr/bin/env python3
"""Run the frozen basketball default versus auto-LR refit experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import basketball_guardrails as guardrails  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_basketball_darkofit_ablation as ablation  # noqa: E402


CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    "auto_lr_early_stopping_refit": {
        "early_stopping": True,
        "early_stopping_rounds": None,
        "validation_fraction": 0.1,
        "use_best_model": True,
        "refit": True,
        "refit_strategy": "exact",
    },
}
CONFIG_ORDER = tuple(CONFIGS)
TIMING_SCHEDULE = (
    ("default", "auto_lr_early_stopping_refit"),
    ("auto_lr_early_stopping_refit", "default"),
    ("default", "auto_lr_early_stopping_refit"),
)
WORKER_RESULT_PREFIX = "BASKETBALL_AUTO_LR_REFIT_RESULT="
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "basketball_auto_lr_refit.json"
MIN_FOLD_WINS = 6
MAX_TIMING_SPREAD_RATIO = 1.20
MAX_CANDIDATE_RUNTIME_RATIO = 0.80
EXPECTED_DEFAULT_MEAN_R2 = 0.5267495183883605


def _build_model(config_name: str):
    from darkofit import DarkoRegressor

    if config_name not in CONFIGS:
        raise ValueError(f"unknown basketball config: {config_name}")
    return DarkoRegressor(
        random_state=creator.RANDOM_STATE,
        verbose_timing=True,
        **CONFIGS[config_name],
    )


def _fit_and_predict(config_name, X_train, y_train, X_test):
    model = _build_model(config_name)
    started = time.perf_counter_ns()
    model.fit(X_train, y_train)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    if prediction.ndim != 1 or not np.all(np.isfinite(prediction)):
        raise RuntimeError("DarkoFit produced invalid basketball predictions")
    metadata = ablation.extract_fit_metadata(model)
    selection = getattr(model, "selection_model_", None)
    metadata["selection_early_stopping_rounds"] = (
        None
        if selection is None or selection.early_stopping_rounds_ is None
        else int(selection.early_stopping_rounds_)
    )
    metadata["final_early_stopping_rounds"] = (
        None
        if model.model_.early_stopping_rounds_ is None
        else int(model.model_.early_stopping_rounds_)
    )
    return (
        model,
        prediction,
        float(fit_seconds),
        float(predict_seconds),
        metadata,
    )


def _warmup(config_name: str, X, y) -> float:
    train_indices, test_indices = next(creator.creator_cv().split(X, y))
    started = time.perf_counter_ns()
    _fit_and_predict(
        config_name,
        X.iloc[train_indices],
        y.iloc[train_indices],
        X.iloc[test_indices],
    )
    return float((time.perf_counter_ns() - started) / 1e9)


def _without_phase_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_phase_timing(item)
            for key, item in value.items()
            if key != "phase_seconds"
        }
    if isinstance(value, list):
        return [_without_phase_timing(item) for item in value]
    return value


def _behavior_fingerprint(result: dict[str, Any]) -> str:
    payload = {
        "config": result["config"],
        "folds": [
            {
                "fold": fold["fold"],
                "r2": fold["r2"],
                "prediction_sha256": fold["prediction_sha256"],
                "fit_metadata": _without_phase_timing(fold["fit_metadata"]),
            }
            for fold in result["folds"]
        ],
        "holdout": {
            "scores": result["holdout"]["scores"],
            "prediction_sha256": result["holdout"]["prediction_sha256"],
            "fit_metadata": _without_phase_timing(
                result["holdout"]["fit_metadata"]
            ),
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_fitted_metadata(config_name: str, metadata: dict[str, Any]) -> None:
    learning_rate = float(metadata["resolved_learning_rate"])
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise RuntimeError(f"{config_name} resolved an invalid learning rate")
    if metadata["selected_lane"] != "boosting":
        raise RuntimeError(f"{config_name} unexpectedly selected another lane")
    if metadata["selected_tree_mode"] != "catboost":
        raise RuntimeError(f"{config_name} unexpectedly changed tree mode")
    if int(metadata["resolved_thread_count"]) < 1:
        raise RuntimeError(f"{config_name} resolved no worker threads")
    if config_name == "default":
        if metadata["refit"] or metadata["selection_fit"] is not None:
            raise RuntimeError("default unexpectedly performed selection/refit")
        return
    if not metadata["refit"] or metadata["refit_strategy"] != "exact":
        raise RuntimeError("candidate did not perform exact refit")
    selection = metadata["selection_fit"]
    final = metadata["final_fit"]
    if selection is None:
        raise RuntimeError("candidate is missing selection-fit metadata")
    patience = metadata["selection_early_stopping_rounds"]
    if patience is None or int(patience) < 1:
        raise RuntimeError("candidate did not resolve automatic patience")
    if metadata["final_early_stopping_rounds"] is not None:
        raise RuntimeError("candidate final refit retained early stopping")
    if int(final["iterations_requested"]) != int(metadata["best_iteration"]):
        raise RuntimeError("candidate exact refit requested the wrong tree count")
    final_count = int(metadata["fitted_tree_count"])
    if final_count != int(metadata["best_iteration"]):
        raise RuntimeError("candidate exact refit retained the wrong tree count")
    if int(final["rounds_retained"]) != final_count:
        raise RuntimeError("candidate final metadata disagrees on tree count")
    if int(selection["rounds_retained"]) != int(metadata["best_iteration"]):
        raise RuntimeError("candidate selected-prefix metadata changed")


def run_worker(config_name: str, cache_path: Path) -> dict[str, Any]:
    frame, raw_metadata = creator.load_raw_data(cache_path)
    X, y, processed_metadata = creator.prepare_creator_data(frame)
    guardrail = guardrails.prepare_player_guardrail(frame)
    if not X.equals(guardrail.X_train) or not y.equals(guardrail.y_train):
        raise RuntimeError("basketball guardrail changed creator training data")

    warmup_seconds = _warmup(config_name, X, y)
    folds = []
    steady_started = time.perf_counter_ns()
    for fold, (train_indices, test_indices) in enumerate(
        creator.creator_cv().split(X, y)
    ):
        _, prediction, fit_seconds, predict_seconds, metadata = _fit_and_predict(
            config_name,
            X.iloc[train_indices],
            y.iloc[train_indices],
            X.iloc[test_indices],
        )
        validate_fitted_metadata(config_name, metadata)
        folds.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "test_indices": [int(value) for value in test_indices],
                "r2": float(r2_score(y.iloc[test_indices], prediction)),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "prediction_sha256": guardrails.prediction_sha256(prediction),
                "predictions": [float(value) for value in prediction],
                "fit_metadata": metadata,
            }
        )
    steady_seconds = (time.perf_counter_ns() - steady_started) / 1e9

    _, held_prediction, held_fit, held_predict, held_metadata = _fit_and_predict(
        config_name,
        guardrail.X_train,
        guardrail.y_train,
        guardrail.X_holdout,
    )
    validate_fitted_metadata(config_name, held_metadata)
    scores = np.asarray([fold["r2"] for fold in folds], dtype=np.float64)
    result = {
        "config": config_name,
        "config_params": creator._jsonable(CONFIGS[config_name]),
        "mean_r2": float(np.mean(scores)),
        "std_r2": float(np.std(scores)),
        "fold_scores": [float(value) for value in scores],
        "folds": folds,
        "steady_wall_seconds": float(steady_seconds),
        "warmup_seconds_outside_timing": warmup_seconds,
        "summed_fit_seconds": float(sum(row["fit_seconds"] for row in folds)),
        "summed_predict_seconds": float(
            sum(row["predict_seconds"] for row in folds)
        ),
        "summed_final_phase_seconds": ablation._sum_phase_times(
            folds, "final_fit"
        ),
        "summed_selection_phase_seconds": ablation._sum_phase_times(
            folds, "selection_fit"
        ),
        "holdout": {
            "fit_seconds": held_fit,
            "predict_seconds": held_predict,
            "prediction_sha256": guardrails.prediction_sha256(held_prediction),
            "predictions": [float(value) for value in held_prediction],
            "fit_metadata": held_metadata,
            "scores": guardrails.score_player_guardrails(
                guardrail.y_holdout,
                held_prediction,
                guardrail.cold_player_mask,
            ),
        },
        "guardrail": guardrail.metadata,
        "raw_data": raw_metadata,
        "processed_data": processed_metadata,
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }
    result["behavior_fingerprint_sha256"] = _behavior_fingerprint(result)
    return result


def _run_worker_process(args: argparse.Namespace, config_name: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config",
        config_name,
        "--data-cache",
        str(args.data_cache),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=ablation._worker_environment(args.threads),
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
            f"basketball worker {config_name!r} failed with exit code "
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


def _timing_summary(values: list[float]) -> dict[str, Any]:
    if len(values) != len(TIMING_SCHEDULE):
        raise RuntimeError("basketball timing requires three repetitions")
    minimum = min(values)
    maximum = max(values)
    return {
        "repetitions": len(values),
        "values_seconds": [float(value) for value in values],
        "minimum_seconds": float(minimum),
        "median_seconds": float(statistics.median(values)),
        "maximum_seconds": float(maximum),
        "maximum_over_minimum": float(maximum / minimum),
    }


def analyze_results(
    canonical: dict[str, dict[str, Any]],
    timing: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    default = canonical["default"]
    candidate = canonical["auto_lr_early_stopping_refit"]
    default_scores = np.asarray(default["fold_scores"], dtype=np.float64)
    candidate_scores = np.asarray(candidate["fold_scores"], dtype=np.float64)
    if default_scores.shape != (creator.N_SPLITS,) or candidate_scores.shape != (
        creator.N_SPLITS,
    ):
        raise RuntimeError("basketball result does not contain ten folds")
    if not math.isclose(
        float(default["mean_r2"]),
        EXPECTED_DEFAULT_MEAN_R2,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("basketball default score no longer reproduces")
    deltas = candidate_scores - default_scores
    jackknife = [
        float(np.mean(np.delete(deltas, fold)))
        for fold in range(creator.N_SPLITS)
    ]
    default_guardrail = default["holdout"]["scores"]
    candidate_guardrail = candidate["holdout"]["scores"]
    team_delta = float(
        candidate_guardrail["overlap_exposed_team_holdout"]["r2"]
        - default_guardrail["overlap_exposed_team_holdout"]["r2"]
    )
    cold_delta = float(
        candidate_guardrail["cold_player_subset"]["r2"]
        - default_guardrail["cold_player_subset"]["r2"]
    )
    runtime_ratio = float(
        timing["auto_lr_early_stopping_refit"]["median_seconds"]
        / timing["default"]["median_seconds"]
    )
    quality_gates = {
        "mean_r2_no_regression": float(np.mean(deltas)) >= 0.0,
        "fold_breadth": int(np.count_nonzero(deltas > 0.0)) >= MIN_FOLD_WINS,
        "leave_one_fold_out_no_regression": min(jackknife) >= 0.0,
        "overlap_exposed_team_no_regression": team_delta >= 0.0,
        "cold_player_no_regression": cold_delta >= 0.0,
    }
    timing_gates = {
        "default_timing_stable": (
            timing["default"]["maximum_over_minimum"]
            <= MAX_TIMING_SPREAD_RATIO
        ),
        "candidate_timing_stable": (
            timing["auto_lr_early_stopping_refit"]["maximum_over_minimum"]
            <= MAX_TIMING_SPREAD_RATIO
        ),
        "material_speedup": runtime_ratio <= MAX_CANDIDATE_RUNTIME_RATIO,
    }
    passes_quality = all(quality_gates.values())
    passes_timing = all(timing_gates.values())
    return {
        "candidate": "auto_lr_early_stopping_refit",
        "mean_r2_delta": float(np.mean(deltas)),
        "fold_wins": int(np.count_nonzero(deltas > 0.0)),
        "fold_losses": int(np.count_nonzero(deltas < 0.0)),
        "fold_ties": int(np.count_nonzero(deltas == 0.0)),
        "fold_r2_deltas": [float(value) for value in deltas],
        "leave_one_fold_out_mean_deltas": jackknife,
        "overlap_exposed_team_r2_delta": team_delta,
        "cold_player_r2_delta": cold_delta,
        "candidate_over_default_median_runtime": runtime_ratio,
        "quality_gates": quality_gates,
        "timing_gates": timing_gates,
        "passes_quality_gates": passes_quality,
        "passes_timing_gates": passes_timing,
        "passes_all_gates": passes_quality and passes_timing,
        "recommendation": (
            "advance_candidate_for_external_validation"
            if passes_quality and passes_timing
            else "advance_none"
        ),
        "kernel_profile": {
            "required": passes_quality and passes_timing,
            "disposition": (
                "required_before_closeout"
                if passes_quality and passes_timing
                else "skipped_candidate_failed_quality_or_timing_gate"
            ),
        },
    }


def _source_state(allow_dirty: bool) -> dict[str, Any]:
    state = creator.git_state(REPO_ROOT)
    if not allow_dirty and not state["clean"]:
        raise RuntimeError("refusing to benchmark a dirty DarkoFit source tree")
    return state


def _assert_source_unchanged(expected, observed, boundary: str) -> None:
    fields = ("path", "head", "branch", "clean", "status")
    changed = [field for field in fields if expected[field] != observed[field]]
    if changed:
        raise RuntimeError(
            f"DarkoFit source changed {boundary}: " + ", ".join(changed)
        )


def _repeat_record(block: int, position: int, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "block": int(block),
        "position": int(position),
        "config": result["config"],
        "steady_wall_seconds": float(result["steady_wall_seconds"]),
        "warmup_seconds_outside_timing": float(
            result["warmup_seconds_outside_timing"]
        ),
        "summed_fit_seconds": float(result["summed_fit_seconds"]),
        "summed_predict_seconds": float(result["summed_predict_seconds"]),
        "holdout_fit_seconds": float(result["holdout"]["fit_seconds"]),
        "holdout_predict_seconds": float(result["holdout"]["predict_seconds"]),
        "fold_fit_seconds": [float(row["fit_seconds"]) for row in result["folds"]],
        "fold_predict_seconds": [
            float(row["predict_seconds"]) for row in result["folds"]
        ],
        "behavior_fingerprint_sha256": result["behavior_fingerprint_sha256"],
        "worker_stdout": result["worker_stdout"],
        "worker_stderr": result["worker_stderr"],
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_symlink():
        raise RuntimeError(f"refusing symlink benchmark output: {args.output}")
    source = _source_state(args.allow_dirty_source)
    frame, raw_metadata = creator.load_raw_data(args.data_cache)
    X, y, processed_metadata = creator.prepare_creator_data(frame)
    fold_digest, fold_sizes = creator.fold_fingerprint(X, y)
    guardrail_metadata = guardrails.prepare_player_guardrail(frame).metadata

    canonical: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, set[str]] = {name: set() for name in CONFIG_ORDER}
    timing_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    repeats = []
    for block, order in enumerate(TIMING_SCHEDULE):
        for position, config_name in enumerate(order):
            _assert_source_unchanged(
                source,
                _source_state(args.allow_dirty_source),
                boundary=f"before block {block} {config_name}",
            )
            print(
                f"running block {block + 1}/{len(TIMING_SCHEDULE)} "
                f"position {position + 1}: {config_name}...",
                flush=True,
            )
            result = _run_worker_process(args, config_name)
            fingerprints[config_name].add(result["behavior_fingerprint_sha256"])
            timing_values[config_name].append(result["steady_wall_seconds"])
            repeats.append(_repeat_record(block, position, result))
            canonical.setdefault(config_name, result)
            print(
                f"  mean R2={result['mean_r2']:.12f}, "
                f"cold={result['holdout']['scores']['cold_player_subset']['r2']:.12f}, "
                f"steady={result['steady_wall_seconds']:.2f}s",
                flush=True,
            )
    for config_name, values in fingerprints.items():
        if len(values) != 1:
            raise RuntimeError(f"{config_name} behavior changed across timing repeats")
    _assert_source_unchanged(
        source,
        _source_state(args.allow_dirty_source),
        boundary="during the experiment",
    )
    timing = {
        config_name: _timing_summary(values)
        for config_name, values in timing_values.items()
    }
    decision = analyze_results(canonical, timing)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_auto_lr_early_stopping_exact_refit",
            "diagnostic_only": True,
            "creator_benchmark_changed": False,
            "configs": creator._jsonable(CONFIGS),
            "timing_schedule": [list(order) for order in TIMING_SCHEDULE],
            "timing_repetitions_per_config": len(TIMING_SCHEDULE),
            "maximum_timing_spread_ratio": MAX_TIMING_SPREAD_RATIO,
            "maximum_candidate_runtime_ratio": MAX_CANDIDATE_RUNTIME_RATIO,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "random_state": creator.RANDOM_STATE,
            "threads_per_fit": args.threads,
            "cv": {
                "kind": "KFold",
                "n_splits": creator.N_SPLITS,
                "shuffle": False,
                "n_jobs": 1,
                "fold_fingerprint_sha256": fold_digest,
                "fold_test_sizes": fold_sizes,
            },
            "warmup": "one full first-fold fit and prediction outside each worker timer",
            "weights_used": False,
            "lockbox_data_used": False,
        },
        "raw_data": raw_metadata,
        "processed_data": processed_metadata,
        "guardrail": guardrail_metadata,
        "source": source,
        "environment": {
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
        },
        "canonical_results": [canonical[name] for name in CONFIG_ORDER],
        "timing_repeats": repeats,
        "timing_summary": timing,
        "decision": decision,
    }
    creator._atomic_write_bytes(
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=creator.DEFAULT_CACHE / "basketball_reference_toy_data.csv",
    )
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument(
        "--worker-config", choices=CONFIG_ORDER, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.output = creator._absolute_lexical_path(args.output)
    args.data_cache = creator._absolute_lexical_path(args.data_cache)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_config:
        result = run_worker(args.worker_config, args.data_cache)
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
