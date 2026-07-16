#!/usr/bin/env python3
"""Run the frozen basketball validation-selected linear-leaf screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


DEFAULT_CONFIG = "default"
CANDIDATE_CONFIG = "linear_leaf_select_refit"
CONFIG_ORDER = (DEFAULT_CONFIG, CANDIDATE_CONFIG)
SELECTION_FRACTION = 0.1
MIN_FOLD_WINS = 6
MAX_FIT_RUNTIME_RATIO = 3.5
MAX_PREDICT_RUNTIME_RATIO = 1.25
MAX_MODEL_SIZE_RATIO = 3.0
MAX_PEAK_RSS_RATIO = 2.0
EXPECTED_DEFAULT_MEAN_R2 = 0.5267495183883605
WORKER_RESULT_PREFIX = "BASKETBALL_LINEAR_LEAVES_RESULT="
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "basketball_linear_leaves.json"


def _index_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def selection_split(X, y) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reproduce DarkoRegressor's deterministic random validation split."""
    from darkofit.sklearn_api import _make_eval_split

    train, validation, policy = _make_eval_split(
        X,
        y,
        SELECTION_FRACTION,
        creator.RANDOM_STATE,
        validation_strategy="random",
    )
    metadata = {
        "policy": policy,
        "random_state": creator.RANDOM_STATE,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_positions_sha256": _index_sha256(train),
        "validation_positions_sha256": _index_sha256(validation),
    }
    return train, validation, metadata


def _model_bytes(model: Any) -> int:
    with tempfile.TemporaryDirectory(prefix="darkofit-linear-leaf-") as root:
        path = Path(root) / "model.npz"
        model.save_model(path)
        size = path.stat().st_size
    if size <= 0:
        raise RuntimeError("serialized basketball model is empty")
    return int(size)


def _fit_model(X_train, y_train, *, linear_leaves: bool, eval_set=None):
    from darkofit import DarkoRegressor

    params = {
        "random_state": creator.RANDOM_STATE,
        "tree_mode": "catboost",
        "linear_leaves": bool(linear_leaves),
        "verbose_timing": True,
    }
    if eval_set is not None:
        params.update(
            early_stopping=True,
            early_stopping_rounds=None,
            use_best_model=True,
            refit=False,
        )
    model = DarkoRegressor(**params)
    started = time.perf_counter_ns()
    model.fit(X_train, y_train, eval_set=eval_set)
    elapsed = (time.perf_counter_ns() - started) / 1e9
    return model, float(elapsed)


def _fit_default(X_train, y_train, X_test):
    model, fit_seconds = _fit_model(X_train, y_train, linear_leaves=False)
    started = time.perf_counter_ns()
    prediction = harness.validate_prediction(model.predict(X_test), len(X_test))
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    metadata = {
        "kind": "single",
        "selected_linear_leaves": False,
        "final_fit": harness.extract_fit_metadata(model),
    }
    model_bytes = _model_bytes(model)
    return (
        prediction,
        fit_seconds,
        float(predict_seconds),
        model_bytes,
        metadata,
    )


def _selection_record(name: str, model: Any, fit_seconds: float):
    fitted = harness.extract_fit_metadata(model)
    score = float(model.best_score_)
    if not math.isfinite(score):
        raise RuntimeError(f"{name} selection score is not finite")
    validation_history = [
        float(value) for value in model.model_.valid_history_
    ]
    if not validation_history or any(
        not math.isfinite(value) for value in validation_history
    ):
        raise RuntimeError(f"{name} selection validation curve is invalid")
    if not math.isclose(
        min(validation_history), score, rel_tol=0.0, abs_tol=1e-15
    ):
        raise RuntimeError(
            f"{name} selection score does not match its validation curve"
        )
    validation = dict(
        getattr(model.model_, "auto_params_", {}).get("validation_split", {})
    )
    return {
        "name": name,
        "linear_leaves": name == "linear",
        "validation_rmse": score,
        "validation_rmse_history": validation_history,
        "fit_seconds": float(fit_seconds),
        "validation": validation,
        "fit_metadata": fitted,
    }


def _fit_selected_linear_leaf(X_train, y_train, X_test):
    selection_train, validation, split = selection_split(X_train, y_train)
    X_selection = X_train.iloc[selection_train]
    y_selection = y_train.iloc[selection_train]
    eval_set = (
        X_train.iloc[validation],
        y_train.iloc[validation],
    )

    constant, constant_seconds = _fit_model(
        X_selection,
        y_selection,
        linear_leaves=False,
        eval_set=eval_set,
    )
    constant_record = _selection_record("constant", constant, constant_seconds)
    del constant
    gc.collect()

    linear, linear_seconds = _fit_model(
        X_selection,
        y_selection,
        linear_leaves=True,
        eval_set=eval_set,
    )
    linear_record = _selection_record("linear", linear, linear_seconds)
    del linear
    gc.collect()

    selected_linear = (
        linear_record["validation_rmse"] < constant_record["validation_rmse"]
    )
    final, final_seconds = _fit_model(
        X_train,
        y_train,
        linear_leaves=selected_linear,
    )
    started = time.perf_counter_ns()
    prediction = harness.validate_prediction(final.predict(X_test), len(X_test))
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    model_bytes = _model_bytes(final)
    metadata = {
        "kind": "validation_selector",
        "split": split,
        "tie_policy": "constant",
        "selection_fits": [constant_record, linear_record],
        "selected_linear_leaves": bool(selected_linear),
        "selected_name": "linear" if selected_linear else "constant",
        "final_fit_seconds": float(final_seconds),
        "final_fit": harness.extract_fit_metadata(final),
    }
    total_fit = constant_seconds + linear_seconds + final_seconds
    return (
        prediction,
        float(total_fit),
        float(predict_seconds),
        model_bytes,
        metadata,
    )


def fit_and_predict(config_name: str, X_train, y_train, X_test):
    if config_name == DEFAULT_CONFIG:
        return _fit_default(X_train, y_train, X_test)
    if config_name == CANDIDATE_CONFIG:
        return _fit_selected_linear_leaf(X_train, y_train, X_test)
    raise ValueError(f"unknown basketball config: {config_name}")


def validate_fitted_metadata(config_name: str, metadata: dict[str, Any]) -> None:
    final = metadata["final_fit"]
    if final["selected_tree_mode"] != "catboost":
        raise RuntimeError("basketball fit changed tree mode")
    if int(final["resolved_thread_count"]) < 1:
        raise RuntimeError("basketball fit resolved no worker threads")
    if not math.isfinite(float(final["resolved_learning_rate"])):
        raise RuntimeError("basketball fit resolved an invalid learning rate")
    if final["refit"] or final["selection_fit"] is not None:
        raise RuntimeError("final basketball fit unexpectedly used wrapper refit")
    if int(final["fitted_tree_count"]) != int(final["best_iteration"]):
        raise RuntimeError("final basketball fit retained the wrong tree count")
    if int(final["final_fit"]["iterations_requested"]) != 1000:
        raise RuntimeError("final basketball fit changed the default horizon")
    selected_linear = bool(metadata["selected_linear_leaves"])
    expected_lane = "linear_leaves" if selected_linear else "boosting"
    if final["selected_lane"] != expected_lane:
        raise RuntimeError("final basketball lane disagrees with selection")

    if config_name == DEFAULT_CONFIG:
        if metadata["kind"] != "single" or selected_linear:
            raise RuntimeError("default unexpectedly used linear leaves")
        return
    if metadata["kind"] != "validation_selector":
        raise RuntimeError("candidate did not report its selector")
    records = metadata["selection_fits"]
    if [record["name"] for record in records] != ["constant", "linear"]:
        raise RuntimeError("candidate selection lanes changed")
    for record, lane in zip(records, ("boosting", "linear_leaves")):
        if record["fit_metadata"]["selected_lane"] != lane:
            raise RuntimeError("candidate selection fit used the wrong lane")
        if record["validation"].get("source") != "explicit_eval_set":
            raise RuntimeError("candidate selection changed validation source")
        history = record.get("validation_rmse_history")
        if (
            not isinstance(history, list)
            or not history
            or any(not math.isfinite(float(value)) for value in history)
        ):
            raise RuntimeError("candidate selection curve is invalid")
        if not math.isclose(
            min(float(value) for value in history),
            float(record["validation_rmse"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError("candidate selection curve disagrees with score")
    expected_linear = float(records[1]["validation_rmse"]) < float(
        records[0]["validation_rmse"]
    )
    if selected_linear != expected_linear:
        raise RuntimeError("candidate did not apply its tie/score policy")


def _warmup(config_name: str, dataset: harness.BasketballDataset) -> float:
    train, test = next(creator.creator_cv().split(dataset.X, dataset.y))
    started = time.perf_counter_ns()
    result = fit_and_predict(
        config_name,
        dataset.X.iloc[train],
        dataset.y.iloc[train],
        dataset.X.iloc[test],
    )
    if config_name == CANDIDATE_CONFIG:
        selected_linear = bool(result[4]["selected_linear_leaves"])
        opposite, _ = _fit_model(
            dataset.X.iloc[train],
            dataset.y.iloc[train],
            linear_leaves=not selected_linear,
        )
        harness.validate_prediction(opposite.predict(dataset.X.iloc[test]), len(test))
    return float((time.perf_counter_ns() - started) / 1e9)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def _behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": result["config"],
        "folds": [
            {
                "fold": row["fold"],
                "r2": row["r2"],
                "prediction_sha256": row["prediction_sha256"],
                "fit_metadata": row["fit_metadata"],
            }
            for row in result["folds"]
        ],
        "holdout": {
            "scores": result["holdout"]["scores"],
            "prediction_sha256": result["holdout"]["prediction_sha256"],
            "fit_metadata": result["holdout"]["fit_metadata"],
        },
    }


def run_worker(config_name: str, cache_path: Path) -> dict[str, Any]:
    dataset = harness.load_basketball_dataset(cache_path)
    warmup_seconds = _warmup(config_name, dataset)
    folds = []
    steady_started = time.perf_counter_ns()
    for fold, (train, test) in enumerate(
        creator.creator_cv().split(dataset.X, dataset.y)
    ):
        prediction, fit_seconds, predict_seconds, model_bytes, metadata = (
            fit_and_predict(
                config_name,
                dataset.X.iloc[train],
                dataset.y.iloc[train],
                dataset.X.iloc[test],
            )
        )
        validate_fitted_metadata(config_name, metadata)
        folds.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "test_indices": [int(value) for value in test],
                "r2": float(r2_score(dataset.y.iloc[test], prediction)),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "model_bytes": model_bytes,
                "prediction_sha256": harness.prediction_sha256(prediction),
                "predictions": [float(value) for value in prediction],
                "fit_metadata": metadata,
            }
        )
    steady_seconds = (time.perf_counter_ns() - steady_started) / 1e9

    guardrail = dataset.player_guardrail
    prediction, fit_seconds, predict_seconds, model_bytes, metadata = fit_and_predict(
        config_name,
        guardrail.X_train,
        guardrail.y_train,
        guardrail.X_holdout,
    )
    validate_fitted_metadata(config_name, metadata)
    scores = np.asarray([row["r2"] for row in folds], dtype=np.float64)
    result = {
        "config": config_name,
        "mean_r2": float(np.mean(scores)),
        "std_r2": float(np.std(scores)),
        "fold_scores": [float(value) for value in scores],
        "folds": folds,
        "steady_wall_seconds": float(steady_seconds),
        "warmup_seconds_outside_timing": warmup_seconds,
        "summed_fit_seconds": float(sum(row["fit_seconds"] for row in folds)),
        "summed_predict_seconds": float(sum(row["predict_seconds"] for row in folds)),
        "peak_rss_bytes": _peak_rss_bytes(),
        "holdout": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "model_bytes": model_bytes,
            "prediction_sha256": harness.prediction_sha256(prediction),
            "predictions": [float(value) for value in prediction],
            "fit_metadata": metadata,
            "scores": harness.guardrails.score_player_guardrails(
                guardrail.y_holdout,
                prediction,
                guardrail.cold_player_mask,
            ),
        },
        "guardrail": guardrail.metadata,
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }
    result["behavior_fingerprint_sha256"] = harness.behavior_fingerprint(
        _behavior_payload(result)
    )
    return result


def _run_worker_process(args: argparse.Namespace, config_name: str):
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
        env=harness.worker_environment(args.threads),
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
    result["worker_stdout"] = (
        "\n".join(
            line
            for line in completed.stdout.splitlines()
            if not line.startswith(WORKER_RESULT_PREFIX)
        ).strip()
        or None
    )
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def analyze_quality(canonical: dict[str, dict[str, Any]]) -> dict[str, Any]:
    default = canonical[DEFAULT_CONFIG]
    candidate = canonical[CANDIDATE_CONFIG]
    default_scores = np.asarray(default["fold_scores"], dtype=np.float64)
    candidate_scores = np.asarray(candidate["fold_scores"], dtype=np.float64)
    expected_shape = (creator.N_SPLITS,)
    if (
        default_scores.shape != expected_shape
        or candidate_scores.shape != expected_shape
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
        float(np.mean(np.delete(deltas, fold))) for fold in range(creator.N_SPLITS)
    ]
    default_holdout = default["holdout"]["scores"]
    candidate_holdout = candidate["holdout"]["scores"]
    team_delta = float(
        candidate_holdout["overlap_exposed_team_holdout"]["r2"]
        - default_holdout["overlap_exposed_team_holdout"]["r2"]
    )
    cold_delta = float(
        candidate_holdout["cold_player_subset"]["r2"]
        - default_holdout["cold_player_subset"]["r2"]
    )
    gates = {
        "mean_r2_no_regression": float(np.mean(deltas)) >= 0.0,
        "fold_breadth": int(np.count_nonzero(deltas > 0.0)) >= MIN_FOLD_WINS,
        "leave_one_fold_out_no_regression": min(jackknife) >= 0.0,
        "overlap_exposed_team_no_regression": team_delta >= 0.0,
        "cold_player_no_regression": cold_delta >= 0.0,
    }
    return {
        "mean_r2_delta": float(np.mean(deltas)),
        "fold_wins": int(np.count_nonzero(deltas > 0.0)),
        "fold_losses": int(np.count_nonzero(deltas < 0.0)),
        "fold_ties": int(np.count_nonzero(deltas == 0.0)),
        "fold_r2_deltas": [float(value) for value in deltas],
        "leave_one_fold_out_mean_deltas": jackknife,
        "overlap_exposed_team_r2_delta": team_delta,
        "cold_player_r2_delta": cold_delta,
        "quality_gates": gates,
        "passes_quality_gates": all(gates.values()),
    }


def _model_size_ratios(canonical: dict[str, dict[str, Any]]) -> list[float]:
    default = canonical[DEFAULT_CONFIG]
    candidate = canonical[CANDIDATE_CONFIG]
    default_sizes = [row["model_bytes"] for row in default["folds"]]
    candidate_sizes = [row["model_bytes"] for row in candidate["folds"]]
    default_sizes.append(default["holdout"]["model_bytes"])
    candidate_sizes.append(candidate["holdout"]["model_bytes"])
    if len(default_sizes) != len(candidate_sizes) or any(
        int(value) <= 0 for value in default_sizes + candidate_sizes
    ):
        raise RuntimeError("basketball model-size telemetry is incomplete")
    return [
        float(candidate_size) / float(default_size)
        for default_size, candidate_size in zip(default_sizes, candidate_sizes)
    ]


def analyze_runtime(
    canonical,
    wall_timing,
    fit_timing,
    predict_timing,
    peak_rss_values,
) -> dict[str, Any]:
    fit_ratio = float(
        fit_timing[CANDIDATE_CONFIG]["median_seconds"]
        / fit_timing[DEFAULT_CONFIG]["median_seconds"]
    )
    predict_ratio = float(
        predict_timing[CANDIDATE_CONFIG]["median_seconds"]
        / predict_timing[DEFAULT_CONFIG]["median_seconds"]
    )
    rss_ratio = float(
        statistics.median(peak_rss_values[CANDIDATE_CONFIG])
        / statistics.median(peak_rss_values[DEFAULT_CONFIG])
    )
    size_ratios = _model_size_ratios(canonical)
    gates = {
        "default_timing_stable": wall_timing[DEFAULT_CONFIG]["stable"],
        "candidate_timing_stable": wall_timing[CANDIDATE_CONFIG]["stable"],
        "fit_cost_within_budget": fit_ratio <= MAX_FIT_RUNTIME_RATIO,
        "prediction_cost_within_budget": (predict_ratio <= MAX_PREDICT_RUNTIME_RATIO),
        "model_size_within_budget": max(size_ratios) <= MAX_MODEL_SIZE_RATIO,
        "peak_rss_within_budget": rss_ratio <= MAX_PEAK_RSS_RATIO,
    }
    return {
        "candidate_over_default_median_fit_time": fit_ratio,
        "candidate_over_default_median_predict_time": predict_ratio,
        "candidate_over_default_median_peak_rss": rss_ratio,
        "candidate_over_default_model_size_ratios": size_ratios,
        "maximum_model_size_ratio": max(size_ratios),
        "runtime_gates": gates,
        "passes_runtime_gates": all(gates.values()),
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
        raise RuntimeError(f"DarkoFit source changed {boundary}: " + ", ".join(changed))


def _repeat_record(block: int, position: int, result: dict[str, Any]):
    return {
        "block": int(block),
        "position": int(position),
        "config": result["config"],
        "steady_wall_seconds": result["steady_wall_seconds"],
        "warmup_seconds_outside_timing": result["warmup_seconds_outside_timing"],
        "summed_fit_seconds": result["summed_fit_seconds"],
        "summed_predict_seconds": result["summed_predict_seconds"],
        "peak_rss_bytes": result["peak_rss_bytes"],
        "behavior_fingerprint_sha256": result["behavior_fingerprint_sha256"],
        "worker_stdout": result["worker_stdout"],
        "worker_stderr": result["worker_stderr"],
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_symlink():
        raise RuntimeError(f"refusing symlink benchmark output: {args.output}")
    source = _source_state(args.allow_dirty_source)
    dataset = harness.load_basketball_dataset(args.data_cache)
    schedule = harness.reciprocal_schedule(DEFAULT_CONFIG, CANDIDATE_CONFIG)
    canonical: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, set[str]] = {name: set() for name in CONFIG_ORDER}
    wall_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    fit_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    predict_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    rss_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    repeats = []

    def run_block(block: int, order: tuple[str, str]) -> None:
        for position, config_name in enumerate(order):
            _assert_source_unchanged(
                source,
                _source_state(args.allow_dirty_source),
                boundary=f"before block {block} {config_name}",
            )
            print(
                f"running block {block + 1}/{len(schedule)} "
                f"position {position + 1}: {config_name}...",
                flush=True,
            )
            result = _run_worker_process(args, config_name)
            fingerprint = result["behavior_fingerprint_sha256"]
            fingerprints[config_name].add(fingerprint)
            wall_values[config_name].append(result["steady_wall_seconds"])
            fit_values[config_name].append(result["summed_fit_seconds"])
            predict_values[config_name].append(result["summed_predict_seconds"])
            rss_values[config_name].append(result["peak_rss_bytes"])
            repeats.append(_repeat_record(block, position, result))
            canonical.setdefault(config_name, result)
            cold = result["holdout"]["scores"]["cold_player_subset"]["r2"]
            print(
                f"  mean R2={result['mean_r2']:.12f}, cold={cold:.12f}, "
                f"steady={result['steady_wall_seconds']:.2f}s",
                flush=True,
            )

    run_block(0, schedule[0])
    quality = analyze_quality(canonical)
    timing_skipped = not quality["passes_quality_gates"]
    if not timing_skipped:
        for block, order in enumerate(schedule[1:], start=1):
            run_block(block, order)

    for config_name, values in fingerprints.items():
        if len(values) != 1:
            raise RuntimeError(f"{config_name} behavior changed across repeats")
    _assert_source_unchanged(
        source,
        _source_state(args.allow_dirty_source),
        boundary="during the experiment",
    )

    wall_timing = fit_timing = predict_timing = runtime = None
    if not timing_skipped:
        wall_timing = {
            name: harness.timing_summary(values) for name, values in wall_values.items()
        }
        fit_timing = {
            name: harness.timing_summary(values) for name, values in fit_values.items()
        }
        predict_timing = {
            name: harness.timing_summary(values)
            for name, values in predict_values.items()
        }
        runtime = analyze_runtime(
            canonical,
            wall_timing,
            fit_timing,
            predict_timing,
            rss_values,
        )

    evidence_eligible = bool(source["clean"])
    passes_runtime = runtime is not None and runtime["passes_runtime_gates"]
    passes_all = (
        quality["passes_quality_gates"] and passes_runtime and evidence_eligible
    )
    decision = {
        "candidate": CANDIDATE_CONFIG,
        "candidate_scope": "experimental_opt_in_only",
        "default_promotion_authorized": False,
        **quality,
        "timing_confirmation_status": (
            "skipped_fatal_quality_failure" if timing_skipped else "completed"
        ),
        "runtime": runtime,
        "passes_runtime_gates": passes_runtime,
        "evidence_gates": {"committed_clean_source": evidence_eligible},
        "evidence_eligible": evidence_eligible,
        "passes_all_gates": passes_all,
        "recommendation": (
            "advance_to_spent_development_panel"
            if passes_all
            else (
                "development_only_dirty_source"
                if not evidence_eligible
                else "advance_none"
            )
        ),
    }
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_validation_selected_linear_leaves",
            "diagnostic_only": True,
            "candidate_scope": "experimental_opt_in_only",
            "default_promotion_authorized": False,
            "creator_benchmark_changed": False,
            "selection_fraction": SELECTION_FRACTION,
            "selection_tie_policy": "constant",
            "selection_lanes": ["constant", "linear"],
            "final_horizon": 1000,
            "final_learning_rate": "auto_full_external_training_fold",
            "quality_fail_fast": True,
            "planned_timing_schedule": [list(order) for order in schedule],
            "executed_timing_blocks": 1 if timing_skipped else len(schedule),
            "maximum_timing_spread_ratio": harness.MAX_TIMING_SPREAD_RATIO,
            "maximum_fit_runtime_ratio": MAX_FIT_RUNTIME_RATIO,
            "maximum_predict_runtime_ratio": MAX_PREDICT_RUNTIME_RATIO,
            "maximum_model_size_ratio": MAX_MODEL_SIZE_RATIO,
            "maximum_peak_rss_ratio": MAX_PEAK_RSS_RATIO,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "random_state": creator.RANDOM_STATE,
            "threads_per_fit": args.threads,
            "cv": {
                "kind": "KFold",
                "n_splits": creator.N_SPLITS,
                "shuffle": False,
                "n_jobs": 1,
                "fold_fingerprint_sha256": dataset.fold_fingerprint_sha256,
                "fold_test_sizes": dataset.fold_test_sizes,
            },
            "warmup": "one complete first-fold fit and prediction per worker",
            "weights_used": False,
            "lockbox_data_used": False,
        },
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "guardrail": dataset.player_guardrail.metadata,
        "source": source,
        "environment": {
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
        },
        "canonical_results": [canonical[name] for name in CONFIG_ORDER],
        "timing_repeats": repeats,
        "wall_timing_summary": wall_timing,
        "fit_timing_summary": fit_timing,
        "predict_timing_summary": predict_timing,
        "peak_rss_values": rss_values,
        "decision": decision,
    }
    creator._atomic_write_bytes(
        args.output,
        (json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    print(f"decision: {decision['recommendation']}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-cache", type=Path, default=harness.DEFAULT_CACHE)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--worker-config", choices=CONFIG_ORDER, help=argparse.SUPPRESS)
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
            WORKER_RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
