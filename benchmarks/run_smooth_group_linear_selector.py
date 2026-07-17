#!/usr/bin/env python3
"""Run the frozen smooth group-safe linear-leaf selector gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_harness as basketball  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_smooth_linear_leaves_development as base  # noqa: E402


TASKS = base.TASKS
FOLDS = base.FOLDS
CONTROL = "darko_default"
SELECTOR = "smooth_margin_selector"
FIXED = "darko_linear_current"
CHIMERA = "chimera_product"
CONFIGS = (CONTROL, SELECTOR, FIXED, CHIMERA)
VALIDATION_FRACTION = 0.2
MIN_RELATIVE_IMPROVEMENT = 0.03
MIN_EQUAL_TASK_GAIN = 0.02
MIN_DATASET_WINS = 2
MIN_SPLIT_WINS = 14
MIN_SELECTIONS = 14
MIN_FIXED_BENEFIT_RETENTION = 0.9
MAX_SELECTOR_OVER_FIXED_DATASET_RATIO = 1.01
PROTOCOL = ROOT / "benchmarks" / "smooth_group_linear_selector_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "smooth_group_linear_selector.json"
WORKER_RESULT_PREFIX = "SMOOTH_GROUP_LINEAR_SELECTOR_RESULT="


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_sha256(values) -> str:
    return base._array_sha256(values, dtype="<i8")


def selection_split(X, y):
    from darkofit.sklearn_api import _make_eval_split

    train, validation, policy = _make_eval_split(
        X,
        y,
        VALIDATION_FRACTION,
        creator.RANDOM_STATE,
        validation_strategy="weighted_stratified",
    )
    return train, validation, {
        "policy": policy,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_positions_sha256": _index_sha256(train),
        "validation_positions_sha256": _index_sha256(validation),
    }


def _fit_darko(X, y, *, linear_leaves: bool, threads: int, eval_set=None):
    from darkofit import DarkoRegressor

    params = {
        "random_state": creator.RANDOM_STATE,
        "thread_count": int(threads),
        "linear_leaves": bool(linear_leaves),
        "verbose_timing": True,
    }
    if eval_set is not None:
        params.update(
            {
                "early_stopping": True,
                "early_stopping_rounds": None,
                "use_best_model": True,
                "refit": False,
            }
        )
    model = DarkoRegressor(**params)
    started = time.perf_counter_ns()
    model.fit(X, y, eval_set=eval_set)
    seconds = (time.perf_counter_ns() - started) / 1e9
    return model, float(seconds)


def _selection_fit_record(name, model, fit_seconds):
    fitted = basketball.extract_fit_metadata(model)
    validation = dict(model.model_.auto_params_.get("validation_split", {}))
    score = float(model.best_score_)
    if not math.isfinite(score) or score <= 0.0:
        raise RuntimeError("smooth selector validation score is invalid")
    if validation.get("source") != "explicit_eval_set":
        raise RuntimeError("smooth selector did not use its explicit eval set")
    if fitted["final_fit"]["stop_reason"] != "early_stopping":
        raise RuntimeError("smooth selector candidate did not early-stop")
    return {
        "name": name,
        "linear_leaves": name == "linear",
        "validation_rmse": score,
        "fit_seconds": float(fit_seconds),
        "validation": validation,
        "fit_metadata": fitted,
    }


def _evaluate_selector_fold(task, X, y, fold: int, threads: int):
    outer_train, outer_test = task.get_train_test_split_indices(
        repeat=0, fold=int(fold), sample=0
    )
    X_outer = X.iloc[outer_train]
    y_outer = y.iloc[outer_train]
    train, validation, split = selection_split(X_outer, y_outer)
    eval_set = (X_outer.iloc[validation], y_outer.iloc[validation])
    selection_fits = []
    for name, linear_leaves in (("constant", False), ("linear", True)):
        model, seconds = _fit_darko(
            X_outer.iloc[train],
            y_outer.iloc[train],
            linear_leaves=linear_leaves,
            threads=threads,
            eval_set=eval_set,
        )
        selection_fits.append(
            _selection_fit_record(name, model, seconds)
        )
    constant_score = selection_fits[0]["validation_rmse"]
    linear_score = selection_fits[1]["validation_rmse"]
    margin = float((constant_score - linear_score) / constant_score)
    selected = margin >= MIN_RELATIVE_IMPROVEMENT

    final, final_fit_seconds = _fit_darko(
        X_outer,
        y_outer,
        linear_leaves=selected,
        threads=threads,
    )
    started = time.perf_counter_ns()
    prediction = np.asarray(
        final.predict(X.iloc[outer_test]), dtype=np.float64
    )
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    if prediction.shape != (len(outer_test),) or not np.all(
        np.isfinite(prediction)
    ):
        raise RuntimeError("smooth selector produced invalid predictions")
    rmse = float(
        mean_squared_error(
            np.asarray(y.iloc[outer_test], dtype=np.float64),
            prediction,
        )
        ** 0.5
    )
    return {
        "fold": int(fold),
        "train_rows": int(len(outer_train)),
        "test_rows": int(len(outer_test)),
        "train_index_sha256": _index_sha256(outer_train),
        "test_index_sha256": _index_sha256(outer_test),
        "rmse": rmse,
        "fit_seconds": float(
            final_fit_seconds
            + sum(row["fit_seconds"] for row in selection_fits)
        ),
        "predict_seconds": float(predict_seconds),
        "prediction_sha256": base._array_sha256(prediction),
        "fit_metadata": {
            "kind": "smooth_margin_selector",
            "split": split,
            "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
            "relative_validation_improvement": margin,
            "selected_linear_leaves": bool(selected),
            "selection_fits": selection_fits,
            "final_fit_seconds": float(final_fit_seconds),
            "final_fit": base._fit_metadata(
                FIXED if selected else CONTROL, final
            ),
        },
    }


def _evaluate_fold(task, X, y, config: str, fold: int, threads: int):
    if config == SELECTOR:
        return _evaluate_selector_fold(task, X, y, fold, threads)
    return base._evaluate_fold(task, X, y, config, fold, threads)


def run_worker(task_id: int, config: str, threads: int) -> dict[str, Any]:
    task, X, y, task_metadata = base._load_task(task_id)
    _evaluate_fold(task, X, y, config, FOLDS[0], threads)
    started = time.perf_counter_ns()
    folds = [
        _evaluate_fold(task, X, y, config, fold, threads)
        for fold in FOLDS
    ]
    wall_seconds = (time.perf_counter_ns() - started) / 1e9
    result = {
        "task": task_metadata,
        "config": config,
        "folds": folds,
        "fold_count": len(folds),
        "geomean_rmse": base._geomean(
            [row["rmse"] for row in folds]
        ),
        "wall_seconds": float(wall_seconds),
        "summed_fit_seconds": float(
            sum(row["fit_seconds"] for row in folds)
        ),
        "summed_predict_seconds": float(
            sum(row["predict_seconds"] for row in folds)
        ),
        "peak_rss_bytes": base._peak_rss_bytes(),
    }
    result["behavior_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "task_id": task_id,
                "config": config,
                "folds": [
                    {
                        "fold": row["fold"],
                        "rmse": row["rmse"],
                        "prediction_sha256": row["prediction_sha256"],
                        "fit_metadata": row["fit_metadata"],
                    }
                    for row in folds
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return result


def _worker_command(task_id: int, config: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-task",
        str(task_id),
        "--worker-config",
        config,
        "--worker-threads",
        str(base.THREADS_PER_WORKER),
    ]


def _run_wave(config: str) -> list[dict[str, Any]]:
    processes = []
    for task_id in TASKS:
        process = subprocess.Popen(
            _worker_command(task_id, config),
            cwd=ROOT,
            env=base._worker_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((task_id, process))
    results = []
    for task_id, process in processes:
        stdout, stderr = process.communicate()
        lines = [
            line for line in stdout.splitlines()
            if line.startswith(WORKER_RESULT_PREFIX)
        ]
        if process.returncode or len(lines) != 1:
            raise RuntimeError(
                f"smooth selector worker {config}/{task_id} failed with "
                f"{process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        result = json.loads(lines[0][len(WORKER_RESULT_PREFIX):])
        result["worker_stdout"] = (
            "\n".join(
                line for line in stdout.splitlines()
                if not line.startswith(WORKER_RESULT_PREFIX)
            ).strip()
            or None
        )
        result["worker_stderr"] = stderr.strip() or None
        results.append(result)
    return results


def analyze(results) -> dict[str, Any]:
    expected = len(TASKS) * len(CONFIGS)
    if len(results) != expected:
        raise RuntimeError(f"smooth selector requires {expected} workers")
    for config in CONFIGS:
        if sum(row["config"] == config for row in results) != len(TASKS):
            raise RuntimeError(f"smooth selector config incomplete: {config}")

    contrasts = {
        "selector_over_default": base._contrast(
            results, SELECTOR, CONTROL
        ),
        "fixed_over_default": base._contrast(results, FIXED, CONTROL),
        "selector_over_fixed": base._contrast(results, SELECTOR, FIXED),
        "selector_over_chimera_product": base._contrast(
            results, SELECTOR, CHIMERA
        ),
    }
    selected = [
        fold["fit_metadata"]["selected_linear_leaves"]
        for result in results
        if result["config"] == SELECTOR
        for fold in result["folds"]
    ]
    policies = [
        fold["fit_metadata"]["split"]["policy"]
        for result in results
        if result["config"] == SELECTOR
        for fold in result["folds"]
    ]
    selector_ratio = contrasts["selector_over_default"][
        "equal_task_geomean_ratio"
    ]
    fixed_ratio = contrasts["fixed_over_default"][
        "equal_task_geomean_ratio"
    ]
    fixed_gain = 1.0 - fixed_ratio
    benefit_retention = (
        float((1.0 - selector_ratio) / fixed_gain)
        if fixed_gain > 0.0
        else float("-inf")
    )
    primary = contrasts["selector_over_default"]
    gates = {
        "equal_task_gain_at_least_2pct": (
            selector_ratio <= 1.0 - MIN_EQUAL_TASK_GAIN
        ),
        "at_least_two_dataset_wins": (
            primary["dataset_wins"] >= MIN_DATASET_WINS
        ),
        "no_dataset_regression": primary["worst_dataset_ratio"] <= 1.0,
        "at_least_14_split_wins": (
            primary["split_wins"] >= MIN_SPLIT_WINS
        ),
        "selects_at_least_14_coordinates": (
            sum(selected) >= MIN_SELECTIONS
        ),
        "declines_at_least_one_coordinate": sum(selected) < len(selected),
        "retains_at_least_90pct_of_fixed_benefit": (
            benefit_retention >= MIN_FIXED_BENEFIT_RETENTION
        ),
        "no_dataset_over_1pct_worse_than_fixed": (
            contrasts["selector_over_fixed"]["worst_dataset_ratio"]
            <= MAX_SELECTOR_OVER_FIXED_DATASET_RATIO
        ),
        "all_internal_splits_target_stratified": all(
            policy == "weighted_target_stratified" for policy in policies
        ),
    }
    passes = all(gates.values())
    return {
        "contrasts": contrasts,
        "selection_count": int(sum(selected)),
        "decline_count": int(len(selected) - sum(selected)),
        "selection_total": len(selected),
        "fixed_benefit_retention": benefit_retention,
        "gates": gates,
        "passes_all_gates": passes,
        "recommendation": (
            "advance_selector_to_fresh_confirmation_design"
            if passes
            else "close_smooth_margin_selector"
        ),
    }


def run_parent(args) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    boundary = base._partition_boundary()
    source = creator.git_state(ROOT)
    chimera_source = creator.git_state(base.CHIMERA_ROOT)
    if not source["clean"] or not chimera_source["clean"]:
        raise RuntimeError("smooth selector requires clean source trees")
    if chimera_source["head"] != base.EXPECTED_CHIMERA_HEAD:
        raise RuntimeError("ChimeraBoost source is not frozen v0.15.0")

    results = []
    for wave, config in enumerate(CONFIGS):
        if creator.git_state(ROOT) != source:
            raise RuntimeError("DarkoFit changed during smooth selector gate")
        if creator.git_state(base.CHIMERA_ROOT) != chimera_source:
            raise RuntimeError("ChimeraBoost changed during selector gate")
        print(
            f"wave {wave + 1}/{len(CONFIGS)}: {config} "
            f"({base.CONCURRENT_TASK_WORKERS} concurrent tasks)",
            flush=True,
        )
        results.extend(_run_wave(config))
    if creator.git_state(ROOT) != source:
        raise RuntimeError("DarkoFit changed during smooth selector gate")
    if creator.git_state(base.CHIMERA_ROOT) != chimera_source:
        raise RuntimeError("ChimeraBoost changed during selector gate")

    analysis = analyze(results)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "smooth_group_linear_selector",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "tasks": TASKS,
            "folds": list(FOLDS),
            "coordinate_count": len(TASKS) * len(FOLDS),
            "configs": list(CONFIGS),
            "validation_fraction": VALIDATION_FRACTION,
            "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
            "threads_per_worker": base.THREADS_PER_WORKER,
            "concurrent_task_workers": base.CONCURRENT_TASK_WORKERS,
            "lockbox_data_used": False,
            "development_only": True,
            "public_selector_authorized": False,
            "default_promotion_authorized": False,
        },
        "partition_boundary": boundary,
        "sources": {
            "darkofit": source,
            "chimeraboost": chimera_source,
        },
        "environment": {
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
            "python": sys.version,
        },
        "results": results,
        "analysis": analysis,
    }
    creator._atomic_write_bytes(
        args.output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    print(f"decision: {analysis['recommendation']}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-task", type=int, choices=tuple(TASKS))
    parser.add_argument("--worker-config", choices=CONFIGS)
    parser.add_argument(
        "--worker-threads", type=int, default=base.THREADS_PER_WORKER
    )
    args = parser.parse_args(argv)
    args.output = creator._absolute_lexical_path(args.output)
    if bool(args.worker_task is not None) != bool(args.worker_config):
        parser.error("--worker-task and --worker-config must be used together")
    if args.worker_threads < 1:
        parser.error("--worker-threads must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker_config:
        result = run_worker(
            args.worker_task, args.worker_config, args.worker_threads
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
