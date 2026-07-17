#!/usr/bin/env python3
"""Run the frozen smooth linear-leaf development comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
CHIMERA_ROOT = ROOT.parent / "chimeraboost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_harness as basketball  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


TASKS = {
    361251: "grid_stability",
    361258: "kin8nm",
    361623: "space_ga",
}
FOLDS = tuple(range(3, 10))
CONFIGS = (
    "darko_default",
    "darko_linear_current",
    "darko_linear_matched",
    "darko_linear_residual",
    "chimera_linear_only",
    "chimera_product",
)
THREADS_PER_WORKER = 6
CONCURRENT_TASK_WORKERS = 3
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
MAX_ADVANCING_RATIO = 0.98
MIN_DATASET_WINS = 2
MIN_SPLIT_WINS = 14
PROTOCOL = ROOT / "benchmarks" / "smooth_linear_leaves_development_protocol.md"
PARTITION = ROOT / "benchmarks" / "ctr23_partition.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "smooth_linear_leaves_development.json"
WORKER_RESULT_PREFIX = "SMOOTH_LINEAR_DEVELOPMENT_RESULT="


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value, dtype="<f8") -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def _partition_boundary() -> dict[str, Any]:
    payload = json.loads(PARTITION.read_text())
    confirmation = set(payload["confirmation_task_ids"])
    lockbox = set(payload["lockbox_task_ids"])
    tasks = set(TASKS)
    if not tasks <= confirmation:
        raise RuntimeError("smooth development task left confirmation panel")
    if tasks & lockbox:
        raise RuntimeError("smooth development task intersects lockbox")
    metadata = payload["task_allocation_metadata"]
    for task_id, name in TASKS.items():
        row = metadata[str(task_id)]
        if row["lineage_cluster"] not in {
            "electrical_grid_stability",
            "delve_kin8nm",
            "space_ga",
        }:
            raise RuntimeError(f"{name} lineage changed")
        if row["has_categorical"] != 0.0 or row["has_missing_features"] != 0.0:
            raise RuntimeError(f"{name} is no longer numeric and complete")
    return {
        "partition_sha256": _sha256(PARTITION),
        "declared_partition_sha256": payload["partition_sha256"],
        "confirmation_task_ids": sorted(confirmation),
        "lockbox_task_ids": sorted(lockbox),
    }


def _load_task(task_id: int):
    import openml

    if task_id not in TASKS:
        raise ValueError(f"undeclared smooth task: {task_id}")
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
        raise RuntimeError(f"{TASKS[task_id]} unexpectedly has nonfinite values")
    if task.get_split_dimensions() != (1, 10, 1):
        raise RuntimeError(f"{TASKS[task_id]} split dimensions changed")
    metadata = {
        "task_id": int(task_id),
        "dataset_id": int(dataset.dataset_id),
        "dataset_name": str(dataset.name),
        "target_name": str(task.target_name),
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "X_sha256": _array_sha256(X_array),
        "y_sha256": _array_sha256(y_array),
        "split_dimensions": [1, 10, 1],
    }
    return task, X, y, metadata


def _make_model(config: str, threads: int):
    if config.startswith("darko_"):
        from darkofit import DarkoRegressor

        params = {
            "random_state": 4,
            "thread_count": int(threads),
            "verbose_timing": True,
        }
        if config == "darko_linear_current":
            params["linear_leaves"] = True
        elif config == "darko_linear_matched":
            params.update(
                {
                    "linear_leaves": True,
                    "l2_leaf_reg": 1.0,
                    "max_bins": 128,
                    "learning_rate": 0.1,
                    "iterations": 1000,
                }
            )
        elif config == "darko_linear_residual":
            params["linear_residual"] = True
        elif config != "darko_default":
            raise ValueError(f"unknown DarkoFit config: {config}")
        return DarkoRegressor(**params)

    if str(CHIMERA_ROOT) not in sys.path:
        sys.path.insert(0, str(CHIMERA_ROOT))
    from chimeraboost import ChimeraBoostRegressor

    params = {"random_state": 4, "thread_count": int(threads)}
    if config == "chimera_linear_only":
        params.update({"linear_leaves": True, "cross_features": False})
    elif config != "chimera_product":
        raise ValueError(f"unknown ChimeraBoost config: {config}")
    return ChimeraBoostRegressor(**params)


def _fit_metadata(config: str, model) -> dict[str, Any]:
    if config.startswith("darko_"):
        metadata = basketball.extract_fit_metadata(model)
        return {
            "library": "darkofit",
            "version": "0.9.0",
            "fitted_tree_count": metadata["fitted_tree_count"],
            "best_iteration": metadata["best_iteration"],
            "resolved_learning_rate": metadata["resolved_learning_rate"],
            "selected_tree_mode": metadata["selected_tree_mode"],
            "selected_lane": metadata["selected_lane"],
            "linear_leaves_active": metadata["linear_leaves_active"],
            "linear_residual_active": metadata["linear_residual_active"],
            "stop_reason": metadata["final_fit"]["stop_reason"],
        }
    core = model.model_
    return {
        "library": "chimeraboost",
        "version": "0.15.0",
        "fitted_tree_count": int(len(core.trees_)),
        "resolved_learning_rate": float(core.lr_),
        "linear_leaves_selected": bool(model.linear_leaves_selected_),
        "cross_features_selected": bool(model.cross_features_selected_),
        "cross_pair_count": int(len(model.cross_pairs_ or ())),
    }


def _evaluate_fold(task, X, y, config: str, fold: int, threads: int):
    train, test = task.get_train_test_split_indices(
        repeat=0, fold=int(fold), sample=0
    )
    model = _make_model(config, threads)
    started = time.perf_counter_ns()
    model.fit(X.iloc[train], y.iloc[train])
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X.iloc[test]), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    if prediction.shape != (len(test),) or not np.all(np.isfinite(prediction)):
        raise RuntimeError("smooth worker produced invalid predictions")
    rmse = float(
        mean_squared_error(np.asarray(y.iloc[test], dtype=np.float64), prediction)
        ** 0.5
    )
    return {
        "fold": int(fold),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_index_sha256": _array_sha256(train, dtype="<i8"),
        "test_index_sha256": _array_sha256(test, dtype="<i8"),
        "rmse": rmse,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "prediction_sha256": _array_sha256(prediction),
        "fit_metadata": _fit_metadata(config, model),
    }


def run_worker(task_id: int, config: str, threads: int) -> dict[str, Any]:
    task, X, y, task_metadata = _load_task(task_id)
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
        "geomean_rmse": float(
            np.exp(np.mean(np.log([row["rmse"] for row in folds])))
        ),
        "wall_seconds": float(wall_seconds),
        "summed_fit_seconds": float(
            sum(row["fit_seconds"] for row in folds)
        ),
        "summed_predict_seconds": float(
            sum(row["predict_seconds"] for row in folds)
        ),
        "peak_rss_bytes": _peak_rss_bytes(),
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


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in creator.THREAD_LIMIT_ENV_KEYS:
        environment[key] = str(THREADS_PER_WORKER)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "NUMBA_DISABLE_JIT": "0",
            "PYTHONHASHSEED": "0",
            "OPENML_CACHE_DIRECTORY": str(Path.home() / ".cache" / "openml"),
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), str(CHIMERA_ROOT), environment.get("PYTHONPATH", "")]
            ),
        }
    )
    return environment


def _worker_command(task_id: int, config: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-task",
        str(task_id),
        "--worker-config",
        config,
        "--worker-threads",
        str(THREADS_PER_WORKER),
    ]


def _run_wave(config: str) -> list[dict[str, Any]]:
    processes = []
    for task_id in TASKS:
        process = subprocess.Popen(
            _worker_command(task_id, config),
            cwd=ROOT,
            env=_worker_environment(),
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
                f"smooth worker {config}/{task_id} failed with "
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


def _geomean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(values))))


def _contrast(results, numerator: str, denominator: str) -> dict[str, Any]:
    per_dataset = {}
    split_ratios = []
    for task_id, dataset_name in TASKS.items():
        top = next(
            row for row in results
            if row["config"] == numerator
            and row["task"]["task_id"] == task_id
        )
        bottom = next(
            row for row in results
            if row["config"] == denominator
            and row["task"]["task_id"] == task_id
        )
        ratios = [
            float(a["rmse"] / b["rmse"])
            for a, b in zip(top["folds"], bottom["folds"])
        ]
        split_ratios.extend(ratios)
        per_dataset[dataset_name] = {
            "task_id": task_id,
            "ratio": _geomean(ratios),
            "pct": (_geomean(ratios) - 1.0) * 100.0,
            "split_ratios": ratios,
            "split_wins": int(np.count_nonzero(np.asarray(ratios) < 1.0)),
            "split_losses": int(np.count_nonzero(np.asarray(ratios) > 1.0)),
        }
    dataset_ratios = [row["ratio"] for row in per_dataset.values()]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "equal_task_geomean_ratio": _geomean(dataset_ratios),
        "equal_task_pct": (_geomean(dataset_ratios) - 1.0) * 100.0,
        "dataset_wins": int(np.count_nonzero(np.asarray(dataset_ratios) < 1.0)),
        "dataset_losses": int(np.count_nonzero(np.asarray(dataset_ratios) > 1.0)),
        "split_wins": int(np.count_nonzero(np.asarray(split_ratios) < 1.0)),
        "split_losses": int(np.count_nonzero(np.asarray(split_ratios) > 1.0)),
        "worst_dataset_ratio": float(max(dataset_ratios)),
        "worst_split_ratio": float(max(split_ratios)),
        "per_dataset": per_dataset,
    }


def analyze(results) -> dict[str, Any]:
    expected = len(TASKS) * len(CONFIGS)
    if len(results) != expected:
        raise RuntimeError(f"smooth development requires {expected} workers")
    for config in CONFIGS:
        if sum(row["config"] == config for row in results) != len(TASKS):
            raise RuntimeError(f"smooth development config incomplete: {config}")
    names = {
        "current_over_default": (
            "darko_linear_current", "darko_default"
        ),
        "matched_over_default": (
            "darko_linear_matched", "darko_default"
        ),
        "residual_over_default": (
            "darko_linear_residual", "darko_default"
        ),
        "current_over_chimera_product": (
            "darko_linear_current", "chimera_product"
        ),
        "matched_over_chimera_product": (
            "darko_linear_matched", "chimera_product"
        ),
        "current_over_chimera_linear_only": (
            "darko_linear_current", "chimera_linear_only"
        ),
        "matched_over_chimera_linear_only": (
            "darko_linear_matched", "chimera_linear_only"
        ),
        "chimera_linear_only_over_product": (
            "chimera_linear_only", "chimera_product"
        ),
    }
    contrasts = {
        name: _contrast(results, numerator, denominator)
        for name, (numerator, denominator) in names.items()
    }
    candidates = {}
    for short, config in (
        ("current", "darko_linear_current"),
        ("matched", "darko_linear_matched"),
    ):
        contrast = contrasts[f"{short}_over_default"]
        gates = {
            "equal_task_ratio_at_most_0_98": (
                contrast["equal_task_geomean_ratio"] <= MAX_ADVANCING_RATIO
            ),
            "at_least_two_dataset_wins": (
                contrast["dataset_wins"] >= MIN_DATASET_WINS
            ),
            "no_dataset_regression": contrast["worst_dataset_ratio"] <= 1.0,
            "at_least_14_split_wins": (
                contrast["split_wins"] >= MIN_SPLIT_WINS
            ),
        }
        candidates[config] = {
            "gates": gates,
            "passes": all(gates.values()),
        }
    passing = [
        config for config, decision in candidates.items()
        if decision["passes"]
    ]
    advancing = None
    if passing:
        advancing = min(
            passing,
            key=lambda config: (
                contrasts[
                    "current_over_default"
                    if config == "darko_linear_current"
                    else "matched_over_default"
                ]["equal_task_geomean_ratio"],
                config != "darko_linear_current",
            ),
        )
    selected_short = (
        None
        if advancing is None
        else ("current" if advancing.endswith("current") else "matched")
    )
    residual_deprecation = False
    if advancing is not None:
        selected_vs_residual = _contrast(
            results, advancing, "darko_linear_residual"
        )
        contrasts["advancing_over_linear_residual"] = selected_vs_residual
        residual_deprecation = (
            selected_vs_residual["dataset_wins"] == len(TASKS)
            and selected_vs_residual["worst_dataset_ratio"] < 1.0
        )
    parity = (
        False
        if selected_short is None
        else contrasts[
            f"{selected_short}_over_chimera_product"
        ]["equal_task_geomean_ratio"] <= 1.0
    )
    return {
        "contrasts": contrasts,
        "candidate_gates": candidates,
        "advancing_candidate": advancing,
        "advances_to_selector_design": advancing is not None,
        "chimera_product_parity_reached": parity,
        "recommend_linear_residual_deprecation": residual_deprecation,
        "recommendation": (
            "design_selector_and_deprecate_linear_residual"
            if advancing is not None and residual_deprecation
            else (
                "design_selector_keep_linear_residual"
                if advancing is not None
                else "close_linear_leaf_policy_route"
            )
        ),
    }


def run_parent(args) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    boundary = _partition_boundary()
    source = creator.git_state(ROOT)
    chimera_source = creator.git_state(CHIMERA_ROOT)
    if not source["clean"] or not chimera_source["clean"]:
        raise RuntimeError("smooth development requires clean source trees")
    if chimera_source["head"] != EXPECTED_CHIMERA_HEAD:
        raise RuntimeError("ChimeraBoost source is not frozen v0.15.0")
    results = []
    for wave, config in enumerate(CONFIGS):
        if creator.git_state(ROOT) != source:
            raise RuntimeError("DarkoFit source changed during smooth campaign")
        if creator.git_state(CHIMERA_ROOT) != chimera_source:
            raise RuntimeError("ChimeraBoost source changed during smooth campaign")
        print(
            f"wave {wave + 1}/{len(CONFIGS)}: {config} "
            f"({CONCURRENT_TASK_WORKERS} concurrent tasks)",
            flush=True,
        )
        results.extend(_run_wave(config))
    if creator.git_state(ROOT) != source:
        raise RuntimeError("DarkoFit source changed during smooth campaign")
    if creator.git_state(CHIMERA_ROOT) != chimera_source:
        raise RuntimeError("ChimeraBoost source changed during smooth campaign")
    analysis = analyze(results)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "smooth_linear_leaves_development",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "tasks": TASKS,
            "folds": list(FOLDS),
            "coordinate_count": len(TASKS) * len(FOLDS),
            "configs": list(CONFIGS),
            "threads_per_worker": THREADS_PER_WORKER,
            "concurrent_task_workers": CONCURRENT_TASK_WORKERS,
            "lockbox_data_used": False,
            "development_only": True,
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
    parser.add_argument("--worker-threads", type=int, default=THREADS_PER_WORKER)
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
