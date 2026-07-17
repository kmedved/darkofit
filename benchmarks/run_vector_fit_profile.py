#!/usr/bin/env python3
"""Profile unmeasured classification and distributional fit paths."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


CASES = (
    "scalar_rmse_catboost",
    "binary_catboost",
    "multiclass_catboost_per_class",
    "multiclass_lightgbm_shared",
    "gaussian_lightgbm",
    "student_t_lightgbm",
)
PHASES = (
    "preprocess",
    "grad_hess",
    "tree_build",
    "train_update",
    "validation_predict",
    "loss_eval",
)
ROWS = 50_000
FEATURES = 24
ITERATIONS = 40
REPETITIONS = 3
THREADS = 18
WORKER_RESULT_PREFIX = "VECTOR_FIT_PROFILE_RESULT="
PROTOCOL = ROOT / "benchmarks" / "vector_fit_profile_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "vector_fit_profile.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _data(rows: int):
    rng = np.random.default_rng(20_260_717)
    X = rng.normal(size=(rows, FEATURES))
    signal = (
        1.4 * X[:, 0]
        - 0.9 * X[:, 1]
        + 0.35 * X[:, 2] * X[:, 3]
        + 0.2 * X[:, 4] ** 2
    )
    rmse = signal + rng.normal(0.0, 0.5, rows)
    binary = (signal + rng.normal(0.0, 0.8, rows) > 0.0).astype(np.int64)
    logits = np.column_stack(
        [
            signal,
            -signal + 0.4 * X[:, 5],
            0.8 * X[:, 6] - 0.5 * X[:, 7],
            -0.6 * X[:, 6] + 0.7 * X[:, 8],
        ]
    )
    logits += rng.normal(0.0, 0.4, logits.shape)
    multiclass = np.argmax(logits, axis=1).astype(np.int64)
    scale = np.exp(np.clip(0.2 * X[:, 9], -0.8, 0.8))
    distributional = signal + rng.normal(0.0, scale, rows)
    return X, {
        "scalar_rmse_catboost": rmse,
        "binary_catboost": binary,
        "multiclass_catboost_per_class": multiclass,
        "multiclass_lightgbm_shared": multiclass,
        "gaussian_lightgbm": distributional,
        "student_t_lightgbm": distributional,
    }


def _base_params(iterations: int) -> dict[str, Any]:
    return {
        "iterations": int(iterations),
        "learning_rate": 0.1,
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "subsample": 1.0,
        "colsample": 1.0,
        "min_child_weight": 1.0,
        "min_child_samples": 20,
        "ordered_boosting": False,
        "early_stopping": False,
        "use_best_model": False,
        "eval_train_loss": False,
        "thread_count": THREADS,
        "random_state": 4,
        "verbose_timing": True,
        "diagnostic_warnings": "never",
    }


def _estimator(case: str, iterations: int):
    from darkofit import DarkoClassifier, DarkoRegressor

    params = _base_params(iterations)
    if case == "scalar_rmse_catboost":
        return DarkoRegressor(
            **params, loss="RMSE", tree_mode="catboost", depth=6
        )
    if case == "binary_catboost":
        return DarkoClassifier(**params, tree_mode="catboost", depth=6)
    if case == "multiclass_catboost_per_class":
        return DarkoClassifier(
            **params,
            tree_mode="catboost",
            depth=6,
            multiclass_tree_strategy="per_class",
        )
    if case == "multiclass_lightgbm_shared":
        return DarkoClassifier(
            **params,
            tree_mode="lightgbm",
            num_leaves=64,
            multiclass_tree_strategy="shared_vector",
        )
    if case == "gaussian_lightgbm":
        return DarkoRegressor(
            **params, loss="Gaussian", tree_mode="lightgbm", num_leaves=64
        )
    if case == "student_t_lightgbm":
        return DarkoRegressor(
            **params, loss="StudentT", tree_mode="lightgbm", num_leaves=64
        )
    raise ValueError(f"unknown vector profile case: {case}")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("peak RSS is unavailable")
    return value


def run_worker(case: str) -> dict[str, Any]:
    X, targets = _data(ROWS)
    warm = _estimator(case, 3)
    warm.fit(X[:5000], targets[case][:5000])

    model = _estimator(case, ITERATIONS)
    started = time.perf_counter_ns()
    model.fit(X, targets[case])
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    prediction = np.asarray(model.predict(X[:2048]))
    if prediction.shape != (2048,) or not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"{case} produced invalid predictions")
    timing = {
        name: float((model.timing_ or {}).get(name, 0.0))
        for name in PHASES
    }
    if any(not math.isfinite(value) or value < 0.0 for value in timing.values()):
        raise RuntimeError(f"{case} produced invalid phase timings")
    phase_total = float(sum(timing.values()))
    if phase_total <= 0.0:
        raise RuntimeError(f"{case} produced no phase attribution")
    core = model.model_
    fitted_trees = len(core.trees_)
    expected_trees = (
        ITERATIONS * 4
        if case == "multiclass_catboost_per_class"
        else ITERATIONS
    )
    if fitted_trees != expected_trees:
        raise RuntimeError(
            f"{case} retained {fitted_trees} trees, expected {expected_trees}"
        )
    auto = dict(getattr(core, "auto_params_", {}) or {})
    return {
        "case": case,
        "rows": ROWS,
        "features": FEATURES,
        "iterations": ITERATIONS,
        "fit_seconds": float(fit_seconds),
        "seconds_per_round": float(fit_seconds / ITERATIONS),
        "phase_seconds": timing,
        "phase_attributed_seconds": phase_total,
        "phase_shares_of_attributed": {
            name: float(value / phase_total) for name, value in timing.items()
        },
        "unattributed_seconds": max(0.0, float(fit_seconds - phase_total)),
        "fitted_tree_count": int(fitted_trees),
        "selected_tree_mode": str(core.tree_mode_),
        "resolved_thread_count": int(core.n_threads_),
        "multiclass_strategy": auto.get("multiclass_tree_strategy"),
        "loss": str(getattr(core, "loss_name", "Multiclass")),
        "prediction_sha256": _array_sha256(prediction),
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in creator.THREAD_LIMIT_ENV_KEYS:
        environment[key] = str(THREADS)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "NUMBA_DISABLE_JIT": "0",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(ROOT),
        }
    )
    return environment


def _run_worker(case: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-case",
            case,
        ],
        cwd=ROOT,
        env=_worker_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if completed.returncode or len(lines) != 1:
        raise RuntimeError(
            f"profile worker {case} failed with {completed.returncode}"
            f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result = json.loads(lines[0][len(WORKER_RESULT_PREFIX) :])
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


def analyze(results: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for case in CASES:
        rows = [row for row in results if row["case"] == case]
        if len(rows) != REPETITIONS:
            raise RuntimeError(f"{case} does not have {REPETITIONS} profiles")
        hashes = {row["prediction_sha256"] for row in rows}
        if len(hashes) != 1:
            raise RuntimeError(f"{case} predictions changed across workers")
        fit_values = np.asarray(
            [row["fit_seconds"] for row in rows], dtype=np.float64
        )
        phase_medians = {
            phase: float(
                np.median([row["phase_seconds"][phase] for row in rows])
            )
            for phase in PHASES
        }
        attributed = sum(phase_medians.values())
        phase_shares = {
            phase: value / attributed
            for phase, value in phase_medians.items()
        }
        summaries[case] = {
            "median_fit_seconds": float(np.median(fit_values)),
            "fit_iqr_over_median": float(
                np.subtract(*np.percentile(fit_values, [75.0, 25.0]))
                / np.median(fit_values)
            ),
            "median_seconds_per_round": float(
                np.median([row["seconds_per_round"] for row in rows])
            ),
            "median_phase_seconds": phase_medians,
            "phase_shares_of_attributed": phase_shares,
            "largest_phase": max(phase_shares, key=phase_shares.get),
            "prediction_sha256": hashes.pop(),
            "fitted_tree_count": rows[0]["fitted_tree_count"],
            "selected_tree_mode": rows[0]["selected_tree_mode"],
            "loss": rows[0]["loss"],
            "peak_rss_bytes": [row["peak_rss_bytes"] for row in rows],
        }
    noncontrol = [case for case in CASES if case != "scalar_rmse_catboost"]
    opportunity = max(
        (
            (
                summaries[case]["phase_shares_of_attributed"][phase],
                case,
                phase,
            )
            for case in noncontrol
            for phase in PHASES
        ),
        key=lambda row: row[0],
    )
    return {
        "summaries": summaries,
        "selected_opportunity": {
            "case": opportunity[1],
            "phase": opportunity[2],
            "share_of_attributed": opportunity[0],
        },
        "recommendation": (
            f"profile_{opportunity[2]}_inside_{opportunity[1]}_before_e1"
        ),
    }


def _source_state() -> dict[str, Any]:
    state = creator.git_state(ROOT)
    if not state["clean"]:
        raise RuntimeError("vector profile requires a clean source")
    return state


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    source = _source_state()
    results = []
    orders = (CASES, tuple(reversed(CASES)), CASES)
    for block, order in enumerate(orders):
        for position, case in enumerate(order):
            if creator.git_state(ROOT) != source:
                raise RuntimeError("source changed during vector profile")
            print(
                f"running block {block + 1}/{REPETITIONS} "
                f"position {position + 1}: {case}",
                flush=True,
            )
            result = _run_worker(case)
            result["block"] = int(block)
            result["position"] = int(position)
            results.append(result)
    if creator.git_state(ROOT) != source:
        raise RuntimeError("source changed during vector profile")
    analysis = analyze(results)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "classification_distributional_fit_profile",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "rows": ROWS,
            "features": FEATURES,
            "iterations": ITERATIONS,
            "repetitions": REPETITIONS,
            "threads": THREADS,
            "cases": list(CASES),
            "default_change_authorized": False,
            "lockbox_data_used": False,
        },
        "source": source,
        "environment": {
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output = creator._absolute_lexical_path(args.output)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_case:
        result = run_worker(args.worker_case)
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
