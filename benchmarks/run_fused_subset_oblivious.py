#!/usr/bin/env python3
"""Run the frozen fused subset oblivious-tree performance gate."""

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
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_campaign_harness as campaign  # noqa: E402
from benchmarks import basketball_harness as basketball  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_fused_variable_hessian as prior_gate  # noqa: E402


REFERENCE = "reference"
CANDIDATE = "candidate"
CONFIGS = (REFERENCE, CANDIDATE)
HESSIAN_CASES = ("unit_rmse", "weighted_rmse")
SAMPLING_LANES = {
    "full": {"subsample": 1.0, "colsample": 1.0},
    "features": {"subsample": 1.0, "colsample": 2.0 / 3.0},
    "rows": {"subsample": 0.8, "colsample": 1.0},
    "both": {"subsample": 0.8, "colsample": 2.0 / 3.0},
}
BLOCK_ORDERS = (
    (REFERENCE, CANDIDATE),
    (CANDIDATE, REFERENCE),
    (REFERENCE, CANDIDATE),
)
ITERATIONS = 600
THREADS = 18
MAX_IQR_OVER_MEDIAN = 0.15
MAX_LANE_REGRESSION_RATIO = 1.02
MAX_GEOMEAN_FIT_RATIO = 0.95
MAX_GEOMEAN_TREE_RATIO = 0.90
MAX_PEAK_RSS_RATIO = 1.05
WORKER_RESULT_PREFIX = "FUSED_SUBSET_RESULT="
PROTOCOL = ROOT / "benchmarks" / "fused_subset_oblivious_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "fused_subset_oblivious.json"
DATA_CACHE = (
    creator.DEFAULT_CACHE / "basketball_reference_toy_data.csv"
)
SUPPORT_PATHS = (
    "darkofit/tree.py",
    "darkofit/booster.py",
    "benchmarks/basketball_campaign_harness.py",
    "benchmarks/basketball_harness.py",
    "benchmarks/run_basketball_creator_benchmark.py",
    "benchmarks/run_fused_variable_hessian.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dataset = basketball.load_basketball_dataset(DATA_CACHE)
    X = np.ascontiguousarray(dataset.X.to_numpy(dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(dataset.y, dtype=np.float64))
    weights = np.linspace(0.5, 1.5, len(y), dtype=np.float64)
    metadata = {
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "X_sha256": _array_sha256(X),
        "y_sha256": _array_sha256(y),
        "weights_sha256": _array_sha256(weights),
        "raw_data_sha256": _sha256(DATA_CACHE),
        "cache_path": str(DATA_CACHE),
    }
    if X.shape != (5241, 15) or y.shape != (5241,):
        raise RuntimeError("basketball creator matrix shape changed")
    return X, y, weights, metadata


def _estimator(lane: str, iterations: int):
    from darkofit import DarkoRegressor

    if lane not in SAMPLING_LANES:
        raise ValueError(f"unknown sampling lane: {lane}")
    return DarkoRegressor(
        iterations=iterations,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        max_bins=128,
        thread_count=THREADS,
        random_state=4,
        tree_mode="catboost",
        ordered_boosting=False,
        verbose_timing=True,
        diagnostic_warnings="never",
        **SAMPLING_LANES[lane],
    )


def _fit(
    hessian_case: str,
    lane: str,
    config: str,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    iterations: int,
):
    import darkofit.booster as booster_module

    original = booster_module.build_oblivious_tree
    counter = np.zeros(1, dtype=np.int64)
    booster_module.build_oblivious_tree = partial(
        original,
        fused_oblivious_kernel=config == CANDIDATE,
        fused_oblivious_counter=counter,
    )
    sample_weight = weights if hessian_case == "weighted_rmse" else None
    try:
        model = _estimator(lane, iterations)
        started = time.perf_counter_ns()
        model.fit(X, y, sample_weight=sample_weight)
        fit_seconds = (time.perf_counter_ns() - started) / 1e9
    finally:
        booster_module.build_oblivious_tree = original
    return model, float(fit_seconds), int(counter[0])


def run_worker(
    hessian_case: str, lane: str, config: str
) -> dict[str, Any]:
    if hessian_case not in HESSIAN_CASES:
        raise ValueError(f"unknown Hessian case: {hessian_case}")
    X, y, weights, data_metadata = _data()
    _fit(
        hessian_case,
        lane,
        config,
        X[:200],
        y[:200],
        weights[:200],
        3,
    )
    model, fit_seconds, engagement = _fit(
        hessian_case, lane, config, X, y, weights, ITERATIONS
    )
    prediction = np.asarray(model.predict(X), dtype=np.float64)
    if prediction.shape != y.shape or not np.all(np.isfinite(prediction)):
        raise RuntimeError("fused subset worker produced invalid predictions")
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.npz"
        model.save_model(model_path)
        model_payload_sha256 = (
            prior_gate._canonical_model_payload_sha256(model_path)
        )
        archive_sha256_diagnostic = _sha256(model_path)
    timing = dict(model.timing_ or {})
    tree_seconds = float(timing.get("tree_build", 0.0))
    if (
        not math.isfinite(fit_seconds)
        or fit_seconds <= 0.0
        or not math.isfinite(tree_seconds)
        or tree_seconds <= 0.0
    ):
        raise RuntimeError("fused subset timing is invalid")
    core = model.model_
    result = {
        "hessian_case": hessian_case,
        "sampling_lane": lane,
        "config": config,
        "sampling": dict(SAMPLING_LANES[lane]),
        "iterations": ITERATIONS,
        "fit_seconds": fit_seconds,
        "tree_build_seconds": tree_seconds,
        "prediction_sha256": _array_sha256(prediction),
        "model_payload_sha256": model_payload_sha256,
        "archive_sha256_diagnostic": archive_sha256_diagnostic,
        "engagement_count": engagement,
        "fitted_tree_count": int(len(core.trees_)),
        "selected_tree_mode": str(core.tree_mode_),
        "resolved_thread_count": int(core.n_threads_),
        "resolved_learning_rate": float(model.learning_rate_),
        "stop_reason": str(core.stop_reason_),
        "peak_rss_bytes": prior_gate._peak_rss_bytes(),
        "data": data_metadata,
        "support_sha256": {
            path: _sha256(ROOT / path) for path in SUPPORT_PATHS
        },
        "executable": sys.executable,
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }
    behavior = {
        key: result[key]
        for key in (
            "hessian_case",
            "sampling_lane",
            "sampling",
            "iterations",
            "prediction_sha256",
            "model_payload_sha256",
            "fitted_tree_count",
            "selected_tree_mode",
            "resolved_learning_rate",
            "stop_reason",
        )
    }
    result["behavior_fingerprint_sha256"] = creator.sha256_bytes(
        json.dumps(
            behavior, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    return result


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


def _run_worker_process(
    hessian_case: str, lane: str, config: str
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-hessian-case",
            hessian_case,
            "--worker-lane",
            lane,
            "--worker-config",
            config,
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
            f"worker {hessian_case}/{lane}/{config} failed with "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result = json.loads(lines[0][len(WORKER_RESULT_PREFIX) :])
    expected = (hessian_case, lane, config)
    actual = (
        result.get("hessian_case"),
        result.get("sampling_lane"),
        result.get("config"),
    )
    if actual != expected:
        raise RuntimeError(
            f"worker coordinate mismatch: expected {expected}, got {actual}"
        )
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


def _geomean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(array))))


def _paired_summary(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    return campaign.paired_ratio_summary(
        [float(row[metric]) for row in candidate],
        [float(row[metric]) for row in reference],
        max_iqr_over_median=MAX_IQR_OVER_MEDIAN,
    )


def analyze(results: list[dict[str, Any]]) -> dict[str, Any]:
    expected_rows = (
        len(HESSIAN_CASES)
        * len(SAMPLING_LANES)
        * len(CONFIGS)
        * len(BLOCK_ORDERS)
    )
    if len(results) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} worker rows, got {len(results)}"
        )
    cells: dict[str, Any] = {}
    for hessian_case in HESSIAN_CASES:
        for lane in SAMPLING_LANES:
            cell_name = f"{hessian_case}/{lane}"
            cell_rows = [
                row
                for row in results
                if row["hessian_case"] == hessian_case
                and row["sampling_lane"] == lane
            ]
            reference = sorted(
                (
                    row
                    for row in cell_rows
                    if row["config"] == REFERENCE
                ),
                key=lambda row: row["block"],
            )
            candidate = sorted(
                (
                    row
                    for row in cell_rows
                    if row["config"] == CANDIDATE
                ),
                key=lambda row: row["block"],
            )
            if len(reference) != 3 or len(candidate) != 3:
                raise RuntimeError(
                    f"{cell_name} does not contain three paired blocks"
                )
            if [row["block"] for row in reference] != [0, 1, 2]:
                raise RuntimeError(f"{cell_name} reference blocks are invalid")
            if [row["block"] for row in candidate] != [0, 1, 2]:
                raise RuntimeError(f"{cell_name} candidate blocks are invalid")
            exactness = {
                "prediction_hashes_match": (
                    len(
                        {
                            row["prediction_sha256"]
                            for row in cell_rows
                        }
                    )
                    == 1
                ),
                "model_payload_hashes_match": (
                    len(
                        {
                            row["model_payload_sha256"]
                            for row in cell_rows
                        }
                    )
                    == 1
                ),
                "behavior_fingerprints_match": (
                    len(
                        {
                            row["behavior_fingerprint_sha256"]
                            for row in cell_rows
                        }
                    )
                    == 1
                ),
                "reference_does_not_engage": all(
                    row["engagement_count"] == 0 for row in reference
                ),
                "candidate_engages": all(
                    row["engagement_count"] > 0 for row in candidate
                ),
                "metadata_matches_protocol": all(
                    row["resolved_thread_count"] == THREADS
                    and row["selected_tree_mode"] == "catboost"
                    and row["iterations"] == ITERATIONS
                    and row["sampling"] == SAMPLING_LANES[lane]
                    for row in cell_rows
                ),
                "data_and_support_match": (
                    len(
                        {
                            json.dumps(row["data"], sort_keys=True)
                            for row in cell_rows
                        }
                    )
                    == 1
                    and len(
                        {
                            json.dumps(
                                row["support_sha256"], sort_keys=True
                            )
                            for row in cell_rows
                        }
                    )
                    == 1
                ),
            }
            ratios = {
                metric: _paired_summary(candidate, reference, metric)
                for metric in (
                    "fit_seconds",
                    "tree_build_seconds",
                    "peak_rss_bytes",
                )
            }
            cells[cell_name] = {
                "hessian_case": hessian_case,
                "sampling_lane": lane,
                "exactness": exactness,
                "exact": all(exactness.values()),
                "paired_ratios": ratios,
            }

    subset_cells = [
        value
        for value in cells.values()
        if value["sampling_lane"] != "full"
    ]
    full_cells = [
        value
        for value in cells.values()
        if value["sampling_lane"] == "full"
    ]
    fit_ratios = [
        value["paired_ratios"]["fit_seconds"]["median_ratio"]
        for value in cells.values()
    ]
    tree_ratios = [
        value["paired_ratios"]["tree_build_seconds"]["median_ratio"]
        for value in cells.values()
    ]
    subset_fit = [
        value["paired_ratios"]["fit_seconds"]["median_ratio"]
        for value in subset_cells
    ]
    subset_tree = [
        value["paired_ratios"]["tree_build_seconds"]["median_ratio"]
        for value in subset_cells
    ]
    full_fit = [
        value["paired_ratios"]["fit_seconds"]["median_ratio"]
        for value in full_cells
    ]
    full_tree = [
        value["paired_ratios"]["tree_build_seconds"]["median_ratio"]
        for value in full_cells
    ]
    gates = {
        "all_exact": all(value["exact"] for value in cells.values()),
        "all_ratios_stable": all(
            summary["stable"]
            for value in cells.values()
            for summary in value["paired_ratios"].values()
        ),
        "no_fit_regression_over_2pct": max(fit_ratios)
        <= MAX_LANE_REGRESSION_RATIO,
        "no_tree_regression_over_2pct": max(tree_ratios)
        <= MAX_LANE_REGRESSION_RATIO,
        "subset_fit_geomean_at_most_0_95": _geomean(subset_fit)
        <= MAX_GEOMEAN_FIT_RATIO,
        "subset_tree_geomean_at_most_0_90": _geomean(subset_tree)
        <= MAX_GEOMEAN_TREE_RATIO,
        "full_fit_geomean_at_most_0_95": _geomean(full_fit)
        <= MAX_GEOMEAN_FIT_RATIO,
        "full_tree_geomean_at_most_0_90": _geomean(full_tree)
        <= MAX_GEOMEAN_TREE_RATIO,
        "rss_at_most_1_05": all(
            value["paired_ratios"]["peak_rss_bytes"]["median_ratio"]
            <= MAX_PEAK_RSS_RATIO
            for value in cells.values()
        ),
    }
    passed = all(gates.values())
    return {
        "cells": cells,
        "subset_fit_geomean_ratio": _geomean(subset_fit),
        "subset_tree_build_geomean_ratio": _geomean(subset_tree),
        "full_fit_geomean_ratio": _geomean(full_fit),
        "full_tree_build_geomean_ratio": _geomean(full_tree),
        "gates": gates,
        "passes_all_gates": passed,
        "recommendation": (
            "retain_fused_subset_lanes"
            if passed
            else "restore_full_lane_only_dispatch"
        ),
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    source = creator.git_state(ROOT)
    if not source["clean"]:
        raise RuntimeError("fused subset gate requires a clean source tree")
    support_sha256 = {
        path: _sha256(ROOT / path) for path in SUPPORT_PATHS
    }
    results = []
    for block, order in enumerate(BLOCK_ORDERS):
        for hessian_case in HESSIAN_CASES:
            for lane in SAMPLING_LANES:
                for position, config in enumerate(order):
                    if creator.git_state(ROOT) != source:
                        raise RuntimeError(
                            "source changed during fused subset gate"
                        )
                    print(
                        f"block {block + 1}/{len(BLOCK_ORDERS)} "
                        f"{hessian_case}/{lane} position {position + 1}: "
                        f"{config}",
                        flush=True,
                    )
                    result = _run_worker_process(
                        hessian_case, lane, config
                    )
                    if result["support_sha256"] != support_sha256:
                        raise RuntimeError("worker support lineage mismatch")
                    result["block"] = int(block)
                    result["position"] = int(position)
                    results.append(result)
    if creator.git_state(ROOT) != source:
        raise RuntimeError("source changed during fused subset gate")
    analysis = analyze(results)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "fused_subset_oblivious_tree_gate",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "hessian_cases": list(HESSIAN_CASES),
            "sampling_lanes": SAMPLING_LANES,
            "iterations": ITERATIONS,
            "threads": THREADS,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "max_iqr_over_median": MAX_IQR_OVER_MEDIAN,
            "max_lane_regression_ratio": MAX_LANE_REGRESSION_RATIO,
            "max_geomean_fit_ratio": MAX_GEOMEAN_FIT_RATIO,
            "max_geomean_tree_ratio": MAX_GEOMEAN_TREE_RATIO,
            "max_peak_rss_ratio": MAX_PEAK_RSS_RATIO,
            "default_promotion_authorized": False,
            "lockbox_data_used": False,
        },
        "source": source,
        "support_sha256": support_sha256,
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--worker-hessian-case",
        choices=HESSIAN_CASES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-lane",
        choices=tuple(SAMPLING_LANES),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-config", choices=CONFIGS, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    args.output = creator._absolute_lexical_path(args.output)
    worker_values = (
        args.worker_hessian_case,
        args.worker_lane,
        args.worker_config,
    )
    if any(worker_values) and not all(worker_values):
        parser.error("all worker arguments must be used together")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker_hessian_case:
        result = run_worker(
            args.worker_hessian_case, args.worker_lane, args.worker_config
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
