#!/usr/bin/env python3
"""Run the fused variable-Hessian oblivious-tree performance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
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
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_vector_fit_profile as vector_profile  # noqa: E402


REFERENCE = "reference"
CANDIDATE = "candidate"
CONFIGS = (REFERENCE, CANDIDATE)
CASES = ("binary_logloss", "weighted_rmse")
BLOCK_ORDERS = (
    (REFERENCE, CANDIDATE),
    (CANDIDATE, REFERENCE),
    (REFERENCE, CANDIDATE),
)
ITERATIONS = 300
THREADS = 18
MAX_LANE_REGRESSION_RATIO = 1.02
MAX_GEOMEAN_FIT_RATIO = 0.95
MAX_GEOMEAN_TREE_RATIO = 0.90
MAX_PEAK_RSS_RATIO = 1.05
WORKER_RESULT_PREFIX = "FUSED_VARIABLE_HESSIAN_RESULT="
PROTOCOL = ROOT / "benchmarks" / "fused_variable_hessian_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "fused_variable_hessian.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: Any) -> str:
    value = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(value.tobytes()).hexdigest()


def _canonical_model_payload_sha256(path: Path) -> str:
    """Hash serialized model state while excluding runtime-only telemetry."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            value = archive[key]
            if key == "header":
                header = json.loads(str(value.item()))
                header["timing"] = None
                value = np.asarray(
                    json.dumps(
                        header,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
            value = np.ascontiguousarray(value)
            digest.update(key.encode("utf-8"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(
                np.asarray(value.shape, dtype="<i8").tobytes()
            )
            digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("peak RSS is unavailable")
    return value


def _data():
    X, targets = vector_profile._data(vector_profile.ROWS)
    weights = np.linspace(0.5, 1.5, len(X), dtype=np.float64)
    return X, {
        "binary_logloss": targets["binary_catboost"],
        "weighted_rmse": targets["scalar_rmse_catboost"],
    }, weights


def _estimator(case: str, iterations: int):
    from darkofit import DarkoClassifier, DarkoRegressor

    params = vector_profile._base_params(iterations)
    params.update(
        {
            "tree_mode": "catboost",
            "depth": 6,
            "verbose_timing": True,
        }
    )
    if case == "binary_logloss":
        return DarkoClassifier(**params)
    if case == "weighted_rmse":
        return DarkoRegressor(**params, loss="RMSE")
    raise ValueError(f"unknown case: {case}")


def _fit(
    case: str,
    config: str,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
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
    try:
        model = _estimator(case, iterations)
        started = time.perf_counter_ns()
        model.fit(X, y, sample_weight=sample_weight)
        fit_seconds = (time.perf_counter_ns() - started) / 1e9
    finally:
        booster_module.build_oblivious_tree = original
    return model, float(fit_seconds), int(counter[0])


def run_worker(case: str, config: str) -> dict[str, Any]:
    X, targets, weights = _data()
    sample_weight = weights if case == "weighted_rmse" else None
    _fit(
        case,
        config,
        X[:5000],
        targets[case][:5000],
        None if sample_weight is None else sample_weight[:5000],
        3,
    )
    model, fit_seconds, engagement = _fit(
        case,
        config,
        X,
        targets[case],
        sample_weight,
        ITERATIONS,
    )
    prediction = np.asarray(model.predict(X[:4096]), dtype=np.float64)
    if prediction.shape != (4096,) or not np.all(np.isfinite(prediction)):
        raise RuntimeError("fused variable-Hessian worker produced bad output")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.npz"
        model.save_model(path)
        archive_file_sha256 = _sha256(path)
        model_payload_sha256 = _canonical_model_payload_sha256(path)
    timing = dict(model.timing_ or {})
    tree_seconds = float(timing.get("tree_build", 0.0))
    if (
        not math.isfinite(fit_seconds)
        or fit_seconds <= 0.0
        or not math.isfinite(tree_seconds)
        or tree_seconds <= 0.0
    ):
        raise RuntimeError("fused variable-Hessian timing is invalid")
    core = model.model_
    result = {
        "case": case,
        "config": config,
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "iterations": ITERATIONS,
        "fit_seconds": fit_seconds,
        "tree_build_seconds": tree_seconds,
        "seconds_per_round": float(fit_seconds / ITERATIONS),
        "tree_seconds_per_round": float(tree_seconds / ITERATIONS),
        "prediction_sha256": _array_sha256(prediction),
        "model_payload_sha256": model_payload_sha256,
        "archive_file_sha256_diagnostic": archive_file_sha256,
        "engagement_count": engagement,
        "fitted_tree_count": int(len(core.trees_)),
        "selected_tree_mode": str(core.tree_mode_),
        "resolved_thread_count": int(core.n_threads_),
        "resolved_learning_rate": float(model.learning_rate_),
        "peak_rss_bytes": _peak_rss_bytes(),
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }
    result["behavior_fingerprint_sha256"] = creator.sha256_bytes(
        json.dumps(
            {
                "case": case,
                "prediction_sha256": result["prediction_sha256"],
                "model_payload_sha256": model_payload_sha256,
                "fitted_tree_count": result["fitted_tree_count"],
                "selected_tree_mode": result["selected_tree_mode"],
                "resolved_learning_rate": result["resolved_learning_rate"],
            },
            sort_keys=True,
            allow_nan=False,
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


def _run_worker_process(case: str, config: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-case",
            case,
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
            f"fused worker {case}/{config} failed with {completed.returncode}"
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


def _geomean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(values))))


def analyze(results: list[dict[str, Any]]) -> dict[str, Any]:
    cases = {}
    for case in CASES:
        reference = [
            row for row in results
            if row["case"] == case and row["config"] == REFERENCE
        ]
        candidate = [
            row for row in results
            if row["case"] == case and row["config"] == CANDIDATE
        ]
        if len(reference) != 3 or len(candidate) != 3:
            raise RuntimeError("fused gate requires three results per arm")
        exact = {
            "prediction_hashes_match": (
                {row["prediction_sha256"] for row in reference}
                == {row["prediction_sha256"] for row in candidate}
                and len({row["prediction_sha256"] for row in results
                         if row["case"] == case}) == 1
            ),
            "model_payload_hashes_match": (
                {row["model_payload_sha256"] for row in reference}
                == {row["model_payload_sha256"] for row in candidate}
                and len({row["model_payload_sha256"] for row in results
                         if row["case"] == case}) == 1
            ),
            "behavior_fingerprints_match": (
                len(
                    {
                        row["behavior_fingerprint_sha256"]
                        for row in results
                        if row["case"] == case
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
        }
        summaries = {}
        for metric in (
            "fit_seconds",
            "tree_build_seconds",
            "peak_rss_bytes",
        ):
            summaries[metric] = campaign.paired_ratio_summary(
                [float(row[metric]) for row in candidate],
                [float(row[metric]) for row in reference],
            )
        cases[case] = {
            "exactness": exact,
            "exact": all(exact.values()),
            "paired_ratios": summaries,
        }

    fit_ratios = [
        cases[case]["paired_ratios"]["fit_seconds"]["median_ratio"]
        for case in CASES
    ]
    tree_ratios = [
        cases[case]["paired_ratios"]["tree_build_seconds"]["median_ratio"]
        for case in CASES
    ]
    gates = {
        "all_exact": all(cases[case]["exact"] for case in CASES),
        "all_ratios_stable": all(
            summary["stable"]
            for case in CASES
            for summary in cases[case]["paired_ratios"].values()
        ),
        "no_fit_regression_over_2pct": max(fit_ratios)
        <= MAX_LANE_REGRESSION_RATIO,
        "no_tree_regression_over_2pct": max(tree_ratios)
        <= MAX_LANE_REGRESSION_RATIO,
        "fit_geomean_at_most_0_95": _geomean(fit_ratios)
        <= MAX_GEOMEAN_FIT_RATIO,
        "tree_geomean_at_most_0_90": _geomean(tree_ratios)
        <= MAX_GEOMEAN_TREE_RATIO,
        "rss_at_most_1_05": all(
            cases[case]["paired_ratios"]["peak_rss_bytes"]["median_ratio"]
            <= MAX_PEAK_RSS_RATIO
            for case in CASES
        ),
    }
    passed = all(gates.values())
    return {
        "cases": cases,
        "fit_geomean_ratio": _geomean(fit_ratios),
        "tree_build_geomean_ratio": _geomean(tree_ratios),
        "gates": gates,
        "passes_all_gates": passed,
        "recommendation": (
            "retain_fused_variable_hessian_lane"
            if passed
            else "restore_reference_variable_hessian_dispatch"
        ),
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    source = creator.git_state(ROOT)
    if not source["clean"]:
        raise RuntimeError("fused gate requires a clean source tree")
    results = []
    for block, order in enumerate(BLOCK_ORDERS):
        for case in CASES:
            for position, config in enumerate(order):
                if creator.git_state(ROOT) != source:
                    raise RuntimeError("source changed during fused gate")
                print(
                    f"block {block + 1}/{len(BLOCK_ORDERS)} "
                    f"{case} position {position + 1}: {config}",
                    flush=True,
                )
                result = _run_worker_process(case, config)
                result["block"] = int(block)
                result["position"] = int(position)
                results.append(result)
    if creator.git_state(ROOT) != source:
        raise RuntimeError("source changed during fused gate")
    analysis = analyze(results)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "fused_variable_hessian_oblivious_tree_gate",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "cases": list(CASES),
            "iterations": ITERATIONS,
            "rows": vector_profile.ROWS,
            "features": vector_profile.FEATURES,
            "threads": THREADS,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "max_lane_regression_ratio": MAX_LANE_REGRESSION_RATIO,
            "max_geomean_fit_ratio": MAX_GEOMEAN_FIT_RATIO,
            "max_geomean_tree_ratio": MAX_GEOMEAN_TREE_RATIO,
            "max_peak_rss_ratio": MAX_PEAK_RSS_RATIO,
            "paired_ratio_max_iqr_over_median": (
                campaign.MAX_PAIRED_RATIO_IQR_OVER_MEDIAN
            ),
            "default_promotion_authorized": False,
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-config", choices=CONFIGS, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    args.output = creator._absolute_lexical_path(args.output)
    if bool(args.worker_case) != bool(args.worker_config):
        parser.error("--worker-case and --worker-config must be used together")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker_case:
        result = run_worker(args.worker_case, args.worker_config)
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
