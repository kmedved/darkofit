#!/usr/bin/env python3
"""Run the seconds-integrated matched prediction-throughput protocol."""

from __future__ import annotations

import argparse
import gc
import json
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

from benchmarks import run_predict_throughput as baseline  # noqa: E402


INTEGRATED_REPEATS = {
    8_192: 256,
    65_536: 32,
    524_288: 4,
    2_000_000: 2,
}
MIN_INTERVAL_SECONDS = 0.75
MAX_PEAK_RSS_RATIO = 1.05
STRETCH_PUBLIC_RATIO = 1.0
WORKER_RESULT_PREFIX = "PREDICT_THROUGHPUT_INTEGRATED_RESULT="
PROTOCOL = ROOT / "benchmarks" / "predict_throughput_integrated_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "predict_throughput_integrated.json"


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("peak RSS is unavailable")
    return value


def _integrated_case(
    arm: str, model: Any, template, rows: int
) -> dict[str, Any]:
    X = baseline._repeat_frame(template, rows)
    input_sha256 = baseline._frame_sha256(X)
    baseline._clear_forest_cache(arm, model)
    warm = baseline._validate_output(model.predict(X), rows)
    warm_sha256 = baseline._array_sha256(warm)
    repeats = INTEGRATED_REPEATS[rows]
    output = None
    gc.disable()
    started = time.perf_counter_ns()
    try:
        for _ in range(repeats):
            output = model.predict(X)
    finally:
        elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
        gc.enable()
    output = baseline._validate_output(output, rows)
    exact = bool(np.array_equal(warm, output))
    if not exact:
        raise RuntimeError("integrated public prediction changed output")
    result = {
        "rows": int(rows),
        "columns": int(X.shape[1]),
        "input_sha256": input_sha256,
        "input_dtypes": [str(value) for value in X.dtypes],
        "prediction_sha256": warm_sha256,
        "warm_equals_integrated": exact,
        "integrated_public": {
            "repetitions": int(repeats),
            "elapsed_seconds": float(elapsed_seconds),
            "seconds_per_call": float(elapsed_seconds / repeats),
            "minimum_interval_seconds": MIN_INTERVAL_SECONDS,
            "minimum_interval_passed": (
                elapsed_seconds >= MIN_INTERVAL_SECONDS
            ),
        },
    }
    del X, warm, output
    gc.collect()
    return result


def _behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "models": {
            name: value["fit_metadata"]
            for name, value in result["models"].items()
        },
        "cases": {
            dataset: {
                rows: {
                    "input_sha256": case["input_sha256"],
                    "prediction_sha256": case["prediction_sha256"],
                    "warm_equals_integrated": case[
                        "warm_equals_integrated"
                    ],
                }
                for rows, case in cases.items()
            }
            for dataset, cases in result["cases"].items()
        },
    }


def run_worker(
    arm: str, cache_path: Path, chimera_repo: Path
) -> dict[str, Any]:
    models, templates, metadata = baseline._fit_models(
        arm,
        cache_path,
        chimera_repo,
    )
    cases = {}
    for dataset in baseline.DATASETS:
        cases[dataset] = {}
        for rows in baseline.BATCH_SIZES:
            print(
                f"worker {arm}: {dataset} rows={rows}",
                flush=True,
            )
            cases[dataset][str(rows)] = _integrated_case(
                arm, models[dataset], templates[dataset], rows
            )
    result = {
        "arm": arm,
        "models": metadata,
        "cases": cases,
        "peak_rss_bytes": _peak_rss_bytes(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in baseline.creator.THREAD_ENV_KEYS
        },
    }
    result["behavior_fingerprint_sha256"] = (
        baseline.basketball.behavior_fingerprint(_behavior_payload(result))
    )
    return result


def _run_worker_process(
    args: argparse.Namespace, arm: str
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-arm",
            arm,
            "--threads",
            str(args.threads),
            "--data-cache",
            str(args.data_cache),
            "--chimeraboost-repo",
            str(args.chimeraboost_repo),
        ],
        cwd=ROOT,
        env=baseline._worker_environment(args, arm),
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
            f"integrated worker {arm} failed with {completed.returncode}"
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


def analyze(
    canonical: dict[str, dict[str, Any]],
    block_results: list[dict[str, Any]],
    rss: dict[str, list[int]],
) -> dict[str, Any]:
    fingerprints = {
        arm: {
            result["behavior_fingerprint_sha256"]
            for result in block_results
            if result["arm"] == arm
        }
        for arm in baseline.ARMS
    }
    fingerprint_stable = all(
        len(values) == 1 for values in fingerprints.values()
    )
    paired = {}
    gates = {
        "behavior_fingerprints_stable": fingerprint_stable,
        "all_predictions_exact": all(
            case["warm_equals_integrated"]
            for arm in baseline.ARMS
            for cases in canonical[arm]["cases"].values()
            for case in cases.values()
        ),
        "all_intervals_at_least_0_75_seconds": all(
            case["integrated_public"]["minimum_interval_passed"]
            for result in block_results
            for cases in result["cases"].values()
            for case in cases.values()
        ),
    }
    stretch = 0
    for dataset in baseline.DATASETS:
        paired[dataset] = {}
        for rows in baseline.BATCH_SIZES:
            key = str(rows)
            darko = [
                float(
                    result["cases"][dataset][key]["integrated_public"][
                        "seconds_per_call"
                    ]
                )
                for result in block_results
                if result["arm"] == baseline.DARKOFIT
            ]
            chimera = [
                float(
                    result["cases"][dataset][key]["integrated_public"][
                        "seconds_per_call"
                    ]
                )
                for result in block_results
                if result["arm"] == baseline.CHIMERABOOST
            ]
            summary = baseline.campaign.paired_ratio_summary(darko, chimera)
            paired[dataset][key] = summary
            gates[f"{dataset}_{rows}_stable"] = summary["stable"]
            gates[f"{dataset}_{rows}_ratio_at_most_1_30"] = (
                summary["median_ratio"] <= baseline.TARGET_PUBLIC_RATIO
            )
            stretch += int(
                summary["stable"]
                and summary["median_ratio"] <= STRETCH_PUBLIC_RATIO
            )
    paired_rss = baseline.campaign.paired_ratio_summary(
        rss[baseline.DARKOFIT],
        rss[baseline.CHIMERABOOST],
    )
    gates["peak_rss_ratio_at_most_1_05"] = (
        paired_rss["median_ratio"] <= MAX_PEAK_RSS_RATIO
    )
    passed = all(gates.values())
    return {
        "paired_ratios": paired,
        "paired_peak_rss": paired_rss,
        "gates": gates,
        "passes_all_gates": passed,
        "stretch_public_cases_at_or_below_chimera": stretch,
        "stretch_public_case_count": (
            len(baseline.DATASETS) * len(baseline.BATCH_SIZES)
        ),
        "recommendation": (
            "close_p2_matched_prediction_target"
            if passed
            else "p2_target_remains_open"
        ),
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    if args.threads != baseline.EXPECTED_THREADS:
        raise RuntimeError("throughput protocol requires exactly 18 threads")
    sources = baseline._source_states(args)
    canonical = {}
    block_results = []
    rss = {arm: [] for arm in baseline.ARMS}
    for block, order in enumerate(baseline.BLOCK_ORDERS):
        for position, arm in enumerate(order):
            baseline._assert_sources_unchanged(
                sources,
                baseline._source_states(args),
                f"before block {block} {arm}",
            )
            print(
                f"running block {block + 1}/{len(baseline.BLOCK_ORDERS)} "
                f"position {position + 1}: {arm}",
                flush=True,
            )
            result = _run_worker_process(args, arm)
            result["block"] = int(block)
            result["position"] = int(position)
            canonical.setdefault(arm, result)
            block_results.append(result)
            rss[arm].append(int(result["peak_rss_bytes"]))
    baseline._assert_sources_unchanged(
        sources,
        baseline._source_states(args),
        "during integrated throughput campaign",
    )
    analysis = analyze(canonical, block_results, rss)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "seconds_integrated_matched_prediction_throughput",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": baseline._sha256(PROTOCOL),
            "runner_sha256": baseline._sha256(Path(__file__).resolve()),
            "predecessor_artifact_sha256": (
                "430341e101194b8bc3fbb98014b568d4bd460517686acea6188d88958619ad61"
            ),
            "chimeraboost_head": baseline.EXPECTED_CHIMERA_HEAD,
            "threads": baseline.EXPECTED_THREADS,
            "batch_sizes": list(baseline.BATCH_SIZES),
            "integrated_repeats": INTEGRATED_REPEATS,
            "minimum_interval_seconds": MIN_INTERVAL_SECONDS,
            "block_orders": [list(order) for order in baseline.BLOCK_ORDERS],
            "target_public_ratio": baseline.TARGET_PUBLIC_RATIO,
            "stretch_public_ratio": STRETCH_PUBLIC_RATIO,
            "max_peak_rss_ratio": MAX_PEAK_RSS_RATIO,
            "paired_ratio_max_iqr_over_median": (
                baseline.campaign.MAX_PAIRED_RATIO_IQR_OVER_MEDIAN
            ),
            "default_promotion_authorized": False,
            "lockbox_data_used": False,
        },
        "sources": sources,
        "environment": {
            "machine": baseline.creator._machine_details(),
            "dependencies": baseline.creator._dependency_versions(),
        },
        "canonical_results": canonical,
        "block_results": block_results,
        "analysis": analysis,
    }
    baseline.creator._atomic_write_bytes(
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
        "--data-cache", type=Path, default=baseline.basketball.DEFAULT_CACHE
    )
    parser.add_argument(
        "--chimeraboost-repo",
        type=Path,
        default=baseline.DEFAULT_CHIMERA_REPO,
    )
    parser.add_argument(
        "--threads", type=int, default=baseline.EXPECTED_THREADS
    )
    parser.add_argument(
        "--worker-arm", choices=baseline.ARMS, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    args.output = baseline.creator._absolute_lexical_path(args.output)
    args.data_cache = baseline.creator._absolute_lexical_path(args.data_cache)
    args.chimeraboost_repo = baseline.creator._absolute_lexical_path(
        args.chimeraboost_repo
    )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker_arm:
        result = run_worker(
            args.worker_arm,
            args.data_cache,
            args.chimeraboost_repo,
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
