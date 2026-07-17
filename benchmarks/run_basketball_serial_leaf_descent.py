#!/usr/bin/env python3
"""Run the frozen behavior-exact small-row leaf-descent campaign."""

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

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_basketball_fused_oblivious as engine_base  # noqa: E402
import darkofit.tree as tree_module  # noqa: E402


REFERENCE_CONFIG = "forced_parallel"
CANDIDATE_CONFIG = "automatic_serial"
CONFIG_ORDER = (REFERENCE_CONFIG, CANDIDATE_CONFIG)
WORKER_RESULT_PREFIX = "BASKETBALL_SERIAL_DESCENT_RESULT="
MAX_WALL_RATIO = 0.70
MAX_FIT_RATIO = 0.70
MAX_PEAK_RSS_RATIO = 1.05
MAX_KERNEL_RATIO = 0.10
EXPECTED_DEFAULT_MEAN_R2 = engine_base.EXPECTED_DEFAULT_MEAN_R2
EXPECTED_THREADS = 18
EXPECTED_PROTOCOL_SHA256 = (
    "0f835ef4abae31e3c6accac9ed12b5f1d763fbc41030b4525cd5be96518442c4"
)
EXPECTED_DARKOFIT_SUBTREE = "b75e9d9ec67db7140d8434a3b5e3382cd93270a5"
EXPECTED_TESTS_SUBTREE = "fbfd8b7a9ad6cdbbaede2487721eed3fa93ac6e1"
EXPECTED_REPOSITORY_MANIFEST_SHA256 = "35c04a1a9d1db1487a6cd28cc2900123479346be7a69eb430f33180cfd0e9e94"
EXPECTED_SUPPORT_MANIFEST_SHA256 = (
    "fd94987faf0b96af58f2cb6000eb20d8aecf6e968e2aa686a13342dc3c0f971d"
)
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / (
    "basketball_serial_leaf_descent_protocol.md"
)
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / (
    "basketball_serial_leaf_descent.json"
)
MICROBENCHMARK_REPEATS = 500
MICROBENCHMARK_BLOCKS = 7
BOUND_SUPPORT_PATHS = (
    Path("NOTICE"),
    Path("pyproject.toml"),
    Path("benchmarks/basketball_harness.py"),
    Path("benchmarks/run_basketball_creator_benchmark.py"),
    Path("benchmarks/run_basketball_fused_oblivious.py"),
)
PREREQUISITE_TESTS = (
    "tests/test_oblivious_kernel_oracle.py",
    "tests/test_prediction_goldens.py",
    "tests/test_serial_leaf_descent.py",
    "tests/test_basketball_serial_leaf_descent.py",
)

_SERIAL_CALLS = 0
_PARALLEL_CALLS = 0
_PRODUCTION_UPDATE = tree_module._update_leaves_with_split
_SERIAL_KERNEL = tree_module._update_leaves_with_split_serial
_PARALLEL_KERNEL = tree_module._update_leaves_with_split_parallel


def configure_descent(config_name: str) -> None:
    """Install an observed reference or automatic leaf-update route."""
    global _PARALLEL_CALLS, _SERIAL_CALLS

    if config_name not in CONFIG_ORDER:
        raise ValueError(f"unknown descent config: {config_name}")
    _SERIAL_CALLS = 0
    _PARALLEL_CALLS = 0

    def observed_serial(X_binned, leaf, split_feat, split_thr):
        global _SERIAL_CALLS
        _SERIAL_CALLS += 1
        return _SERIAL_KERNEL(X_binned, leaf, split_feat, split_thr)

    def observed_parallel(X_binned, leaf, split_feat, split_thr):
        global _PARALLEL_CALLS
        _PARALLEL_CALLS += 1
        return _PARALLEL_KERNEL(X_binned, leaf, split_feat, split_thr)

    tree_module._update_leaves_with_split_serial = observed_serial
    tree_module._update_leaves_with_split_parallel = observed_parallel
    if config_name == REFERENCE_CONFIG:
        tree_module._update_leaves_with_split = observed_parallel
    else:
        # Exercise the production router itself. Its kernel globals are
        # observed above so the artifact proves which branch actually ran.
        tree_module._update_leaves_with_split = _PRODUCTION_UPDATE


def reset_dispatch_counts() -> None:
    global _PARALLEL_CALLS, _SERIAL_CALLS
    _SERIAL_CALLS = 0
    _PARALLEL_CALLS = 0


def dispatch_counts() -> dict[str, int]:
    return {
        "serial_calls": int(_SERIAL_CALLS),
        "parallel_calls": int(_PARALLEL_CALLS),
    }


def _warmup(dataset: harness.BasketballDataset):
    train, test = next(creator.creator_cv().split(dataset.X, dataset.y))
    X_train = dataset.X.iloc[train]
    y_train = dataset.y.iloc[train]
    started = time.perf_counter_ns()
    model, _ = engine_base._fit_model(X_train, y_train)
    prediction = model.predict(dataset.X.iloc[test])
    harness.validate_prediction(prediction, len(test))
    elapsed = (time.perf_counter_ns() - started) / 1e9
    return model, X_train, float(elapsed)


def _descent_microbenchmark(model, X_train) -> dict[str, Any]:
    import darkofit.tree as tree_module

    core = model.model_
    X_binned = core.prep_.transform(np.asarray(X_train, dtype=np.float64))
    if not core.trees_ or core.trees_[0].depth == 0:
        raise RuntimeError("basketball warmup tree has no split")
    split_feat = int(core.trees_[0].splits_feat[0])
    split_thr = int(core.trees_[0].splits_thr[0])
    initial = np.zeros(X_binned.shape[0], dtype=np.int64)
    serial_result = initial.copy()
    parallel_result = initial.copy()
    tree_module._update_leaves_with_split_serial(
        X_binned, serial_result, split_feat, split_thr
    )
    tree_module._update_leaves_with_split_parallel(
        X_binned, parallel_result, split_feat, split_thr
    )
    exact = np.array_equal(serial_result, parallel_result)
    if not exact:
        raise RuntimeError("serial and parallel leaf descent differ")

    functions = {
        "serial": tree_module._update_leaves_with_split_serial,
        "parallel": tree_module._update_leaves_with_split_parallel,
    }
    timings = {name: [] for name in functions}
    work = initial.copy()
    for block in range(MICROBENCHMARK_BLOCKS):
        order = ("parallel", "serial") if block % 2 == 0 else ("serial", "parallel")
        for name in order:
            function = functions[name]
            started = time.perf_counter_ns()
            for _ in range(MICROBENCHMARK_REPEATS):
                work[:] = initial
                function(X_binned, work, split_feat, split_thr)
            timings[name].append(
                (time.perf_counter_ns() - started)
                / MICROBENCHMARK_REPEATS
                / 1e9
            )
    serial_median = float(statistics.median(timings["serial"]))
    parallel_median = float(statistics.median(timings["parallel"]))
    return {
        "rows": int(X_binned.shape[0]),
        "split_feature": split_feat,
        "split_threshold": split_thr,
        "repeats_per_block": MICROBENCHMARK_REPEATS,
        "blocks": MICROBENCHMARK_BLOCKS,
        "serial_seconds": [float(value) for value in timings["serial"]],
        "parallel_seconds": [float(value) for value in timings["parallel"]],
        "median_serial_seconds": serial_median,
        "median_parallel_seconds": parallel_median,
        "serial_over_parallel": float(serial_median / parallel_median),
        "outputs_exact": exact,
    }


def run_worker(config_name: str, cache_path: Path) -> dict[str, Any]:
    configure_descent(config_name)
    dataset = harness.load_basketball_dataset(cache_path)
    warmup_model, warmup_X, warmup_seconds = _warmup(dataset)
    microbenchmark = _descent_microbenchmark(warmup_model, warmup_X)
    reset_dispatch_counts()

    folds = []
    steady_started = time.perf_counter_ns()
    for fold, (train, test) in enumerate(
        creator.creator_cv().split(dataset.X, dataset.y)
    ):
        fitted = engine_base.fit_and_predict(
            dataset.X.iloc[train],
            dataset.y.iloc[train],
            dataset.X.iloc[test],
        )
        engine_base.validate_fitted_metadata(fitted["fit_metadata"])
        prediction = fitted.pop("prediction")
        folds.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "test_indices": [int(value) for value in test],
                "r2": float(r2_score(dataset.y.iloc[test], prediction)),
                "prediction_sha256": harness.prediction_sha256(prediction),
                "predictions": [float(value) for value in prediction],
                **fitted,
            }
        )
    steady_seconds = (time.perf_counter_ns() - steady_started) / 1e9

    guardrail = dataset.player_guardrail
    fitted = engine_base.fit_and_predict(
        guardrail.X_train,
        guardrail.y_train,
        guardrail.X_holdout,
    )
    engine_base.validate_fitted_metadata(fitted["fit_metadata"])
    prediction = fitted.pop("prediction")
    scores = np.asarray([row["r2"] for row in folds], dtype=np.float64)
    result = {
        "config": config_name,
        "descent_dispatch": dispatch_counts(),
        "microbenchmark": microbenchmark,
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
        "peak_rss_bytes": engine_base._peak_rss_bytes(),
        "holdout": {
            "prediction_sha256": harness.prediction_sha256(prediction),
            "predictions": [float(value) for value in prediction],
            "scores": harness.guardrails.score_player_guardrails(
                guardrail.y_holdout,
                prediction,
                guardrail.cold_player_mask,
            ),
            **fitted,
        },
        "guardrail": guardrail.metadata,
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }
    result["behavior_fingerprint_sha256"] = harness.behavior_fingerprint(
        engine_base._behavior_payload(result)
    )
    return result


def _run_worker_process(args: argparse.Namespace, config_name: str):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config",
        config_name,
        "--threads",
        str(args.threads),
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
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if completed.returncode or len(lines) != 1:
        raise RuntimeError(
            f"basketball worker {config_name!r} failed with exit code "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
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


def analyze_exactness(canonical: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = canonical[REFERENCE_CONFIG]
    candidate = canonical[CANDIDATE_CONFIG]
    gates: dict[str, bool] = {}
    gates["reference_reproduces"] = math.isclose(
        float(reference["mean_r2"]),
        EXPECTED_DEFAULT_MEAN_R2,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    gates["mean_r2_exact"] = candidate["mean_r2"] == reference["mean_r2"]
    gates["fold_scores_exact"] = (
        candidate["fold_scores"] == reference["fold_scores"]
    )
    gates["fold_count_exact"] = len(candidate["folds"]) == len(reference["folds"])
    fold_fields = (
        "test_indices",
        "r2",
        "prediction_sha256",
        "archive",
        "feature_importance_sha256",
        "fit_metadata",
    )
    gates["fold_payloads_exact"] = gates["fold_count_exact"] and all(
        all(candidate_row[field] == reference_row[field] for field in fold_fields)
        for reference_row, candidate_row in zip(
            reference["folds"], candidate["folds"]
        )
    )
    holdout_fields = (
        "scores",
        "prediction_sha256",
        "archive",
        "feature_importance_sha256",
        "fit_metadata",
    )
    gates["holdout_exact"] = all(
        candidate["holdout"][field] == reference["holdout"][field]
        for field in holdout_fields
    )
    gates["behavior_fingerprint_exact"] = (
        candidate["behavior_fingerprint_sha256"]
        == reference["behavior_fingerprint_sha256"]
    )
    reference_dispatch = reference["descent_dispatch"]
    candidate_dispatch = candidate["descent_dispatch"]
    gates["serial_dispatch_engaged"] = (
        int(reference_dispatch["serial_calls"]) == 0
        and int(reference_dispatch["parallel_calls"]) > 0
        and int(candidate_dispatch["serial_calls"]) > 0
        and int(candidate_dispatch["parallel_calls"]) == 0
    )
    return {
        "exactness_gates": gates,
        "passes_exactness_gates": all(gates.values()),
    }


def analyze_runtime(
    wall_timing,
    fit_timing,
    predict_timing,
    peak_rss_values,
    canonical,
    microbenchmarks,
) -> dict[str, Any]:
    wall_ratio = float(
        wall_timing[CANDIDATE_CONFIG]["median_seconds"]
        / wall_timing[REFERENCE_CONFIG]["median_seconds"]
    )
    fit_ratio = float(
        fit_timing[CANDIDATE_CONFIG]["median_seconds"]
        / fit_timing[REFERENCE_CONFIG]["median_seconds"]
    )
    predict_ratio = float(
        predict_timing[CANDIDATE_CONFIG]["median_seconds"]
        / predict_timing[REFERENCE_CONFIG]["median_seconds"]
    )
    rss_ratio = float(
        statistics.median(peak_rss_values[CANDIDATE_CONFIG])
        / statistics.median(peak_rss_values[REFERENCE_CONFIG])
    )
    kernel_ratios = [
        float(item["serial_over_parallel"]) for item in microbenchmarks
    ]
    reference_archives = [
        row["archive"] for row in canonical[REFERENCE_CONFIG]["folds"]
    ] + [canonical[REFERENCE_CONFIG]["holdout"]["archive"]]
    candidate_archives = [
        row["archive"] for row in canonical[CANDIDATE_CONFIG]["folds"]
    ] + [canonical[CANDIDATE_CONFIG]["holdout"]["archive"]]
    gates = {
        "reference_timing_stable": wall_timing[REFERENCE_CONFIG]["stable"],
        "candidate_timing_stable": wall_timing[CANDIDATE_CONFIG]["stable"],
        "wall_speedup": wall_ratio <= MAX_WALL_RATIO,
        "fit_speedup": fit_ratio <= MAX_FIT_RATIO,
        "archive_bytes_exact": candidate_archives == reference_archives,
        "peak_rss_within_budget": rss_ratio <= MAX_PEAK_RSS_RATIO,
        "microbenchmark_outputs_exact": all(
            bool(item["outputs_exact"]) for item in microbenchmarks
        ),
        "microbenchmark_speedup": (
            statistics.median(kernel_ratios) <= MAX_KERNEL_RATIO
        ),
    }
    return {
        "candidate_over_reference_median_wall_time": wall_ratio,
        "candidate_over_reference_median_fit_time": fit_ratio,
        "candidate_over_reference_median_predict_time": predict_ratio,
        "candidate_over_reference_median_peak_rss": rss_ratio,
        "median_serial_over_parallel_kernel_time": float(
            statistics.median(kernel_ratios)
        ),
        "prediction_timing_disposition": "diagnostic_training_only_candidate",
        "runtime_gates": gates,
        "passes_runtime_gates": all(gates.values()),
    }


def _git_object(revision: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode or len(value) != 40:
        raise RuntimeError(f"could not attest Git object {revision!r}")
    return value


def _repository_manifest_sha256() -> str:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError("could not enumerate tracked repository files")
    paths = [
        Path(value.decode("utf-8"))
        for value in completed.stdout.split(b"\0")
        if value
    ]
    runner_path = Path(__file__).resolve().relative_to(REPO_ROOT)
    expected = (
        "EXPECTED_REPOSITORY_MANIFEST_SHA256 = "
        f'"{EXPECTED_REPOSITORY_MANIFEST_SHA256}"'
    ).encode("utf-8")
    normalized = b'EXPECTED_REPOSITORY_MANIFEST_SHA256 = "<FROZEN>"'
    digest = hashlib.sha256()
    for relative in paths:
        payload = (REPO_ROOT / relative).read_bytes()
        if relative == runner_path:
            if payload.count(expected) != 1:
                raise RuntimeError(
                    "could not normalize repository manifest binding"
                )
            payload = payload.replace(expected, normalized, 1)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _support_manifest_sha256() -> str:
    digest = hashlib.sha256()
    for relative in BOUND_SUPPORT_PATHS:
        payload = (REPO_ROOT / relative).read_bytes()
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _run_prerequisite_suite() -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["DARKOFIT_STRICT_GOLDENS"] = "1"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    base_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
    ]
    collect_command = [*base_command, "--collect-only", "tests"]
    collected = subprocess.run(
        collect_command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    collected_node_ids = [
        line.strip()
        for line in collected.stdout.splitlines()
        if "::" in line and line.lstrip().startswith("tests/")
    ]
    required_collected = {
        path: any(node_id.startswith(f"{path}::") for node_id in collected_node_ids)
        for path in PREREQUISITE_TESTS
    }
    if collected.returncode or not all(required_collected.values()):
        raise RuntimeError(
            "serial-descent prerequisite collection failed or omitted required "
            f"tests: {required_collected}\nstdout:\n{collected.stdout}\n"
            f"stderr:\n{collected.stderr}"
        )
    command = [*base_command, "tests"]
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    evidence = {
        "command": command,
        "collect_command": collect_command,
        "darkofit_strict_goldens": True,
        "pytest_environment_scrubbed": True,
        "plugin_autoload_disabled": True,
        "required_test_files": list(PREREQUISITE_TESTS),
        "required_test_files_collected": required_collected,
        "collected_node_count": len(collected_node_ids),
        "collection_stdout_sha256": hashlib.sha256(
            collected.stdout.encode("utf-8")
        ).hexdigest(),
        "returncode": int(completed.returncode),
        "passed": completed.returncode == 0,
        "elapsed_seconds": float(elapsed),
        "stdout_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest(),
        "stderr_sha256": hashlib.sha256(
            completed.stderr.encode("utf-8")
        ).hexdigest(),
        "stdout_tail": completed.stdout.splitlines()[-3:],
        "stderr_tail": completed.stderr.splitlines()[-3:],
    }
    if completed.returncode:
        raise RuntimeError(
            "serial-descent prerequisite suite failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return evidence


def _validate_frozen_execution(args: argparse.Namespace) -> None:
    if args.threads != EXPECTED_THREADS:
        raise RuntimeError("serial-descent campaign requires exactly 18 threads")
    if args.output != DEFAULT_OUTPUT:
        raise RuntimeError("serial-descent campaign output path is not exact")
    if hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() != (
        EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("serial-descent protocol changed")
    if _git_object("HEAD:darkofit") != EXPECTED_DARKOFIT_SUBTREE:
        raise RuntimeError("serial-descent candidate subtree changed")
    if _git_object("HEAD:tests") != EXPECTED_TESTS_SUBTREE:
        raise RuntimeError("serial-descent test subtree changed")
    if _repository_manifest_sha256() != EXPECTED_REPOSITORY_MANIFEST_SHA256:
        raise RuntimeError("serial-descent repository manifest changed")
    if _support_manifest_sha256() != EXPECTED_SUPPORT_MANIFEST_SHA256:
        raise RuntimeError("serial-descent support manifest changed")


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
        "descent_dispatch": result["descent_dispatch"],
        "microbenchmark": result["microbenchmark"],
        "worker_stdout": result["worker_stdout"],
        "worker_stderr": result["worker_stderr"],
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_symlink():
        raise RuntimeError(f"refusing symlink benchmark output: {args.output}")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite benchmark output: {args.output}")
    _validate_frozen_execution(args)
    source = engine_base._source_state(False)
    print("running strict complete-suite prerequisite...", flush=True)
    prerequisite_suite = _run_prerequisite_suite()
    print(
        "  prerequisite suite passed in "
        f"{prerequisite_suite['elapsed_seconds']:.2f}s",
        flush=True,
    )
    dataset = harness.load_basketball_dataset(args.data_cache)
    schedule = harness.reciprocal_schedule(REFERENCE_CONFIG, CANDIDATE_CONFIG)
    canonical: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, set[str]] = {name: set() for name in CONFIG_ORDER}
    wall_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    fit_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    predict_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    rss_values: dict[str, list[float]] = {name: [] for name in CONFIG_ORDER}
    repeats = []
    microbenchmarks = []

    def run_block(block: int, order: tuple[str, str]) -> None:
        for position, config_name in enumerate(order):
            engine_base._assert_source_unchanged(
                source,
                engine_base._source_state(False),
                boundary=f"before block {block} {config_name}",
            )
            print(
                f"running block {block + 1}/{len(schedule)} "
                f"position {position + 1}: {config_name}...",
                flush=True,
            )
            result = _run_worker_process(args, config_name)
            fingerprints[config_name].add(result["behavior_fingerprint_sha256"])
            wall_values[config_name].append(result["steady_wall_seconds"])
            fit_values[config_name].append(result["summed_fit_seconds"])
            predict_values[config_name].append(result["summed_predict_seconds"])
            rss_values[config_name].append(result["peak_rss_bytes"])
            microbenchmarks.append(result["microbenchmark"])
            repeats.append(_repeat_record(block, position, result))
            canonical.setdefault(config_name, result)
            print(
                f"  mean R2={result['mean_r2']:.12f}, "
                f"steady={result['steady_wall_seconds']:.2f}s",
                flush=True,
            )

    run_block(0, schedule[0])
    exactness = analyze_exactness(canonical)
    timing_skipped = not exactness["passes_exactness_gates"]
    if not timing_skipped:
        for block, order in enumerate(schedule[1:], start=1):
            run_block(block, order)

    for config_name, values in fingerprints.items():
        if len(values) != 1:
            raise RuntimeError(f"{config_name} behavior changed across repeats")
    if not timing_skipped and (
        fingerprints[REFERENCE_CONFIG] != fingerprints[CANDIDATE_CONFIG]
    ):
        raise RuntimeError("reference and candidate behavior fingerprints differ")
    engine_base._assert_source_unchanged(
        source,
        engine_base._source_state(False),
        boundary="during the experiment",
    )

    wall_timing = fit_timing = predict_timing = runtime = None
    if not timing_skipped:
        wall_timing = {
            name: harness.timing_summary(values)
            for name, values in wall_values.items()
        }
        fit_timing = {
            name: harness.timing_summary(values)
            for name, values in fit_values.items()
        }
        predict_timing = {
            name: harness.timing_summary(values)
            for name, values in predict_values.items()
        }
        runtime = analyze_runtime(
            wall_timing,
            fit_timing,
            predict_timing,
            rss_values,
            canonical,
            microbenchmarks,
        )

    passes_runtime = runtime is not None and runtime["passes_runtime_gates"]
    collected_files = prerequisite_suite["required_test_files_collected"]
    prerequisite_gates = {
        "strict_complete_suite_passed": bool(prerequisite_suite["passed"]),
        "readable_oracle_collected": collected_files[
            "tests/test_oblivious_kernel_oracle.py"
        ],
        "prediction_goldens_collected": collected_files[
            "tests/test_prediction_goldens.py"
        ],
    }
    passes_prerequisites = all(prerequisite_gates.values())
    passes_all = (
        exactness["passes_exactness_gates"]
        and passes_runtime
        and passes_prerequisites
    )
    decision = {
        "candidate": CANDIDATE_CONFIG,
        **exactness,
        "prerequisite_gates": prerequisite_gates,
        "passes_prerequisite_gates": passes_prerequisites,
        "timing_confirmation_status": (
            "skipped_exactness_failure" if timing_skipped else "completed"
        ),
        "runtime": runtime,
        "passes_runtime_gates": passes_runtime,
        "evidence_gates": {"committed_clean_source": bool(source["clean"])},
        "evidence_eligible": bool(source["clean"]),
        "passes_all_gates": passes_all and bool(source["clean"]),
        "recommendation": (
            "promote_internal_serial_descent"
            if passes_all and source["clean"]
            else "advance_none"
        ),
    }
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_behavior_exact_small_serial_leaf_descent",
            "candidate_scope": "automatic_internal_training_dispatch",
            "public_parameter_added": False,
            "creator_benchmark_changed": False,
            "quality_tuning": False,
            "timing_schedule": [list(order) for order in schedule],
            "executed_timing_blocks": 1 if timing_skipped else len(schedule),
            "maximum_timing_spread_ratio": harness.MAX_TIMING_SPREAD_RATIO,
            "maximum_wall_ratio": MAX_WALL_RATIO,
            "maximum_fit_ratio": MAX_FIT_RATIO,
            "maximum_peak_rss_ratio": MAX_PEAK_RSS_RATIO,
            "maximum_kernel_ratio": MAX_KERNEL_RATIO,
            "prediction_timing_is_decision_gate": False,
            "threads_per_fit": args.threads,
            "random_state": creator.RANDOM_STATE,
            "weights_used": False,
            "lockbox_data_used": False,
            "small_row_cutoff": 32_768,
            "cv": {
                "kind": "KFold",
                "n_splits": creator.N_SPLITS,
                "shuffle": False,
                "n_jobs": 1,
                "fold_fingerprint_sha256": dataset.fold_fingerprint_sha256,
                "fold_test_sizes": dataset.fold_test_sizes,
            },
            "warmup": "one complete first-fold fit and prediction per worker",
        },
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "guardrail": dataset.player_guardrail.metadata,
        "source": source,
        "prerequisite_suite": prerequisite_suite,
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
        "microbenchmarks": microbenchmarks,
        "decision": decision,
    }
    engine_base._atomic_write_new_bytes(
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
    parser.add_argument("--threads", type=int, default=EXPECTED_THREADS)
    parser.add_argument("--worker-config", choices=CONFIG_ORDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output = creator._absolute_lexical_path(args.output)
    args.data_cache = creator._absolute_lexical_path(args.data_cache)
    if args.threads != EXPECTED_THREADS:
        parser.error("--threads must be exactly 18 for the frozen campaign")
    if not args.worker_config and args.output != DEFAULT_OUTPUT:
        parser.error("--output must be the frozen serial-descent artifact path")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_config:
        result = run_worker(args.worker_config, args.data_cache)
        print(WORKER_RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
