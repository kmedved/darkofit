#!/usr/bin/env python3
"""Run the matched large-batch prediction-throughput characterization."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_campaign_harness as campaign  # noqa: E402
from benchmarks import basketball_harness as basketball  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


DARKOFIT = "darkofit"
CHIMERABOOST = "chimeraboost"
ARMS = (DARKOFIT, CHIMERABOOST)
DATASETS = ("basketball_numeric", "synthetic_mixed")
BATCH_SIZES = (8_192, 65_536, 524_288, 2_000_000)
BLOCK_ORDERS = (
    (DARKOFIT, CHIMERABOOST),
    (CHIMERABOOST, DARKOFIT),
    (DARKOFIT, CHIMERABOOST),
)
WARM_REPEATS = {8_192: 5, 65_536: 4, 524_288: 3, 2_000_000: 2}
EXPECTED_THREADS = 18
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
TARGET_PUBLIC_RATIO = 1.30
WORKER_RESULT_PREFIX = "PREDICT_THROUGHPUT_RESULT="
PROTOCOL = ROOT / "benchmarks" / "predict_throughput_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "predict_throughput.json"
DEFAULT_CHIMERA_REPO = ROOT.parent / "chimeraboost"

DARKO_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 1.0,
    "max_bins": 128,
    "subsample": 1.0,
    "colsample": 1.0,
    "min_child_weight": 1.0,
    "min_child_samples": 1,
    "ordered_boosting": False,
    "early_stopping": False,
    "tree_mode": "catboost",
    "linear_leaves": False,
    "thread_count": EXPECTED_THREADS,
    "random_state": creator.RANDOM_STATE,
    "diagnostic_warnings": "never",
}
CHIMERA_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 1.0,
    "max_bins": 128,
    "subsample": 1.0,
    "colsample": 1.0,
    "min_child_weight": 1.0,
    "ordered_boosting": False,
    "early_stopping": False,
    "linear_leaves": False,
    "cross_features": False,
    "cat_combinations": False,
    "thread_count": EXPECTED_THREADS,
    "random_state": creator.RANDOM_STATE,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(
        dtype="<u8"
    )
    digest = hashlib.sha256()
    digest.update("|".join(str(value) for value in frame.columns).encode())
    digest.update("|".join(str(value) for value in frame.dtypes).encode())
    digest.update(np.ascontiguousarray(row_hashes).tobytes())
    return digest.hexdigest()


def _prepend(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _assert_source(model: Any, repository: Path) -> None:
    package = model.__class__.__module__.split(".", 1)[0]
    module_path = Path(importlib.import_module(package).__file__).resolve()
    if not module_path.is_relative_to(repository.resolve()):
        raise RuntimeError(
            f"{package} imported from {module_path}, outside {repository}"
        )


def _build_model(arm: str, chimera_repo: Path):
    if arm == DARKOFIT:
        from darkofit import DarkoRegressor

        model = DarkoRegressor(**DARKO_PARAMS)
        _assert_source(model, ROOT)
        return model
    if arm == CHIMERABOOST:
        _prepend(chimera_repo)
        from chimeraboost import ChimeraBoostRegressor

        model = ChimeraBoostRegressor(**CHIMERA_PARAMS)
        _assert_source(model, chimera_repo)
        return model
    raise ValueError(f"unknown arm: {arm}")


def _mixed_training_data() -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    rng = np.random.default_rng(20_260_717)
    rows = 20_000
    numeric = rng.normal(size=(rows, 6))
    cat_a = np.asarray([f"a{value}" for value in rng.integers(0, 32, rows)])
    cat_b = np.asarray([f"b{value}" for value in rng.integers(0, 8, rows)])
    frame = pd.DataFrame(numeric, columns=[f"x{i}" for i in range(6)])
    frame["cat_a"] = cat_a
    frame["cat_b"] = cat_b
    effects_a = np.asarray([int(value[1:]) for value in cat_a], dtype=float)
    effects_b = np.asarray([int(value[1:]) for value in cat_b], dtype=float)
    target = (
        1.2 * numeric[:, 0]
        - 0.7 * numeric[:, 1]
        + 0.25 * numeric[:, 2] ** 2
        + 0.04 * effects_a
        - 0.08 * effects_b
        + rng.normal(0.0, 0.2, rows)
    )
    return frame, target, [6, 7]


def _fit_models(
    arm: str, cache_path: Path, chimera_repo: Path
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
    dataset = basketball.load_basketball_dataset(cache_path)
    train, test = next(creator.creator_cv().split(dataset.X, dataset.y))
    numeric_model = _build_model(arm, chimera_repo)
    started = time.perf_counter_ns()
    numeric_model.fit(dataset.X.iloc[train], dataset.y.iloc[train])
    numeric_fit = (time.perf_counter_ns() - started) / 1e9

    mixed_X, mixed_y, cat_features = _mixed_training_data()
    mixed_model = _build_model(arm, chimera_repo)
    started = time.perf_counter_ns()
    mixed_model.fit(mixed_X, mixed_y, cat_features=cat_features)
    mixed_fit = (time.perf_counter_ns() - started) / 1e9

    models = {
        "basketball_numeric": numeric_model,
        "synthetic_mixed": mixed_model,
    }
    templates = {
        "basketball_numeric": dataset.X.iloc[test].reset_index(drop=True),
        "synthetic_mixed": mixed_X.iloc[:4096].reset_index(drop=True),
    }
    metadata = {
        "basketball_numeric": {
            "fit_seconds": float(numeric_fit),
            "fit_metadata": _fit_metadata(arm, numeric_model),
            "training_rows": int(len(train)),
            "template_sha256": _frame_sha256(templates["basketball_numeric"]),
        },
        "synthetic_mixed": {
            "fit_seconds": float(mixed_fit),
            "fit_metadata": _fit_metadata(arm, mixed_model),
            "training_rows": int(len(mixed_X)),
            "template_sha256": _frame_sha256(templates["synthetic_mixed"]),
        },
    }
    return models, templates, metadata


def _fit_metadata(arm: str, model: Any) -> dict[str, Any]:
    core = model.model_
    trees = list(core.trees_)
    if len(trees) != 1000:
        raise RuntimeError(f"{arm} retained {len(trees)} trees, expected 1000")
    depths = [int(tree.depth) for tree in trees]
    if set(depths) != {6}:
        raise RuntimeError(f"{arm} produced unexpected tree depths")
    if arm == DARKOFIT:
        return basketball.extract_fit_metadata(model)
    return {
        "fitted_tree_count": len(trees),
        "best_iteration": int(core.best_iteration_),
        "resolved_learning_rate": float(core.lr_),
        "resolved_thread_count": int(core.n_threads_),
        "selected_tree_mode": "oblivious",
        "selected_lane": (
            "linear_leaves"
            if bool(getattr(model, "linear_leaves_selected_", False))
            else "boosting"
        ),
        "tree_depths_unique": sorted(set(depths)),
    }


def _repeat_frame(template: pd.DataFrame, rows: int) -> pd.DataFrame:
    indices = np.arange(int(rows), dtype=np.int64) % len(template)
    return template.iloc[indices].reset_index(drop=True)


def _validate_output(value: Any, rows: int) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64)
    if output.shape != (rows,) or not np.all(np.isfinite(output)):
        raise RuntimeError("prediction output is invalid")
    return output


def _time_calls(function: Callable[[], Any], repeats: int) -> dict[str, Any]:
    values = []
    output = None
    for _ in range(int(repeats)):
        gc.disable()
        started = time.perf_counter_ns()
        try:
            output = function()
        finally:
            elapsed = (time.perf_counter_ns() - started) / 1e9
            gc.enable()
        values.append(float(elapsed))
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise RuntimeError("timing produced an invalid value")
    return {
        "values_seconds": values,
        "median_seconds": float(np.median(array)),
        "minimum_seconds": float(array.min()),
        "maximum_seconds": float(array.max()),
        "repetitions": int(repeats),
        "last_output": output,
    }


def _clear_forest_cache(arm: str, model: Any) -> None:
    core = model.model_
    if arm == DARKOFIT:
        core._flat_cache_ = None
    else:
        core._forest_ = None


def _prepare_and_bin(arm: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    core = model.model_
    if arm == DARKOFIT:
        prepared = core._prepare_predict_X(X)
    else:
        prepared = np.asarray(X)
    return core.prep_.transform(prepared)


def _packed_predict(arm: str, model: Any, X_binned: np.ndarray) -> np.ndarray:
    core = model.model_
    if arm == DARKOFIT:
        flat = core._flat_ensemble()
        if flat is None:
            raise RuntimeError("DarkoFit did not build a packed forest")
        output = np.full(len(X_binned), core.init_, dtype=np.float64)
        flat.add_predict(X_binned, output)
        return output

    if core._forest_ is None:
        raise RuntimeError("ChimeraBoost did not build a packed forest")
    from chimeraboost.tree import _predict_forest_rm

    features, thresholds, depths, values, offsets = core._forest_
    return _predict_forest_rm(
        X_binned,
        features,
        thresholds,
        depths,
        values,
        offsets,
        core.init_,
    )


def _case(arm: str, model: Any, template: pd.DataFrame, rows: int):
    X = _repeat_frame(template, rows)
    input_sha256 = _frame_sha256(X)
    _clear_forest_cache(arm, model)

    started = time.perf_counter_ns()
    cold_output = _validate_output(model.predict(X), rows)
    cold_seconds = (time.perf_counter_ns() - started) / 1e9
    warm = _time_calls(
        lambda: _validate_output(model.predict(X), rows),
        WARM_REPEATS[rows],
    )
    warm_output = warm.pop("last_output")

    binning = _time_calls(
        lambda: _prepare_and_bin(arm, model, X),
        WARM_REPEATS[rows],
    )
    X_binned = np.asarray(binning.pop("last_output"))
    core = _time_calls(
        lambda: _packed_predict(arm, model, X_binned),
        WARM_REPEATS[rows],
    )
    core_output = _validate_output(core.pop("last_output"), rows)
    exactness = {
        "cold_equals_warm": bool(np.array_equal(cold_output, warm_output)),
        "public_equals_packed_core": bool(
            np.array_equal(warm_output, core_output)
        ),
    }
    if not all(exactness.values()):
        raise RuntimeError(f"{arm}/{rows}: packed prediction exactness failed")
    result = {
        "rows": int(rows),
        "columns": int(X.shape[1]),
        "input_sha256": input_sha256,
        "input_dtypes": [str(value) for value in X.dtypes],
        "binned_shape": [int(value) for value in X_binned.shape],
        "binned_dtype": str(X_binned.dtype),
        "prediction_sha256": _array_sha256(warm_output),
        "exactness": exactness,
        "cold_public_seconds": float(cold_seconds),
        "warm_public": warm,
        "binning": binning,
        "packed_core": core,
    }
    del X, X_binned, cold_output, warm_output, core_output
    gc.collect()
    return result


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("peak RSS is unavailable")
    return value


def _behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm": result["arm"],
        "models": {
            name: value["fit_metadata"]
            for name, value in result["models"].items()
        },
        "cases": {
            dataset: {
                rows: {
                    "input_sha256": case["input_sha256"],
                    "prediction_sha256": case["prediction_sha256"],
                    "exactness": case["exactness"],
                    "binned_shape": case["binned_shape"],
                    "binned_dtype": case["binned_dtype"],
                }
                for rows, case in cases.items()
            }
            for dataset, cases in result["cases"].items()
        },
    }


def run_worker(
    arm: str, cache_path: Path, chimera_repo: Path
) -> dict[str, Any]:
    models, templates, metadata = _fit_models(arm, cache_path, chimera_repo)
    cases = {}
    for dataset in DATASETS:
        cases[dataset] = {}
        for rows in BATCH_SIZES:
            print(f"worker {arm}: {dataset} rows={rows}", flush=True)
            cases[dataset][str(rows)] = _case(
                arm, models[dataset], templates[dataset], rows
            )
    result = {
        "arm": arm,
        "models": metadata,
        "cases": cases,
        "peak_rss_bytes": _peak_rss_bytes(),
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }
    result["behavior_fingerprint_sha256"] = basketball.behavior_fingerprint(
        _behavior_payload(result)
    )
    return result


def _source_states(args: argparse.Namespace) -> dict[str, Any]:
    states = {
        DARKOFIT: creator.git_state(ROOT),
        CHIMERABOOST: creator.git_state(args.chimeraboost_repo),
    }
    if not all(state["clean"] for state in states.values()):
        raise RuntimeError("throughput campaign requires clean repositories")
    chimera = states[CHIMERABOOST]
    if chimera["head"] != EXPECTED_CHIMERA_HEAD:
        raise RuntimeError("ChimeraBoost is not the frozen 0.15.0 comparator")
    for ref in ("origin/main", "upstream/main"):
        observed = chimera["tracked_main_refs"].get(ref)
        if observed is not None and observed != EXPECTED_CHIMERA_HEAD:
            raise RuntimeError(f"ChimeraBoost {ref} differs from v0.15.0")
    return states


def _assert_sources_unchanged(expected, observed, boundary: str) -> None:
    fields = ("path", "head", "branch", "clean", "status")
    for repository in expected:
        if any(
            expected[repository][field] != observed[repository][field]
            for field in fields
        ):
            raise RuntimeError(f"source changed {boundary}: {repository}")


def _worker_environment(args: argparse.Namespace, arm: str) -> dict[str, str]:
    environment = basketball.worker_environment(args.threads)
    paths = [str(ROOT)]
    if arm == CHIMERABOOST:
        paths.insert(0, str(args.chimeraboost_repo))
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["CHIMERABOOST_WARMUP"] = "0"
    environment["DARKOFIT_WARMUP"] = "0"
    return environment


def _run_worker_process(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    command = [
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
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_worker_environment(args, arm),
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
            f"throughput worker {arm} failed with {completed.returncode}"
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
) -> dict[str, Any]:
    fingerprints = {
        arm: {
            result["behavior_fingerprint_sha256"]
            for result in block_results
            if result["arm"] == arm
        }
        for arm in ARMS
    }
    fingerprint_stable = all(len(values) == 1 for values in fingerprints.values())
    paired = {}
    target_gates = {}
    component_excess = {"binning": [], "packed_core": []}
    for dataset in DATASETS:
        paired[dataset] = {}
        for rows in BATCH_SIZES:
            key = str(rows)
            paired[dataset][key] = {}
            for phase in ("warm_public", "binning", "packed_core"):
                darko = [
                    float(result["cases"][dataset][key][phase]["median_seconds"])
                    for result in block_results
                    if result["arm"] == DARKOFIT
                ]
                chimera = [
                    float(result["cases"][dataset][key][phase]["median_seconds"])
                    for result in block_results
                    if result["arm"] == CHIMERABOOST
                ]
                summary = campaign.paired_ratio_summary(darko, chimera)
                paired[dataset][key][phase] = summary
                if phase in component_excess:
                    component_excess[phase].append(
                        max(0.0, summary["median_ratio"] - 1.0)
                    )
            public = paired[dataset][key]["warm_public"]
            target_gates[f"{dataset}_{rows}_stable"] = public["stable"]
            target_gates[f"{dataset}_{rows}_ratio_at_most_1_30"] = (
                public["median_ratio"] <= TARGET_PUBLIC_RATIO
            )
    exactness = all(
        all(case["exactness"].values())
        for arm in ARMS
        for cases in canonical[arm]["cases"].values()
        for case in cases.values()
    )
    target_gates["behavior_fingerprints_stable"] = fingerprint_stable
    target_gates["within_library_packed_exactness"] = exactness
    dominant = max(
        component_excess,
        key=lambda name: float(np.median(component_excess[name])),
    )
    return {
        "paired_ratios": paired,
        "target_gates": target_gates,
        "meets_public_target": all(target_gates.values()),
        "median_component_excess_over_chimera": {
            name: float(np.median(values))
            for name, values in component_excess.items()
        },
        "largest_excess_component": dominant,
        "recommendation": (
            "target_met_no_p2_optimization_required"
            if all(target_gates.values())
            else f"start_p2_with_{dominant}"
        ),
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    if args.threads != EXPECTED_THREADS:
        raise RuntimeError("throughput protocol requires exactly 18 threads")
    sources = _source_states(args)
    canonical = {}
    block_results = []
    rss = {arm: [] for arm in ARMS}
    for block, order in enumerate(BLOCK_ORDERS):
        for position, arm in enumerate(order):
            _assert_sources_unchanged(
                sources,
                _source_states(args),
                f"before block {block} {arm}",
            )
            print(
                f"running block {block + 1}/{len(BLOCK_ORDERS)} "
                f"position {position + 1}: {arm}",
                flush=True,
            )
            result = _run_worker_process(args, arm)
            result["block"] = int(block)
            result["position"] = int(position)
            canonical.setdefault(arm, result)
            block_results.append(result)
            rss[arm].append(int(result["peak_rss_bytes"]))
    _assert_sources_unchanged(
        sources, _source_states(args), "during throughput campaign"
    )
    analysis = analyze(canonical, block_results)
    analysis["paired_peak_rss"] = campaign.paired_ratio_summary(
        rss[DARKOFIT], rss[CHIMERABOOST]
    )
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "matched_prediction_throughput",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "chimeraboost_head": EXPECTED_CHIMERA_HEAD,
            "threads": EXPECTED_THREADS,
            "batch_sizes": list(BATCH_SIZES),
            "warm_repeats": WARM_REPEATS,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "target_public_ratio": TARGET_PUBLIC_RATIO,
            "paired_ratio_max_iqr_over_median": (
                campaign.MAX_PAIRED_RATIO_IQR_OVER_MEDIAN
            ),
            "default_promotion_authorized": False,
            "lockbox_data_used": False,
        },
        "sources": sources,
        "environment": {
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
        },
        "canonical_results": canonical,
        "block_results": block_results,
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
    parser.add_argument(
        "--data-cache", type=Path, default=basketball.DEFAULT_CACHE
    )
    parser.add_argument(
        "--chimeraboost-repo", type=Path, default=DEFAULT_CHIMERA_REPO
    )
    parser.add_argument("--threads", type=int, default=EXPECTED_THREADS)
    parser.add_argument("--worker-arm", choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output = creator._absolute_lexical_path(args.output)
    args.data_cache = creator._absolute_lexical_path(args.data_cache)
    args.chimeraboost_repo = creator._absolute_lexical_path(
        args.chimeraboost_repo
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_arm:
        result = run_worker(
            args.worker_arm, args.data_cache, args.chimeraboost_repo
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
