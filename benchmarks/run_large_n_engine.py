#!/usr/bin/env python3
"""Run the frozen large-n DarkoFit/ChimeraBoost engine certification."""

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
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
CHIMERA_ROOT = ROOT.parent / "chimeraboost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_campaign_harness as campaign  # noqa: E402
from benchmarks import basketball_harness as basketball  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_vector_fit_profile as vector_profile  # noqa: E402


DARKO = "darkofit"
CHIMERA = "chimeraboost"
ARMS = (DARKO, CHIMERA)
TRAIN_ROWS = (500_000, 1_000_000)
HOLDOUT_ROWS = 100_000
ITERATIONS = 300
THREADS = 18
BLOCK_ORDERS = (
    (DARKO, CHIMERA),
    (CHIMERA, DARKO),
    (DARKO, CHIMERA),
)
MAX_RMSE_RATIO = 1.002
MAX_FIT_GEOMEAN_RATIO = 1.0 / 1.30
MAX_SIZE_FIT_RATIO = 0.85
MAX_RSS_RATIO = 1.10
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
PROTOCOL = ROOT / "benchmarks" / "large_n_engine_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "large_n_engine.json"
WORKER_RESULT_PREFIX = "LARGE_N_ENGINE_RESULT="


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("large-n worker peak RSS is unavailable")
    return value


def _data(train_rows: int):
    X, targets = vector_profile._data(train_rows + HOLDOUT_ROWS)
    y = targets["scalar_rmse_catboost"]
    probe = np.concatenate(
        (
            X[:256].ravel(),
            X[-256:].ravel(),
            y[:256],
            y[-256:],
        )
    )
    return (
        X[:train_rows],
        y[:train_rows],
        X[train_rows:],
        y[train_rows:],
        _array_sha256(probe),
    )


def _common_params():
    return {
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "subsample": 1.0,
        "colsample": 1.0,
        "min_child_weight": 1.0,
        "ordered_boosting": False,
        "early_stopping": False,
        "thread_count": THREADS,
        "random_state": 4,
    }


def build_estimator(arm: str, iterations: int):
    if arm == DARKO:
        from darkofit import DarkoRegressor

        return DarkoRegressor(
            iterations=int(iterations),
            min_child_samples=1,
            tree_mode="catboost",
            linear_leaves=False,
            use_best_model=False,
            eval_train_loss=False,
            diagnostic_warnings="never",
            verbose_timing=True,
            **_common_params(),
        )
    if arm == CHIMERA:
        if str(CHIMERA_ROOT) not in sys.path:
            sys.path.insert(0, str(CHIMERA_ROOT))
        from chimeraboost import ChimeraBoostRegressor

        return ChimeraBoostRegressor(
            n_estimators=int(iterations),
            linear_leaves=False,
            cross_features=False,
            cat_combinations=False,
            **_common_params(),
        )
    raise ValueError(f"unknown large-n arm: {arm}")


def _metadata(model, arm, fused_engagement):
    core = model.model_
    if arm == DARKO:
        metadata = basketball.extract_fit_metadata(model)
        metadata["fused_engagement_count"] = int(fused_engagement)
        metadata["bin_sample_count"] = int(core.prep_.binner_.sample_count)
        return metadata
    return {
        "fitted_tree_count": int(len(core.trees_)),
        "resolved_learning_rate": float(core.lr_),
        "resolved_depth": int(core.depth),
        "resolved_thread_count": int(core.n_threads_),
        "linear_leaves_selected": bool(model.linear_leaves_selected_),
        "cross_features_selected": bool(model.cross_features_selected_),
        "fused_engagement_count": 0,
        "bin_sample_count": None,
    }


def _behavior_metadata(metadata):
    result = json.loads(json.dumps(metadata, allow_nan=False))
    for key in ("final_fit", "selection_fit"):
        if result.get(key) is not None:
            result[key]["phase_seconds"] = None
    return result


def _fit(arm, iterations, X, y):
    counter = np.zeros(1, dtype=np.int64)
    original = None
    if arm == DARKO:
        import darkofit.booster as booster_module

        original = booster_module.build_oblivious_tree
        booster_module.build_oblivious_tree = partial(
            original,
            fused_oblivious_counter=counter,
        )
    try:
        model = build_estimator(arm, iterations)
        started = time.perf_counter_ns()
        model.fit(X, y)
        fit_seconds = (time.perf_counter_ns() - started) / 1e9
    finally:
        if arm == DARKO:
            booster_module.build_oblivious_tree = original
    return model, float(fit_seconds), int(counter[0])


def run_worker(arm: str, train_rows: int) -> dict[str, Any]:
    X_train, y_train, X_test, y_test, data_probe = _data(train_rows)
    _fit(arm, 3, X_train[:5000], y_train[:5000])
    model, fit_seconds, engagement = _fit(
        arm, ITERATIONS, X_train, y_train
    )
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    if (
        prediction.shape != (HOLDOUT_ROWS,)
        or not np.all(np.isfinite(prediction))
        or not math.isfinite(fit_seconds)
        or fit_seconds <= 0.0
    ):
        raise RuntimeError("large-n worker produced invalid output")
    metadata = _metadata(model, arm, engagement)
    if metadata["fitted_tree_count"] != ITERATIONS:
        raise RuntimeError("large-n worker retained the wrong tree count")
    if metadata["resolved_thread_count"] != THREADS:
        raise RuntimeError("large-n worker resolved the wrong thread count")
    rmse = float(mean_squared_error(y_test, prediction) ** 0.5)
    timing = (
        {key: float(value) for key, value in (model.timing_ or {}).items()}
        if arm == DARKO
        else None
    )
    behavior = {
        "arm": arm,
        "train_rows": int(train_rows),
        "data_probe_sha256": data_probe,
        "prediction_sha256": _array_sha256(prediction),
        "rmse": rmse,
        "metadata": _behavior_metadata(metadata),
    }
    return {
        **behavior,
        "holdout_rows": HOLDOUT_ROWS,
        "features": int(X_train.shape[1]),
        "iterations": ITERATIONS,
        "fit_seconds": fit_seconds,
        "predict_seconds": float(predict_seconds),
        "darkofit_timing": timing,
        "peak_rss_bytes": _peak_rss_bytes(),
        "behavior_fingerprint_sha256": hashlib.sha256(
            json.dumps(
                behavior,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }


def _worker_environment():
    environment = basketball.worker_environment(THREADS)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join(
                (str(ROOT), str(CHIMERA_ROOT))
            ),
        }
    )
    return environment


def _run_worker(arm, train_rows):
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-arm",
            arm,
            "--worker-rows",
            str(train_rows),
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
            f"large-n worker {arm}/{train_rows} failed with "
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


def _geomean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(values))))


def analyze(results):
    expected = len(TRAIN_ROWS) * len(ARMS) * len(BLOCK_ORDERS)
    if len(results) != expected:
        raise RuntimeError(f"large-n gate requires {expected} workers")
    sizes = {}
    for rows in TRAIN_ROWS:
        darko = [
            row for row in results
            if row["train_rows"] == rows and row["arm"] == DARKO
        ]
        chimera = [
            row for row in results
            if row["train_rows"] == rows and row["arm"] == CHIMERA
        ]
        if len(darko) != 3 or len(chimera) != 3:
            raise RuntimeError("large-n gate is missing an arm repeat")
        fit = campaign.paired_ratio_summary(
            [row["fit_seconds"] for row in darko],
            [row["fit_seconds"] for row in chimera],
        )
        rss = campaign.paired_ratio_summary(
            [row["peak_rss_bytes"] for row in darko],
            [row["peak_rss_bytes"] for row in chimera],
        )
        rmse_ratio = float(darko[0]["rmse"] / chimera[0]["rmse"])
        sizes[str(rows)] = {
            "darkofit_rmse": float(darko[0]["rmse"]),
            "chimeraboost_rmse": float(chimera[0]["rmse"]),
            "rmse_ratio": rmse_ratio,
            "fit_ratio": fit,
            "rss_ratio": rss,
            "darkofit_predict_seconds_median": float(
                np.median([row["predict_seconds"] for row in darko])
            ),
            "chimeraboost_predict_seconds_median": float(
                np.median([row["predict_seconds"] for row in chimera])
            ),
            "behavior_stable": (
                len({row["behavior_fingerprint_sha256"] for row in darko})
                == 1
                and len(
                    {row["behavior_fingerprint_sha256"] for row in chimera}
                )
                == 1
            ),
            "fused_engages": all(
                row["metadata"]["fused_engagement_count"] > 0
                for row in darko
            ),
        }
    fit_ratios = [
        sizes[str(rows)]["fit_ratio"]["median_ratio"]
        for rows in TRAIN_ROWS
    ]
    gates = {
        "all_behavior_stable": all(
            row["behavior_stable"] for row in sizes.values()
        ),
        "fused_engages": all(
            row["fused_engages"] for row in sizes.values()
        ),
        "quality_noninferior": all(
            row["rmse_ratio"] <= MAX_RMSE_RATIO for row in sizes.values()
        ),
        "fit_ratios_stable": all(
            row["fit_ratio"]["stable"] for row in sizes.values()
        ),
        "rss_ratios_stable": all(
            row["rss_ratio"]["stable"] for row in sizes.values()
        ),
        "fit_geomean_at_most_1_over_1_30": (
            _geomean(fit_ratios) <= MAX_FIT_GEOMEAN_RATIO
        ),
        "no_size_fit_ratio_over_0_85": max(fit_ratios)
        <= MAX_SIZE_FIT_RATIO,
        "rss_at_most_1_10": all(
            row["rss_ratio"]["median_ratio"] <= MAX_RSS_RATIO
            for row in sizes.values()
        ),
    }
    passed = all(gates.values())
    return {
        "sizes": sizes,
        "fit_geomean_ratio": _geomean(fit_ratios),
        "fit_geomean_speedup": 1.0 / _geomean(fit_ratios),
        "gates": gates,
        "passes_all_gates": passed,
        "recommendation": (
            "certify_large_n_engine_advantage"
            if passed
            else "do_not_claim_large_n_engine_advantage"
        ),
    }


def run_parent(args):
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    darko_source = creator.git_state(ROOT)
    chimera_source = creator.git_state(CHIMERA_ROOT)
    if not darko_source["clean"] or not chimera_source["clean"]:
        raise RuntimeError("large-n gate requires clean source trees")
    if chimera_source["head"] != EXPECTED_CHIMERA_HEAD:
        raise RuntimeError("large-n gate ChimeraBoost head changed")
    results = []
    for block, order in enumerate(BLOCK_ORDERS):
        for train_rows in TRAIN_ROWS:
            for position, arm in enumerate(order):
                if creator.git_state(ROOT) != darko_source:
                    raise RuntimeError("DarkoFit changed during large-n gate")
                if creator.git_state(CHIMERA_ROOT) != chimera_source:
                    raise RuntimeError(
                        "ChimeraBoost changed during large-n gate"
                    )
                print(
                    f"block {block + 1}/{len(BLOCK_ORDERS)} "
                    f"rows={train_rows} position={position + 1} arm={arm}",
                    flush=True,
                )
                result = _run_worker(arm, train_rows)
                result["block"] = int(block)
                result["position"] = int(position)
                results.append(result)
    if creator.git_state(ROOT) != darko_source:
        raise RuntimeError("DarkoFit changed during large-n gate")
    if creator.git_state(CHIMERA_ROOT) != chimera_source:
        raise RuntimeError("ChimeraBoost changed during large-n gate")
    analysis = analyze(results)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "large_n_matched_core_engine",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "train_rows": list(TRAIN_ROWS),
            "holdout_rows": HOLDOUT_ROWS,
            "iterations": ITERATIONS,
            "threads": THREADS,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "max_rmse_ratio": MAX_RMSE_RATIO,
            "max_fit_geomean_ratio": MAX_FIT_GEOMEAN_RATIO,
            "max_size_fit_ratio": MAX_SIZE_FIT_RATIO,
            "max_rss_ratio": MAX_RSS_RATIO,
            "lockbox_data_used": False,
            "default_change_authorized": False,
        },
        "sources": {
            "darkofit": darko_source,
            "chimeraboost": chimera_source,
        },
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
        ).encode()
    )
    print(f"decision: {analysis['recommendation']}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-arm", choices=ARMS)
    parser.add_argument("--worker-rows", type=int, choices=TRAIN_ROWS)
    args = parser.parse_args(argv)
    args.output = creator._absolute_lexical_path(args.output)
    if bool(args.worker_arm) != bool(args.worker_rows):
        parser.error("--worker-arm and --worker-rows must be used together")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker_arm:
        result = run_worker(args.worker_arm, args.worker_rows)
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
