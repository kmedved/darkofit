"""Bounded current-main performance preflight for integrated hot paths.

The lanes compare optimized and reference implementations on identical data.
They are regression alarms, not general-purpose performance claims.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numba import get_num_threads, set_num_threads
from sklearn.datasets import make_classification, make_regression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import darkofit
from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.binning import Binner
from darkofit.flat_model import flat_predict_preferred


def _timing_summary(values: list[float]) -> dict[str, float | list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    iqr = float(np.subtract(*np.percentile(arr, [75, 25])))
    return {
        "raw_seconds": [float(value) for value in arr],
        "median_seconds": median,
        "iqr_seconds": iqr,
        "iqr_fraction": iqr / median if median > 0.0 else float("inf"),
    }


def _time_alternating(
    optimized,
    reference,
    repeats: int,
) -> tuple[dict[str, float | list[float]], dict[str, float | list[float]]]:
    timings = {"optimized": [], "reference": []}
    functions = {"optimized": optimized, "reference": reference}
    for repeat in range(repeats):
        order = ("optimized", "reference") if repeat % 2 == 0 else (
            "reference",
            "optimized",
        )
        for name in order:
            start = time.perf_counter()
            functions[name]()
            timings[name].append(time.perf_counter() - start)
            gc.collect()
    return _timing_summary(timings["optimized"]), _timing_summary(
        timings["reference"]
    )


def _speedup(
    optimized: dict[str, float | list[float]],
    reference: dict[str, float | list[float]],
) -> float:
    return float(reference["median_seconds"]) / float(
        optimized["median_seconds"]
    )


def _flat_preflight(seed: int, threads: int) -> dict:
    X_train, y_train = make_regression(
        n_samples=10_000,
        n_features=20,
        n_informative=14,
        noise=5.0,
        random_state=seed,
    )
    model = DarkoRegressor(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        early_stopping=False,
        eval_train_loss=False,
        ordered_boosting=False,
        thread_count=threads,
        random_state=seed,
    ).fit(X_train, y_train)
    booster = model.model_
    X_predict = np.random.default_rng(seed + 1).normal(size=(100_000, 20))
    X_binned = booster.prep_.transform(X_predict)
    flat = booster._flat_ensemble()
    router_selected = flat is not None and flat_predict_preferred(
        flat, X_binned.shape[0], booster.tree_mode_
    )

    def optimized():
        out = np.full(X_binned.shape[0], booster.init_, dtype=np.float64)
        flat.add_predict(X_binned, out)
        return out

    def reference():
        out = np.full(X_binned.shape[0], booster.init_, dtype=np.float64)
        for tree in booster.trees_:
            tree.add_predict(X_binned, out)
        return out

    optimized_out = optimized()
    reference_out = reference()
    parity = bool(np.array_equal(optimized_out, reference_out))
    optimized_timing, reference_timing = _time_alternating(
        optimized, reference, repeats=7
    )
    reran_for_noise = False
    max_iqr = max(
        float(optimized_timing["iqr_fraction"]),
        float(reference_timing["iqr_fraction"]),
    )
    if max_iqr > 0.15:
        reran_for_noise = True
        optimized_timing, reference_timing = _time_alternating(
            optimized, reference, repeats=7
        )
        max_iqr = max(
            float(optimized_timing["iqr_fraction"]),
            float(reference_timing["iqr_fraction"]),
        )
    speedup = _speedup(optimized_timing, reference_timing)
    start = time.perf_counter()
    public_out = model.predict(X_predict)
    public_seconds = time.perf_counter() - start
    return {
        "case": {
            "train_rows": 10_000,
            "prediction_rows": 100_000,
            "features": 20,
            "trees": int(booster.best_iteration_),
            "depth": 6,
        },
        "optimized": optimized_timing,
        "reference": reference_timing,
        "speedup": speedup,
        "parity": parity,
        "public_prediction_parity": bool(np.array_equal(public_out, optimized_out)),
        "router_selected": bool(router_selected),
        "public_end_to_end_seconds": public_seconds,
        "reran_for_noise": reran_for_noise,
        "max_iqr_fraction": max_iqr,
        "thresholds": {"minimum_speedup": 1.25, "maximum_iqr_fraction": 0.15},
        "passed": bool(
            parity
            and np.array_equal(public_out, optimized_out)
            and router_selected
            and speedup >= 1.25
            and max_iqr <= 0.15
        ),
    }


def _block_binning_preflight(seed: int) -> dict:
    rng = np.random.default_rng(seed + 2)
    blocks = [
        np.ascontiguousarray(rng.normal(size=(250_000, width)))
        for width in (32, 4, 4)
    ]

    # Compile the transform kernel without warming the measured arrays/cache.
    warm_blocks = [rng.normal(size=(1_000, width)) for width in (32, 4, 4)]
    Binner(max_bins=254, sample_count=800, random_state=seed).fit_transform_blocks(
        warm_blocks
    )
    Binner(max_bins=254, sample_count=800, random_state=seed).fit_transform(
        np.hstack(warm_blocks)
    )

    def block_result():
        binner = Binner(max_bins=254, sample_count=200_000, random_state=seed)
        return binner, binner.fit_transform_blocks(blocks)

    def stacked_result():
        binner = Binner(max_bins=254, sample_count=200_000, random_state=seed)
        return binner, binner.fit_transform(np.hstack(blocks))

    block_binner, block_out = block_result()
    stack_binner, stack_out = stacked_result()
    border_parity = all(
        np.array_equal(left, right)
        for left, right in zip(block_binner.borders_, stack_binner.borders_)
    )
    output_parity = bool(np.array_equal(block_out, stack_out))

    def optimized():
        block_result()

    def reference():
        stacked_result()

    optimized_timing, reference_timing = _time_alternating(
        optimized, reference, repeats=3
    )
    speedup = _speedup(optimized_timing, reference_timing)
    minimum_non_regression_speedup = 1.0 / 1.05
    return {
        "case": {
            "rows": 250_000,
            "block_widths": [32, 4, 4],
            "features": 40,
            "sample_count": 200_000,
            "avoided_stack_bytes": 250_000 * 40 * 8,
        },
        "optimized": optimized_timing,
        "reference": reference_timing,
        "speedup": speedup,
        "border_parity": bool(border_parity),
        "output_parity": output_parity,
        "thresholds": {
            "minimum_non_regression_speedup": minimum_non_regression_speedup,
            "preferred_speedup": 1.10,
        },
        "preferred_speedup_met": bool(speedup >= 1.10),
        "passed": bool(
            border_parity
            and output_parity
            and speedup >= minimum_non_regression_speedup
        ),
    }


def _training_preflight(seed: int, threads: int) -> dict:
    X, y = make_classification(
        n_samples=100_000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=4,
        n_clusters_per_class=2,
        random_state=seed + 3,
    )
    X_train, X_predict = X[:90_000], X[90_000:]
    y_train = y[:90_000]

    common = dict(
        iterations=50,
        learning_rate=0.1,
        depth=6,
        early_stopping=False,
        ordered_boosting=False,
        thread_count=threads,
        random_state=seed,
    )

    # Compile both evaluation paths on an independent small case.
    for enabled in (False, True):
        DarkoClassifier(
            iterations=2,
            learning_rate=0.1,
            depth=2,
            early_stopping=False,
            ordered_boosting=False,
            thread_count=threads,
            random_state=seed,
            eval_train_loss=enabled,
        ).fit(X_train[:2_000], y_train[:2_000])

    def timed_batch():
        timings = {False: [], True: []}
        final_models = {}
        for repeat in range(7):
            order = (False, True) if repeat % 2 == 0 else (True, False)
            for enabled in order:
                model = DarkoClassifier(eval_train_loss=enabled, **common)
                start = time.perf_counter()
                model.fit(X_train, y_train)
                timings[enabled].append(time.perf_counter() - start)
                final_models[enabled] = model
                gc.collect()
        return (
            _timing_summary(timings[False]),
            _timing_summary(timings[True]),
            final_models,
        )

    optimized_timing, reference_timing, final_models = timed_batch()
    reran_for_noise = False
    max_iqr = max(
        float(optimized_timing["iqr_fraction"]),
        float(reference_timing["iqr_fraction"]),
    )
    if max_iqr > 0.15:
        reran_for_noise = True
        optimized_timing, reference_timing, final_models = timed_batch()
        max_iqr = max(
            float(optimized_timing["iqr_fraction"]),
            float(reference_timing["iqr_fraction"]),
        )
    optimized_prediction = final_models[False].predict_proba(X_predict)
    reference_prediction = final_models[True].predict_proba(X_predict)
    parity = bool(np.array_equal(optimized_prediction, reference_prediction))
    speedup = _speedup(optimized_timing, reference_timing)
    return {
        "case": {
            "train_rows": 90_000,
            "prediction_rows": 10_000,
            "features": 30,
            "classes": 4,
            "rounds": 50,
            "depth": 6,
        },
        "optimized": optimized_timing,
        "reference": reference_timing,
        "speedup": speedup,
        "prediction_parity": parity,
        "timed_scope": "fit_only",
        "reran_for_noise": reran_for_noise,
        "max_iqr_fraction": max_iqr,
        "thresholds": {"minimum_speedup": 1.05, "maximum_iqr_fraction": 0.15},
        "passed": bool(parity and speedup >= 1.05 and max_iqr <= 0.15),
    }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_status() -> list[str] | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        )
        return output.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def _harness_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/perf-preflight-remaining9/hotpaths.json"),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_712)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    previous_threads = get_num_threads()
    set_num_threads(args.threads)
    started = time.time()
    try:
        results = {
            "schema_version": 1,
            "source_commit": _git_commit(),
            "source_status": _git_status(),
            "harness_sha256": _harness_sha256(),
            "command_line": [sys.executable, *sys.argv],
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_model": _cpu_model(),
                "numpy": np.__version__,
                "numba": _version("numba"),
                "scikit_learn": _version("scikit-learn"),
                "darkofit": _version("darkofit"),
                "darkofit_source": str(Path(darkofit.__file__).resolve()),
                "threads": args.threads,
            },
            "seed": args.seed,
            "flat_prediction": _flat_preflight(args.seed, args.threads),
            "block_binning": _block_binning_preflight(args.seed),
            "training": _training_preflight(args.seed, args.threads),
        }
    finally:
        set_num_threads(previous_threads)
    results["elapsed_seconds"] = time.time() - started
    results["passed"] = bool(
        results["flat_prediction"]["passed"]
        and results["block_binning"]["passed"]
        and results["training"]["passed"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
