#!/usr/bin/env python3
"""Measure a benchmark-local packed histogram against post-dispatch DarkoFit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np
from numba import njit, prange
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL = ROOT / "benchmarks" / "q_post_dispatch_microprototype.md"
DEFAULT_RAW = ROOT / "benchmarks" / "q_post_dispatch_microprototype_raw_20260723.json"
DEFAULT_RESULT = (
    ROOT / "benchmarks" / "q_post_dispatch_microprototype_result_20260723.md"
)
ROWS = (500_000, 1_000_000)
HOLDOUT_ROWS = 100_000
FEATURES = 24
BINS = 128
DEPTH = 6
ITERATIONS = 40
THREADS = 14
REPEATS = 3
SEED = 20_260_717
ARMS = ("post_dispatch_control", "packed_candidate")
BLOCK_ORDERS = (
    ARMS,
    tuple(reversed(ARMS)),
    ARMS,
)
MAX_GEOMEAN_RATIO = 0.90
MAX_SIZE_RATIO = 1.02
MAX_IQR_OVER_MEDIAN = 0.10
QMAX_CAP = 32_767
WORKER_PREFIX = "Q_POST_DISPATCH_RESULT="


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(value) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def geomean(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
        raise RuntimeError("geometric mean requires positive values")
    return float(np.exp(np.mean(np.log(values))))


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


@njit(cache=True, parallel=True)
def gradient_absmax(grad):
    result = 0.0
    for i in prange(grad.shape[0]):
        result = max(result, abs(grad[i]))
    return result


@njit(cache=True, parallel=True)
def quantize_pack_unit_hessian(grad, inv_scale, qmax, qseed, out):
    """Pack stochastic-rounded signed gradients and exact unit counts."""
    for i in prange(grad.shape[0]):
        z = (qseed + np.uint64(i)) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        uniform = (z & np.uint64(0xFFFFFFFF)) * (1.0 / 4294967296.0)
        qgrad = np.int64(np.floor(grad[i] * inv_scale + uniform))
        qgrad = min(max(qgrad, -qmax), qmax)
        out[i] = (qgrad << np.int64(32)) + np.int64(1)


@njit(cache=True, parallel=True)
def packed_unit_hessian_best_split(
    X_binned,
    packed,
    grad_scale,
    leaf,
    n_leaves,
    histogram,
    n_bins_per_feature,
    l2,
    feat_mask,
    min_child_weight,
    scratch_Gt,
    scratch_Ht,
    scratch_GL,
    scratch_HL,
    scratch_parent,
    scratch_last_positive,
):
    """DarkoFit-row-major packed twin of the fused unit-Hessian kernel."""
    n_samples, n_features = X_binned.shape
    max_bins = histogram.shape[2]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.full(n_features, -1, dtype=np.int64)
    low_mask = np.int64(0xFFFFFFFF)

    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                histogram[f, l, b] = 0
        for i in range(n_samples):
            l = leaf[i]
            b = X_binned[i, f]
            histogram[f, l, b] += packed[i]

        if feat_mask[f] == 0:
            continue
        nb = n_bins_per_feature[f]
        for l in range(n_leaves):
            total = np.int64(0)
            last_positive = -1
            for b in range(nb):
                total += histogram[f, l, b]
                if (histogram[f, l, b] & low_mask) > 0:
                    last_positive = b
            scratch_Gt[f, l] = (total >> np.int64(32)) * grad_scale
            scratch_Ht[f, l] = total & low_mask
            scratch_last_positive[f, l] = last_positive
            scratch_GL[f, l] = 0.0
            scratch_HL[f, l] = 0.0
            parent_denom = scratch_Ht[f, l] + l2
            if scratch_Ht[f, l] > 0.0 and parent_denom > 0.0:
                scratch_parent[f, l] = (
                    scratch_Gt[f, l] * scratch_Gt[f, l] / parent_denom
                )
            else:
                scratch_parent[f, l] = 0.0

        best_gain = -np.inf
        best_threshold = -1
        for threshold in range(nb - 1):
            gain = 0.0
            legal = True
            any_nonempty = False
            for l in range(n_leaves):
                cell = histogram[f, l, threshold]
                scratch_GL[f, l] += (
                    (cell >> np.int64(32)) * grad_scale
                )
                scratch_HL[f, l] += cell & low_mask
                if scratch_Ht[f, l] > 0.0:
                    any_nonempty = True
                    hl = scratch_HL[f, l]
                    hr = scratch_Ht[f, l] - hl
                    if (
                        hl <= 0.0
                        or threshold >= scratch_last_positive[f, l]
                    ):
                        continue
                    left_denom = hl + l2
                    right_denom = hr + l2
                    parent_denom = scratch_Ht[f, l] + l2
                    if (
                        hl < min_child_weight
                        or hr < min_child_weight
                        or hr <= 0.0
                        or left_denom <= 0.0
                        or right_denom <= 0.0
                        or parent_denom <= 0.0
                    ):
                        legal = False
                    else:
                        gl = scratch_GL[f, l]
                        gr = scratch_Gt[f, l] - gl
                        gain += (
                            gl * gl / left_denom
                            + gr * gr / right_denom
                            - scratch_parent[f, l]
                        )
            if legal and any_nonempty and gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        feat_gain[f] = best_gain
        feat_thr[f] = best_threshold

    best_feature = 0
    best_gain = -np.inf
    for f in range(n_features):
        if feat_gain[f] > best_gain:
            best_gain = feat_gain[f]
            best_feature = f
    return best_feature, feat_thr[best_feature], best_gain


def qmax_for_rows(n_rows: int) -> int:
    return min(QMAX_CAP, (2**31 - 1) // max(int(n_rows), 1))


@contextmanager
def packed_kernel_patch():
    import darkofit.tree as tree

    original = tree._build_histograms_unit_hess_and_best_split
    state = {
        "packed_levels": 0,
        "quantized_trees": 0,
        "quantization_seconds": 0.0,
        "kernel_seconds": 0.0,
        "qmax": None,
        "gradient_scale": None,
        "packed": None,
        "histogram": None,
    }

    def replacement(
        X_binned,
        grad,
        leaf,
        n_leaves,
        hg,
        hh,
        n_bins_per_feature,
        l2,
        feat_mask,
        min_child_weight,
        scratch_Gt,
        scratch_Ht,
        scratch_GL,
        scratch_HL,
        scratch_parent,
        scratch_last_positive,
    ):
        if state["histogram"] is None:
            state["histogram"] = np.zeros(hg.shape, dtype=np.int64)
        if n_leaves == 1:
            started = time.perf_counter()
            n_rows = grad.shape[0]
            qmax = qmax_for_rows(n_rows)
            gmax = gradient_absmax(grad)
            inv_scale = qmax / gmax if gmax > 0.0 else 0.0
            grad_scale = gmax / qmax if gmax > 0.0 else 0.0
            if state["packed"] is None or state["packed"].shape != grad.shape:
                state["packed"] = np.empty(grad.shape, dtype=np.int64)
            tree_index = int(state["quantized_trees"])
            quantize_pack_unit_hessian(
                grad,
                inv_scale,
                np.int64(qmax),
                np.uint64(SEED + tree_index),
                state["packed"],
            )
            state["qmax"] = qmax
            state["gradient_scale"] = grad_scale
            state["quantized_trees"] = tree_index + 1
            state["quantization_seconds"] += time.perf_counter() - started
        if state["packed"] is None:
            raise RuntimeError("packed kernel reached a child before its root")
        started = time.perf_counter()
        result = packed_unit_hessian_best_split(
            X_binned,
            state["packed"],
            state["gradient_scale"],
            leaf,
            n_leaves,
            state["histogram"],
            n_bins_per_feature,
            l2,
            feat_mask,
            min_child_weight,
            scratch_Gt,
            scratch_Ht,
            scratch_GL,
            scratch_HL,
            scratch_parent,
            scratch_last_positive,
        )
        state["kernel_seconds"] += time.perf_counter() - started
        state["packed_levels"] += 1
        return result

    tree._build_histograms_unit_hess_and_best_split = replacement
    try:
        yield state
    finally:
        tree._build_histograms_unit_hess_and_best_split = original


def make_data(train_rows: int):
    rows = train_rows + HOLDOUT_ROWS
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(rows, FEATURES))
    signal = (
        1.4 * X[:, 0]
        - 0.9 * X[:, 1]
        + 0.35 * X[:, 2] * X[:, 3]
        + 0.2 * X[:, 4] ** 2
    )
    y = signal + rng.normal(0.0, 0.5, rows)
    return X[:train_rows], y[:train_rows], X[train_rows:], y[train_rows:]


def model_params(*, iterations: int, kernel: str):
    return {
        "iterations": iterations,
        "loss": "RMSE",
        "learning_rate": 0.1,
        "depth": DEPTH,
        "l2_leaf_reg": 1.0,
        "max_bins": BINS,
        "min_child_samples": 1,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample": 1.0,
        "ordered_boosting": False,
        "tree_mode": "catboost",
        "linear_leaves": False,
        "early_stopping": False,
        "use_best_model": False,
        "eval_train_loss": False,
        "diagnostic_warnings": "never",
        "thread_count": THREADS,
        "random_state": 4,
        "oblivious_kernel": kernel,
    }


def fitted_fingerprint(model) -> str:
    payload = []
    for fitted_tree in model.model_.trees_:
        payload.extend(np.asarray(fitted_tree.splits_feat, dtype="<i8").tobytes())
        payload.extend(np.asarray(fitted_tree.splits_thr, dtype="<i8").tobytes())
        payload.extend(np.asarray(fitted_tree.values, dtype="<f8").tobytes())
    return hashlib.sha256(bytes(payload)).hexdigest()


def fit_once(arm: str, train_rows: int):
    from darkofit import DarkoRegressor

    ambient_threads = numba.get_num_threads()
    X, y, X_holdout, y_holdout = make_data(train_rows)
    warm_rows = 8_192
    if arm == "packed_candidate":
        with packed_kernel_patch():
            DarkoRegressor(**model_params(iterations=2, kernel="fused")).fit(
                X[:warm_rows], y[:warm_rows]
            )
        with packed_kernel_patch() as state:
            model = DarkoRegressor(
                **model_params(iterations=ITERATIONS, kernel="fused")
            )
            started = time.perf_counter()
            model.fit(X, y)
            fit_seconds = time.perf_counter() - started
            packed_stats = {
                key: value
                for key, value in state.items()
                if key not in {"packed", "histogram"}
            }
    else:
        expected = "fused" if train_rows == 500_000 else "unfused"
        DarkoRegressor(**model_params(iterations=2, kernel=expected)).fit(
            X[:warm_rows], y[:warm_rows]
        )
        model = DarkoRegressor(
            **model_params(iterations=ITERATIONS, kernel="auto")
        )
        started = time.perf_counter()
        model.fit(X, y)
        fit_seconds = time.perf_counter() - started
        packed_stats = None

    started = time.perf_counter()
    prediction = model.predict(X_holdout)
    predict_seconds = time.perf_counter() - started
    dispatch = model.model_.oblivious_kernel_dispatch_
    return {
        "arm": arm,
        "train_rows": train_rows,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "holdout_rmse": float(mean_squared_error(
            y_holdout, prediction
        ) ** 0.5),
        "prediction_sha256": array_sha256(prediction),
        "fitted_sha256": fitted_fingerprint(model),
        "tree_count": len(model.model_.trees_),
        "retained_levels": sum(tree.depth for tree in model.model_.trees_),
        "dispatch": dispatch,
        "packed": packed_stats,
        "numba_threads_before": ambient_threads,
        "numba_threads_after": numba.get_num_threads(),
    }


def source_state(expected_sha: str):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    )
    if head != expected_sha:
        raise RuntimeError(f"source SHA mismatch: expected {expected_sha}, got {head}")
    if status:
        raise RuntimeError("source tree must be clean before benchmark execution")
    return head


def worker_command(args, arm: str, train_rows: int):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--arm",
        arm,
        "--train-rows",
        str(train_rows),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.worker_timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"worker failed ({completed.returncode}): {completed.stderr}"
        )
    lines = [
        line[len(WORKER_PREFIX):]
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(lines) != 1:
        raise RuntimeError("worker did not emit exactly one result")
    row = json.loads(lines[0])
    row["worker_stderr"] = completed.stderr
    return row


def iqr_over_median(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    q1, q3 = np.percentile(values, [25.0, 75.0])
    return float((q3 - q1) / median)


def analyze(rows):
    expected = {
        (train_rows, block, arm)
        for train_rows in ROWS
        for block in range(REPEATS)
        for arm in ARMS
    }
    actual = [
        (int(row["train_rows"]), int(row["block"]), row["arm"])
        for row in rows
    ]
    if len(actual) != len(set(actual)):
        raise RuntimeError("duplicate benchmark rows")
    if set(actual) != expected:
        raise RuntimeError("benchmark rows do not match the exact grid")

    sizes = {}
    integrity = {
        "no_worker_stderr": all(not row["worker_stderr"] for row in rows),
        "tree_counts": all(row["tree_count"] == ITERATIONS for row in rows),
        "thread_restore": all(
            row["numba_threads_after"] == row["numba_threads_before"]
            for row in rows
        ),
        "packed_engaged": True,
        "packed_bounds": True,
        "dispatch": True,
        "candidate_deterministic": True,
    }
    for train_rows in ROWS:
        selected = [row for row in rows if row["train_rows"] == train_rows]
        by_key = {(row["block"], row["arm"]): row for row in selected}
        fit_ratios = [
            by_key[(block, "packed_candidate")]["fit_seconds"]
            / by_key[(block, "post_dispatch_control")]["fit_seconds"]
            for block in range(REPEATS)
        ]
        quality_ratios = [
            by_key[(block, "packed_candidate")]["holdout_rmse"]
            / by_key[(block, "post_dispatch_control")]["holdout_rmse"]
            for block in range(REPEATS)
        ]
        packed_rows = [
            row for row in selected if row["arm"] == "packed_candidate"
        ]
        control_rows = [
            row for row in selected if row["arm"] == "post_dispatch_control"
        ]
        expected_control = "fused" if train_rows == 500_000 else "unfused"
        integrity["dispatch"] &= all(
            row["dispatch"]["requested"] == "auto"
            and row["dispatch"]["resolved"] == expected_control
            for row in control_rows
        )
        integrity["dispatch"] &= all(
            row["dispatch"]["requested"] == "fused"
            and row["dispatch"]["resolved"] == "fused"
            for row in packed_rows
        )
        qmax = qmax_for_rows(train_rows)
        integrity["packed_bounds"] &= (
            train_rows < 2**32
            and train_rows * qmax <= 2**31 - 1
            and all(row["packed"]["qmax"] == qmax for row in packed_rows)
        )
        integrity["packed_engaged"] &= all(
            row["packed"]["packed_levels"] == row["retained_levels"]
            and row["packed"]["quantized_trees"] == ITERATIONS
            for row in packed_rows
        )
        integrity["candidate_deterministic"] &= (
            len({row["prediction_sha256"] for row in packed_rows}) == 1
            and len({row["fitted_sha256"] for row in packed_rows}) == 1
        )
        sizes[str(train_rows)] = {
            "fit_ratios": fit_ratios,
            "paired_median_fit_ratio": float(np.median(fit_ratios)),
            "fit_iqr_over_median": iqr_over_median(fit_ratios),
            "quality_ratios": quality_ratios,
            "quality_geomean_ratio": geomean(quality_ratios),
            "predict_geomean_ratio": geomean(
                by_key[(block, "packed_candidate")]["predict_seconds"]
                / by_key[(block, "post_dispatch_control")]["predict_seconds"]
                for block in range(REPEATS)
            ),
            "rss_geomean_ratio": geomean(
                by_key[(block, "packed_candidate")]["peak_rss_bytes"]
                / by_key[(block, "post_dispatch_control")]["peak_rss_bytes"]
                for block in range(REPEATS)
            ),
            "control_dispatch": expected_control,
            "qmax": qmax,
        }
    integrity["passed"] = all(bool(value) for value in integrity.values())
    size_ratios = [
        sizes[str(train_rows)]["paired_median_fit_ratio"]
        for train_rows in ROWS
    ]
    geomean_ratio = geomean(size_ratios)
    speed = {
        "equal_size_geomean_fit_ratio": geomean_ratio,
        "all_sizes_at_most_1_02": max(size_ratios) <= MAX_SIZE_RATIO,
        "geomean_at_most_0_90": geomean_ratio <= MAX_GEOMEAN_RATIO,
        "stable": all(
            sizes[str(train_rows)]["fit_iqr_over_median"]
            <= MAX_IQR_OVER_MEDIAN
            for train_rows in ROWS
        ),
    }
    funded = integrity["passed"] and all(
        bool(value) for value in speed.values()
        if isinstance(value, bool)
    )
    return {
        "sizes": sizes,
        "integrity": integrity,
        "speed": speed,
        "q1_funded": funded,
        "disposition": (
            "fund_private_q1_design"
            if funded
            else "close_q_at_microprototype"
        ),
    }


def render(artifact, raw_hash: str) -> str:
    analysis = artifact["analysis"]
    lines = [
        "# Q post-dispatch packed-histogram microprototype result",
        "",
        "Normal development evidence; no holdout ship-check or TabArena data was consulted.",
        "",
        f"- Source: `{artifact['source_sha']}`",
        f"- Integrity: `{str(analysis['integrity']['passed']).lower()}`",
        f"- Q1 funded: `{str(analysis['q1_funded']).lower()}`",
        f"- Disposition: `{analysis['disposition']}`",
        "",
        "| Rows | Control dispatch | Fit ratio | IQR / median | RMSE ratio | Predict ratio | RSS ratio |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for train_rows in ROWS:
        item = analysis["sizes"][str(train_rows)]
        lines.append(
            f"| {train_rows:,} | {item['control_dispatch']} | "
            f"{item['paired_median_fit_ratio']:.6f} | "
            f"{item['fit_iqr_over_median']:.6f} | "
            f"{item['quality_geomean_ratio']:.6f} | "
            f"{item['predict_geomean_ratio']:.6f} | "
            f"{item['rss_geomean_ratio']:.6f} |"
        )
    lines.extend([
        "",
        "Equal-size geomean fit ratio: "
        f"`{analysis['speed']['equal_size_geomean_fit_ratio']:.6f}` "
        "(funding bar `<= 0.90`).",
        "",
        "The candidate is benchmark-local and changes split selection through "
        "stochastic gradient quantization. Leaf values remain float64. This "
        "result cannot ship an option or default.",
        "",
        f"Raw SHA-256: `{raw_hash}`",
        "",
    ])
    return "\n".join(lines)


def run(args):
    raw_path = Path(args.raw_output)
    result_path = Path(args.result_output)
    if raw_path.exists() or result_path.exists():
        raise FileExistsError("Q outputs are create-only")
    head = source_state(args.expected_source_sha)
    rows = []
    for train_rows in ROWS:
        for block, order in enumerate(BLOCK_ORDERS):
            for position, arm in enumerate(order):
                row = worker_command(args, arm, train_rows)
                row.update({"block": block, "position": position})
                rows.append(row)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha": head,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "numba": numba.__version__,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "threads": THREADS,
        },
        "rows": rows,
    }
    artifact["analysis"] = analyze(rows)
    raw_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    raw_hash = sha256(raw_path)
    result_path.write_text(render(artifact, raw_hash))
    print(json.dumps(artifact["analysis"], indent=2, sort_keys=True))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW))
    parser.add_argument("--result-output", default=str(DEFAULT_RESULT))
    parser.add_argument("--worker-timeout", type=int, default=1_800)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--train-rows", type=int, choices=ROWS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        if args.arm is None or args.train_rows is None:
            raise SystemExit("--worker requires --arm and --train-rows")
        print(
            WORKER_PREFIX
            + json.dumps(fit_once(args.arm, args.train_rows), sort_keys=True)
        )
        return
    if not args.expected_source_sha:
        raise SystemExit("--expected-source-sha is required")
    run(args)


if __name__ == "__main__":
    main()
