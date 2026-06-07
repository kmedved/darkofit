"""Directly compare tree kernel speed across ChimeraBoost revisions.

This is a narrow follow-up to ``bench_tree_phase_compare.py``. The phase
benchmark showed catboost-mode numeric overhead inside histogram fill and
``_best_split``. This script removes estimator fitting from the loop: it builds
one deterministic set of binned arrays, imports each revision in an isolated
subprocess, warms the numba kernels, and then times the same direct kernel and
``build_oblivious_tree`` calls.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from bench_compare_revisions import _json_default, _truncate_error
except ImportError:  # pragma: no cover - supports module execution
    from benchmarks.bench_compare_revisions import _json_default, _truncate_error


CSV_FIELDS = [
    "status",
    "error",
    "variant",
    "revision_path",
    "case",
    "seed",
    "threads",
    "n_samples",
    "n_features",
    "n_leaves",
    "max_bins",
    "repeat",
    "hist_seconds",
    "hist_repeat_seconds",
    "hist_checksum",
    "split_seconds",
    "split_repeat_seconds",
    "split_feature",
    "split_threshold",
    "split_gain",
    "split_v2_seconds",
    "split_v2_repeat_seconds",
    "split_v2_feature",
    "split_v2_threshold",
    "split_v2_gain",
    "split_v2_matches_legacy",
    "build_seconds",
    "build_repeat_seconds",
    "build_depth",
    "build_leaf_checksum",
    "build_value_checksum",
    "build_split_signature",
    "linear_leaves",
    "boost_iterations",
    "boost_seconds",
    "boost_repeat_seconds",
    "boost_grad_seconds",
    "boost_tree_seconds",
    "boost_update_seconds",
    "boost_depth_sum",
    "boost_raw_checksum",
]


@dataclass
class KernelCase:
    name: str
    n_samples: int
    n_features: int
    n_leaves: int
    max_bins: int


def _prepare_revision_import(revision_path):
    repo_root = Path(__file__).resolve().parents[1]
    resolved = str(Path(revision_path).resolve())
    for name in list(sys.modules):
        if name == "chimeraboost" or name.startswith("chimeraboost."):
            sys.modules.pop(name, None)
    sys.path = [
        p for p in sys.path
        if p and str(Path(p).resolve()) not in {str(repo_root), resolved}
    ]
    sys.path.insert(0, resolved)


def _make_case(path, case: KernelCase, seed: int):
    rng = np.random.default_rng(seed)
    Xb = rng.integers(
        0,
        case.max_bins,
        size=(case.n_features, case.n_samples),
        dtype=np.uint8 if case.max_bins <= 256 else np.uint16,
    )
    grad = rng.normal(0.0, 1.0, size=case.n_samples).astype(np.float64)
    hess = rng.lognormal(0.0, 0.15, size=case.n_samples).astype(np.float64)
    logits = (
        0.045 * Xb[: min(6, case.n_features)].astype(np.float64).sum(axis=0)
        - 0.045 * min(6, case.n_features) * (case.max_bins - 1) / 2.0
    )
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob).astype(np.float64)
    leaf = rng.integers(
        0,
        case.n_leaves,
        size=case.n_samples,
        dtype=np.int64,
    )
    n_bins_per_feature = np.full(case.n_features, case.max_bins, dtype=np.int64)
    feat_mask = np.ones(case.n_features, dtype=np.uint8)
    centers_std = rng.normal(
        0.0,
        1.0,
        size=(case.n_features, case.max_bins),
    ).astype(np.float64)
    is_numeric = np.ones(case.n_features, dtype=np.bool_)
    np.savez_compressed(
        path,
        Xb=Xb,
        y=y,
        grad=grad,
        hess=hess,
        leaf=leaf,
        n_bins_per_feature=n_bins_per_feature,
        feat_mask=feat_mask,
        centers_std=centers_std,
        is_numeric=is_numeric,
    )


def _time_min(fn, repeat):
    times = []
    best = None
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if best is None or elapsed < best[0]:
            best = (elapsed, value)
    return best[0], best[1], times


def _worker(payload):
    _prepare_revision_import(payload["revision_path"])
    import numba
    import chimeraboost.tree as tree

    numba.set_num_threads(int(payload["threads"]))
    data = np.load(payload["data_path"])
    Xb = data["Xb"]
    y = data["y"]
    grad = data["grad"]
    hess = data["hess"]
    leaf = data["leaf"]
    n_bins_per_feature = data["n_bins_per_feature"]
    feat_mask = data["feat_mask"]
    centers_std = data["centers_std"]
    is_numeric = data["is_numeric"]
    n_leaves = int(payload["n_leaves"])
    max_bins = int(payload["max_bins"])
    max_depth = int(payload["max_depth"])
    repeat = max(1, int(payload["repeat"]))
    hist = np.empty((Xb.shape[0], n_leaves, max_bins, 2), dtype=np.float64)

    def hist_call():
        tree._build_histograms_into(Xb, grad, hess, leaf, n_leaves, hist)
        return None

    # Compile and populate ``hist`` once before timing split search.
    hist_call()
    split_warm = tree._best_split(
        hist,
        n_bins_per_feature,
        float(payload["l2"]),
        feat_mask,
        float(payload["min_child_weight"]),
        n_leaves,
    )

    hist_seconds, _, hist_repeats = _time_min(hist_call, repeat)
    hist_checksum = float(hist.sum())

    def split_call():
        return tree._best_split(
            hist,
            n_bins_per_feature,
            float(payload["l2"]),
            feat_mask,
            float(payload["min_child_weight"]),
            n_leaves,
        )

    split_seconds, split_result, split_repeats = _time_min(split_call, repeat)
    # Keep the warm result alive in case an over-eager optimizer ever gets cute.
    if split_warm[0] < -1:  # pragma: no cover - impossible guard
        split_result = split_warm
    if hasattr(tree, "_best_split_v2"):
        split_v2_warm = tree._best_split_v2(
            hist,
            n_bins_per_feature,
            float(payload["l2"]),
            feat_mask,
            float(payload["min_child_weight"]),
            n_leaves,
        )

        def split_v2_call():
            return tree._best_split_v2(
                hist,
                n_bins_per_feature,
                float(payload["l2"]),
                feat_mask,
                float(payload["min_child_weight"]),
                n_leaves,
            )

        split_v2_seconds, split_v2_result, split_v2_repeats = _time_min(
            split_v2_call, repeat)
        if split_v2_warm[0] < -1:  # pragma: no cover - impossible guard
            split_v2_result = split_v2_warm
        split_v2_matches = (
            int(split_result[0]) == int(split_v2_result[0])
            and int(split_result[1]) == int(split_v2_result[1])
            and float(split_result[2]) == float(split_v2_result[2])
        )
    else:
        split_v2_seconds = ""
        split_v2_result = ("", "", "")
        split_v2_repeats = []
        split_v2_matches = ""

    def sigmoid(raw):
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -35.0, 35.0)))

    def grad_hess(raw):
        p = sigmoid(raw)
        return p - y, np.maximum(p * (1.0 - p), 1e-6)

    build_hist = np.zeros((Xb.shape[0], 1 << max_depth, max_bins, 2))
    linear_leaves = bool(payload["linear_leaves"])
    build_kwargs = {
        "feature_mask": None,
        "min_child_weight": float(payload["min_child_weight"]),
        "hist_buffers": build_hist,
        "hs_lambda": 0.0,
        "linear_leaves": linear_leaves,
        "centers_std": centers_std if linear_leaves else None,
        "is_numeric": is_numeric,
        "linear_lambda": 1.0,
        "constant_hessian": False,
        "feature_indices": None,
        "row_indices": None,
    }
    accepted = set(inspect.signature(tree.build_oblivious_tree).parameters)
    build_kwargs = {k: v for k, v in build_kwargs.items() if k in accepted}

    def build_call():
        return tree.build_oblivious_tree(
            Xb,
            grad,
            hess,
            n_bins_per_feature,
            max_depth,
            float(payload["l2"]),
            float(payload["learning_rate"]),
            **build_kwargs,
        )

    # Warm the enclosing builder after direct-kernel timing so build-specific
    # numba specializations do not leak into the measured repeats.
    build_call()
    build_seconds, build_result, build_repeats = _time_min(build_call, repeat)
    built_tree, built_leaf = build_result
    split_sig = ",".join(
        f"{int(f)}:{int(t)}"
        for f, t in zip(built_tree.splits_feat, built_tree.splits_thr)
    )

    boost_iterations = int(payload["boost_iterations"])
    boost_hist = np.zeros((Xb.shape[0], 1 << max_depth, max_bins, 2))
    boost_kwargs = dict(build_kwargs)
    boost_kwargs["hist_buffers"] = boost_hist

    def boost_loop():
        p0 = np.clip(y.mean(), 1e-6, 1.0 - 1e-6)
        raw = np.full(y.shape[0], np.log(p0 / (1.0 - p0)), dtype=np.float64)
        depth_sum = 0
        grad_seconds = 0.0
        tree_seconds = 0.0
        update_seconds = 0.0
        for _ in range(boost_iterations):
            start = time.perf_counter()
            g_loop, h_loop = grad_hess(raw)
            grad_seconds += time.perf_counter() - start
            start = time.perf_counter()
            tree_obj, train_leaf = tree.build_oblivious_tree(
                Xb,
                g_loop,
                h_loop,
                n_bins_per_feature,
                max_depth,
                float(payload["l2"]),
                float(payload["learning_rate"]),
                **boost_kwargs,
            )
            tree_seconds += time.perf_counter() - start
            depth_sum += int(getattr(tree_obj, "depth", len(tree_obj.splits_feat)))
            if tree_obj.depth == 0:
                break
            start = time.perf_counter()
            if linear_leaves and getattr(tree_obj, "lin_coef", None) is not None:
                raw += tree._linear_predict(
                    train_leaf,
                    tree_obj.lin_feats,
                    tree_obj.lin_coef,
                    centers_std,
                    Xb,
                )
            else:
                raw += tree_obj.values[train_leaf]
            update_seconds += time.perf_counter() - start
        return depth_sum, float(raw.sum()), grad_seconds, tree_seconds, update_seconds

    # Warm the repeated loop separately; this catches the realistic repeated
    # tree-build call pattern without including its compilation in timings.
    boost_loop()
    boost_seconds, boost_result, boost_repeats = _time_min(boost_loop, repeat)
    return {
        "status": "ok",
        "error": "",
        "hist_seconds": float(hist_seconds),
        "hist_repeat_seconds": ";".join(f"{v:.12g}" for v in hist_repeats),
        "hist_checksum": hist_checksum,
        "split_seconds": float(split_seconds),
        "split_repeat_seconds": ";".join(f"{v:.12g}" for v in split_repeats),
        "split_feature": int(split_result[0]),
        "split_threshold": int(split_result[1]),
        "split_gain": float(split_result[2]),
        "split_v2_seconds": (
            "" if split_v2_seconds == "" else float(split_v2_seconds)
        ),
        "split_v2_repeat_seconds": ";".join(
            f"{v:.12g}" for v in split_v2_repeats),
        "split_v2_feature": (
            "" if split_v2_result[0] == "" else int(split_v2_result[0])
        ),
        "split_v2_threshold": (
            "" if split_v2_result[1] == "" else int(split_v2_result[1])
        ),
        "split_v2_gain": (
            "" if split_v2_result[2] == "" else float(split_v2_result[2])
        ),
        "split_v2_matches_legacy": split_v2_matches,
        "build_seconds": float(build_seconds),
        "build_repeat_seconds": ";".join(f"{v:.12g}" for v in build_repeats),
        "build_depth": int(getattr(built_tree, "depth", len(built_tree.splits_feat))),
        "build_leaf_checksum": int(np.asarray(built_leaf, dtype=np.int64).sum()),
        "build_value_checksum": float(np.asarray(built_tree.values).sum()),
        "build_split_signature": split_sig,
        "linear_leaves": linear_leaves,
        "boost_iterations": boost_iterations,
        "boost_seconds": float(boost_seconds),
        "boost_repeat_seconds": ";".join(f"{v:.12g}" for v in boost_repeats),
        "boost_grad_seconds": float(boost_result[2]),
        "boost_tree_seconds": float(boost_result[3]),
        "boost_update_seconds": float(boost_result[4]),
        "boost_depth_sum": int(boost_result[0]),
        "boost_raw_checksum": float(boost_result[1]),
    }


def _worker_main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.payload).read_text())
    try:
        row = _worker(payload)
    except Exception:
        row = {"status": "error", "error": _truncate_error(traceback.format_exc())}
    print(json.dumps(row, default=_json_default))


def _run_worker(payload_path):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--payload",
        str(payload_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return {"status": "error", "error": _truncate_error(proc.stderr or proc.stdout)}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "status": "error",
            "error": _truncate_error(
                f"worker returned invalid JSON:\n{proc.stdout}\n{proc.stderr}"
            ),
        }


def _case_specs(names):
    specs = {
        "numeric_medium_depth5": KernelCase(
            "numeric_medium_depth5",
            n_samples=6000,
            n_features=40,
            n_leaves=32,
            max_bins=128,
        ),
        "numeric_medium_depth6": KernelCase(
            "numeric_medium_depth6",
            n_samples=6000,
            n_features=40,
            n_leaves=64,
            max_bins=128,
        ),
        "numeric_large_depth5": KernelCase(
            "numeric_large_depth5",
            n_samples=30000,
            n_features=40,
            n_leaves=32,
            max_bins=128,
        ),
        "numeric_large_depth6": KernelCase(
            "numeric_large_depth6",
            n_samples=30000,
            n_features=40,
            n_leaves=64,
            max_bins=128,
        ),
        "wide_medium_depth5": KernelCase(
            "wide_medium_depth5",
            n_samples=6000,
            n_features=120,
            n_leaves=32,
            max_bins=128,
        ),
        "wide_medium_depth6": KernelCase(
            "wide_medium_depth6",
            n_samples=6000,
            n_features=120,
            n_leaves=64,
            max_bins=128,
        ),
        "shallow_medium": KernelCase(
            "shallow_medium",
            n_samples=6000,
            n_features=40,
            n_leaves=4,
            max_bins=128,
        ),
    }
    unknown = sorted(set(names) - set(specs))
    if unknown:
        raise SystemExit(f"unknown case(s): {unknown}")
    return [specs[name] for name in names]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--payload")
    parser.add_argument("--upstream")
    parser.add_argument("--candidate", default=".")
    parser.add_argument("--cases", nargs="+", default=["numeric_medium_depth5"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--linear-leaves", action="store_true")
    parser.add_argument("--boost-iterations", type=int, default=50)
    parser.add_argument("--csv")
    args = parser.parse_args(argv)
    if args.worker:
        _worker_main(["--payload", args.payload])
        return
    if not args.upstream:
        raise SystemExit("--upstream is required unless --worker is used")
    if not args.csv:
        raise SystemExit("--csv is required")

    variants = [
        ("upstream_matched", args.upstream),
        ("candidate_catboost", args.candidate),
    ]
    out_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cb-tree-kernel-") as td, out_path.open(
        "w", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for case in _case_specs(args.cases):
            for seed in range(args.seeds):
                data_path = Path(td) / f"{case.name}-seed{seed}.npz"
                _make_case(data_path, case, seed)
                for label, revision_path in variants:
                    payload = {
                        "revision_path": revision_path,
                        "data_path": str(data_path),
                        "threads": args.threads,
                        "repeat": args.repeat,
                        "n_leaves": case.n_leaves,
                        "max_bins": case.max_bins,
                        "max_depth": int(np.log2(case.n_leaves)),
                        "l2": args.l2,
                        "learning_rate": args.learning_rate,
                        "min_child_weight": args.min_child_weight,
                        "linear_leaves": args.linear_leaves,
                        "boost_iterations": args.boost_iterations,
                    }
                    payload_path = Path(td) / f"{case.name}-seed{seed}-{label}.json"
                    payload_path.write_text(json.dumps(payload, default=_json_default))
                    row = _run_worker(payload_path)
                    full = {
                        "variant": label,
                        "revision_path": revision_path,
                        "case": case.name,
                        "seed": seed,
                        "threads": args.threads,
                        "n_samples": case.n_samples,
                        "n_features": case.n_features,
                        "n_leaves": case.n_leaves,
                        "max_bins": case.max_bins,
                        "repeat": args.repeat,
                    }
                    full.update(row)
                    writer.writerow({k: full.get(k, "") for k in CSV_FIELDS})
                    fh.flush()
                    print(
                        f"{full.get('status')} {label:20s} "
                        f"{case.name:22s} seed={seed}",
                        flush=True,
                    )
    print(f"wrote kernel micro rows to {out_path}")


if __name__ == "__main__":
    main()
