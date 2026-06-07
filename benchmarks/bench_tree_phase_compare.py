"""Compare tree-build phase time across ChimeraBoost revisions.

This is a narrow diagnostic harness for the best-of-both-worlds audit. The
main revision benchmark can record phase timings only for revisions that expose
``verbose_timing``. Upstream bbstats v2 does not, so this script instruments the
imported revision directly in an isolated subprocess by wrapping
``chimeraboost.booster.build_oblivious_tree`` before fitting.
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
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    from benchmark_adapters import (
        DATASETS,
        FitConfig,
        RevisionSpec,
        build_dataset,
        estimator_kwargs,
        make_sample_weight,
        split_case,
    )
    from bench_compare_revisions import (
        _best_iteration,
        _json_default,
        _load_case,
        _path_token,
        _prepare_revision_import,
        _save_case,
        _truncate_error,
        _validation_eval_set,
    )
    from weighted_metrics import metric_bundle
except ImportError:  # pragma: no cover - supports module execution
    from benchmarks.benchmark_adapters import (
        DATASETS,
        FitConfig,
        RevisionSpec,
        build_dataset,
        estimator_kwargs,
        make_sample_weight,
        split_case,
    )
    from benchmarks.bench_compare_revisions import (
        _best_iteration,
        _json_default,
        _load_case,
        _path_token,
        _prepare_revision_import,
        _save_case,
        _truncate_error,
        _validation_eval_set,
    )
    from benchmarks.weighted_metrics import metric_bundle


CSV_FIELDS = [
    "status",
    "error",
    "variant",
    "revision_path",
    "tree_mode",
    "dataset",
    "task",
    "loss",
    "alpha",
    "size",
    "seed",
    "weight_mode",
    "validation_weight_policy",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "fit_seconds",
    "fit_repeat_seconds",
    "tree_build_seconds",
    "tree_build_repeat_seconds",
    "tree_build_calls",
    "tree_build_repeat_calls",
    "tree_build_share",
    "tree_build_ms_per_call",
    "hist_seconds",
    "hist_repeat_seconds",
    "hist_calls",
    "split_seconds",
    "split_repeat_seconds",
    "split_calls",
    "leaf_seconds",
    "leaf_repeat_seconds",
    "leaf_calls",
    "linear_leaf_seconds",
    "linear_leaf_repeat_seconds",
    "linear_leaf_calls",
    "tree_other_seconds",
    "tree_other_repeat_seconds",
    "boost_seconds",
    "best_iteration",
    "primary_metric",
    "primary_value",
    "rmse",
    "pinball",
    "f1_macro",
    "log_loss",
    "weighted_rmse",
    "weighted_pinball",
    "weighted_f1_macro",
    "weighted_log_loss",
]


def _install_tree_timer():
    import chimeraboost.booster as bm
    import chimeraboost.tree as tm

    state = {
        "seconds": 0.0,
        "calls": 0,
        "hist_seconds": 0.0,
        "hist_calls": 0,
        "split_seconds": 0.0,
        "split_calls": 0,
        "leaf_seconds": 0.0,
        "leaf_calls": 0,
        "linear_leaf_seconds": 0.0,
        "linear_leaf_calls": 0,
    }
    original = bm.build_oblivious_tree

    def timed_build(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            state["seconds"] += time.perf_counter() - start
            state["calls"] += 1

    bm.build_oblivious_tree = timed_build

    def wrap(name, seconds_key, calls_key):
        if not hasattr(tm, name):
            return
        original_fn = getattr(tm, name)

        def timed_fn(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original_fn(*args, **kwargs)
            finally:
                state[seconds_key] += time.perf_counter() - start
                state[calls_key] += 1

        setattr(tm, name, timed_fn)

    for name in (
        "_build_histograms_into",
        "_build_histograms_unit_hess_into",
        "_build_histograms_selected_into",
        "_build_histograms_selected_unit_hess_into",
        "_build_histograms_rows_into",
        "_build_histograms_rows_unit_hess_into",
        "_build_histograms_selected_rows_into",
        "_build_histograms_selected_rows_unit_hess_into",
    ):
        wrap(name, "hist_seconds", "hist_calls")
    wrap("_best_split", "split_seconds", "split_calls")
    for name in (
        "_leaf_values",
        "_leaf_values_rows",
        "_leaf_values_hs",
        "_leaf_values_hs_rows",
    ):
        wrap(name, "leaf_seconds", "leaf_calls")
    wrap("_linear_leaf_fit", "linear_leaf_seconds", "linear_leaf_calls")
    return state


def _reset_timer(timer):
    for key in list(timer):
        timer[key] = 0.0 if key.endswith("seconds") else 0


def _tree_other_seconds(timer):
    sub = (
        timer["hist_seconds"]
        + timer["split_seconds"]
        + timer["leaf_seconds"]
        + timer["linear_leaf_seconds"]
    )
    return timer["seconds"] - sub


def _fit_worker(payload):
    variant = RevisionSpec(**payload["variant"])
    config = FitConfig(**payload["fit_config"])
    data = _load_case(payload["data_path"])
    _prepare_revision_import(variant.path)

    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

    timer = _install_tree_timer()
    task = payload["task"]
    estimator_cls = (
        ChimeraBoostRegressor
        if task in ("regression", "quantile")
        else ChimeraBoostClassifier
    )
    kwargs = estimator_kwargs(estimator_cls, config, variant, payload["seed"])
    accepted = set(inspect.signature(estimator_cls.__init__).parameters)
    loss = payload.get("loss")
    alpha = payload.get("alpha")
    if loss is not None:
        if "loss" not in accepted:
            raise TypeError(
                f"{estimator_cls.__name__} does not support loss={loss!r}")
        kwargs["loss"] = loss
    if alpha is not None:
        if "alpha" not in accepted:
            raise TypeError(
                f"{estimator_cls.__name__} does not support alpha={alpha!r}")
        kwargs["alpha"] = alpha

    fit_args = (data["X_fit"], data["y_fit"])
    fit_kwargs = {
        "cat_features": payload["cat_features"],
        "eval_set": _validation_eval_set(data, variant, config),
    }
    if data["w_fit"] is not None:
        fit_kwargs["sample_weight"] = data["w_fit"]

    best_model = None
    best_fit = None
    best_tree = None
    best_calls = None
    fit_repeats = []
    tree_repeats = []
    call_repeats = []
    hist_repeats = []
    split_repeats = []
    leaf_repeats = []
    linear_leaf_repeats = []
    other_repeats = []
    repeat = max(1, int(payload["repeat"]))
    for _ in range(repeat):
        model = estimator_cls(**kwargs)
        _reset_timer(timer)
        start = time.perf_counter()
        model.fit(*fit_args, **fit_kwargs)
        fit_seconds = time.perf_counter() - start
        tree_seconds = timer["seconds"]
        tree_calls = timer["calls"]
        fit_repeats.append(fit_seconds)
        tree_repeats.append(tree_seconds)
        call_repeats.append(tree_calls)
        hist_repeats.append(timer["hist_seconds"])
        split_repeats.append(timer["split_seconds"])
        leaf_repeats.append(timer["leaf_seconds"])
        linear_leaf_repeats.append(timer["linear_leaf_seconds"])
        other_repeats.append(_tree_other_seconds(timer))
        if best_fit is None or fit_seconds < best_fit:
            best_model = model
            best_fit = fit_seconds
            best_tree = tree_seconds
            best_calls = tree_calls
            best_timer = dict(timer)

    pred = best_model.predict(data["X_test"])
    if task in ("regression", "quantile"):
        proba = None
    else:
        proba = best_model.predict_proba(data["X_test"])
    labels = getattr(best_model, "classes_", None)
    metrics = metric_bundle(
        task,
        data["y_test"],
        pred,
        proba=proba,
        labels=labels,
        sample_weight=data["w_test"],
        alpha=payload.get("alpha"),
    )
    boost_seconds = getattr(getattr(best_model, "model_", None), "fit_time_", None)
    row = {
        "status": "ok",
        "error": "",
        "fit_seconds": float(best_fit),
        "fit_repeat_seconds": ";".join(f"{v:.12g}" for v in fit_repeats),
        "tree_build_seconds": float(best_tree),
        "tree_build_repeat_seconds": ";".join(f"{v:.12g}" for v in tree_repeats),
        "tree_build_calls": int(best_calls),
        "tree_build_repeat_calls": ";".join(str(int(v)) for v in call_repeats),
        "tree_build_share": float(best_tree / best_fit) if best_fit else "",
        "tree_build_ms_per_call": (
            float(1000.0 * best_tree / best_calls) if best_calls else ""
        ),
        "hist_seconds": float(best_timer["hist_seconds"]),
        "hist_repeat_seconds": ";".join(f"{v:.12g}" for v in hist_repeats),
        "hist_calls": int(best_timer["hist_calls"]),
        "split_seconds": float(best_timer["split_seconds"]),
        "split_repeat_seconds": ";".join(f"{v:.12g}" for v in split_repeats),
        "split_calls": int(best_timer["split_calls"]),
        "leaf_seconds": float(best_timer["leaf_seconds"]),
        "leaf_repeat_seconds": ";".join(f"{v:.12g}" for v in leaf_repeats),
        "leaf_calls": int(best_timer["leaf_calls"]),
        "linear_leaf_seconds": float(best_timer["linear_leaf_seconds"]),
        "linear_leaf_repeat_seconds": ";".join(
            f"{v:.12g}" for v in linear_leaf_repeats
        ),
        "linear_leaf_calls": int(best_timer["linear_leaf_calls"]),
        "tree_other_seconds": float(_tree_other_seconds(best_timer)),
        "tree_other_repeat_seconds": ";".join(f"{v:.12g}" for v in other_repeats),
        "boost_seconds": "" if boost_seconds is None else float(boost_seconds),
        "best_iteration": _best_iteration(best_model) or "",
    }
    row.update(metrics)
    return row


def _worker_main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.payload).read_text())
    try:
        row = _fit_worker(payload)
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--payload")
    parser.add_argument("--upstream", required=False)
    parser.add_argument("--candidate", required=False, default=".")
    parser.add_argument("--models", nargs="+", default=[
        "upstream_matched",
        "candidate_catboost",
    ])
    parser.add_argument("--datasets", nargs="+", default=[
        "categorical_reg",
        "categorical_binary",
        "numeric_binary",
        "quantile_reg_10",
    ])
    parser.add_argument("--sizes", nargs="+", default=["medium"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--weight-modes", nargs="+", default=["none", "stress"])
    parser.add_argument(
        "--validation-weight-policy",
        choices=["product", "upstream-compatible"],
        default="upstream-compatible",
    )
    parser.add_argument("--csv")
    args = parser.parse_args(argv)
    if args.worker:
        _worker_main(["--payload", args.payload])
        return
    if not args.upstream:
        raise SystemExit("--upstream is required unless --worker is used")
    if not args.csv:
        raise SystemExit("--csv is required")

    specs = {
        "upstream_matched": RevisionSpec("upstream_matched", args.upstream),
        "candidate_catboost": RevisionSpec(
            "candidate_catboost", args.candidate, tree_mode="catboost"
        ),
    }
    unknown = sorted(set(args.models) - set(specs))
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}")
    variants = [specs[name] for name in args.models]

    fit_config = FitConfig(
        iterations=args.iterations,
        patience=args.patience,
        threads=args.threads,
        validation_weight_policy=args.validation_weight_policy,
    )
    out_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cb-tree-phase-") as td, out_path.open(
        "w", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for dataset in args.datasets:
            for size in args.sizes:
                for seed in range(args.seeds):
                    spec, X, y, cat_features = build_dataset(dataset, size, seed)
                    for weight_mode in args.weight_modes:
                        sample_weight = make_sample_weight(y, spec.task, weight_mode)
                        split = split_case(
                            X, y, spec.task, seed, sample_weight=sample_weight
                        )
                        data_path = Path(td) / f"{_path_token(dataset, size, seed, weight_mode)}.npz"
                        _save_case(data_path, split)
                        for variant in variants:
                            payload = {
                                "variant": asdict(variant),
                                "fit_config": asdict(fit_config),
                                "data_path": str(data_path),
                                "task": spec.task,
                                "loss": spec.loss,
                                "alpha": spec.alpha,
                                "cat_features": cat_features,
                                "seed": seed,
                                "repeat": args.repeat,
                            }
                            payload_path = Path(td) / f"{_path_token(dataset, size, seed, weight_mode, variant.label)}.json"
                            payload_path.write_text(json.dumps(payload, default=_json_default))
                            row = _run_worker(payload_path)
                            full = {
                                "variant": variant.label,
                                "revision_path": variant.path,
                                "tree_mode": variant.tree_mode or "",
                                "dataset": dataset,
                                "task": spec.task,
                                "loss": spec.loss or "",
                                "alpha": "" if spec.alpha is None else spec.alpha,
                                "size": size,
                                "seed": seed,
                                "weight_mode": weight_mode,
                                "validation_weight_policy": args.validation_weight_policy,
                                "n_train": split["n_train"],
                                "n_val": split["n_val"],
                                "n_test": split["n_test"],
                                "n_features": split["n_features"],
                            }
                            full.update(row)
                            writer.writerow({k: full.get(k, "") for k in CSV_FIELDS})
                            fh.flush()
                            print(
                                f"{full.get('status')} {variant.label:20s} "
                                f"{dataset:22s} seed={seed} weights={weight_mode}",
                                flush=True,
                            )
    print(f"wrote tree phase rows to {out_path}")


if __name__ == "__main__":
    main()
