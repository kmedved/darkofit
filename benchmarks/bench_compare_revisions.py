"""Compare multiple ChimeraBoost revisions in isolated subprocesses.

This is the benchmark-first integration harness for the fork/upstream merge
work. It compares current bbstats upstream, the local fork, and an integration
candidate without importing two ``chimeraboost`` packages in one Python process.

Example
-------
    python benchmarks/bench_compare_revisions.py \
      --upstream /tmp/chimeraboost-upstream \
      --fork /tmp/chimeraboost-fork \
      --candidate . \
      --sizes medium large \
      --datasets numeric_binary numeric_multiclass categorical_binary \
      --seeds 3 --repeat 2 --threads 8 \
      --csv benchmarks/tri_compare_raw.csv
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
        SIZE_SAMPLES,
        FitConfig,
        RevisionSpec,
        build_dataset,
        default_revision_specs,
        estimator_kwargs,
        make_sample_weight,
        policy_suite_specs,
        split_case,
    )
    from weighted_metrics import metric_bundle
except ImportError:  # pragma: no cover - supports `python -m benchmarks...`
    from benchmarks.benchmark_adapters import (
        DATASETS,
        SIZE_SAMPLES,
        FitConfig,
        RevisionSpec,
        build_dataset,
        default_revision_specs,
        estimator_kwargs,
        make_sample_weight,
        policy_suite_specs,
        split_case,
    )
    from benchmarks.weighted_metrics import metric_bundle


CSV_FIELDS = [
    "status",
    "error",
    "variant",
    "revision_path",
    "tree_mode",
    "selected_tree_mode",
    "max_bins",
    "sampling",
    "top_rate",
    "other_rate",
    "use_defaults",
    "dataset",
    "task",
    "size",
    "seed",
    "weight_mode",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "fit_seconds",
    "predict_seconds",
    "boost_seconds",
    "selection_overhead_seconds",
    "timing_scope",
    "best_iteration",
    "primary_metric",
    "primary_value",
    "rmse",
    "mae",
    "r2",
    "accuracy",
    "f1_macro",
    "log_loss",
    "brier",
    "weighted_rmse",
    "weighted_mae",
    "weighted_r2",
    "weighted_accuracy",
    "weighted_f1_macro",
    "weighted_log_loss",
    "weighted_brier",
    "timing_preprocess",
    "timing_grad_hess",
    "timing_tree_build",
    "timing_train_update",
    "timing_validation_predict",
    "timing_loss_eval",
]


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _save_case(path, split):
    has_weights = split["w_fit"] is not None
    np.savez_compressed(
        path,
        X_fit=split["X_fit"],
        X_val=split["X_val"],
        X_test=split["X_test"],
        y_fit=split["y_fit"],
        y_val=split["y_val"],
        y_test=split["y_test"],
        w_fit=split["w_fit"] if has_weights else np.array([], dtype=np.float64),
        w_val=split["w_val"] if has_weights else np.array([], dtype=np.float64),
        w_test=split["w_test"] if has_weights else np.array([], dtype=np.float64),
        has_weights=np.array([has_weights], dtype=np.bool_),
    )


def _load_case(path):
    data = np.load(path, allow_pickle=True)
    has_weights = bool(data["has_weights"][0])
    return {
        "X_fit": data["X_fit"],
        "X_val": data["X_val"],
        "X_test": data["X_test"],
        "y_fit": data["y_fit"],
        "y_val": data["y_val"],
        "y_test": data["y_test"],
        "w_fit": data["w_fit"] if has_weights else None,
        "w_val": data["w_val"] if has_weights else None,
        "w_test": data["w_test"] if has_weights else None,
    }


def _truncate_error(text):
    text = str(text or "")
    return text if len(text) <= 4000 else text[:3997] + "..."


def _prepare_revision_import(revision_path):
    repo_root = Path(__file__).resolve().parents[1]
    resolved = str(Path(revision_path).resolve())
    sys.modules.pop("chimeraboost", None)
    sys.modules.pop("chimeraboost.sklearn_api", None)
    sys.path = [
        p for p in sys.path
        if p and str(Path(p).resolve()) not in {str(repo_root), resolved}
    ]
    sys.path.insert(0, resolved)


def _best_iteration(model):
    value = getattr(model, "best_iteration_", None)
    if value is None:
        value = getattr(model, "best_iteration", None)
    if callable(value):
        value = value()
    return None if value is None else int(value)


def _timing_fields(model):
    timing = getattr(model, "timing_", None)
    if timing is None:
        timing = getattr(getattr(model, "model_", None), "timing_", None)
    out = {}
    for key in (
        "preprocess",
        "grad_hess",
        "tree_build",
        "train_update",
        "validation_predict",
        "loss_eval",
    ):
        out[f"timing_{key}"] = "" if not timing else float(timing.get(key, 0.0))
    return out


def _selection_timing_fields(model, fit_seconds, boost_seconds):
    selected_tree_mode = getattr(getattr(model, "model_", None), "tree_mode_", "")
    tree_mode_selection = getattr(model, "tree_mode_selection_", None)
    out = {
        "selected_tree_mode": selected_tree_mode,
        "selection_overhead_seconds": "",
        "timing_scope": "fit_model",
    }
    if tree_mode_selection is not None:
        out["timing_scope"] = "selected_model"
        if boost_seconds is not None:
            out["selection_overhead_seconds"] = max(
                0.0, float(fit_seconds) - float(boost_seconds)
            )
    return out


def _fit_worker(payload):
    variant = RevisionSpec(**payload["variant"])
    config = FitConfig(**payload["fit_config"])
    data = _load_case(payload["data_path"])
    _prepare_revision_import(variant.path)

    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

    task = payload["task"]
    estimator_cls = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    kwargs = estimator_kwargs(estimator_cls, config, variant, payload["seed"])
    if "sampling" in kwargs:
        kwargs["sampling"] = _effective_sampling(task, config)
    fit_params = set(inspect.signature(estimator_cls.fit).parameters)
    cat_features = payload["cat_features"]
    repeat = max(1, int(payload["repeat"]))

    if variant.use_defaults:
        X_train = np.concatenate([data["X_fit"], data["X_val"]], axis=0)
        y_train = np.concatenate([data["y_fit"], data["y_val"]], axis=0)
        if data["w_fit"] is None:
            w_train = None
        else:
            w_train = np.concatenate([data["w_fit"], data["w_val"]], axis=0)
        fit_args = (X_train, y_train)
        fit_kwargs = {"cat_features": cat_features}
        if w_train is not None:
            fit_kwargs["sample_weight"] = w_train
    else:
        fit_args = (data["X_fit"], data["y_fit"])
        fit_kwargs = {
            "cat_features": cat_features,
            "eval_set": (data["X_val"], data["y_val"]),
        }
        if data["w_fit"] is not None:
            fit_kwargs["sample_weight"] = data["w_fit"]
            if "eval_sample_weight" in fit_params:
                fit_kwargs["eval_sample_weight"] = data["w_val"]

    best_model = None
    fit_seconds = None
    for _ in range(repeat):
        model = estimator_cls(**kwargs)
        start = time.perf_counter()
        model.fit(*fit_args, **fit_kwargs)
        elapsed = time.perf_counter() - start
        if fit_seconds is None or elapsed < fit_seconds:
            best_model = model
            fit_seconds = elapsed

    def predict_once():
        start = time.perf_counter()
        pred = best_model.predict(data["X_test"])
        if task == "regression":
            proba = None
        else:
            proba = best_model.predict_proba(data["X_test"])
        return pred, proba, time.perf_counter() - start

    pred, proba, predict_seconds = predict_once()
    for _ in range(repeat - 1):
        cand_pred, cand_proba, elapsed = predict_once()
        if elapsed < predict_seconds:
            pred, proba, predict_seconds = cand_pred, cand_proba, elapsed

    labels = getattr(best_model, "classes_", None)
    metrics = metric_bundle(
        task,
        data["y_test"],
        pred,
        proba=proba,
        labels=labels,
        sample_weight=data["w_test"],
    )
    boost_seconds = getattr(getattr(best_model, "model_", None), "fit_time_", None)
    row = {
        "status": "ok",
        "error": "",
        "max_bins": kwargs.get("max_bins", ""),
        "sampling": kwargs.get("sampling", "uniform"),
        "top_rate": kwargs.get("top_rate", ""),
        "other_rate": kwargs.get("other_rate", ""),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "boost_seconds": "" if boost_seconds is None else float(boost_seconds),
        "best_iteration": _best_iteration(best_model) or "",
    }
    row.update(_selection_timing_fields(best_model, fit_seconds, boost_seconds))
    row.update(metrics)
    row.update(_timing_fields(best_model))
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
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--payload", str(payload_path)]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return {
            "status": "error",
            "error": _truncate_error(proc.stderr or proc.stdout),
        }
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "status": "error",
            "error": _truncate_error(f"worker returned invalid JSON:\n{proc.stdout}\n{proc.stderr}"),
        }


def _effective_sampling(task, config):
    if task == "multiclass" and config.sampling == "goss":
        return "uniform"
    return config.sampling


def _base_row(variant, spec, size, seed, weight_mode, split, config):
    if variant.use_defaults:
        n_train = split["n_train"] + split["n_val"]
        n_val = 0
    else:
        n_train = split["n_train"]
        n_val = split["n_val"]
    return {
        "variant": variant.label,
        "revision_path": str(Path(variant.path).resolve()),
        "tree_mode": variant.tree_mode or "",
        "max_bins": config.max_bins,
        "sampling": _effective_sampling(spec.task, config),
        "top_rate": config.top_rate,
        "other_rate": config.other_rate,
        "use_defaults": variant.use_defaults,
        "dataset": spec.name,
        "task": spec.task,
        "size": size,
        "seed": seed,
        "weight_mode": weight_mode,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": split["n_test"],
        "n_features": split["n_features"],
    }


def _complete_row(row):
    return {field: row.get(field, "") for field in CSV_FIELDS}


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=None)
    parser.add_argument("--fork", type=Path, default=None)
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument(
        "--policy-suite",
        choices=["revision", "default-regret"],
        default="revision",
        help=(
            "'revision' preserves the historical upstream/fork/candidate "
            "comparison. 'default-regret' runs named policies for one candidate "
            "checkout so benchmarks can be summarized by default regret."
        ),
    )
    parser.add_argument("--sizes", nargs="+", choices=SIZE_SAMPLES, default=["tiny"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["friedman_numeric", "numeric_binary", "categorical_binary"],
        help="dataset names, or 'all'",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=1_500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--max-bins", type=int, default=128)
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--min-gain-to-split", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--sampling",
        choices=["uniform", "goss"],
        default="uniform",
        help=(
            "ChimeraBoost row sampling policy. Experimental 'goss' applies "
            "to scalar regression/binary rows; multiclass rows stay uniform."
        ),
    )
    parser.add_argument("--top-rate", type=float, default=0.2)
    parser.add_argument("--other-rate", type=float, default=0.1)
    parser.add_argument("--ordered-boosting", action="store_true", default=False)
    parser.add_argument(
        "--weight-modes",
        nargs="+",
        choices=["none", "uniform", "stress"],
        default=["none", "uniform"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="optional variant labels to run after expansion",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmarks/tri_compare_raw.csv"),
    )
    parser.add_argument("--keep-case-files", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    if args.policy_suite == "default-regret":
        if not args.candidate:
            raise SystemExit("--policy-suite default-regret requires --candidate")
        variants = policy_suite_specs(str(args.candidate), suite=args.policy_suite)
    else:
        variants = default_revision_specs(
            upstream=str(args.upstream) if args.upstream else None,
            fork=str(args.fork) if args.fork else None,
            candidate=str(args.candidate) if args.candidate else None,
        )
    if args.models:
        wanted = set(args.models)
        variants = [v for v in variants if v.label in wanted]
        missing = wanted - {v.label for v in variants}
        if missing:
            raise SystemExit(f"unknown or unavailable variants: {sorted(missing)}")
    if not variants:
        raise SystemExit("no revisions to run; pass --upstream/--fork/--candidate")

    datasets = list(DATASETS) if args.datasets == ["all"] else list(args.datasets)
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise SystemExit(f"unknown dataset(s): {unknown}; known: {sorted(DATASETS)}")

    config = FitConfig(
        iterations=args.iterations,
        patience=args.patience,
        depth=args.depth,
        max_bins=args.max_bins,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        threads=args.threads,
        ordered_boosting=args.ordered_boosting,
        min_child_samples=args.min_child_samples,
        min_gain_to_split=args.min_gain_to_split,
        sampling=args.sampling,
        top_rate=args.top_rate,
        other_rate=args.other_rate,
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    if args.keep_case_files:
        tmp_ctx = None
        tmpdir = Path(tempfile.mkdtemp(prefix="cb-revision-bench-", dir=None))
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="cb-revision-bench-", dir=None)
        tmpdir = Path(tmp_ctx.name)
    try:
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            fh.flush()
            for size in args.sizes:
                for dataset in datasets:
                    for seed in range(args.seeds):
                        spec, X, y, cat_features = build_dataset(dataset, size, seed)
                        for weight_mode in args.weight_modes:
                            weights = make_sample_weight(y, spec.task, weight_mode)
                            split = split_case(X, y, spec.task, seed, weights)
                            data_path = tmpdir / f"{dataset}-{size}-{seed}-{weight_mode}.npz"
                            _save_case(data_path, split)
                            for variant in variants:
                                payload = {
                                    "variant": asdict(variant),
                                    "fit_config": asdict(config),
                                    "data_path": str(data_path),
                                    "task": spec.task,
                                    "cat_features": cat_features,
                                    "seed": seed,
                                    "repeat": args.repeat,
                                }
                                payload_path = tmpdir / f"payload-{variant.label}-{dataset}-{size}-{seed}-{weight_mode}.json"
                                payload_path.write_text(json.dumps(payload, default=_json_default))
                                row = _base_row(
                                    variant, spec, size, seed, weight_mode,
                                    split, config,
                                )
                                row.update(_run_worker(payload_path))
                                writer.writerow(_complete_row(row))
                                fh.flush()
                                print(
                                    f"{row['status']:5s} {variant.label:24s} "
                                    f"{dataset:23s} {size:6s} seed={seed} "
                                    f"weights={weight_mode}",
                                    flush=True,
                                )
    finally:
        if args.keep_case_files:
            print(f"kept case files in {tmpdir}")
        else:
            tmp_ctx.cleanup()

    print(f"wrote raw rows to {args.csv}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2:])
    else:
        main()
