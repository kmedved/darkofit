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
import resource
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
        make_groups,
        make_sample_weight,
        register_external_datasets,
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
        make_groups,
        make_sample_weight,
        register_external_datasets,
        split_case,
    )
    from benchmarks.weighted_metrics import metric_bundle


CSV_FIELDS = [
    "status",
    "error",
    "variant",
    "revision_path",
    "tree_mode",
    "use_defaults",
    "dataset",
    "task",
    "loss",
    "alpha",
    "size",
    "seed",
    "split_mode",
    "weight_mode",
    "ensemble_size",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "n_groups_train",
    "n_groups_val",
    "n_groups_test",
    "fit_seconds",
    "predict_seconds",
    "peak_rss_mb",
    "boost_seconds",
    "best_iteration",
    "primary_metric",
    "primary_value",
    "rmse",
    "mae",
    "r2",
    "pinball",
    "coverage",
    "accuracy",
    "f1_macro",
    "log_loss",
    "brier",
    "weighted_rmse",
    "weighted_mae",
    "weighted_r2",
    "weighted_pinball",
    "weighted_coverage",
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


def _path_token(*parts):
    text = "-".join(str(part) for part in parts)
    return "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in text
    )


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


def _peak_rss_mb():
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _fit_worker(payload):
    variant = RevisionSpec(**payload["variant"])
    config = FitConfig(**payload["fit_config"])
    data = _load_case(payload["data_path"])
    _prepare_revision_import(variant.path)

    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

    task = payload["task"]
    estimator_cls = (
        ChimeraBoostRegressor
        if task in ("regression", "quantile")
        else ChimeraBoostClassifier
    )
    kwargs = estimator_kwargs(estimator_cls, config, variant, payload["seed"])
    if not variant.use_defaults:
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
        eval_set = (data["X_val"], data["y_val"])
        if data["w_val"] is not None and variant.label.startswith("candidate"):
            eval_set = (data["X_val"], data["y_val"], data["w_val"])
        fit_kwargs = {
            "cat_features": cat_features,
            "eval_set": eval_set,
        }
        if data["w_fit"] is not None:
            fit_kwargs["sample_weight"] = data["w_fit"]

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
        if task in ("regression", "quantile"):
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
        alpha=payload.get("alpha"),
    )
    boost_seconds = getattr(getattr(best_model, "model_", None), "fit_time_", None)
    row = {
        "status": "ok",
        "error": "",
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "peak_rss_mb": _peak_rss_mb(),
        "boost_seconds": "" if boost_seconds is None else float(boost_seconds),
        "best_iteration": _best_iteration(best_model) or "",
    }
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


def _base_row(
    variant,
    spec,
    size,
    seed,
    split_mode,
    weight_mode,
    ensemble_size,
    split,
):
    return {
        "variant": variant.label,
        "revision_path": str(Path(variant.path).resolve()),
        "tree_mode": variant.tree_mode or "",
        "use_defaults": variant.use_defaults,
        "dataset": spec.name,
        "task": spec.task,
        "loss": "default" if variant.use_defaults else (spec.loss or ""),
        "alpha": (
            "" if variant.use_defaults or spec.alpha is None
            else spec.alpha
        ),
        "size": size,
        "seed": seed,
        "split_mode": split_mode,
        "weight_mode": weight_mode,
        "ensemble_size": "default" if variant.use_defaults else ensemble_size,
        "n_train": split["n_train"],
        "n_val": split["n_val"],
        "n_test": split["n_test"],
        "n_features": split["n_features"],
        "n_groups_train": split.get("n_groups_train", ""),
        "n_groups_val": split.get("n_groups_val", ""),
        "n_groups_test": split.get("n_groups_test", ""),
    }


def _complete_row(row):
    return {field: row.get(field, "") for field in CSV_FIELDS}


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=None)
    parser.add_argument("--fork", type=Path, default=None)
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--sizes", nargs="+", choices=SIZE_SAMPLES, default=["tiny"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["friedman_numeric", "numeric_binary", "categorical_binary"],
        help="dataset names, or 'all'",
    )
    parser.add_argument(
        "--openml",
        action="store_true",
        help="register the opt-in OpenML real-tabular dataset suite",
    )
    parser.add_argument(
        "--grinsztajn",
        action="store_true",
        help="register the opt-in Grinsztajn real-tabular dataset suite",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=1_500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--ensemble-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="single-model size 1 plus optional bag sizes to pass as n_ensembles",
    )
    parser.add_argument(
        "--ensemble-n-jobs",
        type=int,
        default=1,
        help="parallel jobs for bagged ensemble members when supported",
    )
    parser.add_argument(
        "--max-bins-ts",
        type=int,
        default=None,
        help="optional target-stat encoded column bin cap for revisions that support it",
    )
    parser.add_argument(
        "--weighted-target-stats",
        action="store_true",
        default=False,
        help="let sample weights affect ordered target-stat encodings when supported",
    )
    parser.add_argument("--ordered-boosting", action="store_true", default=False)
    parser.add_argument(
        "--weight-modes",
        nargs="+",
        choices=["none", "uniform", "stress"],
        default=["none", "uniform"],
    )
    parser.add_argument(
        "--split-modes",
        nargs="+",
        choices=["row", "group"],
        default=["row"],
        help="row = ordinary random splits; group = hold out whole synthetic groups",
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

    register_external_datasets(
        args.datasets,
        include_openml=args.openml,
        include_grinsztajn=args.grinsztajn,
    )
    datasets = list(DATASETS) if args.datasets == ["all"] else list(args.datasets)
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise SystemExit(f"unknown dataset(s): {unknown}; known: {sorted(DATASETS)}")
    if any(size < 1 for size in args.ensemble_sizes):
        raise SystemExit("--ensemble-sizes values must be positive integers")
    if args.ensemble_n_jobs == 0:
        raise SystemExit("--ensemble-n-jobs must be nonzero")

    base_config = FitConfig(
        iterations=args.iterations,
        patience=args.patience,
        depth=args.depth,
        learning_rate=args.learning_rate,
        ensemble_n_jobs=args.ensemble_n_jobs,
        max_bins_ts=args.max_bins_ts,
        weighted_target_stats=args.weighted_target_stats,
        threads=args.threads,
        ordered_boosting=args.ordered_boosting,
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)

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
                        for split_mode in args.split_modes:
                            groups = (
                                make_groups(len(y), seed)
                                if split_mode == "group"
                                else None
                            )
                            for weight_mode in args.weight_modes:
                                weights = make_sample_weight(y, spec.task, weight_mode)
                                split = split_case(
                                    X, y, spec.task, seed, weights, groups=groups)
                                data_path = tmpdir / (
                                    _path_token(
                                        dataset, size, seed, split_mode,
                                        weight_mode,
                                    ) + ".npz"
                                )
                                _save_case(data_path, split)
                                for variant in variants:
                                    ensemble_sizes = (
                                        ["default"]
                                        if variant.use_defaults
                                        else args.ensemble_sizes
                                    )
                                    for ensemble_size in ensemble_sizes:
                                        n_ensembles = (
                                            None
                                            if ensemble_size in ("default", 1)
                                            else int(ensemble_size)
                                        )
                                        config = FitConfig(
                                            **{
                                                **asdict(base_config),
                                                "n_ensembles": n_ensembles,
                                            }
                                        )
                                        payload = {
                                            "variant": asdict(variant),
                                            "fit_config": asdict(config),
                                            "data_path": str(data_path),
                                            "task": spec.task,
                                            "loss": spec.loss,
                                            "alpha": spec.alpha,
                                            "cat_features": cat_features,
                                            "seed": seed,
                                            "repeat": args.repeat,
                                        }
                                        ensemble_token = str(ensemble_size)
                                        payload_path = tmpdir / (
                                            _path_token(
                                                "payload", variant.label,
                                                dataset, size, seed, split_mode,
                                                weight_mode,
                                                f"ens{ensemble_token}",
                                            ) + ".json"
                                        )
                                        payload_path.write_text(
                                            json.dumps(payload, default=_json_default)
                                        )
                                        row = _base_row(
                                            variant, spec, size, seed,
                                            split_mode, weight_mode,
                                            ensemble_size, split)
                                        row.update(_run_worker(payload_path))
                                        writer.writerow(_complete_row(row))
                                        fh.flush()
                                        print(
                                            f"{row['status']:5s} "
                                            f"{variant.label:24s} "
                                            f"{dataset:23s} {size:6s} "
                                            f"seed={seed} split={split_mode} "
                                            f"weights={weight_mode} "
                                            f"ensemble={row['ensemble_size']}",
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
