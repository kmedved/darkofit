"""Compare full estimator fit phases across ChimeraBoost revisions.

This is the estimator-level companion to ``bench_tree_phase_compare.py``. It
wraps upstream/candidate internals in isolated subprocesses so older upstream
revisions do not need to expose ``verbose_timing``. The goal is to explain
blocker-manifest timing gaps after direct tree/context replays have tied.
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
        _case_selected,
        _json_default,
        _load_case,
        _load_case_manifest,
        _manifest_axis,
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
        _case_selected,
        _json_default,
        _load_case,
        _load_case_manifest,
        _manifest_axis,
        _path_token,
        _prepare_revision_import,
        _save_case,
        _truncate_error,
        _validation_eval_set,
    )
    from benchmarks.weighted_metrics import metric_bundle


TIMER_KEYS = (
    "preprocess_fit_transform",
    "preprocess_transform",
    "grad_hess",
    "tree_build",
    "hist",
    "split",
    "leaf",
    "linear_leaf",
    "tree_predict",
    "forest_pack",
    "forest_predict",
    "loss_eval",
    "temperature",
)


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
    "split_mode",
    "weight_mode",
    "validation_weight_policy",
    "ensemble_size",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "fit_seconds",
    "fit_repeat_seconds",
    "predict_seconds",
    "predict_repeat_seconds",
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
    "preprocess_fit_transform_seconds",
    "preprocess_transform_seconds",
    "grad_hess_seconds",
    "tree_build_seconds",
    "hist_seconds",
    "split_seconds",
    "leaf_seconds",
    "linear_leaf_seconds",
    "tree_predict_seconds",
    "forest_pack_seconds",
    "forest_predict_seconds",
    "loss_eval_seconds",
    "temperature_seconds",
    "fit_residual_seconds",
    "preprocess_fit_transform_calls",
    "preprocess_transform_calls",
    "grad_hess_calls",
    "tree_build_calls",
    "hist_calls",
    "split_calls",
    "leaf_calls",
    "linear_leaf_calls",
    "tree_predict_calls",
    "forest_pack_calls",
    "forest_predict_calls",
    "loss_eval_calls",
    "temperature_calls",
]


def _blank_timer():
    state = {}
    for key in TIMER_KEYS:
        state[f"{key}_seconds"] = 0.0
        state[f"{key}_calls"] = 0
    return state


def _reset_timer(timer):
    for key in list(timer):
        timer[key] = 0.0 if key.endswith("_seconds") else 0


def _timed(timer, key, fn):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            timer[f"{key}_seconds"] += time.perf_counter() - start
            timer[f"{key}_calls"] += 1

    return wrapper


def _wrap_attr(obj, name, timer, key):
    if hasattr(obj, name):
        setattr(obj, name, _timed(timer, key, getattr(obj, name)))


def _install_phase_timer():
    import chimeraboost.booster as bm
    import chimeraboost.losses as lm
    import chimeraboost.preprocessing as pm
    import chimeraboost.sklearn_api as sm
    import chimeraboost.tree as tm

    timer = _blank_timer()

    _wrap_attr(pm.FeaturePreprocessor, "fit_transform", timer, "preprocess_fit_transform")
    _wrap_attr(pm.FeaturePreprocessor, "transform", timer, "preprocess_transform")

    for cls_name in ("RMSE", "Logloss", "MAE", "Quantile", "MultiSoftmax"):
        cls = getattr(lm, cls_name, None)
        if cls is None:
            continue
        for name in ("grad_hess", "grad_hess_class_major"):
            _wrap_attr(cls, name, timer, "grad_hess")
        for name in ("eval", "eval_class_major"):
            _wrap_attr(cls, name, timer, "loss_eval")

    for cls_name in ("ObliviousTree", "LevelwiseTree", "MultiLevelwiseTree"):
        cls = getattr(tm, cls_name, None)
        if cls is not None:
            _wrap_attr(cls, "predict", timer, "tree_predict")

    _wrap_attr(sm, "_fit_temperature", timer, "temperature")

    _wrap_attr(bm, "pack_forest", timer, "forest_pack")
    _wrap_attr(bm, "pack_forest_linear", timer, "forest_pack")
    _wrap_attr(bm, "_predict_forest", timer, "forest_predict")
    _wrap_attr(bm, "_predict_forest_linear", timer, "forest_predict")

    original_build = bm.build_oblivious_tree

    def timed_build(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_build(*args, **kwargs)
        finally:
            timer["tree_build_seconds"] += time.perf_counter() - start
            timer["tree_build_calls"] += 1

    bm.build_oblivious_tree = timed_build

    def wrap_tree_fn(name, key):
        if hasattr(tm, name):
            setattr(tm, name, _timed(timer, key, getattr(tm, name)))

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
        wrap_tree_fn(name, "hist")
    wrap_tree_fn("_best_split", "split")
    for name in (
        "_leaf_values",
        "_leaf_values_rows",
        "_leaf_values_hs",
        "_leaf_values_hs_rows",
    ):
        wrap_tree_fn(name, "leaf")
    wrap_tree_fn("_linear_leaf_fit", "linear_leaf")
    return timer


def _fit_worker(payload):
    variant = RevisionSpec(**payload["variant"])
    config = FitConfig(**payload["fit_config"])
    data = _load_case(payload["data_path"])
    _prepare_revision_import(variant.path)

    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

    timer = _install_phase_timer()
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
    best_timer = None
    fit_repeats = []
    repeat = max(1, int(payload["repeat"]))
    for _ in range(repeat):
        model = estimator_cls(**kwargs)
        _reset_timer(timer)
        start = time.perf_counter()
        model.fit(*fit_args, **fit_kwargs)
        elapsed = time.perf_counter() - start
        fit_repeats.append(elapsed)
        if best_fit is None or elapsed < best_fit:
            best_model = model
            best_fit = elapsed
            best_timer = dict(timer)

    def predict_once():
        start = time.perf_counter()
        pred = best_model.predict(data["X_test"])
        if task in ("regression", "quantile"):
            proba = None
        else:
            proba = best_model.predict_proba(data["X_test"])
        return pred, proba, time.perf_counter() - start

    pred, proba, predict_seconds = predict_once()
    predict_repeats = [predict_seconds]
    for _ in range(repeat - 1):
        cand_pred, cand_proba, elapsed = predict_once()
        predict_repeats.append(elapsed)
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
    measured = sum(best_timer[f"{key}_seconds"] for key in TIMER_KEYS)
    row = {
        "status": "ok",
        "error": "",
        "fit_seconds": float(best_fit),
        "fit_repeat_seconds": ";".join(f"{v:.12g}" for v in fit_repeats),
        "predict_seconds": float(predict_seconds),
        "predict_repeat_seconds": ";".join(f"{v:.12g}" for v in predict_repeats),
        "best_iteration": _best_iteration(best_model) or "",
        "fit_residual_seconds": float(best_fit - measured),
    }
    for key in TIMER_KEYS:
        row[f"{key}_seconds"] = float(best_timer[f"{key}_seconds"])
        row[f"{key}_calls"] = int(best_timer[f"{key}_calls"])
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
    parser.add_argument("--case-manifest", type=Path, default=None)
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
        "candidate_matched": RevisionSpec("candidate_matched", args.candidate),
        "candidate_catboost": RevisionSpec(
            "candidate_catboost", args.candidate, tree_mode="catboost"
        ),
    }
    unknown = sorted(set(args.models) - set(specs))
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}")
    variants = [specs[name] for name in args.models]
    manifest = _load_case_manifest(args.case_manifest)
    datasets = _manifest_axis(manifest, "dataset", args.datasets)
    sizes = _manifest_axis(manifest, "size", args.sizes)
    seeds = _manifest_axis(manifest, "seed", range(args.seeds))
    weight_modes = _manifest_axis(manifest, "weight_mode", args.weight_modes)

    fit_config = FitConfig(
        iterations=args.iterations,
        patience=args.patience,
        threads=args.threads,
        validation_weight_policy=args.validation_weight_policy,
    )
    out_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cb-estimator-phase-") as td, out_path.open(
        "w", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for dataset in datasets:
            for size in sizes:
                for seed in seeds:
                    spec, X, y, cat_features = build_dataset(dataset, size, seed)
                    for weight_mode in weight_modes:
                        if not _case_selected(
                            manifest,
                            dataset=dataset,
                            size=size,
                            seed=seed,
                            split_mode="row",
                            weight_mode=weight_mode,
                        ):
                            continue
                        sample_weight = make_sample_weight(y, spec.task, weight_mode)
                        split = split_case(
                            X, y, spec.task, seed, sample_weight=sample_weight)
                        data_path = Path(td) / (
                            _path_token(dataset, size, seed, weight_mode) + ".npz"
                        )
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
                            payload_path = Path(td) / (
                                _path_token(dataset, size, seed, weight_mode, variant.label)
                                + ".json"
                            )
                            payload_path.write_text(
                                json.dumps(payload, default=_json_default))
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
                                "split_mode": "row",
                                "weight_mode": weight_mode,
                                "validation_weight_policy": args.validation_weight_policy,
                                "ensemble_size": 1,
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
    print(f"wrote estimator phase rows to {out_path}")


if __name__ == "__main__":
    main()
