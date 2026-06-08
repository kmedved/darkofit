"""Replay real scalar booster contexts across ChimeraBoost revisions.

The random kernel microbench isolates low-level tree kernels, while the revision
benchmark times full estimators. This script sits between them: it builds the
same benchmark split as ``bench_compare_revisions.py``, lets each revision
preprocess it with its own product code, then replays a fixed number of scalar
boosting rounds while timing gradient, tree, histogram, split, and update work.
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
        FitConfig,
        RevisionSpec,
        build_dataset,
        make_sample_weight,
        split_case,
    )
    from bench_compare_revisions import (
        _case_selected,
        _json_default,
        _load_case,
        _load_case_manifest,
        _manifest_axis,
        _path_token,
        _prepare_revision_import,
        _save_case,
        _truncate_error,
    )
except ImportError:  # pragma: no cover - supports module execution
    from benchmarks.benchmark_adapters import (
        FitConfig,
        RevisionSpec,
        build_dataset,
        make_sample_weight,
        split_case,
    )
    from benchmarks.bench_compare_revisions import (
        _case_selected,
        _json_default,
        _load_case,
        _load_case_manifest,
        _manifest_axis,
        _path_token,
        _prepare_revision_import,
        _save_case,
        _truncate_error,
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
    "weight_mode",
    "n_train",
    "n_features_raw",
    "n_features_binned",
    "max_bins",
    "threads",
    "repeat",
    "boost_iterations",
    "linear_leaves",
    "constant_hessian",
    "leaf_estimation_iterations",
    "lr",
    "min_child_weight",
    "boost_seconds",
    "boost_repeat_seconds",
    "grad_seconds",
    "tree_seconds",
    "update_seconds",
    "tree_build_seconds",
    "hist_seconds",
    "split_seconds",
    "leaf_seconds",
    "linear_leaf_seconds",
    "tree_other_seconds",
    "tree_build_calls",
    "hist_calls",
    "split_calls",
    "leaf_calls",
    "linear_leaf_calls",
    "depth_sum",
    "raw_checksum",
    "layout_summary",
    "signature_summary",
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
    original_build = bm.build_oblivious_tree

    def timed_build(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_build(*args, **kwargs)
        finally:
            state["seconds"] += time.perf_counter() - start
            state["calls"] += 1

    bm.build_oblivious_tree = timed_build

    def wrap(name, seconds_key, calls_key):
        if not hasattr(tm, name):
            return
        original = getattr(tm, name)

        def timed_fn(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
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
    wrap("_best_split_no_sparse_veto", "split_seconds", "split_calls")
    wrap("_best_split_v2", "split_seconds", "split_calls")
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


def _timer_other(timer):
    return (
        timer["seconds"]
        - timer["hist_seconds"]
        - timer["split_seconds"]
        - timer["leaf_seconds"]
        - timer["linear_leaf_seconds"]
    )


def _accepted_kwargs(cls, kwargs):
    accepted = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}


def _array_summary(name, arr):
    arr = np.asarray(arr)
    return {
        "name": name,
        "dtype": str(arr.dtype),
        "shape": tuple(int(v) for v in arr.shape),
        "strides": tuple(int(v) for v in arr.strides),
        "c_contig": bool(arr.flags.c_contiguous),
        "f_contig": bool(arr.flags.f_contiguous),
    }


def _signature_summary(tree_module):
    names = (
        "_build_histograms_into",
        "_build_histograms_unit_hess_into",
        "_best_split",
        "_best_split_no_sparse_veto",
        "_leaf_values",
        "_linear_leaf_fit",
    )
    out = {}
    for name in names:
        fn = getattr(tree_module, name, None)
        out[name] = [str(sig) for sig in getattr(fn, "signatures", [])]
    return out


def _build_model(payload, booster_module):
    loss = payload["loss"]
    kwargs = {
        "n_estimators": int(payload["iterations"]),
        "learning_rate": payload.get("learning_rate"),
        "depth": int(payload["depth"]),
        "l2_leaf_reg": float(payload["l2_leaf_reg"]),
        "max_bins": int(payload["max_bins"]),
        "max_bins_ts": payload.get("max_bins_ts"),
        "subsample": 1.0,
        "colsample": 1.0,
        "cat_smoothing": 1.0,
        "cat_n_permutations": 4,
        "early_stopping_rounds": int(payload["patience"]),
        "min_child_weight": float(payload["min_child_weight"]),
        "thread_count": int(payload["threads"]),
        "random_state": int(payload["seed"]),
        "verbose": False,
        "ordered_boosting": bool(payload["ordered_boosting"]),
        "cat_combinations": False,
        "leaf_estimation_iterations": int(payload["leaf_estimation_iterations"]),
        "hs_lambda": 0.0,
        "linear_leaves": bool(payload["linear_leaves"]),
        "linear_lambda": 1.0,
        "tree_mode": payload.get("tree_mode") or "catboost",
        "verbose_timing": False,
        "weighted_target_stats": False,
    }
    loss_kwargs = {"alpha": payload["alpha"]} if loss == "Quantile" else {}
    cls = booster_module.GradientBoosting
    return cls(loss=loss, loss_kwargs=loss_kwargs, **_accepted_kwargs(cls, kwargs))


def _worker(payload):
    _prepare_revision_import(payload["revision_path"])
    import chimeraboost.booster as bm
    import chimeraboost.tree as tm

    if hasattr(bm, "_apply_thread_count"):
        bm._apply_thread_count(int(payload["threads"]))
    else:
        import numba

        numba.set_num_threads(int(payload["threads"]))

    timer = _install_tree_timer()
    data = _load_case(payload["data_path"])
    cat_features = payload["cat_features"]
    X = (
        np.asarray(data["X_fit"], dtype=object)
        if cat_features
        else np.asarray(data["X_fit"], dtype=np.float64)
    )
    y = np.asarray(data["y_fit"], dtype=np.float64)
    w = None if data["w_fit"] is None else np.asarray(data["w_fit"], dtype=np.float64)

    model = _build_model(payload, bm)
    model.loss_ = bm.LOSSES[payload["loss"]](**(
        {"alpha": payload["alpha"]} if payload["loss"] == "Quantile" else {}
    ))
    if hasattr(model, "_normalize_weights"):
        w = model._normalize_weights(w, len(y))
    elif w is not None:
        w = w * (len(y) / w.sum())
    model.lr_ = (
        model._resolve_lr(len(y), eval_set=True)
        if hasattr(model, "_resolve_lr")
        else (0.1 if payload.get("learning_rate") is None else payload["learning_rate"])
    )
    model.prep_ = model._new_preprocessor()
    ts_weight = w if getattr(model, "weighted_target_stats", False) else None
    prep_kwargs = {}
    if "sample_weight" in inspect.signature(model.prep_.fit_transform).parameters:
        prep_kwargs["sample_weight"] = ts_weight
    Xb = np.ascontiguousarray(
        model.prep_.fit_transform(
            X, [y], cat_features, **prep_kwargs).T)
    n_bins = model.prep_.n_bins_
    max_bins = int(n_bins.max()) if len(n_bins) else 1
    hist_buffers = model._alloc_hist_buffers(Xb.shape[0], n_bins)
    use_constant_hessian = (
        getattr(model.loss_, "constant_hessian", False)
        and w is None
        and getattr(model, "subsample", 1.0) >= 1.0
    )
    adjusts_leaves = getattr(model.loss_, "adjusts_leaves", False)
    supports_linear = getattr(model, "supports_linear_leaves", True)
    ll_active = (
        bool(getattr(model, "linear_leaves", False))
        and supports_linear
        and not adjusts_leaves
        and len(y) >= getattr(bm, "LINEAR_LEAVES_MIN_SAMPLES", 1000)
    )
    centers_std = model._build_centers_std(Xb, n_bins) if ll_active else None
    is_numeric = getattr(model.prep_, "is_numeric_binned_", np.ones(Xb.shape[0], dtype=bool))
    builder = model._tree_builder() if hasattr(model, "_tree_builder") else bm.build_oblivious_tree

    accepted = set(inspect.signature(builder).parameters)

    def tree_kwargs():
        kwargs = {
            "feature_mask": None,
            "min_child_weight": float(model.min_child_weight),
            "hist_buffers": hist_buffers,
            "hs_lambda": float(getattr(model, "hs_lambda", 0.0)),
            "linear_leaves": ll_active,
            "centers_std": centers_std,
            "is_numeric": is_numeric,
            "linear_lambda": float(getattr(model, "linear_lambda", 1.0)),
            "constant_hessian": use_constant_hessian,
            "feature_indices": None,
            "row_indices": None,
        }
        return {k: v for k, v in kwargs.items() if k in accepted}

    def update_with_tree(F, tree, leaf, grad, hess):
        if adjusts_leaves:
            if hasattr(model, "_correct_leaves"):
                model._correct_leaves(tree, leaf, y - F, w)
            return F + tree.values[leaf]
        if ll_active and getattr(tree, "lin_coef", None) is not None:
            return F + bm._linear_predict(
                leaf, tree.lin_feats, tree.lin_coef, centers_std, Xb)
        if getattr(model, "ordered_boosting", False) and not adjusts_leaves:
            return F + model._loo_update(tree, leaf, grad, hess)
        for _ in range(int(getattr(model, "leaf_estimation_iterations", 1)) - 1):
            F_tmp = F + tree.values[leaf]
            g2, h2 = model.loss_.grad_hess(y, F_tmp)
            if w is not None:
                g2, h2 = g2 * w, h2 * w
            n_lv = tree.values.shape[0]
            if getattr(model, "hs_lambda", 0.0) > 0.0:
                tree.values += bm._leaf_values_hs(
                    leaf, g2, h2, n_lv, model.l2_leaf_reg, model.lr_, model.hs_lambda)
            else:
                tree.values += bm._leaf_values(
                    leaf, g2, h2, n_lv, model.l2_leaf_reg, model.lr_)
        return F + tree.values[leaf]

    def replay_once():
        _reset_timer(timer)
        F = np.full(len(y), model.loss_.init(y, w), dtype=np.float64)
        depth_sum = 0
        grad_seconds = 0.0
        tree_seconds = 0.0
        update_seconds = 0.0
        last_grad = last_hess = None
        for _ in range(int(payload["boost_iterations"])):
            start = time.perf_counter()
            grad, hess = model.loss_.grad_hess(y, F)
            if w is not None:
                grad, hess = grad * w, hess * w
            grad_seconds += time.perf_counter() - start
            start = time.perf_counter()
            tree, leaf = builder(
                Xb,
                grad,
                hess,
                n_bins,
                int(model.depth),
                float(model.l2_leaf_reg),
                float(model.lr_),
                **tree_kwargs(),
            )
            tree_seconds += time.perf_counter() - start
            depth_sum += int(getattr(tree, "depth", len(tree.splits_feat)))
            last_grad, last_hess = grad, hess
            if tree.depth == 0:
                break
            start = time.perf_counter()
            F = update_with_tree(F, tree, leaf, grad, hess)
            update_seconds += time.perf_counter() - start
        return {
            "depth_sum": depth_sum,
            "raw_checksum": float(F.sum()),
            "grad_seconds": grad_seconds,
            "tree_seconds": tree_seconds,
            "update_seconds": update_seconds,
            "last_grad": last_grad,
            "last_hess": last_hess,
            "timer": dict(timer),
        }

    # Compile/warm once outside measured repeats.
    replay_once()
    repeats = []
    best = None
    for _ in range(max(1, int(payload["repeat"]))):
        start = time.perf_counter()
        result = replay_once()
        elapsed = time.perf_counter() - start
        repeats.append(elapsed)
        if best is None or elapsed < best[0]:
            best = (elapsed, result)
    elapsed, result = best
    t = result["timer"]
    layout = [
        _array_summary("Xb", Xb),
        _array_summary("n_bins", n_bins),
    ]
    if result["last_grad"] is not None:
        layout.append(_array_summary("grad", result["last_grad"]))
    if result["last_hess"] is not None:
        layout.append(_array_summary("hess", result["last_hess"]))

    return {
        "status": "ok",
        "error": "",
        "n_train": len(y),
        "n_features_raw": np.asarray(data["X_fit"]).shape[1],
        "n_features_binned": Xb.shape[0],
        "max_bins": max_bins,
        "threads": int(payload["threads"]),
        "repeat": int(payload["repeat"]),
        "boost_iterations": int(payload["boost_iterations"]),
        "linear_leaves": ll_active,
        "constant_hessian": use_constant_hessian,
        "leaf_estimation_iterations": int(getattr(model, "leaf_estimation_iterations", 1)),
        "lr": float(model.lr_),
        "min_child_weight": float(model.min_child_weight),
        "boost_seconds": float(elapsed),
        "boost_repeat_seconds": ";".join(f"{v:.12g}" for v in repeats),
        "grad_seconds": float(result["grad_seconds"]),
        "tree_seconds": float(result["tree_seconds"]),
        "update_seconds": float(result["update_seconds"]),
        "tree_build_seconds": float(t["seconds"]),
        "hist_seconds": float(t["hist_seconds"]),
        "split_seconds": float(t["split_seconds"]),
        "leaf_seconds": float(t["leaf_seconds"]),
        "linear_leaf_seconds": float(t["linear_leaf_seconds"]),
        "tree_other_seconds": float(_timer_other(t)),
        "tree_build_calls": int(t["calls"]),
        "hist_calls": int(t["hist_calls"]),
        "split_calls": int(t["split_calls"]),
        "leaf_calls": int(t["leaf_calls"]),
        "linear_leaf_calls": int(t["linear_leaf_calls"]),
        "depth_sum": int(result["depth_sum"]),
        "raw_checksum": float(result["raw_checksum"]),
        "layout_summary": json.dumps(layout, sort_keys=True),
        "signature_summary": json.dumps(_signature_summary(tm), sort_keys=True),
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


def _scalar_loss_for_task(task):
    if task == "binary":
        return "Logloss", None
    if task == "regression":
        return "RMSE", None
    if task == "quantile":
        return None, None
    return None, None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--payload")
    parser.add_argument("--upstream")
    parser.add_argument("--candidate", default=".")
    parser.add_argument("--models", nargs="+", default=[
        "upstream_matched",
        "candidate_catboost",
    ])
    parser.add_argument("--datasets", nargs="+", default=[
        "categorical_reg",
        "friedman_numeric",
        "numeric_binary",
    ])
    parser.add_argument("--sizes", nargs="+", default=["medium"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--weight-modes", nargs="+", default=["none", "stress"])
    parser.add_argument("--case-manifest", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--l2-leaf-reg", type=float, default=1.0)
    parser.add_argument("--max-bins", type=int, default=128)
    parser.add_argument("--max-bins-ts", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--boost-iterations", type=int, default=80)
    parser.add_argument("--ordered-boosting", action="store_true", default=False)
    parser.add_argument("--csv", required=False)
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
    manifest = _load_case_manifest(args.case_manifest)
    datasets = _manifest_axis(manifest, "dataset", args.datasets)
    sizes = _manifest_axis(manifest, "size", args.sizes)
    seeds = _manifest_axis(manifest, "seed", range(args.seeds))
    weight_modes = _manifest_axis(manifest, "weight_mode", args.weight_modes)
    fit_config = FitConfig(
        iterations=args.iterations,
        patience=args.patience,
        depth=args.depth,
        learning_rate=args.learning_rate,
        max_bins_ts=args.max_bins_ts,
        threads=args.threads,
        ordered_boosting=args.ordered_boosting,
    )

    out_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cb-context-replay-") as td, out_path.open(
        "w", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for dataset in datasets:
            for size in sizes:
                for seed in seeds:
                    spec, X, y, cat_features = build_dataset(dataset, size, seed)
                    loss, alpha = _scalar_loss_for_task(spec.task)
                    if spec.task == "quantile":
                        loss, alpha = spec.loss, spec.alpha
                    if loss is None or spec.task == "multiclass":
                        continue
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
                            # Mirror sklearn wrapper auto-defaults used by the
                            # matched benchmark rows.
                            linear_leaves = (
                                spec.task == "binary"
                                and (variant.tree_mode in (None, "catboost"))
                            )
                            leaf_iters = 3 if spec.task == "binary" else 1
                            min_child_weight = 1.0
                            payload = {
                                "revision_path": variant.path,
                                "tree_mode": variant.tree_mode,
                                "data_path": str(data_path),
                                "task": spec.task,
                                "loss": loss,
                                "alpha": alpha,
                                "cat_features": cat_features,
                                "seed": seed,
                                "iterations": fit_config.iterations,
                                "patience": fit_config.patience,
                                "depth": fit_config.depth,
                                "learning_rate": fit_config.learning_rate,
                                "l2_leaf_reg": args.l2_leaf_reg,
                                "max_bins": args.max_bins,
                                "max_bins_ts": fit_config.max_bins_ts,
                                "threads": fit_config.threads,
                                "ordered_boosting": fit_config.ordered_boosting,
                                "min_child_weight": min_child_weight,
                                "linear_leaves": linear_leaves,
                                "leaf_estimation_iterations": leaf_iters,
                                "boost_iterations": args.boost_iterations,
                                "repeat": args.repeat,
                            }
                            payload_path = Path(td) / (
                                _path_token(
                                    dataset, size, seed, weight_mode, variant.label
                                ) + ".json"
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
                                "loss": loss,
                                "alpha": "" if alpha is None else alpha,
                                "size": size,
                                "seed": seed,
                                "weight_mode": weight_mode,
                            }
                            full.update(row)
                            writer.writerow({k: full.get(k, "") for k in CSV_FIELDS})
                            fh.flush()
                            print(
                                f"{full.get('status')} {variant.label:20s} "
                                f"{dataset:22s} seed={seed} weights={weight_mode}",
                                flush=True,
                            )
    print(f"wrote context replay rows to {out_path}")


if __name__ == "__main__":
    main()
