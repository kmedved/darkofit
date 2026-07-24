#!/usr/bin/env python3
"""Compare three ensemble-v3 member recipes on the spent general slice."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from benchmark_adapters import (
        build_dataset,
        make_sample_weight,
        split_case,
    )
    from m6_quality_rule_v3 import (
        DATASETS,
        SEEDS,
        THREADS,
        WEIGHT_MODES,
        quality_decision,
    )
    from weighted_metrics import metric_bundle
except ImportError:  # pragma: no cover
    from benchmarks.benchmark_adapters import (
        build_dataset,
        make_sample_weight,
        split_case,
    )
    from benchmarks.m6_quality_rule_v3 import (
        DATASETS,
        SEEDS,
        THREADS,
        WEIGHT_MODES,
        quality_decision,
    )
    from benchmarks.weighted_metrics import metric_bundle


ROOT = Path(__file__).resolve().parents[1]
SIZE = "medium"
ARMS = {
    "single": None,
    "legacy_auto": {
        "ensemble_member_learning_rate": None,
        "ensemble_member_colsample": 1.0,
    },
    "intermediate": {
        "ensemble_member_learning_rate": 0.125,
        "ensemble_member_colsample": 0.925,
    },
    "current": {
        "ensemble_member_learning_rate": "policy",
        "ensemble_member_colsample": "policy",
    },
}
ENSEMBLE_ARMS = tuple(name for name in ARMS if name != "single")
CURRENT_ARM = "current"
ITERATIONS = 1_500
PATIENCE = 50


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def source_state(expected_sha: str) -> dict[str, object]:
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if head != expected_sha:
        raise RuntimeError(f"source SHA differs: expected {expected_sha}, got {head}")
    if status:
        raise RuntimeError(f"member-policy benchmark requires a clean tree: {status}")
    return {"head": head, "tree": _git("rev-parse", "HEAD^{tree}"), "clean": True}


def _geomean(values) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array <= 0):
        raise RuntimeError("geometric mean inputs must be positive and finite")
    return float(np.exp(np.mean(np.log(array))))


def expected_cell_keys() -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (dataset, int(seed), weight_mode)
        for dataset in DATASETS
        for seed in SEEDS
        for weight_mode in WEIGHT_MODES
    )


def arm_params(name: str) -> dict[str, object]:
    if name not in ARMS:
        raise ValueError(f"unknown arm: {name}")
    common: dict[str, object] = {
        "iterations": ITERATIONS,
        "early_stopping": True,
        "early_stopping_rounds": PATIENCE,
        "use_best_model": True,
        "refit": False,
        "validation_fraction": 0.15,
        "random_state": None,
        "thread_count": THREADS,
        "diagnostic_warnings": "never",
    }
    recipe = ARMS[name]
    if recipe is not None:
        common.update({
            "n_ensembles": 8,
            "ensemble_mode": "v3",
            "ensemble_parallelism": "sequential",
            **recipe,
        })
    return common


def _fit_one(spec, split, cat_features, arm: str, seed: int, archive: Path):
    from darkofit import DarkoClassifier, DarkoRegressor

    estimator_cls = DarkoRegressor if spec.task == "regression" else DarkoClassifier
    params = arm_params(arm)
    params["random_state"] = seed
    if spec.task == "regression":
        # Ensemble-v3 members use constant leaves. Match that representation in
        # the single-model reference rather than invoking the public auto selector.
        params["linear_leaves"] = False
    model = estimator_cls(**params)
    X_train = np.concatenate([split["X_fit"], split["X_val"]], axis=0)
    y_train = np.concatenate([split["y_fit"], split["y_val"]], axis=0)
    if split["w_fit"] is None:
        w_train = None
    else:
        w_train = np.concatenate([split["w_fit"], split["w_val"]], axis=0)
    kwargs = {"cat_features": cat_features}
    if w_train is not None:
        kwargs["sample_weight"] = w_train
    started = time.perf_counter()
    model.fit(X_train, y_train, **kwargs)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(split["X_test"])
    probability = (
        None if spec.task == "regression" else model.predict_proba(split["X_test"])
    )
    predict_seconds = time.perf_counter() - started
    metrics = metric_bundle(
        spec.task,
        split["y_test"],
        prediction,
        proba=probability,
        labels=getattr(model, "classes_", None),
        sample_weight=split["w_test"],
    )
    model.save_model(archive)
    members = list(getattr(model, "estimators_", ()) or ())
    if members:
        member_tree_counts = [
            len(getattr(member.model_, "trees_", ())) for member in members
        ]
    else:
        member_tree_counts = [len(getattr(model.model_, "trees_", ()))]
    return {
        "primary_metric": metrics["primary_metric"],
        "primary_value": metrics["primary_value"],
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "archive_bytes": int(archive.stat().st_size),
        "member_tree_counts": member_tree_counts,
        "total_tree_count": int(sum(member_tree_counts)),
        "resolved_member_policy": (
            None
            if arm == "single"
            else {
                "learning_rate": model.ensemble_metadata_["policy_resolutions"][
                    "learning_rate"
                ]["resolved"],
                "colsample": model.ensemble_metadata_["policy_resolutions"][
                    "colsample"
                ]["resolved"],
            }
        ),
    }


def analyze_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    expected = set(expected_cell_keys())
    cells: dict[tuple[str, int, str], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["seed"]), str(row["weight_mode"]))
        arm = str(row["arm"])
        if key not in expected or arm not in ARMS:
            raise RuntimeError("member-policy rows contain an unexpected identity")
        if arm in cells.setdefault(key, {}):
            raise RuntimeError("member-policy rows contain a duplicate arm")
        cells[key][arm] = row
    if set(cells) != expected or any(set(cell) != set(ARMS) for cell in cells.values()):
        raise RuntimeError("member-policy rows do not cover the exact grid")

    comparisons: dict[str, object] = {}
    for arm in ENSEMBLE_ARMS:
        current_ratios: dict[str, float] = {}
        single_ratios: dict[str, float] = {}
        groups: dict[str, str] = {}
        fit_ratios = []
        predict_ratios = []
        archive_ratios = []
        for dataset, seed, weight_mode in expected_cell_keys():
            key = (dataset, seed, weight_mode)
            current = cells[key][CURRENT_ARM]
            candidate = cells[key][arm]
            single = cells[key]["single"]
            if (
                candidate["primary_metric"] != current["primary_metric"]
                or candidate["primary_metric"] != single["primary_metric"]
            ):
                raise RuntimeError("paired member-policy metrics differ")
            name = f"{dataset}/{seed}/{weight_mode}"
            current_ratios[name] = float(candidate["primary_value"]) / float(
                current["primary_value"]
            )
            single_ratios[name] = float(candidate["primary_value"]) / float(
                single["primary_value"]
            )
            groups[name] = dataset
            fit_ratios.append(
                float(candidate["fit_seconds"]) / float(current["fit_seconds"])
            )
            predict_ratios.append(
                float(candidate["predict_seconds"])
                / float(current["predict_seconds"])
            )
            archive_ratios.append(
                float(candidate["archive_bytes"]) / float(current["archive_bytes"])
            )
        decision = quality_decision(current_ratios, groups=groups)
        comparisons[arm] = {
            "vs_current": decision,
            "vs_single_geometric_mean": _geomean(single_ratios.values()),
            "fit_seconds_vs_current_geometric_mean": _geomean(fit_ratios),
            "predict_seconds_vs_current_geometric_mean": _geomean(predict_ratios),
            "archive_bytes_vs_current_geometric_mean": _geomean(archive_ratios),
        }

    eligible = [
        arm
        for arm in ENSEMBLE_ARMS
        if (
            arm == CURRENT_ARM
            or (
                comparisons[arm]["vs_current"]["disposition"] == "advance"
                and comparisons[arm]["vs_current"]["geometric_mean_ratio"] < 1.0
            )
        )
    ]
    selected = min(
        eligible,
        key=lambda arm: comparisons[arm]["vs_current"]["geometric_mean_ratio"],
    )
    return {
        "cell_count": len(expected),
        "comparisons": comparisons,
        "selected_recipe": selected,
        "policy_changed": selected != CURRENT_ARM,
        "selection_rule": (
            "lowest aggregate ratio among current and strict improvements that "
            "also satisfy the M6-v3 dataset-harm and LOO bounds"
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(result: dict[str, object]) -> str:
    summary = result["summary"]
    lines = [
        "# Ensemble-v3 member-policy retune",
        "",
        "Spent general-development evidence; no holdout was consulted.",
        "",
        f"- Source: `{result['source']['head']}`",
        f"- Cells: `{summary['cell_count']}`",
        f"- Selected recipe: `{summary['selected_recipe']}`",
        f"- Public policy changed: `{str(summary['policy_changed']).lower()}`",
        "",
        "| Recipe | Quality/current | Worst dataset | Worst LOO | Quality/single | Fit/current | Predict/current | Archive/current |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ENSEMBLE_ARMS:
        comparison = summary["comparisons"][arm]
        decision = comparison["vs_current"]
        lines.append(
            f"| {arm} | {decision['geometric_mean_ratio']:.6f} | "
            f"{decision['worst_group_ratio']:.6f} | "
            f"{decision['worst_loo_ratio']:.6f} | "
            f"{comparison['vs_single_geometric_mean']:.6f} | "
            f"{comparison['fit_seconds_vs_current_geometric_mean']:.6f} | "
            f"{comparison['predict_seconds_vs_current_geometric_mean']:.6f} | "
            f"{comparison['archive_bytes_vs_current_geometric_mean']:.6f} |"
        )
    lines.extend([
        "",
        "The slice contains deterministic synthetic and resampled sklearn",
        "development datasets. It can choose an ensemble opt-in recipe; it does",
        "not establish a new automatic default or unseen-data claim.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    for path in (args.raw_output, args.result_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"member-policy output is create-only: {path}")
    source = source_state(args.expected_source_sha)
    rows = []
    with tempfile.TemporaryDirectory(prefix="darkofit-member-policy-") as tmp:
        tmpdir = Path(tmp)
        for cell_index, (dataset, seed, weight_mode) in enumerate(
            expected_cell_keys()
        ):
            spec, X, y, cat_features = build_dataset(dataset, SIZE, seed)
            weights = make_sample_weight(y, spec.task, weight_mode)
            split = split_case(X, y, spec.task, seed, weights)
            arm_order = list(ARMS)
            shift = cell_index % len(arm_order)
            arm_order = arm_order[shift:] + arm_order[:shift]
            for arm in arm_order:
                archive = tmpdir / f"{cell_index}-{arm}.npz"
                row = {
                    "dataset": dataset,
                    "task": spec.task,
                    "size": SIZE,
                    "seed": seed,
                    "weight_mode": weight_mode,
                    "arm": arm,
                    "train_rows": int(split["n_train"] + split["n_val"]),
                    "test_rows": int(split["n_test"]),
                    "feature_count": int(split["n_features"]),
                }
                row.update(_fit_one(spec, split, cat_features, arm, seed, archive))
                rows.append(row)
                archive.unlink()
                gc.collect()
            print(
                f"completed {dataset} seed={seed} weights={weight_mode}",
                flush=True,
            )
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "spent_general_quality_development_slice",
        "source": source,
        "grid": {
            "datasets": list(DATASETS),
            "size": SIZE,
            "seeds": list(SEEDS),
            "weight_modes": list(WEIGHT_MODES),
            "threads": THREADS,
            "iterations": ITERATIONS,
            "patience": PATIENCE,
            "arms": ARMS,
        },
        "rows": rows,
        "summary": analyze_rows(rows),
    }
    raw_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    args.raw_output.write_bytes(raw_bytes)
    rendered = _markdown(result)
    args.result_output.write_text(
        rendered + f"\nRaw SHA-256: `{_sha256(args.raw_output)}`\n"
    )
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
