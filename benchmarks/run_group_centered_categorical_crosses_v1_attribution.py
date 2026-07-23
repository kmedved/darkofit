#!/usr/bin/env python3
"""Attribute catcross value on the two spent categorical M2 datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve()
M2_CONTRACT = ROOT / "benchmarks/v011_m2_broad_panel_contract_v2_20260722.json"
ATTRIBUTION_ID = "group-centered-categorical-crosses-v1-attribution-20260723"
CANDIDATE_HEAD = "c3f2608cd3033cfc00aa0737897a92ed868b5865"
M2_CONTRACT_SHA256 = (
    "0c7ab7659a7b448534bad4372ba9b1db03c95b4ad9b403dd7966baf6fbfbbf66"
)
THREADS = 14
ITERATIONS = 1_000
WORKER_TIMEOUT_SECONDS = 7_200.0
DATASETS = (
    ("diamonds", 363631),
    ("healthcare_insurance_expenses", 363675),
)
COORDINATES = (
    (0, 0, 0),
    (1, 1, 1),
    (2, 2, 2),
)
ARMS = ("constant", "automatic", "forced")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: Sequence[int]) -> str:
    array = np.asarray(value, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    if Path(_git(repository, "rev-parse", "--show-toplevel")).resolve() != repository:
        raise RuntimeError("source path must name its Git root")
    status = _git(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    return {
        "path": str(repository),
        "head": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "clean": not status,
        "status": status,
    }


def build_manifest() -> dict[str, Any]:
    if file_sha256(M2_CONTRACT) != M2_CONTRACT_SHA256:
        raise RuntimeError("M2 coordinate source changed")
    contract = _load_json(M2_CONTRACT)
    protocol = contract.get("protocol", {})
    registered = {
        (
            str(row["dataset"]),
            int(row["repeat"]),
            int(row["fold"]),
        )
        for row in protocol.get("coordinates", ())
    }
    tasks = {
        str(name): int(spec["task_id"])
        for name, spec in protocol.get("task_split_counts", {}).items()
    }
    coordinates = []
    for dataset, task_id in DATASETS:
        if tasks.get(dataset) != task_id:
            raise RuntimeError(f"M2 task binding changed for {dataset}")
        for coordinate, repeat, fold in COORDINATES:
            if (dataset, repeat, fold) not in registered:
                raise RuntimeError(f"M2 coordinate changed for {dataset}/{coordinate}")
            coordinates.append(
                {
                    "dataset": dataset,
                    "task_id": task_id,
                    "coordinate": coordinate,
                    "repeat": repeat,
                    "fold": fold,
                    "seed": repeat * 1_000 + fold,
                }
            )
    return {
        "schema_version": 1,
        "attribution_id": ATTRIBUTION_ID,
        "status": "ready",
        "kind": "spent_development_attribution",
        "candidate_head": CANDIDATE_HEAD,
        "m2_contract_sha256": M2_CONTRACT_SHA256,
        "coordinates": coordinates,
        "planned_workers": len(coordinates),
        "arms": list(ARMS),
        "model": {
            "iterations": ITERATIONS,
            "loss": "RMSE",
            "tree_mode": "catboost",
            "thread_count": THREADS,
            "diagnostic_warnings": "never",
        },
        "execution": {
            "fresh_process_per_coordinate": True,
            "automatic_runs_before_forced_to_supply_the_exact_candidate_pairs": True,
            "constant_forced_order_alternates_by_coordinate": True,
            "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        },
        "interpretation": [
            "These two datasets and coordinates were already observed in M2.",
            "This attribution is development evidence, never holdout evidence.",
            "Automatic engagement is judged by its tested strict validation-win rule.",
        ],
    }


def _categorical_features(X) -> list[str]:
    import pandas as pd

    return [
        str(column)
        for column in X.columns
        if (
            isinstance(X[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(X[column].dtype)
            or pd.api.types.is_string_dtype(X[column].dtype)
            or pd.api.types.is_bool_dtype(X[column].dtype)
        )
    ]


def _load_coordinate(task_id: int, repeat: int, fold: int):
    import openml

    task = openml.tasks.get_task(int(task_id), download_splits=True)
    dataset = task.get_dataset()
    X, target, _categorical, _names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    y = np.asarray(target, dtype=np.float64)
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise RuntimeError("attribution target is invalid")
    train, test = task.get_train_test_split_indices(
        repeat=int(repeat),
        fold=int(fold),
        sample=0,
    )
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    if (
        train.size == 0
        or test.size == 0
        or np.intersect1d(train, test).size
    ):
        raise RuntimeError("attribution split is invalid")
    return {
        "task": task,
        "dataset": dataset,
        "X_train": X.iloc[train].reset_index(drop=True),
        "y_train": y[train],
        "X_test": X.iloc[test].reset_index(drop=True),
        "y_test": y[test],
        "train_index_sha256": _array_sha256(train),
        "test_index_sha256": _array_sha256(test),
    }


def _model(seed: int):
    from darkofit import DarkoRegressor

    return DarkoRegressor(
        iterations=ITERATIONS,
        loss="RMSE",
        tree_mode="catboost",
        random_state=int(seed),
        thread_count=THREADS,
        diagnostic_warnings="never",
    )


def _rmse(truth, prediction) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise RuntimeError("attribution prediction is invalid")
    value = float(np.sqrt(np.mean(np.square(prediction - truth))))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("attribution RMSE is invalid")
    return value


def _fit_arm(
    arm: str,
    *,
    seed: int,
    data: Mapping[str, Any],
    cat_features: Sequence[str],
    pairs: Sequence[Sequence[int]] | None = None,
) -> tuple[dict[str, Any], list[list[int]]]:
    import numba

    model = _model(seed)
    if arm == "constant":
        model._group_centered_crosses_private_mode = "off"
    elif arm == "forced":
        if not pairs:
            raise RuntimeError("forced attribution arm requires candidate pairs")
        model._group_centered_crosses_private_mode = "forced"
        model._group_centered_pairs_override = [tuple(pair) for pair in pairs]
    elif arm != "automatic":
        raise RuntimeError(f"unknown attribution arm: {arm}")
    ambient = int(numba.get_num_threads())
    started = time.perf_counter()
    model.fit(
        data["X_train"],
        data["y_train"],
        cat_features=list(cat_features),
    )
    fit_seconds = time.perf_counter() - started
    prediction = np.asarray(model.predict(data["X_test"]), dtype=np.float64)
    if int(numba.get_num_threads()) != ambient:
        raise RuntimeError("attribution fit/predict leaked the Numba thread mask")
    selector = getattr(model, "group_centered_categorical_crosses_", None)
    fitted_pairs = [
        [int(numeric), int(categorical)]
        for numeric, categorical in getattr(
            model.model_.prep_, "group_centered_pairs_", ()
        )
    ]
    if arm == "automatic":
        if (
            not isinstance(selector, Mapping)
            or selector.get("eligible") is not True
            or not isinstance(selector.get("selected"), bool)
            or not selector.get("pairs")
        ):
            raise RuntimeError("automatic attribution selector state is invalid")
        candidate_pairs = [
            [int(pair[0]), int(pair[1])] for pair in selector["pairs"]
        ]
        expected_final = candidate_pairs if selector["selected"] else []
        if fitted_pairs != expected_final:
            raise RuntimeError("automatic attribution final lane drifted")
    else:
        if selector is not None:
            raise RuntimeError("explicit attribution arm has selector state")
        candidate_pairs = fitted_pairs
    return (
        {
            "arm": arm,
            "rmse": _rmse(data["y_test"], prediction),
            "fit_seconds": float(fit_seconds),
            "prediction_sha256": hashlib.sha256(
                np.ascontiguousarray(prediction, dtype="<f8").tobytes()
            ).hexdigest(),
            "fitted_pairs": fitted_pairs,
            "selector": selector,
            "tree_count": int(len(model.model_.trees_)),
            "resolved_threads": int(model.model_.n_threads_),
            "ambient_thread_restored": True,
        },
        candidate_pairs,
    )


def run_worker(spec: Mapping[str, Any], source: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    state = source_state(source)
    if not state["clean"] or state["head"] != CANDIDATE_HEAD:
        raise RuntimeError("catcross candidate source changed")
    sys.path.insert(0, str(source))
    import darkofit

    if Path(darkofit.__file__).resolve().parents[1] != source:
        raise RuntimeError("worker imported DarkoFit from the wrong source")
    if os.cpu_count() != THREADS:
        raise RuntimeError("attribution requires the 14-CPU host")
    data = _load_coordinate(
        int(spec["task_id"]),
        int(spec["repeat"]),
        int(spec["fold"]),
    )
    if (
        str(data["dataset"].name) != str(spec["dataset"])
        or int(data["task"].task_id) != int(spec["task_id"])
    ):
        raise RuntimeError("attribution dataset binding changed")
    cat_features = _categorical_features(data["X_train"])
    if not cat_features or len(cat_features) == data["X_train"].shape[1]:
        raise RuntimeError("attribution dataset lost its mixed feature schema")

    automatic, pairs = _fit_arm(
        "automatic",
        seed=int(spec["seed"]),
        data=data,
        cat_features=cat_features,
    )
    trailing = (
        ("constant", "forced")
        if int(spec["coordinate"]) % 2 == 0
        else ("forced", "constant")
    )
    arms = {"automatic": automatic}
    for arm in trailing:
        row, _ = _fit_arm(
            arm,
            seed=int(spec["seed"]),
            data=data,
            cat_features=cat_features,
            pairs=pairs,
        )
        arms[arm] = row
    if set(arms) != set(ARMS):
        raise RuntimeError("attribution worker arm census changed")
    return {
        "schema_version": 1,
        "attribution_id": ATTRIBUTION_ID,
        "status": "ok",
        **{name: spec[name] for name in (
            "dataset", "task_id", "coordinate", "repeat", "fold", "seed"
        )},
        "source": state,
        "train_rows": int(len(data["y_train"])),
        "test_rows": int(len(data["y_test"])),
        "feature_count": int(data["X_train"].shape[1]),
        "categorical_features": cat_features,
        "train_index_sha256": data["train_index_sha256"],
        "test_index_sha256": data["test_index_sha256"],
        "arms": arms,
    }


def _geomean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or not array.size
        or np.any(array <= 0.0)
        or not np.isfinite(array).all()
    ):
        raise RuntimeError("invalid attribution geomean input")
    return float(np.exp(np.mean(np.log(array))))


def analyze(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 6:
        raise RuntimeError("attribution row census changed")
    indexed = {}
    expected_specs = {
        (dataset, coordinate): {
            "dataset": dataset,
            "task_id": task_id,
            "coordinate": coordinate,
            "repeat": repeat,
            "fold": fold,
            "seed": repeat * 1_000 + fold,
        }
        for dataset, task_id in DATASETS
        for coordinate, repeat, fold in COORDINATES
    }
    for row in rows:
        key = (str(row["dataset"]), int(row["coordinate"]))
        if key in indexed:
            raise RuntimeError("duplicate attribution coordinate")
        if (
            row.get("attribution_id") != ATTRIBUTION_ID
            or row.get("status") != "ok"
            or set(row.get("arms", {})) != set(ARMS)
            or key not in expected_specs
            or any(
                row.get(name) != value
                for name, value in expected_specs[key].items()
            )
        ):
            raise RuntimeError("invalid attribution worker row")
        automatic = row["arms"]["automatic"]
        forced = row["arms"]["forced"]
        constant = row["arms"]["constant"]
        selector = automatic.get("selector")
        if (
            any(not isinstance(row["arms"][arm], Mapping) for arm in ARMS)
            or any(
                not math.isfinite(float(row["arms"][arm].get("rmse", math.nan)))
                or float(row["arms"][arm]["rmse"]) <= 0.0
                for arm in ARMS
            )
            or not isinstance(selector, Mapping)
            or selector.get("eligible") is not True
            or not isinstance(selector.get("selected"), bool)
            or not isinstance(selector.get("pairs"), list)
            or not selector["pairs"]
            or automatic["fitted_pairs"] != (
                selector["pairs"] if selector["selected"] else []
            )
            or forced["fitted_pairs"] != selector["pairs"]
            or constant["fitted_pairs"] != []
        ):
            raise RuntimeError("attribution lane provenance is invalid")
        indexed[key] = row
    if set(indexed) != set(expected_specs):
        raise RuntimeError("attribution coordinate grid drifted")

    datasets = []
    automatic_all = []
    forced_all = []
    worst_automatic = 0.0
    calibration_findings = []
    for dataset, _task in DATASETS:
        coordinate_rows = [
            indexed[(dataset, coordinate)]
            for coordinate, _repeat, _fold in COORDINATES
        ]
        automatic_ratios = [
            float(row["arms"]["automatic"]["rmse"])
            / float(row["arms"]["constant"]["rmse"])
            for row in coordinate_rows
        ]
        forced_ratios = [
            float(row["arms"]["forced"]["rmse"])
            / float(row["arms"]["constant"]["rmse"])
            for row in coordinate_rows
        ]
        automatic_all.extend(automatic_ratios)
        forced_all.extend(forced_ratios)
        worst_automatic = max(worst_automatic, max(automatic_ratios))
        for row, automatic_ratio, forced_ratio in zip(
            coordinate_rows, automatic_ratios, forced_ratios
        ):
            selected = bool(row["arms"]["automatic"]["selector"]["selected"])
            if not selected and forced_ratio < 1.0:
                calibration_findings.append(
                    {
                        "dataset": dataset,
                        "coordinate": int(row["coordinate"]),
                        "forced_ratio": forced_ratio,
                        "automatic_ratio": automatic_ratio,
                        "reason": "automatic_declined_with_forced_value_left",
                    }
                )
        datasets.append(
            {
                "dataset": dataset,
                "automatic_constant_geomean_ratio": _geomean(automatic_ratios),
                "forced_constant_geomean_ratio": _geomean(forced_ratios),
                "automatic_coordinate_ratios": automatic_ratios,
                "forced_coordinate_ratios": forced_ratios,
                "automatic_selected_coordinates": int(
                    sum(
                        bool(row["arms"]["automatic"]["selector"]["selected"])
                        for row in coordinate_rows
                    )
                ),
            }
        )
    dataset_gate_passes = all(
        row["automatic_constant_geomean_ratio"] <= 1.0 for row in datasets
    )
    harm_gate_passes = worst_automatic <= 1.02
    return {
        "schema_version": 1,
        "attribution_id": ATTRIBUTION_ID,
        "kind": "spent_development_attribution",
        "integrity": {"passes": True, "workers": 6, "arm_rows": 18},
        "quality": {
            "automatic_constant_equal_coordinate_geomean_ratio": _geomean(
                automatic_all
            ),
            "forced_constant_equal_coordinate_geomean_ratio": _geomean(forced_all),
            "worst_automatic_coordinate_ratio": worst_automatic,
            "datasets": datasets,
        },
        "engagement": {
            "selected_coordinates": int(
                sum(row["automatic_selected_coordinates"] for row in datasets)
            ),
            "eligible_coordinates": 6,
            "calibration_findings": calibration_findings,
        },
        "gates": {
            "automatic_not_worse_each_dataset": dataset_gate_passes,
            "automatic_worst_coordinate_at_most_1_02": harm_gate_passes,
            "passes": bool(dataset_gate_passes and harm_gate_passes),
        },
        "disposition": (
            "attribution_supports_opt_in_product_path"
            if dataset_gate_passes and harm_gate_passes
            else "attribution_does_not_support_automatic_path"
        ),
        "limitations": [
            "Spent development evidence on two categorical datasets.",
            "No holdout, sports, release-ladder, or default claim.",
            "Fit times include different numbers of auditions and are telemetry only.",
        ],
    }


def output_paths(prefix: Path) -> dict[str, Path]:
    prefix = prefix.expanduser().resolve()
    try:
        prefix.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("attribution outputs must be outside the source tree")
    return {
        "launch": Path(str(prefix) + "_launch.json"),
        "raw": Path(str(prefix) + "_raw.json"),
        "result": Path(str(prefix) + "_result.json"),
    }


def exclusive_machine_audit() -> dict[str, Any]:
    import psutil

    own = {os.getpid()}
    parent = psutil.Process().parent()
    while parent is not None:
        own.add(parent.pid)
        try:
            parent = parent.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            break
    markers = (
        "run_group_centered_categorical_crosses_v1_attribution",
        "run_m6_quality_successor",
        "run_v011_m2_broad_panel",
        "run_v011_compute_ladder",
        "run_tabarena",
        "run_m3",
        "run_b3",
    )
    conflicts = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid not in own and any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"another benchmark process is active: {conflicts}")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "conflicting_benchmark_processes": [],
        "load_average": [float(value) for value in os.getloadavg()],
    }


def _worker_env(source: Path, cache: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(source.resolve()),
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": str(THREADS),
            "OMP_THREAD_LIMIT": str(THREADS),
            "OPENBLAS_NUM_THREADS": str(THREADS),
            "MKL_NUM_THREADS": str(THREADS),
            "NUMEXPR_NUM_THREADS": str(THREADS),
            "NUMBA_NUM_THREADS": str(THREADS),
            "VECLIB_MAXIMUM_THREADS": str(THREADS),
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "DARKOFIT_WARMUP": "0",
            "NUMBA_CACHE_DIR": str(cache),
        }
    )
    return environment


def execute(
    *,
    manifest_path: Path,
    source: Path,
    prefix: Path,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest != build_manifest():
        raise RuntimeError("attribution manifest is invalid")
    paths = output_paths(prefix)
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise RuntimeError(f"attribution output collision: {collisions}")
    harness = source_state(ROOT)
    candidate = source_state(source)
    if not harness["clean"]:
        raise RuntimeError("attribution harness must be clean")
    if not candidate["clean"] or candidate["head"] != CANDIDATE_HEAD:
        raise RuntimeError("attribution candidate source changed")
    if os.cpu_count() != THREADS:
        raise RuntimeError("attribution requires the 14-CPU host")
    audit = exclusive_machine_audit()
    launch = {
        "schema_version": 1,
        "attribution_id": ATTRIBUTION_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {"harness": harness, "candidate": candidate},
        "source_hashes": {
            "manifest": file_sha256(manifest_path),
            "runner": file_sha256(RUNNER),
            "m2_contract": file_sha256(M2_CONTRACT),
        },
        "exclusive_machine_audit": audit,
        "planned_workers": 6,
    }
    _write_create_only(paths["launch"], launch)

    rows = []
    with tempfile.TemporaryDirectory(prefix="darkofit-catcross-attribution-") as temp:
        temp_path = Path(temp)
        cache = temp_path / "numba-cache"
        cache.mkdir()
        for index, spec in enumerate(manifest["coordinates"]):
            spec_path = temp_path / f"coordinate-{index:02d}.json"
            spec_path.write_bytes(canonical_json_bytes(spec))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "worker",
                    "--spec",
                    str(spec_path),
                    "--source",
                    str(source),
                ],
                cwd=ROOT,
                env=_worker_env(source, cache),
                check=False,
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"catcross attribution worker {index} failed: "
                    + completed.stderr[-5000:]
                )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError("catcross attribution worker returned no row")
            row = json.loads(lines[-1])
            if row.get("status") != "ok":
                raise RuntimeError("catcross attribution worker integrity failed")
            rows.append(row)
    raw = {
        "schema_version": 1,
        "attribution_id": ATTRIBUTION_ID,
        "complete": True,
        "launch_sha256": file_sha256(paths["launch"]),
        "rows": rows,
    }
    _write_create_only(paths["raw"], raw)
    result = analyze(rows)
    result["source_hashes"] = {
        "raw": file_sha256(paths["raw"]),
        "runner": file_sha256(RUNNER),
    }
    _write_create_only(paths["result"], result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--spec", type=Path, required=True)
    worker.add_argument("--source", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "manifest":
        if args.output.exists() or args.output.is_symlink():
            raise RuntimeError(f"refusing existing output: {args.output}")
        manifest = build_manifest()
        _write_create_only(args.output, manifest)
        print(json.dumps({"output": str(args.output), "workers": 6}))
        return 0
    if args.command == "worker":
        print(
            json.dumps(
                run_worker(_load_json(args.spec), args.source),
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    result = execute(
        manifest_path=args.manifest,
        source=args.source,
        prefix=args.output_prefix,
    )
    print(
        json.dumps(
            {
                "automatic_ratio": result["quality"][
                    "automatic_constant_equal_coordinate_geomean_ratio"
                ],
                "forced_ratio": result["quality"][
                    "forced_constant_equal_coordinate_geomean_ratio"
                ],
                "passes": result["gates"]["passes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
