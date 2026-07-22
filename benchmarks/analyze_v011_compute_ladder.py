#!/usr/bin/env python3
"""Verify and analyze the frozen v0.11 release compute ladder."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # Direct execution from a clean checkout.
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from benchmarks import run_v011_compute_ladder as campaign


METRICS = (
    "test_rmse",
    "fit_seconds",
    "prediction_seconds_per_call",
    "fit_rss_peak_bytes",
    "fit_rss_peak_delta_bytes",
)
COUNTERPARTS = (
    (campaign.DARKO_DEFAULT, campaign.CHIMERA_DEFAULT),
    (campaign.DARKO_ACCURACY, campaign.CHIMERA_ACCURACY),
    (campaign.DARKO_ENSEMBLE, campaign.CHIMERA_ENSEMBLE),
)
MANIFEST_FIELDS = {
    "schema_version",
    "kind",
    "contract_id",
    "created_at_utc",
    "contract",
    "harness_head",
    "darkofit_source",
    "chimeraboost_source",
    "tabarena_source",
    "latest_chimeraboost_release",
    "hardware",
    "exclusive_machine",
    "worker_environment",
    "expected_worker_count",
    "ordered_grid_sha256",
    "ordered_grid",
}
WORKER_FIELDS = {
    "schema_version",
    "kind",
    "worker_index",
    "pid",
    "parent_pid",
    "started_at_utc",
    "completed_at_utc",
    "dataset",
    "task_id",
    "repeat",
    "fold",
    "arm",
    "code",
    "engine",
    "profile",
    "seed",
    "train_rows",
    "test_rows",
    "feature_count",
    "categorical_features",
    "fingerprints",
    "test_rmse",
    "fit_seconds",
    "fit_rss",
    "prediction",
    "prediction_sha256",
    "model",
    "implementation",
    "warmup",
    "environment",
    "numba_threads_before_fit",
    "numba_threads_after_fit",
    "warnings",
    "launcher_output",
}


def _as_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be an object")
    return dict(value)


def _finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{field} must be finite and positive")
    return number


def _exact_artifact(
    root: Path,
    metadata: Any,
    *,
    expected_path: str | None = None,
    field: str,
) -> tuple[Path, bytes]:
    record = _as_mapping(metadata, field)
    if set(record) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{field} metadata fields are not exact")
    relative = record["path"]
    if (
        not isinstance(relative, str)
        or not relative
        or type(record["bytes"]) is not int
        or record["bytes"] <= 0
        or not isinstance(record["sha256"], str)
        or len(record["sha256"]) != 64
    ):
        raise RuntimeError(f"{field} path is invalid")
    if expected_path is not None and relative != expected_path:
        raise RuntimeError(f"{field} path drifted")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"{field} path escapes the campaign")
    path = root / candidate
    metadata_on_disk = path.lstat()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{field} is not a regular file")
    payload = path.read_bytes()
    if (
        record["bytes"] != metadata_on_disk.st_size
        or len(payload) != metadata_on_disk.st_size
        or record["sha256"] != campaign._sha256_bytes(payload)
    ):
        raise RuntimeError(f"{field} artifact digest changed")
    return path, payload


def _decode_object(payload: bytes, field: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant in {field}: {value}")

    value = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    return _as_mapping(value, field)


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric == "prediction_seconds_per_call":
        return float(row["prediction"]["seconds_per_call"])
    if metric == "fit_rss_peak_bytes":
        return float(row["fit_rss"]["peak_bytes"])
    if metric == "fit_rss_peak_delta_bytes":
        return float(row["fit_rss"]["peak_delta_bytes"])
    return float(row[metric])


def _verify_source_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if set(manifest) != MANIFEST_FIELDS:
        raise RuntimeError("compute-ladder manifest fields are not exact")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "v011_compute_ladder_manifest"
        or manifest.get("contract_id") != campaign.CONTRACT_ID
        or manifest.get("expected_worker_count") != campaign.EXPECTED_WORKERS
        or manifest.get("ordered_grid_sha256") != campaign.ordered_grid_sha256()
        or manifest.get("worker_environment") != campaign.WORKER_ENVIRONMENT
    ):
        raise RuntimeError("compute-ladder manifest identity drifted")
    expected_grid = [
        {
            "dataset": dataset,
            "repeat": repeat,
            "fold": fold,
            "arm": arm,
        }
        for dataset, repeat, fold, arm in campaign.expected_ordered_grid()
    ]
    if manifest.get("ordered_grid") != expected_grid:
        raise RuntimeError("compute-ladder manifest grid drifted")
    contract = _as_mapping(manifest.get("contract"), "manifest contract")
    if (
        set(contract) != {"path", "sha256", "protocol_sha256"}
        or contract.get("sha256") != campaign.sha256(campaign.CONTRACT_PATH)
        or contract.get("protocol_sha256") != campaign.protocol_sha256()
        or Path(str(contract.get("path"))).resolve() != campaign.CONTRACT_PATH.resolve()
    ):
        raise RuntimeError("manifest contract binding drifted")
    harness_head = campaign.validate_execution_source_pin(campaign.load_contract())
    if manifest.get("harness_head") != harness_head:
        raise RuntimeError("manifest harness source drifted")
    recorded_darko = _as_mapping(manifest.get("darkofit_source"), "DarkoFit source")
    recorded_chimera = _as_mapping(
        manifest.get("chimeraboost_source"), "ChimeraBoost source"
    )
    recorded_tabarena = _as_mapping(manifest.get("tabarena_source"), "TabArena source")
    darko = campaign._validate_source_checkout(
        Path(str(recorded_darko.get("path"))),
        commit=campaign.DARKOFIT_COMMIT,
        tag=campaign.DARKOFIT_TAG,
        package_init="darkofit/__init__.py",
    )
    chimera = campaign._validate_source_checkout(
        Path(str(recorded_chimera.get("path"))),
        commit=campaign.CHIMERABOOST_COMMIT,
        tag=campaign.CHIMERABOOST_TAG,
        package_init="chimeraboost/__init__.py",
    )
    tabarena = campaign.validate_tabarena_source(
        Path(str(recorded_tabarena.get("path")))
    )
    if (
        recorded_darko != darko
        or recorded_chimera != chimera
        or recorded_tabarena != tabarena
    ):
        raise RuntimeError("manifest product-source provenance drifted")
    latest = _as_mapping(
        manifest.get("latest_chimeraboost_release"),
        "latest ChimeraBoost release",
    )
    if (
        set(latest) != {"tag_name", "published_at", "html_url", "verified_at_utc"}
        or latest.get("tag_name") != campaign.CHIMERABOOST_TAG
        or latest.get("published_at") != campaign.CHIMERABOOST_RELEASE_PUBLISHED_AT
        or latest.get("html_url")
        != "https://github.com/bbstats/chimeraboost/releases/tag/v0.20.0"
        or not isinstance(latest.get("verified_at_utc"), str)
    ):
        raise RuntimeError("latest ChimeraBoost release attestation drifted")
    hardware = _as_mapping(manifest.get("hardware"), "hardware")
    if (
        set(hardware)
        != {
            "platform",
            "machine",
            "processor",
            "python",
            "logical_cpus",
            "physical_cpus",
            "memory_bytes",
        }
        or hardware.get("logical_cpus") != campaign.THREADS
        or hardware.get("physical_cpus") != campaign.THREADS
        or int(hardware.get("memory_bytes", 0)) <= 0
    ):
        raise RuntimeError("compute-ladder hardware identity drifted")
    exclusive = _as_mapping(manifest.get("exclusive_machine"), "exclusive machine")
    if (
        set(exclusive)
        != {
            "checked_at_utc",
            "conflicting_benchmark_processes",
            "ignored_launch_ancestor_pids",
            "load_average",
        }
        or exclusive.get("conflicting_benchmark_processes") != []
        or not isinstance(exclusive.get("checked_at_utc"), str)
        or not isinstance(exclusive.get("ignored_launch_ancestor_pids"), list)
        or any(
            type(value) is not int or value <= 0
            for value in exclusive["ignored_launch_ancestor_pids"]
        )
        or not isinstance(exclusive.get("load_average"), list)
        or len(exclusive["load_average"]) != 3
        or any(not math.isfinite(float(value)) for value in exclusive["load_average"])
    ):
        raise RuntimeError("exclusive-machine attestation drifted")
    return {
        "harness_head": harness_head,
        "contract_sha256": contract["sha256"],
        "darkofit_commit": darko["commit"],
        "chimeraboost_commit": chimera["commit"],
        "tabarena_commit": tabarena["commit"],
        "latest_chimeraboost_verified_at_utc": latest["verified_at_utc"],
        "hardware": hardware,
    }


def _verify_worker_model(row: Mapping[str, Any], expected_source: Path) -> None:
    arm = str(row["arm"])
    spec = campaign.ARM_SPECS[arm]
    implementation = _as_mapping(row.get("implementation"), "implementation")
    if set(implementation) != {
        "class",
        "module",
        "module_path",
        "module_sha256",
    }:
        raise RuntimeError("worker implementation fields are not exact")
    module_path = Path(str(implementation["module_path"])).resolve()
    try:
        module_path.relative_to(expected_source.resolve())
    except ValueError as exc:
        raise RuntimeError("worker model did not come from the pinned product") from exc
    if (
        not module_path.is_file()
        or module_path.is_symlink()
        or campaign.sha256(module_path) != implementation["module_sha256"]
    ):
        raise RuntimeError("worker implementation source changed")
    expected_class = (
        "DarkoRegressor" if spec["engine"] == "darkofit" else "ChimeraBoostRegressor"
    )
    if implementation.get("class") != expected_class:
        raise RuntimeError("worker estimator class drifted")

    model = _as_mapping(row.get("model"), "model metadata")
    common = {
        "engine",
        "profile",
        "public_config",
        "member_count",
        "total_tree_count",
        "members",
    }
    engine_fields = (
        {"preset", "tree_mode_selection", "ensemble_mode"}
        if spec["engine"] == "darkofit"
        else {"ensemble_n_jobs", "max_samples"}
    )
    if set(model) != common | engine_fields:
        raise RuntimeError("worker model metadata fields are not exact")
    expected_members = 8 if spec["profile"] == "ensemble" else 1
    if (
        model.get("engine") != spec["engine"]
        or model.get("profile") != spec["profile"]
        or model.get("public_config") != spec["config"]
        or model.get("member_count") != expected_members
        or not isinstance(model.get("members"), list)
        or len(model["members"]) != expected_members
    ):
        raise RuntimeError("worker model metadata identity drifted")
    total_trees = 0
    for member in model["members"]:
        member = _as_mapping(member, "member metadata")
        expected_fields = {"tree_count", "thread_count"}
        if spec["engine"] == "darkofit":
            expected_fields |= {"tree_mode", "stop_reason"}
        else:
            expected_fields |= {
                "linear_leaves_selected",
                "cross_features_selected",
                "cross_pair_count",
            }
        if set(member) != expected_fields:
            raise RuntimeError("member metadata fields are not exact")
        tree_count = int(member["tree_count"])
        thread_count = int(member["thread_count"])
        if tree_count <= 0 or not 1 <= thread_count <= campaign.THREADS:
            raise RuntimeError("member tree/thread metadata is invalid")
        if spec["engine"] == "darkofit" and (
            not isinstance(member["tree_mode"], str)
            or not isinstance(member["stop_reason"], str)
        ):
            raise RuntimeError("DarkoFit resolved member metadata is invalid")
        if spec["engine"] == "chimeraboost" and (
            isinstance(member["cross_pair_count"], bool)
            or int(member["cross_pair_count"]) < 0
        ):
            raise RuntimeError("ChimeraBoost selector metadata is invalid")
        total_trees += tree_count
    if int(model["total_tree_count"]) != total_trees:
        raise RuntimeError("worker total-tree metadata is inconsistent")
    if spec["engine"] == "darkofit":
        if spec["profile"] == "accuracy" and model.get("preset") != "accuracy":
            raise RuntimeError("DarkoFit accuracy preset did not resolve")
        if spec["profile"] == "ensemble" and model.get("ensemble_mode") != "v3":
            raise RuntimeError("DarkoFit v3 ensemble mode did not resolve")
    else:
        if int(model["ensemble_n_jobs"]) != -1 or not math.isclose(
            float(model["max_samples"]),
            0.8,
        ):
            raise RuntimeError("ChimeraBoost ensemble policy metadata drifted")


def _verify_worker(
    raw: Any,
    *,
    expected_index: int,
    parent_pid: int | None,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    row = _as_mapping(raw, f"worker {expected_index}")
    if set(row) != WORKER_FIELDS:
        raise RuntimeError(f"worker {expected_index} fields are not exact")
    dataset, repeat, fold, arm = campaign.expected_ordered_grid()[expected_index]
    spec = campaign.ARM_SPECS[arm]
    expected_identity = {
        "schema_version": 1,
        "kind": "v011_compute_ladder_worker",
        "worker_index": expected_index,
        "dataset": dataset,
        "task_id": campaign.TASKS[dataset],
        "repeat": repeat,
        "fold": fold,
        "arm": arm,
        "code": spec["code"],
        "engine": spec["engine"],
        "profile": spec["profile"],
        "seed": campaign._coordinate_seed(repeat, fold),
    }
    if any(row.get(name) != value for name, value in expected_identity.items()):
        raise RuntimeError(f"worker {expected_index} identity drifted")
    if (
        type(row.get("pid")) is not int
        or row["pid"] <= 0
        or type(row.get("parent_pid")) is not int
        or row["parent_pid"] <= 0
        or row["pid"] == row["parent_pid"]
        or not isinstance(row.get("started_at_utc"), str)
        or not isinstance(row.get("completed_at_utc"), str)
    ):
        raise RuntimeError(f"worker {expected_index} process identity is invalid")
    if parent_pid is not None and row["parent_pid"] != parent_pid:
        raise RuntimeError("compute-ladder workers did not share one parent")
    for name in ("train_rows", "test_rows", "feature_count"):
        if isinstance(row.get(name), bool) or int(row[name]) <= 0:
            raise RuntimeError(f"worker {expected_index} {name} is invalid")
    categorical = row.get("categorical_features")
    if (
        not isinstance(categorical, list)
        or len(categorical) != len(set(categorical))
        or any(not isinstance(value, str) for value in categorical)
    ):
        raise RuntimeError("worker categorical-feature metadata is invalid")
    fingerprints = _as_mapping(row.get("fingerprints"), "split fingerprints")
    if set(fingerprints) != {
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "combined_sha256",
    } or any(
        not isinstance(value, str) or len(value) != 64
        for value in fingerprints.values()
    ):
        raise RuntimeError("worker split fingerprints are invalid")
    _finite_positive(row.get("test_rmse"), "test RMSE")
    _finite_positive(row.get("fit_seconds"), "fit seconds")
    rss = _as_mapping(row.get("fit_rss"), "fit RSS")
    if set(rss) != {
        "scope",
        "start_bytes",
        "peak_bytes",
        "peak_delta_bytes",
        "end_bytes",
        "samples",
        "errors",
        "interval_seconds",
    }:
        raise RuntimeError("fit RSS fields are not exact")
    start = int(rss["start_bytes"])
    peak = int(rss["peak_bytes"])
    end = int(rss["end_bytes"])
    delta = int(rss["peak_delta_bytes"])
    if (
        rss.get("scope") != "worker_plus_recursive_children"
        or min(start, peak, end) <= 0
        or peak < max(start, end)
        or delta != peak - start
        or int(rss["samples"]) < 2
        or rss["errors"] != []
        or not math.isclose(
            float(rss["interval_seconds"]),
            campaign.RSS_INTERVAL_SECONDS,
        )
    ):
        raise RuntimeError("fit RSS telemetry is inconsistent")
    prediction = _as_mapping(row.get("prediction"), "prediction timing")
    if set(prediction) != {
        "rows",
        "pilots_seconds",
        "pilot_median_seconds",
        "calls",
        "interval_seconds",
        "seconds_per_call",
        "rows_per_second",
        "prediction_sha256",
    }:
        raise RuntimeError("prediction timing fields are not exact")
    pilots = prediction["pilots_seconds"]
    calls = int(prediction["calls"])
    interval = _finite_positive(prediction["interval_seconds"], "prediction interval")
    seconds_per_call = _finite_positive(
        prediction["seconds_per_call"],
        "prediction seconds per call",
    )
    rows_per_second = _finite_positive(
        prediction["rows_per_second"],
        "prediction rows per second",
    )
    if (
        prediction["rows"] != row["test_rows"]
        or not isinstance(pilots, list)
        or len(pilots) != campaign.PREDICTION_PILOTS
        or any(_finite_positive(value, "prediction pilot") <= 0 for value in pilots)
        or not math.isclose(
            float(prediction["pilot_median_seconds"]),
            float(np.median(np.asarray(pilots, dtype=np.float64))),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not campaign.PREDICTION_MIN_CALLS <= calls <= campaign.PREDICTION_MAX_CALLS
        or interval < campaign.PREDICTION_MIN_SECONDS
        or not math.isclose(seconds_per_call, interval / calls, rel_tol=1e-12)
        or not math.isclose(
            rows_per_second,
            row["test_rows"] * calls / interval,
            rel_tol=1e-12,
        )
        or prediction.get("prediction_sha256") != row.get("prediction_sha256")
        or not isinstance(row.get("prediction_sha256"), str)
        or len(row["prediction_sha256"]) != 64
    ):
        raise RuntimeError("prediction timing telemetry is inconsistent")
    if (
        row.get("environment") != campaign.WORKER_ENVIRONMENT
        or row.get("numba_threads_before_fit") != campaign.THREADS
        or row.get("numba_threads_after_fit") != campaign.THREADS
    ):
        raise RuntimeError("worker thread environment drifted")
    warmup = _as_mapping(row.get("warmup"), "warmup")
    expected_routes = (
        ["darkofit_catboost", "darkofit_lightgbm", "darkofit_hybrid"]
        if arm == campaign.DARKO_ACCURACY
        else [f"{spec['engine']}_{spec['profile']}"]
    )
    if (
        set(warmup)
        != {"rows", "categorical_feature_count", "routes", "reduced_iteration_budget"}
        or warmup.get("rows") != min(512, row["train_rows"])
        or warmup.get("categorical_feature_count") != len(categorical)
        or warmup.get("routes") != expected_routes
        or warmup.get("reduced_iteration_budget") is not True
    ):
        raise RuntimeError("worker warmup attestation drifted")
    warnings = row.get("warnings")
    if not isinstance(warnings, list) or any(
        not isinstance(item, Mapping)
        or set(item) != {"category", "message"}
        or not isinstance(item["category"], str)
        or not isinstance(item["message"], str)
        for item in warnings
    ):
        raise RuntimeError("worker warning record is invalid")
    launcher = _as_mapping(row.get("launcher_output"), "launcher output")
    if (
        set(launcher) != {"returncode", "stdout_without_result", "stderr"}
        or launcher.get("returncode") != 0
        or not isinstance(launcher.get("stdout_without_result"), list)
        or any(
            not isinstance(value, str) for value in launcher["stdout_without_result"]
        )
        or not isinstance(launcher.get("stderr"), str)
    ):
        raise RuntimeError("worker launcher output is invalid")
    expected_source = Path(
        str(
            manifest[
                (
                    "darkofit_source"
                    if spec["engine"] == "darkofit"
                    else "chimeraboost_source"
                )
            ]["path"]
        )
    )
    _verify_worker_model(row, expected_source)
    clean = dict(row)
    clean["test_rmse"] = float(row["test_rmse"])
    clean["fit_seconds"] = float(row["fit_seconds"])
    return clean, int(row["parent_pid"])


def verify_campaign(
    input_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Verify every immutable campaign artifact and return normalized rows."""
    input_dir = Path(os.path.abspath(input_dir.expanduser()))
    if not input_dir.is_dir() or input_dir.is_symlink():
        raise RuntimeError("compute-ladder input must be a regular directory")
    contract = campaign.load_contract()
    terminal = campaign._read_json(input_dir / "terminal.json")
    if (
        set(terminal)
        != {
            "schema_version",
            "kind",
            "status",
            "contract_id",
            "completed_worker_count",
            "raw",
            "completed_at_utc",
        }
        or terminal.get("schema_version") != 1
        or terminal.get("kind") != "v011_compute_ladder_terminal"
        or terminal.get("status") != "complete"
        or terminal.get("contract_id") != campaign.CONTRACT_ID
        or terminal.get("completed_worker_count") != campaign.EXPECTED_WORKERS
    ):
        raise RuntimeError("compute-ladder terminal record is not complete")
    raw_path, raw_bytes = _exact_artifact(
        input_dir,
        terminal.get("raw"),
        expected_path="raw.json",
        field="raw artifact",
    )
    raw = _decode_object(raw_bytes, "raw artifact")
    if (
        set(raw)
        != {
            "schema_version",
            "kind",
            "contract_id",
            "started_at_utc",
            "completed_at_utc",
            "manifest",
            "workers",
            "rows",
        }
        or raw.get("schema_version") != 1
        or raw.get("kind") != "v011_compute_ladder_raw"
        or raw.get("contract_id") != campaign.CONTRACT_ID
        or raw.get("completed_at_utc") != terminal.get("completed_at_utc")
        or not isinstance(raw.get("started_at_utc"), str)
    ):
        raise RuntimeError("compute-ladder raw artifact identity drifted")
    _manifest_path, manifest_bytes = _exact_artifact(
        input_dir,
        raw.get("manifest"),
        expected_path="manifest.json",
        field="manifest artifact",
    )
    manifest = _decode_object(manifest_bytes, "manifest artifact")
    provenance = _verify_source_manifest(manifest)

    artifacts = raw.get("workers")
    embedded_rows = raw.get("rows")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != campaign.EXPECTED_WORKERS
        or not isinstance(embedded_rows, list)
        or len(embedded_rows) != campaign.EXPECTED_WORKERS
    ):
        raise RuntimeError("compute-ladder worker collection is incomplete")
    expected_paths = {
        f"workers/{index:03d}.json" for index in range(campaign.EXPECTED_WORKERS)
    }
    observed_paths = {
        str(path.relative_to(input_dir)) for path in (input_dir / "workers").rglob("*")
    }
    if observed_paths != expected_paths:
        raise RuntimeError("on-disk compute-ladder worker set drifted")
    rows: list[dict[str, Any]] = []
    parent_pid = None
    seen_paths: set[str] = set()
    for index, artifact in enumerate(artifacts):
        expected_path = f"workers/{index:03d}.json"
        worker_path, worker_bytes = _exact_artifact(
            input_dir,
            artifact,
            expected_path=expected_path,
            field=f"worker artifact {index}",
        )
        seen_paths.add(str(worker_path.relative_to(input_dir)))
        payload = _decode_object(worker_bytes, f"worker {index}")
        if payload != embedded_rows[index]:
            raise RuntimeError(f"embedded worker {index} changed")
        row, observed_parent = _verify_worker(
            payload,
            expected_index=index,
            parent_pid=parent_pid,
            manifest=manifest,
        )
        if parent_pid is None:
            parent_pid = observed_parent
        rows.append(row)
    if seen_paths != expected_paths:
        raise RuntimeError("attested compute-ladder worker set drifted")

    by_coordinate: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_coordinate[(row["dataset"], row["repeat"], row["fold"])].append(row)
    if len(by_coordinate) != campaign.EXPECTED_COORDINATES:
        raise RuntimeError("compute-ladder coordinate count drifted")
    for coordinate, block in by_coordinate.items():
        if {row["arm"] for row in block} != set(campaign.ARM_SPECS):
            raise RuntimeError(f"compute-ladder arm block is incomplete: {coordinate}")
        invariant_fields = (
            "task_id",
            "train_rows",
            "test_rows",
            "feature_count",
            "categorical_features",
            "fingerprints",
        )
        for field in invariant_fields:
            values = [row[field] for row in block]
            if values[1:] != values[:-1]:
                raise RuntimeError(
                    f"cross-arm dataset identity differs at {coordinate}: {field}"
                )
    provenance.update(
        {
            "contract_id": contract["contract_id"],
            "protocol_sha256": contract["protocol_sha256"],
            "ordered_grid_sha256": campaign.ordered_grid_sha256(),
            "manifest_sha256": campaign._sha256_bytes(manifest_bytes),
            "raw_sha256": campaign._sha256_bytes(raw_bytes),
            "raw_bytes": len(raw_bytes),
            "parent_pid": parent_pid,
            "completed_at_utc": terminal["completed_at_utc"],
            "raw_path": str(raw_path),
        }
    )
    return manifest, rows, provenance


def _contrast_definitions() -> list[tuple[str, str, str]]:
    contrasts = [
        ("reference", arm, campaign.CHIMERA_DEFAULT) for arm in campaign.ARM_SPECS
    ]
    contrasts.extend(("counterpart", darko, chimera) for darko, chimera in COUNTERPARTS)
    return contrasts


def pair_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["dataset"], row["repeat"], row["fold"], row["arm"]): row for row in rows
    }
    paired: list[dict[str, Any]] = []
    for kind, numerator_arm, denominator_arm in _contrast_definitions():
        contrast = (
            f"{campaign.ARM_SPECS[numerator_arm]['code']}/"
            f"{campaign.ARM_SPECS[denominator_arm]['code']}"
        )
        for dataset, repeat, fold in campaign.expected_coordinates():
            numerator = index[(dataset, repeat, fold, numerator_arm)]
            denominator = index[(dataset, repeat, fold, denominator_arm)]
            item: dict[str, Any] = {
                "comparison_kind": kind,
                "contrast": contrast,
                "numerator_arm": numerator_arm,
                "numerator_engine": numerator["engine"],
                "numerator_profile": numerator["profile"],
                "denominator_arm": denominator_arm,
                "denominator_engine": denominator["engine"],
                "denominator_profile": denominator["profile"],
                "dataset": dataset,
                "task_id": campaign.TASKS[dataset],
                "repeat": repeat,
                "fold": fold,
                "registered_fold": 3 * repeat + fold,
            }
            for metric in METRICS:
                numerator_value = _metric_value(numerator, metric)
                denominator_value = _metric_value(denominator, metric)
                item[f"numerator_{metric}"] = numerator_value
                item[f"denominator_{metric}"] = denominator_value
                if numerator_value > 0.0 and denominator_value > 0.0:
                    ratio = numerator_value / denominator_value
                    item[f"{metric}_ratio"] = ratio
                    item[f"{metric}_log_ratio"] = math.log(ratio)
                else:
                    item[f"{metric}_ratio"] = None
                    item[f"{metric}_log_ratio"] = None
            paired.append(item)
    expected = len(_contrast_definitions()) * campaign.EXPECTED_COORDINATES
    if len(paired) != expected:
        raise RuntimeError("paired compute-ladder grid is incomplete")
    return paired


def _bootstrap(logs_by_dataset: Mapping[str, Sequence[float]]) -> np.ndarray:
    rng = np.random.default_rng(campaign.BOOTSTRAP_SEED)
    draws = np.empty(campaign.BOOTSTRAP_DRAWS, dtype=np.float64)
    datasets = list(campaign.TASKS)
    for draw in range(campaign.BOOTSTRAP_DRAWS):
        points = []
        for dataset in datasets:
            values = np.asarray(logs_by_dataset[dataset], dtype=np.float64)
            sampled = rng.integers(0, len(values), size=len(values))
            points.append(float(np.mean(values[sampled])))
        draws[draw] = float(np.mean(points))
    return draws


def _head_to_head(values: Sequence[float]) -> dict[str, Any]:
    wins = sum(value < 1.0 for value in values)
    losses = sum(value > 1.0 for value in values)
    ties = len(values) - wins - losses
    decisive = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_excluding_ties": None if decisive == 0 else wins / decisive,
        "win_share_with_half_ties": (wins + 0.5 * ties) / len(values),
    }


def _summarize_contrast(
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(selected) != campaign.EXPECTED_COORDINATES:
        raise RuntimeError("contrast does not contain every coordinate")
    first = selected[0]
    metrics: dict[str, Any] = {}
    dataset_ratios: dict[str, dict[str, float | None]] = defaultdict(dict)
    for metric in METRICS:
        logs_by_dataset: dict[str, list[float]] = {}
        available = True
        for dataset in campaign.TASKS:
            values = [
                row[f"{metric}_log_ratio"]
                for row in selected
                if row["dataset"] == dataset
            ]
            if len(values) != len(campaign.COORDINATE_PAIRS) or any(
                value is None for value in values
            ):
                available = False
                break
            logs_by_dataset[dataset] = [float(value) for value in values]
        if not available:
            metrics[metric] = {
                "available": False,
                "reason": "zero_peak_delta_observation",
            }
            for dataset in campaign.TASKS:
                dataset_ratios[dataset][metric] = None
            continue
        dataset_logs = {
            dataset: math.fsum(values) / len(values)
            for dataset, values in logs_by_dataset.items()
        }
        point = math.fsum(dataset_logs.values()) / len(dataset_logs)
        draws = _bootstrap(logs_by_dataset)
        ratio = math.exp(point)
        metrics[metric] = {
            "available": True,
            "ratio": ratio,
            "pct": 100.0 * (ratio - 1.0),
            "bootstrap_ratio": {
                "draws": campaign.BOOTSTRAP_DRAWS,
                "p025": math.exp(float(np.quantile(draws, 0.025))),
                "median": math.exp(float(np.quantile(draws, 0.5))),
                "p975": math.exp(float(np.quantile(draws, 0.975))),
            },
        }
        for dataset, value in dataset_logs.items():
            dataset_ratios[dataset][metric] = math.exp(value)
    per_dataset: list[dict[str, Any]] = []
    for dataset in campaign.TASKS:
        block = [row for row in selected if row["dataset"] == dataset]
        quality = [float(row["test_rmse_ratio"]) for row in block]
        item: dict[str, Any] = {
            "comparison_kind": first["comparison_kind"],
            "contrast": first["contrast"],
            "numerator_arm": first["numerator_arm"],
            "numerator_engine": first["numerator_engine"],
            "numerator_profile": first["numerator_profile"],
            "denominator_arm": first["denominator_arm"],
            "denominator_engine": first["denominator_engine"],
            "denominator_profile": first["denominator_profile"],
            "dataset": dataset,
            "task_id": campaign.TASKS[dataset],
            "coordinate_count": len(block),
            "quality_wins": sum(value < 1.0 for value in quality),
            "quality_losses": sum(value > 1.0 for value in quality),
            "quality_ties": sum(value == 1.0 for value in quality),
        }
        for metric in METRICS:
            item[f"{metric}_ratio"] = dataset_ratios[dataset][metric]
        per_dataset.append(item)
    quality_coordinates = [float(row["test_rmse_ratio"]) for row in selected]
    quality_datasets = [
        float(item["test_rmse_ratio"])
        for item in per_dataset
        if item["test_rmse_ratio"] is not None
    ]
    summary = {
        "comparison_kind": first["comparison_kind"],
        "contrast": first["contrast"],
        "numerator_arm": first["numerator_arm"],
        "numerator_engine": first["numerator_engine"],
        "numerator_profile": first["numerator_profile"],
        "denominator_arm": first["denominator_arm"],
        "denominator_engine": first["denominator_engine"],
        "denominator_profile": first["denominator_profile"],
        "paired_coordinates": len(selected),
        "datasets": len(campaign.TASKS),
        "metrics": metrics,
        "head_to_head": {
            "coordinate_quality": _head_to_head(quality_coordinates),
            "equal_dataset_quality": _head_to_head(quality_datasets),
            "descriptive_only": True,
        },
    }
    return summary, per_dataset


def summarize_contrasts(
    paired: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    references = []
    counterparts = []
    per_dataset = []
    for kind, numerator, denominator in _contrast_definitions():
        selected = [
            row
            for row in paired
            if row["comparison_kind"] == kind
            and row["numerator_arm"] == numerator
            and row["denominator_arm"] == denominator
        ]
        summary, dataset_rows = _summarize_contrast(selected)
        (references if kind == "reference" else counterparts).append(summary)
        per_dataset.extend(dataset_rows)
    return references, counterparts, per_dataset


def _frontier(
    reference_summaries: Sequence[Mapping[str, Any]],
    cost_metric: str,
) -> dict[str, Any]:
    points = []
    for summary in reference_summaries:
        quality = summary["metrics"]["test_rmse"]
        cost = summary["metrics"][cost_metric]
        if not quality["available"] or not cost["available"]:
            raise RuntimeError(f"frontier metric is unavailable: {cost_metric}")
        points.append(
            {
                "arm": summary["numerator_arm"],
                "code": campaign.ARM_SPECS[summary["numerator_arm"]]["code"],
                "engine": summary["numerator_engine"],
                "profile": summary["numerator_profile"],
                "quality_ratio": float(quality["ratio"]),
                "cost_ratio": float(cost["ratio"]),
            }
        )
    nondominated: dict[str, list[dict[str, Any]]] = {}
    dominated: dict[str, list[str]] = {}
    for engine in ("darkofit", "chimeraboost"):
        engine_points = [point for point in points if point["engine"] == engine]
        keep = []
        drop = []
        for point in engine_points:
            is_dominated = any(
                other["arm"] != point["arm"]
                and other["cost_ratio"] <= point["cost_ratio"]
                and other["quality_ratio"] <= point["quality_ratio"]
                and (
                    other["cost_ratio"] < point["cost_ratio"]
                    or other["quality_ratio"] < point["quality_ratio"]
                )
                for other in engine_points
            )
            (drop if is_dominated else keep).append(point)
        nondominated[engine] = sorted(
            keep,
            key=lambda item: (item["cost_ratio"], item["quality_ratio"]),
        )
        dominated[engine] = sorted(item["arm"] for item in drop)
    budgets = sorted(
        {
            point["cost_ratio"]
            for engine_points in nondominated.values()
            for point in engine_points
        }
    )
    comparable = []
    for budget in budgets:
        achieved = {}
        for engine in ("darkofit", "chimeraboost"):
            eligible = [
                point for point in nondominated[engine] if point["cost_ratio"] <= budget
            ]
            if eligible:
                achieved[engine] = min(
                    eligible,
                    key=lambda item: (
                        item["quality_ratio"],
                        item["cost_ratio"],
                        list(campaign.BASE_ORDER).index(item["arm"]),
                    ),
                )
        if set(achieved) != {"darkofit", "chimeraboost"}:
            continue
        darko_quality = achieved["darkofit"]["quality_ratio"]
        chimera_quality = achieved["chimeraboost"]["quality_ratio"]
        comparable.append(
            {
                "budget_ratio_to_chimeraboost_default": budget,
                "darkofit_arm": achieved["darkofit"]["arm"],
                "darkofit_quality_ratio": darko_quality,
                "chimeraboost_arm": achieved["chimeraboost"]["arm"],
                "chimeraboost_quality_ratio": chimera_quality,
                "darkofit_minus_chimeraboost_quality_ratio": (
                    darko_quality - chimera_quality
                ),
                "darkofit_no_worse": darko_quality <= chimera_quality,
            }
        )
    no_worse = sum(item["darkofit_no_worse"] for item in comparable)
    strict = sum(
        item["darkofit_quality_ratio"] < item["chimeraboost_quality_ratio"]
        for item in comparable
    )
    return {
        "cost_metric": cost_metric,
        "reference_arm": campaign.CHIMERA_DEFAULT,
        "points": points,
        "nondominated_points": nondominated,
        "dominated_arms": dominated,
        "comparable_budgets": comparable,
        "comparable_budget_count": len(comparable),
        "darkofit_no_worse_budget_count": no_worse,
        "darkofit_strictly_better_budget_count": strict,
        "darkofit_full_curve_dominance": bool(comparable)
        and no_worse == len(comparable),
        "interpolation_used": False,
    }


def summarize(
    paired: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    references, counterparts, per_dataset = summarize_contrasts(paired)
    frontiers = {
        metric: _frontier(references, metric)
        for metric in ("fit_seconds", "prediction_seconds_per_call")
    }
    counterpart_peak_rss = {
        summary["contrast"]: summary["metrics"]["fit_rss_peak_bytes"]["ratio"]
        for summary in counterparts
    }
    memory_retained = all(
        value is not None and float(value) <= 1.0
        for value in counterpart_peak_rss.values()
    )
    strict_victory = (
        frontiers["fit_seconds"]["darkofit_full_curve_dominance"]
        and frontiers["prediction_seconds_per_call"]["darkofit_full_curve_dominance"]
        and memory_retained
    )
    summary = {
        "schema_version": 1,
        "campaign": campaign.CONTRACT_ID,
        "decision": "spent_descriptive_release_scoreboard",
        "policy_advancement_allowed": False,
        "counts": {
            "workers": campaign.EXPECTED_WORKERS,
            "coordinates": campaign.EXPECTED_COORDINATES,
            "datasets": len(campaign.TASKS),
            "arms": len(campaign.ARM_SPECS),
            "paired_rows": len(paired),
            "per_dataset_rows": len(per_dataset),
        },
        "aggregation": {
            "datasets_fixed_and_equally_weighted": True,
            "coordinates_per_dataset": len(campaign.COORDINATE_PAIRS),
            "bootstrap_draws": campaign.BOOTSTRAP_DRAWS,
            "bootstrap_seed": campaign.BOOTSTRAP_SEED,
            "resampling_unit": "coordinates_within_each_fixed_dataset",
            "independent_dataset_claim": False,
        },
        "arms_vs_chimeraboost_default": references,
        "matched_profile_contrasts": counterparts,
        "frontiers": frontiers,
        "memory_retention": {
            "metric": "fit_rss_peak_bytes",
            "matched_profile_ratios": counterpart_peak_rss,
            "all_no_worse": memory_retained,
        },
        "strict_program_verdict": {
            "basis": "predeclared_equal_dataset_point_estimates",
            "fit_frontier_dominance": frontiers["fit_seconds"][
                "darkofit_full_curve_dominance"
            ],
            "prediction_frontier_dominance": frontiers["prediction_seconds_per_call"][
                "darkofit_full_curve_dominance"
            ],
            "counterpart_peak_rss_no_worse": memory_retained,
            "strict_pareto_victory": strict_victory,
        },
        "scope": {
            "latest_chimeraboost_release_at_worker_zero": campaign.CHIMERABOOST_TAG,
            "historical_m2_regression_tasks": True,
            "direct_public_estimators": True,
            "catboost_included": False,
            "classification_included": False,
            "tabarena_placement": False,
            "fresh_confirmation": False,
            "lockbox": False,
        },
        "provenance": dict(provenance),
    }
    return summary, per_dataset


def _ratio_text(summary: Mapping[str, Any], metric: str) -> str:
    record = summary["metrics"][metric]
    return "n/a" if not record["available"] else f"{record['ratio']:.4f}x"


def _ratio_interval_text(summary: Mapping[str, Any], metric: str) -> str:
    record = summary["metrics"][metric]
    if not record["available"]:
        return "n/a"
    interval = record["bootstrap_ratio"]
    return (
        f"{record['ratio']:.4f}x " f"[{interval['p025']:.4f}, {interval['p975']:.4f}]"
    )


def render_report(summary: Mapping[str, Any]) -> str:
    verdict = summary["strict_program_verdict"]
    lines = [
        "# DarkoFit v0.11 release compute-ladder scoreboard",
        "",
        (
            "Status: **spent, descriptive release evidence; no policy "
            "advancement is authorized.**"
        ),
        "",
        "All ratios are numerator / pinned ChimeraBoost v0.20 default unless the",
        "table says matched profile; lower is better. Point estimates equally",
        "weight the 13 fixed historical regression datasets after averaging each",
        "dataset's three registered split log ratios. Intervals resample those",
        "three coordinates within each fixed dataset; they do not imply 13",
        "independent datasets.",
        "",
        "## Public compute points",
        "",
        (
            "| Arm | Quality [95%] | Fit | Predict/call | Peak RSS | "
            "Peak-start RSS | Dataset W-L-T |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["arms_vs_chimeraboost_default"]:
        counts = item["head_to_head"]["equal_dataset_quality"]
        quality = _ratio_interval_text(item, "test_rmse")
        lines.append(
            f"| {campaign.ARM_SPECS[item['numerator_arm']]['code']} "
            f"({item['numerator_profile']}) | {quality} | "
            f"{_ratio_text(item, 'fit_seconds')} | "
            f"{_ratio_text(item, 'prediction_seconds_per_call')} | "
            f"{_ratio_text(item, 'fit_rss_peak_bytes')} | "
            f"{_ratio_text(item, 'fit_rss_peak_delta_bytes')} | "
            f"{counts['wins']}-{counts['losses']}-{counts['ties']} |"
        )
    lines.extend(
        [
            "",
            "## Matched public profiles",
            "",
            (
                "| Contrast | Quality [95%] | Fit | Predict/call | Peak RSS | "
                "Dataset W-L-T |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["matched_profile_contrasts"]:
        counts = item["head_to_head"]["equal_dataset_quality"]
        lines.append(
            f"| {item['contrast']} | {_ratio_interval_text(item, 'test_rmse')} | "
            f"{_ratio_text(item, 'fit_seconds')} | "
            f"{_ratio_text(item, 'prediction_seconds_per_call')} | "
            f"{_ratio_text(item, 'fit_rss_peak_bytes')} | "
            f"{counts['wins']}-{counts['losses']}-{counts['ties']} |"
        )
    lines.extend(
        [
            "",
            "## Stepwise frontier verdicts",
            "",
            (
                "| Axis | Comparable budgets | DarkoFit no worse | DarkoFit "
                "strictly better | Full-curve dominance |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric, frontier in summary["frontiers"].items():
        lines.append(
            f"| {metric} | {frontier['comparable_budget_count']} | "
            f"{frontier['darkofit_no_worse_budget_count']} | "
            f"{frontier['darkofit_strictly_better_budget_count']} | "
            f"{'yes' if frontier['darkofit_full_curve_dominance'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The strict program target requires fit-frontier dominance, prediction-",
            "frontier dominance, and no worse peak RSS at all three matched public",
            "profiles. The verdict below is the predeclared point-estimate readout;",
            "the paired intervals and complete coordinate rows remain part of the",
            "evidence and prevent it from being read as a certificate.",
            "",
            (
                "**Strict Pareto victory: "
                f"{'YES' if verdict['strict_pareto_victory'] else 'NO'}.**"
            ),
            "",
            "This result covers regression on the fixed historical M2 task set. It",
            "does not cover classification, CatBoost, fresh confirmation, lockbox",
            "evidence, or TabArena placement.",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], field: str) -> bytes:
    if not rows:
        raise RuntimeError(f"{field} is empty")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise RuntimeError(f"{field} columns are inconsistent")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def analyze(input_dir: Path) -> dict[str, Any]:
    input_dir = Path(os.path.abspath(input_dir.expanduser()))
    baseline = verify_campaign(input_dir)
    _manifest, rows, provenance = baseline
    paired = pair_rows(rows)
    summary, per_dataset = summarize(paired, provenance)
    outputs = {
        input_dir
        / "coordinate_ratios.csv": _csv_bytes(
            paired,
            "coordinate ratios",
        ),
        input_dir / "per_dataset.csv": _csv_bytes(per_dataset, "per-dataset rows"),
        input_dir
        / "summary.json": (
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        input_dir / "report.md": render_report(summary).encode("utf-8"),
    }
    attestation_path = input_dir / "analysis_attestation.json"
    if (
        attestation_path.exists()
        or attestation_path.is_symlink()
        or any(path.exists() or path.is_symlink() for path in outputs)
    ):
        raise RuntimeError("compute-ladder analysis outputs are create-only")
    if verify_campaign(input_dir) != baseline:
        raise RuntimeError("compute-ladder campaign changed during analysis")
    written: list[Path] = []
    try:
        for path, payload in outputs.items():
            _write_create_only(path, payload)
            written.append(path)
        if verify_campaign(input_dir) != baseline:
            raise RuntimeError("compute-ladder campaign changed while publishing")
        attestation = {
            "schema_version": 1,
            "kind": "v011_compute_ladder_analysis_attestation",
            "contract_id": campaign.CONTRACT_ID,
            "decision": "spent_descriptive_release_scoreboard",
            "strict_pareto_victory": summary["strict_program_verdict"][
                "strict_pareto_victory"
            ],
            "input": {
                "raw_sha256": provenance["raw_sha256"],
                "manifest_sha256": provenance["manifest_sha256"],
                "contract_sha256": provenance["contract_sha256"],
            },
            "outputs": {
                path.name: campaign._stable_artifact(path, input_dir)
                for path in outputs
            },
        }
        _write_create_only(
            attestation_path,
            (
                json.dumps(attestation, allow_nan=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
        )
        written.append(attestation_path)
    except BaseException:
        for path in reversed(written):
            path.unlink(missing_ok=True)
        raise
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = analyze(args.input_dir)
    verdict = summary["strict_program_verdict"]["strict_pareto_victory"]
    print(
        f"analyzed {campaign.EXPECTED_WORKERS} workers; "
        f"strict_pareto_victory={str(verdict).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
