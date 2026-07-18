#!/usr/bin/env python3
"""Derive a cost-aware engagement margin from the spent cross-feature screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_smooth_cross_features as source_analyzer  # noqa: E402


SOURCE = ROOT / "benchmarks" / "smooth_cross_features.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "smooth_cross_margin_analysis.json"
MARGIN_GRID = tuple(index / 100 for index in range(0, 11))
FROZEN_ARTIFACT_SHA256 = (
    "cd5e4c2fe97077224e3bdf01f267654ee817a65e786f5baf1a92371c1d88e726"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_mutable_symlink_output_parents(path):
    for directory in (path.parent, *path.parent.parents):
        if directory.is_symlink() and os.access(directory.parent, os.W_OK):
            raise RuntimeError(
                f"refusing symlink output directory: {directory}"
            )


def _create_missing_directories(directory, created):
    missing = []
    current = directory
    while not current.exists():
        if current.is_symlink():
            raise RuntimeError(
                f"refusing symlink output directory: {current}"
            )
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise NotADirectoryError(
            f"output parent is not a directory: {current}"
        )
    for current in reversed(missing):
        try:
            current.mkdir()
        except FileExistsError:
            if not current.is_dir() or current.is_symlink():
                raise
        else:
            metadata = current.lstat()
            created.append(
                (current, (metadata.st_dev, metadata.st_ino))
            )


def _remove_owned_empty_directories(directories):
    for directory, identity in reversed(directories):
        try:
            current = directory.lstat()
            if (
                stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == identity
            ):
                directory.rmdir()
        except OSError:
            pass


def _assert_output_parent_identity(path, identity):
    _reject_mutable_symlink_output_parents(path)
    try:
        current = path.parent.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"output parent changed: {path.parent}"
        ) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
    ):
        raise RuntimeError(f"output parent changed: {path.parent}")


def _open_output_parent(path):
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptor = os.open(path.parent, flags)
    try:
        current = os.fstat(descriptor)
        identity = (current.st_dev, current.st_ino)
        if not stat.S_ISDIR(current.st_mode):
            raise RuntimeError(
                f"output parent is not a directory: {path.parent}"
            )
        _assert_output_parent_identity(path, identity)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _temporary_at(directory_descriptor, output_name):
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _attempt in range(128):
        name = f".{output_name}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                name,
                flags,
                0o600,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            continue
        current = os.fstat(descriptor)
        return (
            descriptor,
            name,
            (current.st_dev, current.st_ino),
        )
    raise FileExistsError(
        f"unable to reserve temporary output for {output_name}"
    )


def _stat_at(directory_descriptor, name):
    return os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )


def _exists_at(directory_descriptor, name):
    try:
        _stat_at(directory_descriptor, name)
    except FileNotFoundError:
        return False
    return True


def _unlink_if_owned_at(directory_descriptor, name, identity):
    try:
        current = _stat_at(directory_descriptor, name)
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == identity
    ):
        os.unlink(name, dir_fd=directory_descriptor)


def _atomic_create(path, value):
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    _reject_mutable_symlink_output_parents(path)
    created_directories = []
    parent_descriptor = None
    parent_identity = None
    temporary_name = None
    identity = None
    created = False
    try:
        _create_missing_directories(path.parent, created_directories)
        parent_descriptor, parent_identity = _open_output_parent(path)
        if _exists_at(parent_descriptor, path.name):
            raise FileExistsError(
                f"refusing to replace existing output: {path}"
            )
        descriptor, temporary_name, identity = _temporary_at(
            parent_descriptor,
            path.name,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_output_parent_identity(path, parent_identity)
        current = _stat_at(parent_descriptor, temporary_name)
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != identity
        ):
            raise RuntimeError(
                "temporary output changed before publication: "
                f"{path.parent / temporary_name}"
            )
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        created = True
        _assert_output_parent_identity(path, parent_identity)
        current = _stat_at(parent_descriptor, path.name)
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != identity
        ):
            raise RuntimeError(f"published output changed: {path}")
        _unlink_if_owned_at(
            parent_descriptor,
            temporary_name,
            identity,
        )
        _assert_output_parent_identity(path, parent_identity)
    except BaseException:
        if parent_descriptor is not None and identity is not None:
            if created:
                try:
                    _unlink_if_owned_at(
                        parent_descriptor,
                        path.name,
                        identity,
                    )
                except OSError:
                    pass
            if temporary_name is not None:
                try:
                    _unlink_if_owned_at(
                        parent_descriptor,
                        temporary_name,
                        identity,
                    )
                except OSError:
                    pass
        _remove_owned_empty_directories(created_directories)
        raise
    finally:
        if parent_descriptor is not None:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass


def _geomean(values):
    values = np.asarray(list(values), dtype=np.float64)
    if (
        values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise RuntimeError("geomean requires finite positive values")
    return float(np.exp(np.mean(np.log(values))))


def _positive_float(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"{label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive")
    return result


def _margin(row):
    if not isinstance(row, dict):
        raise RuntimeError("margin row must be an object")
    base_record = row.get("base")
    selected_record = row.get("selected")
    if not isinstance(base_record, dict) or not isinstance(selected_record, dict):
        raise RuntimeError("margin row fingerprints must be objects")
    base = _positive_float(
        base_record.get("best_validation_rmse"),
        "base validation RMSE",
    )
    selected = _positive_float(
        selected_record.get("best_validation_rmse"),
        "selected validation RMSE",
    )
    return float((base - selected) / base)


def evaluate_margin(rows, threshold):
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or float(threshold) < 0.0
    ):
        raise RuntimeError("margin threshold must be finite and nonnegative")
    threshold = float(threshold)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("margin rows must be a nonempty list")
    per_dataset = {}
    split_records = []
    coordinates = set()
    task_names = {}
    dataset_tasks = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("task_id"), int)
            or isinstance(row.get("task_id"), bool)
            or row["task_id"] <= 0
            or not isinstance(row.get("fold"), int)
            or isinstance(row.get("fold"), bool)
            or row["fold"] < 0
            or not isinstance(row.get("dataset_name"), str)
            or not row["dataset_name"]
            or not isinstance(row.get("cross_selected"), bool)
        ):
            raise RuntimeError("margin row ledger is invalid")
        coordinate = (row["task_id"], row["fold"])
        if coordinate in coordinates:
            raise RuntimeError("margin coordinate ledger repeats a split")
        coordinates.add(coordinate)
        existing_name = task_names.setdefault(
            row["task_id"], row["dataset_name"]
        )
        existing_task = dataset_tasks.setdefault(
            row["dataset_name"], row["task_id"]
        )
        if (
            existing_name != row["dataset_name"]
            or existing_task != row["task_id"]
        ):
            raise RuntimeError("margin task/dataset identity changed")
        margin = _margin(row)
        if row["cross_selected"] is not (margin > 0.0):
            raise RuntimeError("margin cross-selection ledger is invalid")
        engage = row["cross_selected"] and margin >= threshold
        selected_test = _positive_float(
            row["selected"].get("test_rmse"),
            "selected test RMSE",
        )
        base_test = _positive_float(
            row["base"].get("test_rmse"),
            "base test RMSE",
        )
        ratio = (
            selected_test / base_test
            if engage
            else 1.0
        )
        per_dataset.setdefault(row["dataset_name"], []).append(ratio)
        split_records.append(
            {
                "task_id": int(row["task_id"]),
                "dataset_name": row["dataset_name"],
                "fold": int(row["fold"]),
                "validation_improvement": margin,
                "engaged": engage,
                "test_ratio": ratio,
            }
        )
    if len(per_dataset) < 2:
        raise RuntimeError("margin analysis requires at least two datasets")
    dataset_ratios = {
        name: _geomean(ratios) for name, ratios in per_dataset.items()
    }
    leave_one_out = {
        omitted: _geomean(
            [
                ratio
                for name, ratio in dataset_ratios.items()
                if name != omitted
            ]
        )
        for omitted in dataset_ratios
    }
    return {
        "minimum_validation_improvement": float(threshold),
        "engaged_coordinates": int(
            sum(record["engaged"] for record in split_records)
        ),
        "equal_dataset_geomean_ratio": _geomean(dataset_ratios.values()),
        "worst_dataset_ratio": float(max(dataset_ratios.values())),
        "worst_split_ratio": float(
            max(record["test_ratio"] for record in split_records)
        ),
        "leave_one_out_equal_dataset_ratios": leave_one_out,
        "dataset_ratios": dataset_ratios,
        "split_records": split_records,
    }


def analyze(source, *, validate_source=True):
    if validate_source:
        source_analyzer.validate_artifact(source)
    if not isinstance(source, dict) or not isinstance(source.get("results"), list):
        raise RuntimeError("smooth cross-feature source result ledger is invalid")
    rows = source["results"]
    if (
        not isinstance(MARGIN_GRID, tuple)
        or not MARGIN_GRID
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in MARGIN_GRID
        )
        or tuple(sorted(set(MARGIN_GRID))) != MARGIN_GRID
    ):
        raise RuntimeError("smooth cross-margin grid is invalid")
    grid = [evaluate_margin(rows, threshold) for threshold in MARGIN_GRID]
    zero_harm = [
        record for record in grid if record["worst_split_ratio"] <= 1.0
    ]
    if not zero_harm:
        nominee = None
    else:
        nominee = min(
            zero_harm,
            key=lambda record: record["minimum_validation_improvement"],
        )
    return {
        "claim_tier": "development_policy_nomination",
        "fresh_claim_eligible": False,
        "margin_grid": list(MARGIN_GRID),
        "selection_rule": (
            "smallest whole-percentage validation margin on the declared "
            "grid with no observed split regression"
        ),
        "grid_results": grid,
        "nominee": nominee,
        "nominee_requires_fresh_confirmation": nominee is not None,
        "caveats": [
            "margin chosen on spent development outcomes",
            "three datasets are insufficient for a shipping claim",
            "full crossed audition cost is paid even when the guard declines",
            "zero observed harm is not a population guarantee",
        ],
    }


def _valid_aware_timestamp(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def validate_artifact(artifact, *, require_frozen=True):
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != 1
        or isinstance(artifact.get("schema_version"), bool)
        or not _valid_aware_timestamp(artifact.get("created_at"))
    ):
        raise RuntimeError("smooth cross-margin artifact has an unknown schema")
    source_payload = SOURCE.read_bytes()
    source = json.loads(source_payload)
    source_analyzer.validate_artifact(source)
    expected_source = {
        "path": str(SOURCE.relative_to(ROOT)),
        "sha256": hashlib.sha256(source_payload).hexdigest(),
        "source_head": source["sources"]["darkofit"]["head"],
        "source_protocol_sha256": source["protocol"]["sha256"],
    }
    if artifact.get("source") != expected_source:
        raise RuntimeError("smooth cross-margin source ledger changed")
    regenerated = analyze(source, validate_source=False)
    if not source_analyzer._analysis_equal(
        artifact.get("analysis"),
        regenerated,
    ):
        raise RuntimeError(
            "smooth cross-margin stored analysis is not reproducible"
        )
    if (
        require_frozen
        and source_analyzer._json_sha256(artifact)
        != FROZEN_ARTIFACT_SHA256
    ):
        raise RuntimeError("smooth cross-margin frozen artifact changed")
    return regenerated


def run(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    source_payload = SOURCE.read_bytes()
    source = json.loads(source_payload)
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": source_sha256,
            "source_head": source["sources"]["darkofit"]["head"],
            "source_protocol_sha256": source["protocol"]["sha256"],
        },
        "analysis": analyze(source),
    }
    validate_artifact(artifact, require_frozen=False)
    _atomic_create(
        output,
        (
            json.dumps(
                artifact,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = run(args.output.resolve())
    print(json.dumps(artifact["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
