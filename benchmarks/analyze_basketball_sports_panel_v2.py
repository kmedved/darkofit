#!/usr/bin/env python3
"""Analyze the frozen T10 basketball sports panel 2 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import stat
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import build_basketball_sports_panel_v2 as panel_builder  # noqa: E402
from benchmarks import run_basketball_sports_panel_v2 as runner  # noqa: E402


AGGREGATE_BAR = 1.0
BOOTSTRAP_UPPER_BAR = 1.002
LEAVE_ONE_OUT_BAR = 1.003
WORST_LINEAGE_BAR = 1.02
GUARDRAIL_AGGREGATE_BAR = 1.005
GUARDRAIL_WORST_BAR = 1.02
MAX_COST_RATIO = 3.0
MAX_PAIRED_RATIO_IQR_OVER_MEDIAN = 0.20
BOOTSTRAP_SEED = 20_260_718
BOOTSTRAP_RESAMPLES = 100_000
FROZEN_RAW_SHA256 = (
    "787f7f34bf1e5207d231b01bc402c7a32174e24892b2118bb71d5ff4412517b3"
)


def _sha256(path: Path) -> str:
    return runner._sha256(path)


def _atomic_create(path: Path, value: bytes) -> None:
    runner._atomic_create(path, value)


def _reject_mutable_symlink_output_parents(path: Path) -> None:
    for directory in (path.parent, *path.parent.parents):
        if directory.is_symlink() and os.access(directory.parent, os.W_OK):
            raise RuntimeError(
                f"refusing symlink analyzer output directory: {directory}"
            )


def _create_missing_directories(
    directory: Path,
    created: list[tuple[Path, tuple[int, int]]],
) -> None:
    missing = []
    current = directory
    while not current.exists():
        if current.is_symlink():
            raise RuntimeError(
                f"refusing symlink analyzer output directory: {current}"
            )
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise NotADirectoryError(
            f"analyzer output parent is not a directory: {current}"
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


def _remove_owned_empty_directories(
    directories: list[tuple[Path, tuple[int, int]]],
) -> None:
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


def _assert_output_parent_identity(
    path: Path,
    identity: tuple[int, int],
) -> None:
    _reject_mutable_symlink_output_parents(path)
    try:
        current = path.parent.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"analyzer output parent changed: {path.parent}"
        ) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
    ):
        raise RuntimeError(
            f"analyzer output parent changed: {path.parent}"
        )


def _open_output_parent(path: Path) -> tuple[int, tuple[int, int]]:
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
                f"analyzer output parent is not a directory: {path.parent}"
            )
        _assert_output_parent_identity(path, identity)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _temporary_at(
    directory_descriptor: int,
    output_name: str,
) -> tuple[int, str, tuple[int, int]]:
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
        f"unable to reserve analyzer temporary output for {output_name}"
    )


def _stat_at(directory_descriptor: int, name: str) -> os.stat_result:
    return os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )


def _exists_at(directory_descriptor: int, name: str) -> bool:
    try:
        _stat_at(directory_descriptor, name)
    except FileNotFoundError:
        return False
    return True


def _unlink_if_owned_at(
    directory_descriptor: int,
    name: str,
    identity: tuple[int, int],
) -> None:
    try:
        current = _stat_at(directory_descriptor, name)
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == identity
    ):
        os.unlink(name, dir_fd=directory_descriptor)


def _atomic_create_many(payloads: dict[Path, bytes]) -> None:
    normalized = {
        Path(os.path.abspath(os.path.expanduser(os.fspath(path)))): payload
        for path, payload in payloads.items()
    }
    if len(normalized) != len(payloads):
        raise ValueError("sports panel 2 analyzer output paths collide")
    for path in normalized:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing analyzer output: {path}")
        _reject_mutable_symlink_output_parents(path)

    parents: dict[Path, tuple[int, tuple[int, int]]] = {}
    temporary: dict[
        Path,
        tuple[int, str, tuple[int, int], tuple[int, int]],
    ] = {}
    temporary_paths: list[tuple[int, str, tuple[int, int]]] = []
    created: list[tuple[int, str, tuple[int, int]]] = []
    created_directories: list[tuple[Path, tuple[int, int]]] = []
    try:
        for path in normalized:
            _create_missing_directories(path.parent, created_directories)
            _reject_mutable_symlink_output_parents(path)
        for path in normalized:
            if path.parent not in parents:
                parents[path.parent] = _open_output_parent(path)
        for path in normalized:
            parent_descriptor, parent_identity = parents[path.parent]
            _assert_output_parent_identity(path, parent_identity)
            if _exists_at(parent_descriptor, path.name):
                raise FileExistsError(
                    f"refusing existing analyzer output: {path}"
                )
        for path, payload in normalized.items():
            parent_descriptor, parent_identity = parents[path.parent]
            _assert_output_parent_identity(path, parent_identity)
            descriptor, name, identity = _temporary_at(
                parent_descriptor,
                path.name,
            )
            temporary_paths.append(
                (parent_descriptor, name, identity)
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary[path] = (
                parent_descriptor,
                name,
                identity,
                parent_identity,
            )
        for path, (
            parent_descriptor,
            temporary_name,
            identity,
            parent_identity,
        ) in temporary.items():
            _assert_output_parent_identity(path, parent_identity)
            current = _stat_at(parent_descriptor, temporary_name)
            if (
                not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != identity
            ):
                raise RuntimeError(
                    f"analyzer temporary output changed: {path.parent / temporary_name}"
                )
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            created.append((parent_descriptor, path.name, identity))
            _assert_output_parent_identity(path, parent_identity)
            current = _stat_at(parent_descriptor, path.name)
            if (
                not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != identity
            ):
                raise RuntimeError(
                    f"analyzer published output changed: {path}"
                )
        for parent_descriptor, temporary_name, identity in temporary_paths:
            _unlink_if_owned_at(
                parent_descriptor,
                temporary_name,
                identity,
            )
        for path in normalized:
            _assert_output_parent_identity(
                path,
                parents[path.parent][1],
            )
    except BaseException:
        for parent_descriptor, name, identity in reversed(created):
            try:
                _unlink_if_owned_at(
                    parent_descriptor,
                    name,
                    identity,
                )
            except OSError:
                pass
        for parent_descriptor, name, identity in temporary_paths:
            try:
                _unlink_if_owned_at(
                    parent_descriptor,
                    name,
                    identity,
                )
            except OSError:
                pass
        _remove_owned_empty_directories(created_directories)
        raise
    finally:
        for descriptor, _identity in parents.values():
            try:
                os.close(descriptor)
            except OSError:
                pass


def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == identity
    ):
        path.unlink()


def _bound_payload(
    record: Any,
    expected_path: Path,
    *,
    label: str,
    digest_field: str = "sha256",
) -> bytes:
    expected_path = expected_path.resolve()
    expected_relative = str(expected_path.relative_to(REPO_ROOT))
    if not isinstance(record, dict) or record.get("path") != expected_relative:
        raise RuntimeError(f"raw sports panel 2 {label} path changed")
    payload = expected_path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if record.get(digest_field) != actual:
        raise RuntimeError(f"raw sports panel 2 {label} hash changed")
    return payload


def _is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_provenance(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError("raw sports panel 2 artifact must be an object")
    _bound_payload(
        raw.get("runner"),
        Path(runner.__file__),
        label="runner",
    )
    _bound_payload(
        raw.get("protocol"),
        runner.PROTOCOL_PATH,
        label="protocol",
    )
    manifest_payload = _bound_payload(
        raw.get("panel_manifest"),
        runner.MANIFEST_PATH,
        label="panel manifest",
        digest_field="file_sha256",
    )
    manifest = json.loads(manifest_payload)
    power_analysis = (
        manifest.get("power_analysis")
        if isinstance(manifest, dict)
        else None
    )
    processed_panel = (
        manifest.get("processed_panel")
        if isinstance(manifest, dict)
        else None
    )
    split = manifest.get("split") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or isinstance(manifest.get("schema_version"), bool)
        or manifest.get("name") != "darkofit_basketball_sports_panel_v2"
        or manifest.get("candidate_data_scored") is not False
        or manifest.get("comparators_scored") is not False
        or manifest.get("panel_spent") is not False
        or not isinstance(power_analysis, dict)
        or power_analysis.get("passes") is not True
        or not isinstance(processed_panel, dict)
        or not isinstance(split, dict)
    ):
        raise RuntimeError("raw sports panel 2 manifest boundary changed")
    panel_record = raw["panel_manifest"]
    if (
        panel_record.get("processed_panel_sha256")
        != processed_panel.get("sha256")
        or panel_record.get("split_manifest_sha256")
        != split.get("split_manifest_sha256")
        or panel_record.get("power_pass_probability")
        != power_analysis.get("pass_probability")
        or raw["protocol"] != manifest.get("protocol")
    ):
        raise RuntimeError("raw sports panel 2 manifest ledger changed")
    builder = manifest.get("builder", {})
    if not isinstance(builder, dict):
        raise RuntimeError("raw sports panel 2 manifest builder changed")
    _bound_payload(
        {
            "path": builder.get("path"),
            "sha256": builder.get("sha256"),
        },
        Path(panel_builder.__file__),
        label="builder",
    )
    _bound_payload(
        {
            "path": builder.get("shared_builder_path"),
            "sha256": builder.get("shared_builder_sha256"),
        },
        Path(panel_builder.base.__file__),
        label="shared builder",
    )
    execution = raw.get("execution")
    if (
        not isinstance(execution, dict)
        or execution.get("threads") != runner.EXPECTED_THREADS
        or execution.get("worker_count")
        != len(runner.BLOCK_ORDERS) * len(runner.ARM_ORDER)
        or execution.get("block_orders")
        != [list(order) for order in runner.BLOCK_ORDERS]
        or execution.get("candidate_or_comparator_outcomes_previously_scored")
        is not False
    ):
        raise RuntimeError("raw sports panel 2 execution ledger changed")
    sources = raw.get("source")
    if not isinstance(sources, dict) or set(sources) != {
        "darkofit",
        "chimeraboost",
    }:
        raise RuntimeError("raw sports panel 2 source ledger changed")
    darkofit = sources["darkofit"]
    chimeraboost = sources["chimeraboost"]
    tracked_main_refs = (
        chimeraboost.get("tracked_main_refs", {})
        if isinstance(chimeraboost, dict)
        else {}
    )
    if (
        not isinstance(darkofit, dict)
        or darkofit.get("branch") != runner.EXPECTED_BRANCH
        or darkofit.get("clean") is not True
        or darkofit.get("status") != []
        or darkofit.get("published_branch_ref")
        != f"origin/{runner.EXPECTED_BRANCH}"
        or darkofit.get("published_branch_head") != darkofit.get("head")
        or not _is_git_sha(darkofit.get("head"))
        or not isinstance(darkofit.get("path"), str)
        or not Path(darkofit["path"]).is_absolute()
    ):
        raise RuntimeError("raw sports panel 2 DarkoFit source ledger changed")
    if (
        not isinstance(chimeraboost, dict)
        or chimeraboost.get("clean") is not True
        or chimeraboost.get("status") != []
        or chimeraboost.get("head") != runner.EXPECTED_CHIMERABOOST_HEAD
        or not isinstance(chimeraboost.get("path"), str)
        or not Path(chimeraboost["path"]).is_absolute()
        or not isinstance(tracked_main_refs, dict)
        or not tracked_main_refs
        or any(
            value != runner.EXPECTED_CHIMERABOOST_HEAD
            for value in tracked_main_refs.values()
        )
    ):
        raise RuntimeError(
            "raw sports panel 2 ChimeraBoost source ledger changed"
        )
    return manifest


def _geometric_mean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or not len(array)
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        raise RuntimeError(
            "RMSE ratios must be finite, positive, and one-dimensional"
        )
    return float(np.exp(np.mean(np.log(array))))


def _finite_positive(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(
            f"raw sports panel 2 {label} must be finite and positive"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"raw sports panel 2 {label} must be finite and positive")
    return result


def _positive_int(value: Any, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise RuntimeError(
            f"raw sports panel 2 {label} must be a positive integer"
        )
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise RuntimeError(
            f"raw sports panel 2 {label} must be a nonnegative integer"
        )
    return value


def _score_rmse(score: Any, label: str) -> float:
    if not isinstance(score, dict):
        raise RuntimeError(f"raw sports panel 2 {label} score is invalid")
    _positive_int(score.get("rows"), f"{label} rows")
    rmse = _finite_positive(score.get("rmse"), f"{label} RMSE")
    if (
        not isinstance(score.get("r2"), (int, float))
        or isinstance(score.get("r2"), bool)
    ):
        raise RuntimeError(f"raw sports panel 2 {label} R2 must be finite")
    r2 = float(score["r2"])
    if not math.isfinite(r2) or r2 > 1.0:
        raise RuntimeError(
            f"raw sports panel 2 {label} R2 must be finite and at most 1"
        )
    return rmse


def _cell_key(cell: dict[str, Any]) -> tuple[int, str]:
    if not isinstance(cell, dict):
        raise RuntimeError("raw sports panel 2 cell ledger is invalid")
    season = cell.get("season")
    target = cell.get("target")
    if (
        not isinstance(season, int)
        or isinstance(season, bool)
        or not isinstance(target, str)
    ):
        raise RuntimeError("raw sports panel 2 cell ledger is invalid")
    return season, target


def _validate_manifest_evidence(
    raw: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    split = manifest.get("split")
    seasons = split.get("seasons") if isinstance(split, dict) else None
    if not isinstance(seasons, dict):
        raise RuntimeError("raw sports panel 2 manifest split ledger changed")
    for record in raw["repeats"]:
        result = record["result"]
        for cell in result["cells"]:
            season, _target = _cell_key(cell)
            expected = seasons.get(str(season))
            if not isinstance(expected, dict):
                raise RuntimeError(
                    "raw sports panel 2 result season is outside the manifest"
                )
            row_fields = (
                "primary_rows",
                "held_team_rows",
                "seen_player_rows",
                "cold_player_rows",
            )
            if any(
                _positive_int(cell.get(field), field) != expected.get(field)
                for field in row_fields
            ):
                raise RuntimeError(
                    "raw sports panel 2 result row ledger changed"
                )
            folds = cell.get("folds")
            expected_folds = expected.get("folds")
            if (
                not isinstance(folds, list)
                or not isinstance(expected_folds, list)
                or len(folds) != len(expected_folds)
            ):
                raise RuntimeError(
                    "raw sports panel 2 result fold ledger changed"
                )
            for observed, frozen in zip(folds, expected_folds):
                if (
                    not isinstance(observed, dict)
                    or _nonnegative_int(
                        observed.get("fold"),
                        "fold number",
                    )
                    != frozen.get("fold")
                    or not isinstance(observed.get("train_indices"), list)
                    or any(
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or index < 0
                        for index in observed["train_indices"]
                    )
                    or not isinstance(observed.get("test_indices"), list)
                    or any(
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or index < 0
                        for index in observed["test_indices"]
                    )
                    or observed.get("train_indices")
                    != frozen.get("train_indices")
                    or observed.get("test_indices") != frozen.get("test_indices")
                    or _positive_int(
                        observed.get("train_rows"),
                        "fold train rows",
                    )
                    != frozen.get("train_rows")
                    or _positive_int(
                        observed.get("test_rows"),
                        "fold test rows",
                    )
                    != frozen.get("test_rows")
                ):
                    raise RuntimeError(
                        "raw sports panel 2 result fold ledger changed"
                    )
            primary = cell.get("primary")
            guardrail_record = cell.get("guardrail")
            guardrail = (
                guardrail_record.get("scores")
                if isinstance(guardrail_record, dict)
                else None
            )
            if (
                not isinstance(primary, dict)
                or _positive_int(primary.get("rows"), "primary score rows")
                != expected.get("primary_rows")
                or not isinstance(guardrail, dict)
                or set(guardrail)
                != {"held_team", "seen_player", "cold_player"}
                or any(
                    not isinstance(guardrail[view], dict)
                    for view in guardrail
                )
                or _positive_int(
                    guardrail["held_team"].get("rows"),
                    "held-team score rows",
                )
                != expected.get("held_team_rows")
                or _positive_int(
                    guardrail["seen_player"].get("rows"),
                    "seen-player score rows",
                )
                != expected.get("seen_player_rows")
                or _positive_int(
                    guardrail["cold_player"].get("rows"),
                    "cold-player score rows",
                )
                != expected.get("cold_player_rows")
            ):
                raise RuntimeError(
                    "raw sports panel 2 result score-row ledger changed"
                )


def _canonical_results(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        raw.get("schema_version") != 1
        or isinstance(raw.get("schema_version"), bool)
        or raw.get("name") != "darkofit_basketball_sports_panel_raw_v2"
    ):
        raise RuntimeError("raw sports panel 2 artifact has an unknown schema")
    if raw.get("panel_spent_by_this_run") is not True:
        raise RuntimeError("raw sports panel 2 artifact is not marked spent")
    repeats = raw.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != (
        len(runner.BLOCK_ORDERS) * len(runner.ARM_ORDER)
    ):
        raise RuntimeError("raw sports panel 2 has an invalid worker count")
    fingerprint_ledger = raw.get("behavior_fingerprints")
    if (
        not isinstance(fingerprint_ledger, dict)
        or set(fingerprint_ledger) != set(runner.ARM_ORDER)
    ):
        raise RuntimeError("raw sports panel 2 lacks a fingerprint ledger")
    expected_positions = {
        (block, position): arm
        for block, order in enumerate(runner.BLOCK_ORDERS)
        for position, arm in enumerate(order)
    }
    observed_positions: set[tuple[int, int]] = set()
    grouped = {arm: [] for arm in runner.ARM_ORDER}
    for record in repeats:
        if not isinstance(record, dict):
            raise RuntimeError("raw sports panel 2 worker ledger is invalid")
        coordinate = (
            _nonnegative_int(record.get("block"), "worker block"),
            _nonnegative_int(record.get("position"), "worker position"),
        )
        if coordinate in observed_positions:
            raise RuntimeError("raw sports panel 2 repeats a worker coordinate")
        observed_positions.add(coordinate)
        arm = record.get("arm")
        if not isinstance(arm, str):
            raise RuntimeError("raw sports panel 2 worker arm is invalid")
        if expected_positions.get(coordinate) != arm:
            raise RuntimeError("raw sports panel 2 worker order changed")
        if record.get("order") != list(runner.BLOCK_ORDERS[coordinate[0]]):
            raise RuntimeError("raw sports panel 2 block ledger changed")
        result = record.get("result")
        if not isinstance(result, dict) or result.get("arm") != arm:
            raise RuntimeError("raw sports panel 2 worker arm is inconsistent")
        grouped[arm].append(result)
    if observed_positions != set(expected_positions):
        raise RuntimeError("raw sports panel 2 is missing worker coordinates")

    expected_keys = [
        (season, target)
        for season in panel_builder.SEASONS
        for target in panel_builder.TARGET_COLUMNS
    ]
    canonical: dict[str, dict[str, Any]] = {}
    for arm, results in grouped.items():
        if len(results) != len(runner.BLOCK_ORDERS):
            raise RuntimeError(f"raw sports panel 2 has wrong repeats for {arm}")
        fingerprints = set()
        for row in results:
            if not isinstance(row.get("cells"), list):
                raise RuntimeError(
                    f"raw sports panel 2 fingerprint payload changed for {arm}"
                )
            try:
                computed = harness.behavior_fingerprint(
                    runner._behavior_payload(row)
                )
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise RuntimeError(
                    f"raw sports panel 2 fingerprint payload changed for {arm}"
                ) from exc
            if row.get("behavior_fingerprint_sha256") != computed:
                raise RuntimeError(
                    f"raw sports panel 2 fingerprint payload changed for {arm}"
                )
            fingerprints.add(computed)
        if fingerprints != {fingerprint_ledger.get(arm)}:
            raise RuntimeError(f"raw sports panel 2 behavior changed for {arm}")
        keys = [_cell_key(cell) for cell in results[0]["cells"]]
        if keys != expected_keys or len(set(keys)) != len(keys):
            raise RuntimeError(f"raw sports panel 2 cells changed for {arm}")
        canonical[arm] = results[0]
    return canonical


def _cell_map(
    result: dict[str, Any],
) -> dict[tuple[int, str], dict[str, Any]]:
    if not isinstance(result, dict) or not isinstance(
        result.get("cells"), list
    ):
        raise RuntimeError("sports panel 2 arm cell ledger is invalid")
    cells = result["cells"]
    mapped = {_cell_key(cell): cell for cell in cells}
    if len(mapped) != len(cells):
        raise RuntimeError("sports panel 2 arm repeats a cell")
    return mapped


def _guardrail_scores(
    cell: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    guardrail = cell.get("guardrail")
    scores = (
        guardrail.get("scores")
        if isinstance(guardrail, dict)
        else None
    )
    if not isinstance(scores, dict):
        raise RuntimeError(
            f"raw sports panel 2 {label} guardrail ledger is invalid"
        )
    return scores


def _bootstrap_upper(ratios: np.ndarray) -> float:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(ratios),
        size=(BOOTSTRAP_RESAMPLES, len(ratios)),
    )
    aggregates = np.exp(np.mean(np.log(ratios[indices]), axis=1))
    return float(np.quantile(aggregates, 0.95))


def _quality_comparison(
    candidate: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    candidate_cells = _cell_map(candidate)
    control_cells = _cell_map(control)
    expected_keys = [
        (season, target)
        for season in panel_builder.SEASONS
        for target in panel_builder.TARGET_COLUMNS
    ]
    if (
        list(candidate_cells) != expected_keys
        or list(control_cells) != expected_keys
    ):
        raise RuntimeError("sports panel 2 arms do not share cells")
    rows = []
    for key in candidate_cells:
        left = candidate_cells[key]
        right = control_cells[key]
        left_rmse = _score_rmse(
            left.get("primary"),
            f"{candidate['arm']} {key} primary",
        )
        right_rmse = _score_rmse(
            right.get("primary"),
            f"{control['arm']} {key} primary",
        )
        ratio = float(left_rmse / right_rmse)
        rows.append(
            {
                "season": key[0],
                "target": key[1],
                "candidate_rmse": left_rmse,
                "control_rmse": right_rmse,
                "ratio": ratio,
            }
        )
    ratios = np.asarray([row["ratio"] for row in rows], dtype=np.float64)
    aggregate = _geometric_mean(ratios)
    bootstrap_upper = _bootstrap_upper(ratios)
    leave_one_out = [
        _geometric_mean(np.delete(ratios, index)) for index in range(len(ratios))
    ]
    primary_gates = {
        "aggregate_ratio_at_most_1_000": aggregate <= AGGREGATE_BAR,
        "bootstrap_upper_at_most_1_002": bootstrap_upper <= BOOTSTRAP_UPPER_BAR,
        "remove_most_favorable_at_most_1_003": max(leave_one_out) <= LEAVE_ONE_OUT_BAR,
        "worst_lineage_at_most_1_020": float(np.max(ratios)) <= WORST_LINEAGE_BAR,
    }

    guardrails = {}
    for view in ("held_team", "seen_player", "cold_player"):
        view_ratios = np.asarray(
            [
                _score_rmse(
                    _guardrail_scores(
                        candidate_cells[key],
                        f"{candidate['arm']} {key}",
                    ).get(view),
                    f"{candidate['arm']} {key} {view}",
                )
                / _score_rmse(
                    _guardrail_scores(
                        control_cells[key],
                        f"{control['arm']} {key}",
                    ).get(view),
                    f"{control['arm']} {key} {view}",
                )
                for key in candidate_cells
            ],
            dtype=np.float64,
        )
        guardrails[view] = {
            "ratios": view_ratios.tolist(),
            "aggregate_ratio": _geometric_mean(view_ratios),
            "worst_ratio": float(np.max(view_ratios)),
        }
    guardrail_gates = {
        "held_team_aggregate_at_most_1_005": guardrails["held_team"]["aggregate_ratio"]
        <= GUARDRAIL_AGGREGATE_BAR,
        "cold_player_aggregate_at_most_1_005": guardrails["cold_player"][
            "aggregate_ratio"
        ]
        <= GUARDRAIL_AGGREGATE_BAR,
        "held_team_worst_at_most_1_020": guardrails["held_team"]["worst_ratio"]
        <= GUARDRAIL_WORST_BAR,
        "cold_player_worst_at_most_1_020": guardrails["cold_player"]["worst_ratio"]
        <= GUARDRAIL_WORST_BAR,
    }
    return {
        "candidate_arm": candidate["arm"],
        "control_arm": control["arm"],
        "cells": rows,
        "aggregate_rmse_ratio": aggregate,
        "bootstrap_95_upper": bootstrap_upper,
        "leave_one_out_ratios": leave_one_out,
        "remove_most_favorable_ratio": max(leave_one_out),
        "worst_lineage_ratio": float(np.max(ratios)),
        "wins_ties_losses": {
            "wins": int(np.sum(ratios < 1.0)),
            "ties": int(np.sum(ratios == 1.0)),
            "losses": int(np.sum(ratios > 1.0)),
        },
        "primary_gates": primary_gates,
        "guardrails": guardrails,
        "guardrail_gates": guardrail_gates,
        "passes_quality": all(primary_gates.values()) and all(guardrail_gates.values()),
    }


def _group_repeats(
    raw: dict[str, Any],
) -> dict[int, dict[str, dict[str, Any]]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("repeats"), list):
        raise RuntimeError("sports panel 2 timing ledger is invalid")
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in raw["repeats"]:
        if not isinstance(record, dict):
            raise RuntimeError("sports panel 2 timing ledger is invalid")
        block = _nonnegative_int(record.get("block"), "timing block")
        arm = record.get("arm")
        result = record.get("result")
        if (
            arm not in runner.ARM_ORDER
            or not isinstance(result, dict)
            or arm in grouped.setdefault(block, {})
        ):
            raise RuntimeError("sports panel 2 timing ledger is invalid")
        grouped[block][arm] = result
    if set(grouped) != set(range(len(runner.BLOCK_ORDERS))):
        raise RuntimeError("sports panel 2 timing blocks are incomplete")
    for block, arms in grouped.items():
        if set(arms) != set(runner.ARM_ORDER):
            raise RuntimeError(f"sports panel 2 block {block} is incomplete")
    return grouped


def _ratio_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.shape != (len(runner.BLOCK_ORDERS),)
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        raise RuntimeError("sports panel 2 paired ratios are invalid")
    median = float(np.median(array))
    q25, q75 = np.percentile(array, [25.0, 75.0])
    relative = float((q75 - q25) / median)
    return {
        "values": array.tolist(),
        "median": median,
        "q25": float(q25),
        "q75": float(q75),
        "iqr_over_median": relative,
        "stable": relative <= MAX_PAIRED_RATIO_IQR_OVER_MEDIAN,
    }


def _timing_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    blocks = _group_repeats(raw)
    summaries: dict[str, Any] = {}
    metrics = (
        "total_fit_seconds",
        "total_predict_seconds",
        "steady_wall_seconds",
        "peak_rss_bytes",
    )
    for arm in runner.ARM_ORDER:
        results = [blocks[block][arm] for block in sorted(blocks)]
        summaries[arm] = {}
        for metric in metrics:
            if metric == "peak_rss_bytes":
                values = [
                    float(
                        _positive_int(
                            row.get(metric),
                            f"{arm} {metric}",
                        )
                    )
                    for row in results
                ]
            else:
                values = [
                    _finite_positive(
                        row.get(metric),
                        f"{arm} {metric}",
                    )
                    for row in results
                ]
            summaries[arm][metric] = {
                "values": values,
                "median": float(statistics.median(values)),
            }
    ratios = {}
    for name in metrics:
        ratios[name] = _ratio_summary(
            [
                blocks[block][runner.CANDIDATE][name]
                / blocks[block][runner.CONTROL][name]
                for block in sorted(blocks)
            ]
        )
    return {"arms": summaries, "candidate_over_control": ratios}


def analyze(raw: dict[str, Any], raw_sha256: str) -> dict[str, Any]:
    manifest = _validate_provenance(raw)
    canonical = _canonical_results(raw)
    _validate_manifest_evidence(raw, manifest)
    if not _is_sha256(raw_sha256):
        raise RuntimeError("raw sports panel 2 content hash is invalid")
    if raw_sha256 != FROZEN_RAW_SHA256:
        raise RuntimeError("raw sports panel 2 content hash changed")
    candidate = _quality_comparison(
        canonical[runner.CANDIDATE], canonical[runner.CONTROL]
    )
    timing = _timing_analysis(raw)
    paired = timing["candidate_over_control"]
    cost_gates = {
        "fit_ratio_at_most_3": paired["total_fit_seconds"]["median"] <= MAX_COST_RATIO,
        "predict_ratio_at_most_3": paired["total_predict_seconds"]["median"]
        <= MAX_COST_RATIO,
        "rss_ratio_at_most_3": paired["peak_rss_bytes"]["median"] <= MAX_COST_RATIO,
        "fit_ratio_stable": paired["total_fit_seconds"]["stable"],
        "predict_ratio_stable": paired["total_predict_seconds"]["stable"],
        "behavior_reproduced": True,
    }
    passed = candidate["passes_quality"] and all(cost_gates.values())
    external = {
        arm: _quality_comparison(
            canonical[runner.CANDIDATE if passed else runner.CONTROL],
            canonical[arm],
        )
        for arm in (runner.CHIMERABOOST, runner.CATBOOST)
    }
    arm_summary = {}
    for arm, result in canonical.items():
        cells = result["cells"]
        arm_summary[arm] = {
            "geometric_mean_primary_rmse": _geometric_mean(
                [cell["primary"]["rmse"] for cell in cells]
            ),
            "geometric_mean_held_team_rmse": _geometric_mean(
                [cell["guardrail"]["scores"]["held_team"]["rmse"] for cell in cells]
            ),
            "geometric_mean_cold_player_rmse": _geometric_mean(
                [cell["guardrail"]["scores"]["cold_player"]["rmse"] for cell in cells]
            ),
            "median_total_fit_seconds": timing["arms"][arm]["total_fit_seconds"][
                "median"
            ],
            "median_total_predict_seconds": timing["arms"][arm][
                "total_predict_seconds"
            ]["median"],
            "median_peak_rss_bytes": timing["arms"][arm]["peak_rss_bytes"]["median"],
        }
    _validate_provenance(raw)
    return {
        "schema_version": 1,
        "name": "darkofit_basketball_sports_panel_result_v2",
        "raw": {
            "sha256": raw_sha256,
            "runner_sha256": raw["runner"]["sha256"],
            "protocol_sha256": raw["protocol"]["sha256"],
            "panel_sha256": raw["panel_manifest"]["processed_panel_sha256"],
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
        },
        "candidate": {
            "comparison": candidate,
            "cost_gates": cost_gates,
            "passes": passed,
            "decision": (
                "advance_oob_ensemble5_as_sports_automatic_policy"
                if passed
                else "close_oob_ensemble5_as_sports_automatic_policy"
            ),
            "global_default_change_authorized": False,
            "sports_profile_change_authorized": passed,
        },
        "eligible_darkofit_arm": (runner.CANDIDATE if passed else runner.CONTROL),
        "external_context": external,
        "timing": timing,
        "arm_summary": arm_summary,
        "panel_spent": True,
        "retuning_on_panel_authorized": False,
    }


def render_report(result: dict[str, Any]) -> str:
    candidate = result["candidate"]
    comparison = candidate["comparison"]
    cost = result["timing"]["candidate_over_control"]
    lines = [
        "# Basketball sports automatic-policy confirmation, panel 2",
        "",
        "## Decision",
        "",
        (
            "The five-member row-OOB ensemble **passed** the frozen Tier-D "
            "gate and may advance as the named sports-profile automatic policy. "
            "The global default remains unchanged."
            if candidate["passes"]
            else "The five-member row-OOB ensemble **failed** the frozen Tier-D "
            "gate. Close it as a sports automatic policy without retuning on "
            "this now-spent panel."
        ),
        "",
        f"Decision code: `{candidate['decision']}`.",
        "",
        "## Candidate versus control",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Equal-lineage RMSE ratio | {comparison['aggregate_rmse_ratio']:.6f}× |",
        f"| 95% bootstrap upper | {comparison['bootstrap_95_upper']:.6f}× |",
        (
            "| Remove-best-lineage ratio | "
            f"{comparison['remove_most_favorable_ratio']:.6f}× |"
        ),
        f"| Worst lineage ratio | {comparison['worst_lineage_ratio']:.6f}× |",
        (
            "| Held-team aggregate ratio | "
            f"{comparison['guardrails']['held_team']['aggregate_ratio']:.6f}× |"
        ),
        (
            "| Cold-player aggregate ratio | "
            f"{comparison['guardrails']['cold_player']['aggregate_ratio']:.6f}× |"
        ),
        (f"| Median total-fit ratio | {cost['total_fit_seconds']['median']:.3f}× |"),
        (
            "| Median total-predict ratio | "
            f"{cost['total_predict_seconds']['median']:.3f}× |"
        ),
        f"| Median peak-RSS ratio | {cost['peak_rss_bytes']['median']:.3f}× |",
        "",
        "## Same-machine context",
        "",
        "| Arm | Primary RMSE | Cold-player RMSE | Median fit |",
        "|---|---:|---:|---:|",
    ]
    ordered = sorted(
        result["arm_summary"],
        key=lambda arm: result["arm_summary"][arm]["geometric_mean_primary_rmse"],
    )
    for arm in ordered:
        row = result["arm_summary"][arm]
        lines.append(
            f"| `{arm}` | {row['geometric_mean_primary_rmse']:.6f} | "
            f"{row['geometric_mean_cold_player_rmse']:.6f} | "
            f"{row['median_total_fit_seconds']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "The nine target-season lineages receive equal weight. Primary "
            "folds are player-disjoint. External comparisons are descriptive "
            "and cannot rescue the candidate decision. Panel 2 is spent and "
            "may not be used for retuning.",
            "",
            f"Raw artifact SHA-256: `{result['raw']['sha256']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _validate_paths(raw: Path, output: Path, report: Path) -> None:
    resolved = [path.resolve() for path in (raw, output, report)]
    if len(set(resolved)) != 3:
        raise RuntimeError("raw, JSON output, and report paths must be distinct")
    if not raw.is_file() or raw.is_symlink():
        raise RuntimeError(f"raw sports panel 2 artifact is unavailable: {raw}")
    for path in (output, report):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to replace analyzer output: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_paths(args.raw, args.output, args.report)
    raw_payload = args.raw.read_bytes()
    raw_sha256 = hashlib.sha256(raw_payload).hexdigest()
    raw = json.loads(raw_payload)
    result = analyze(raw, raw_sha256)
    _atomic_create_many(
        {
            args.output: (
                json.dumps(
                    result,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
            args.report: render_report(result).encode("utf-8"),
        }
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(args.report),
                "decision": result["candidate"]["decision"],
                "passed": result["candidate"]["passes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
