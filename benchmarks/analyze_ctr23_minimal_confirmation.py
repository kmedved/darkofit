"""Verify and analyze the frozen minimal CTR23 regression confirmation.

The analyzer reads an exact allowlist of runner-attested finite JSON inputs.
Only the runner opens, hashes, or decodes ``results.pkl`` files; this module
validates their attested metadata chain without accessing the raw files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import stat
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmarks import analyze_tabarena_regression_cap_horizon as hardened
    from benchmarks import run_ctr23_minimal_confirmation as campaign
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    import analyze_tabarena_regression_cap_horizon as hardened
    import run_ctr23_minimal_confirmation as campaign


BOOTSTRAP_DRAWS = 10_000
PRIMARY_BOOTSTRAP_SEED = 20_260_719
GUARDRAIL_BOOTSTRAP_SEED = 20_260_720
CATBOOST_BOOTSTRAP_SEED = 20_260_721
QUANTILE_METHOD = "higher"

PRIMARY_RATIO_LIMIT = 1.0
PRODUCT_MAX_REGRET_LIMIT = 1.02
PRODUCT_TASK_FLAG_LIMIT = 1.01
DESIRED_PRIMARY_POINT = 0.995

CORE_ARMS = ("A10", "M", "D")
ALL_ARMS = (*CORE_ARMS, "C")
OUTPUT_KEYS = (
    "split_csv",
    "dataset_csv",
    "child_csv",
    "summary_json",
    "report_md",
)
OUTPUT_NAMES = (
    "paired_splits.csv",
    "per_dataset.csv",
    "paired_children.csv",
    "summary.json",
    "report.md",
)
PERFORMANCE_EVIDENCE_DISPOSITION = (
    "inadmissible_by_quality_only_swap_in_policy"
)
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "created_at_utc",
        "output_dir",
        "protocol_sha256",
        "frozen_protocol_sha256",
        "coordinate_manifest_sha256",
        "schedule_sha256",
        "schedule",
        "expected_jobs",
        "expected_child_fits",
        "time_limit_seconds",
        "resolved_child_num_cpus",
        "execution_mode",
        "swap_policy",
        "timing_admissible",
        "source_freeze",
        "source",
        "runtime",
        "sequential_recovery",
    }
)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("value is not canonical finite JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be an object")
    return value


def _exact_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{field} must be an integer")
    return value


def _finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"{field} must be finite and strictly positive")
    return result


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"{field} must be finite and nonnegative")
    return result


def _strict_json_loads(payload: bytes, field: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{field} is not strict finite JSON") from exc


def _reject_symlink_components(path: Path, field: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"{field} contains a symbolic-link component")


def _read_stable_regular(path: Path, field: str) -> bytes:
    _reject_symlink_components(path, field)
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{field} is not a regular file")
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise RuntimeError(f"cannot read {field}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise RuntimeError(f"{field} changed while being read")
    return payload


def _confined_regular_path(root: Path, relative: str | Path, field: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise RuntimeError(f"{field} path is unsafe")
    path = root / candidate
    _reject_symlink_components(path, field)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{field} path escapes the campaign root") from exc
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise RuntimeError(f"{field} is not a regular file")
    return resolved


def _campaign_json_path(root: Path, expected_name: str, field: str) -> Path:
    if (
        type(expected_name) is not str
        or expected_name not in campaign.ANALYZER_CAMPAIGN_JSON_FILENAMES
        or Path(expected_name).parts != (expected_name,)
        or not expected_name.endswith(".json")
    ):
        raise RuntimeError(f"{field} is outside the analyzer JSON allowlist")
    return _confined_regular_path(root, expected_name, field)


def _campaign_json_artifact_bytes(
    root: Path,
    expected_name: str,
    metadata: Mapping[str, Any],
    field: str,
) -> bytes:
    if set(metadata) != {"sha256", "size_bytes"}:
        raise RuntimeError(f"{field} metadata fields are not exact")
    digest = metadata.get("sha256")
    size = metadata.get("size_bytes")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or _exact_int(size, f"{field} size") < 0
    ):
        raise RuntimeError(f"{field} metadata is invalid")
    payload = _read_stable_regular(
        _campaign_json_path(root, expected_name, field), field
    )
    if len(payload) != size or _sha256(payload) != digest:
        raise RuntimeError(f"{field} hash or size differs")
    return payload


def _singleton_artifact_bytes(
    root: Path,
    metadata: Any,
    *,
    expected_name: str,
    field: str,
) -> tuple[bytes, str]:
    item = _as_mapping(metadata, field)
    if set(item) != {"path", "sha256", "size_bytes"}:
        raise RuntimeError(f"{field} metadata fields are not exact")
    relative = item.get("path")
    if not isinstance(relative, str) or relative != expected_name:
        raise RuntimeError(f"{field} path is not canonical")
    payload = _campaign_json_artifact_bytes(
        root,
        expected_name,
        {"sha256": item.get("sha256"), "size_bytes": item.get("size_bytes")},
        field,
    )
    return payload, _sha256(payload)


def _absolute_artifact_bytes(
    expected_path: Path,
    metadata: Any,
    *,
    field: str,
) -> tuple[bytes, str]:
    """Authenticate one canonical absolute artifact outside the run root."""
    item = _as_mapping(metadata, field)
    if set(item) != {"path", "sha256", "size_bytes"}:
        raise RuntimeError(f"{field} metadata fields are not exact")
    if item.get("path") != str(expected_path):
        raise RuntimeError(f"{field} path is not canonical")
    digest = item.get("sha256")
    size = item.get("size_bytes")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or _exact_int(size, f"{field} size") < 0
    ):
        raise RuntimeError(f"{field} metadata is invalid")
    payload = _read_stable_regular(expected_path, field)
    if len(payload) != size or _sha256(payload) != digest:
        raise RuntimeError(f"{field} hash or size differs")
    return payload, digest


def _exists_including_broken_symlink(path: Path, field: str) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"cannot inspect {field}") from exc
    return True


def _validate_recovery_namespace_policy(path: Path) -> None:
    """Mirror the runner's Git-ignore gate without opening recovery artifacts."""
    repository = campaign.REPOSITORY_ROOT.resolve(strict=True)
    repository_parts = tuple(part.casefold() for part in repository.parts)
    path_parts = tuple(part.casefold() for part in path.parts)
    if path_parts[: len(repository_parts)] != repository_parts:
        return
    suffix = path.parts[len(repository.parts) :]
    relative = Path(*suffix) if suffix else Path(".")
    if relative == Path("."):
        raise RuntimeError("recovery source cannot be the campaign repository")
    try:
        ignored = subprocess.run(
            [
                "git",
                "check-ignore",
                "--quiet",
                "--no-index",
                "--",
                str(relative),
            ],
            cwd=repository,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise RuntimeError(
            "could not validate recovery source Git-ignore state"
        ) from exc
    if ignored.returncode == 0:
        return
    if ignored.returncode == 1:
        raise RuntimeError("in-repository recovery source is not Git-ignored")
    raise RuntimeError(
        "could not validate recovery source Git-ignore state: "
        f"git check-ignore exited with status {ignored.returncode}"
    )


def _validate_recovery_failure_swap_telemetry(value: Any) -> None:
    """Independently prove a recoverable failure retained its shutdown sample."""
    record = _as_mapping(value, "recovery failure swap telemetry")
    if (
        set(record)
        != {
            "capture_status",
            "teardown_confirmed",
            "post_teardown_sample_recorded",
            "worker_session_swap_telemetry",
            "swap_in_bytes",
            "swap_out_bytes",
            "diagnostic",
        }
        or record.get("capture_status") != "captured"
        or record.get("teardown_confirmed") is not True
        or record.get("post_teardown_sample_recorded") is not True
        or record.get("diagnostic") is not None
    ):
        raise RuntimeError("recovery failure swap telemetry is not canonical")
    telemetry = _as_mapping(
        record.get("worker_session_swap_telemetry"),
        "recovery worker-session swap telemetry",
    )
    if set(telemetry) != {
        "sample_count",
        "samples",
        "swap_in_delta",
        "swap_out_delta",
    }:
        raise RuntimeError("recovery worker-session swap fields changed")
    samples = telemetry.get("samples")
    if (
        not isinstance(samples, list)
        or len(samples) < 2
        or _exact_int(telemetry.get("sample_count"), "recovery swap sample count")
        != len(samples)
    ):
        raise RuntimeError("recovery worker-session swap samples are incomplete")
    previous = (-1, -1, -1)
    for sample_value in samples:
        sample = _as_mapping(sample_value, "recovery swap sample")
        if set(sample) != {"monotonic_ns", "swap_in_bytes", "swap_out_bytes"}:
            raise RuntimeError("recovery swap sample fields changed")
        current = (
            _exact_int(sample.get("monotonic_ns"), "recovery swap clock"),
            _exact_int(sample.get("swap_in_bytes"), "recovery swap-in counter"),
            _exact_int(sample.get("swap_out_bytes"), "recovery swap-out counter"),
        )
        if (
            current[0] <= previous[0]
            or current[1] < 0
            or current[2] < 0
            or current[1] < previous[1]
            or current[2] < previous[2]
        ):
            raise RuntimeError("recovery swap counters are not monotonic")
        previous = current
    swap_in = samples[-1]["swap_in_bytes"] - samples[0]["swap_in_bytes"]
    swap_out = samples[-1]["swap_out_bytes"] - samples[0]["swap_out_bytes"]
    if (
        _exact_int(telemetry.get("swap_in_delta"), "recovery lifecycle swap-in")
        != swap_in
        or _exact_int(telemetry.get("swap_out_delta"), "recovery lifecycle swap-out")
        != swap_out
        or _exact_int(record.get("swap_in_bytes"), "recovery retained swap-in")
        != swap_in
        or _exact_int(record.get("swap_out_bytes"), "recovery retained swap-out")
        != swap_out
    ):
        raise RuntimeError("recovery failure swap deltas changed")


def _read_json_file(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_stable_regular(path, field)
    value = _strict_json_loads(payload, field)
    return dict(_as_mapping(value, field)), payload


def _validate_runner_attested_result_manifest(value: Any) -> dict[str, Any]:
    """Validate raw-result metadata without touching the runner-owned files."""
    artifacts = dict(_as_mapping(value, "runner-attested result manifest"))
    expected_paths = {
        campaign.expected_result_relative_path(*key)
        for key in campaign.expected_grid()
    }
    if set(artifacts) != expected_paths:
        raise RuntimeError("runner-attested result manifest does not cover the grid")
    for relative, raw_metadata in artifacts.items():
        metadata = _as_mapping(
            raw_metadata, f"runner-attested raw result {relative}"
        )
        digest = metadata.get("sha256")
        size = metadata.get("size_bytes")
        if (
            set(metadata) != {"sha256", "size_bytes"}
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or _exact_int(size, f"runner-attested raw result {relative} size") < 0
        ):
            raise RuntimeError("runner-attested result metadata is invalid")
    return artifacts


def _ratio_summary(log_ratio: float) -> dict[str, float]:
    ratio = math.exp(log_ratio)
    return {
        "log_ratio": float(log_ratio),
        "ratio": ratio,
        "pct": 100.0 * (ratio - 1.0),
    }


def _counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "wins": sum(value < 1.0 for value in values),
        "losses": sum(value > 1.0 for value in values),
        "ties": sum(value == 1.0 for value in values),
    }


def _quantile(values: np.ndarray, probability: float) -> float:
    if values.ndim != 1 or len(values) != BOOTSTRAP_DRAWS:
        raise RuntimeError("bootstrap vector has the wrong shape")
    result = float(np.quantile(values, probability, method=QUANTILE_METHOD))
    if not math.isfinite(result):
        raise RuntimeError("bootstrap quantile is nonfinite")
    return result


def _primary_bootstrap(
    task_logs: Sequence[Sequence[float]],
) -> dict[str, Any]:
    matrix = np.asarray(task_logs, dtype=np.float64)
    if matrix.shape != (9, 3) or not np.isfinite(matrix).all():
        raise RuntimeError("primary bootstrap requires a finite 9x3 log-ratio grid")
    rng = np.random.Generator(np.random.PCG64(PRIMARY_BOOTSTRAP_SEED))
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        task_indices = rng.integers(0, 9, size=9)
        selected = matrix[task_indices]
        fold_indices = rng.integers(0, 3, size=(9, 3))
        within = np.take_along_axis(selected, fold_indices, axis=1).mean(axis=1)
        draws[index] = math.exp(float(within.mean()))
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": PRIMARY_BOOTSTRAP_SEED,
        "rng": "numpy.random.Generator(PCG64)",
        "quantile_method": QUANTILE_METHOD,
        "upper_95": _quantile(draws, 0.95),
    }


def _product_guardrail_bootstrap(
    task_logs: Sequence[Sequence[float]],
) -> dict[str, Any]:
    matrix = np.asarray(task_logs, dtype=np.float64)
    if matrix.shape != (9, 3) or not np.isfinite(matrix).all():
        raise RuntimeError("guardrail bootstrap requires a finite 9x3 log-ratio grid")
    rng = np.random.Generator(np.random.PCG64(GUARDRAIL_BOOTSTRAP_SEED))
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        fold_indices = rng.integers(0, 3, size=(9, 3))
        within = np.take_along_axis(matrix, fold_indices, axis=1).mean(axis=1)
        draws[index] = math.exp(float(within.max()))
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": GUARDRAIL_BOOTSTRAP_SEED,
        "rng": "numpy.random.Generator(PCG64)",
        "quantile_method": QUANTILE_METHOD,
        "upper_95_simultaneous_max_regret": _quantile(draws, 0.95),
    }


def _catboost_bootstrap(log_ratios: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(log_ratios, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise RuntimeError("CatBoost bootstrap requires nine finite task log ratios")
    rng = np.random.Generator(np.random.PCG64(CATBOOST_BOOTSTRAP_SEED))
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        selected = values[rng.integers(0, 9, size=9)]
        draws[index] = math.exp(float(selected.mean()))
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": CATBOOST_BOOTSTRAP_SEED,
        "rng": "numpy.random.Generator(PCG64)",
        "quantile_method": QUANTILE_METHOD,
        "lower_95": _quantile(draws, 0.025),
        "upper_95": _quantile(draws, 0.975),
    }


def _expected_grid() -> set[tuple[str, int, int, int, int, str]]:
    grid = set(campaign.expected_grid())
    if len(grid) != campaign.EXPECTED_JOBS or len(grid) != 90:
        raise RuntimeError("runner expected grid does not contain exactly 90 jobs")
    arm_counts = Counter(key[-1] for key in grid)
    if arm_counts != {"A10": 27, "M": 27, "D": 27, "C": 9}:
        raise RuntimeError("runner arm grid does not match the frozen campaign")
    return grid


def _expected_child_grid() -> set[tuple[str, int, int, int, int, str, int]]:
    grid = set(campaign.expected_child_grid())
    if len(grid) != campaign.EXPECTED_CHILD_FITS or len(grid) != 720:
        raise RuntimeError("runner expected child grid does not contain 720 rows")
    return grid


def _candidate_metadata_field() -> str:
    fields = set(campaign.CHILD_PAYLOAD_FIELDS)
    if "candidate_metadata" not in fields or "tree_mode_selection" in fields:
        raise RuntimeError("runner child schema changed its A10 candidate field")
    return "candidate_metadata"


_DARKOFIT_STOP_REASONS = frozenset(
    {"early_stopping", "iteration_limit", "no_split"}
)


def _validate_child_stop_causality(
    arm: str,
    reason: Any,
    *,
    requested: int,
    attempted: int,
    completed: int,
    retained: int,
) -> None:
    """Re-prove the exact stop states exposed by each frozen adapter."""
    if arm in {"A10", "D"}:
        if reason not in _DARKOFIT_STOP_REASONS:
            raise RuntimeError("DarkoFit child has an unproven stop state")
        if reason == "iteration_limit" and attempted != requested:
            raise RuntimeError("iteration_limit child did not attempt every round")
        if reason == "no_split" and attempted <= completed:
            raise RuntimeError("no_split child has no failed attempted round")
        if reason == "early_stopping" and (attempted == 0 or completed == 0):
            raise RuntimeError("early_stopping child completed no rounds")
        return

    # Under the registered positive time budget, the official ChimeraBoost
    # adapter can prove only these two terminal causes; lane-selection ambiguity
    # remains null and is reported rather than guessed.
    if arm == "M":
        if reason is None:
            return
        if reason == "early_stopping" and attempted > retained:
            return
        if reason == "iteration_limit" and retained == requested:
            return
        raise RuntimeError("ChimeraBoost child has an invalid stop state")

    # With a positive AutoGluon time callback, CatBoost can prove only full
    # iteration exhaustion. Other ordinary terminal causes remain null.
    if arm == "C":
        if reason is None or (
            reason == "iteration_limit" and attempted == requested
        ):
            return
        raise RuntimeError("CatBoost child has an invalid stop state")
    raise RuntimeError(f"unknown child arm: {arm}")


def _validate_a10_candidate_metadata(value: Any) -> dict[str, Any]:
    metadata = dict(_as_mapping(value, "A10 candidate metadata"))
    expected_fields = {
        "candidate_count",
        "fitted_candidate_count",
        "candidate_order",
        "selected_candidate_index",
        "candidates",
    }
    if set(metadata) != expected_fields:
        raise RuntimeError("A10 candidate metadata fields are not exact")
    order = ["catboost", "lightgbm", "hybrid"]
    candidates = metadata.get("candidates")
    selected = _exact_int(
        metadata.get("selected_candidate_index"), "selected candidate index"
    )
    if (
        metadata.get("candidate_count") != 3
        or metadata.get("fitted_candidate_count") != 3
        or metadata.get("candidate_order") != order
        or not isinstance(candidates, list)
        or len(candidates) != 3
        or selected not in range(3)
    ):
        raise RuntimeError("A10 child does not contain all frozen candidates")
    scores = []
    normalized = []
    candidate_fields = {
        "candidate_index",
        "tree_mode",
        "fitted",
        "validation_rmse",
        "deadline_hit",
        "stop_reason",
    }
    for index, raw in enumerate(candidates):
        candidate = dict(_as_mapping(raw, f"A10 candidate {index}"))
        if set(candidate) != candidate_fields:
            raise RuntimeError("A10 candidate fields are not exact")
        score = _finite_nonnegative(
            candidate.get("validation_rmse"), "A10 candidate validation RMSE"
        )
        if (
            candidate.get("candidate_index") != index
            or candidate.get("tree_mode") != order[index]
            or candidate.get("fitted") is not True
            or candidate.get("deadline_hit") is not False
            or candidate.get("stop_reason") not in _DARKOFIT_STOP_REASONS
        ):
            raise RuntimeError("A10 candidate fitted state is invalid")
        scores.append(score)
        normalized.append(candidate)
    if selected != min(range(3), key=scores.__getitem__):
        raise RuntimeError("A10 selected candidate is not the first validation argmin")
    metadata["candidates"] = normalized
    return metadata


def _validate_arm_child_contract(row: Mapping[str, Any], arm: str) -> None:
    requested = row.get("iterations_requested")
    learning_rate = float(row.get("resolved_learning_rate"))
    requested_mode = row.get("requested_tree_mode")
    selected_mode = row.get("selected_tree_mode")
    lane = row.get("selected_lane")
    if arm == "A10":
        valid = (
            requested == 10_000
            and learning_rate == 0.1
            and requested_mode == "auto"
            and selected_mode in {"catboost", "lightgbm", "hybrid"}
            and lane == "boosting"
        )
    elif arm == "D":
        valid = (
            requested == 1_000
            and learning_rate > 0.0
            and requested_mode == "catboost"
            and selected_mode == "catboost"
            and lane == "boosting"
        )
    elif arm == "M":
        valid = (
            requested == 10_000
            and math.isclose(learning_rate, 0.1, rel_tol=1e-7, abs_tol=1e-12)
            and requested_mode is None
            and selected_mode is None
            and lane in {"constant", "linear"}
        )
    elif arm == "C":
        valid = (
            requested == 10_000
            and math.isclose(learning_rate, 0.05, rel_tol=1e-7, abs_tol=1e-12)
            and requested_mode is None
            and selected_mode is None
            and lane == "cpu"
        )
    else:  # The exact grid makes this unreachable, but keep the boundary local.
        valid = False
    if not valid:
        raise RuntimeError(f"{arm} child fitted behavior changed")


def _outer_key(row: Mapping[str, Any]) -> tuple[str, int, int, int, int, str]:
    return (
        str(row.get("dataset")),
        _exact_int(row.get("task_id"), "outer task id"),
        _exact_int(row.get("repeat"), "outer repeat"),
        _exact_int(row.get("fold"), "outer fold"),
        _exact_int(row.get("sample"), "outer sample"),
        str(row.get("arm")),
    )


def _child_key(
    row: Mapping[str, Any],
) -> tuple[str, int, int, int, int, str, int]:
    return (*_outer_key(row), _exact_int(row.get("child_fold"), "child fold"))


def _validate_payload_rows(
    payload: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_outer = payload.get("outer_rows")
    raw_children = payload.get("child_rows")
    if not isinstance(raw_outer, list) or not isinstance(raw_children, list):
        raise RuntimeError("safe payload rows must be lists")
    if len(raw_outer) != 90 or len(raw_children) != 720:
        raise RuntimeError("safe payload row counts do not match 90 jobs/720 children")
    outer_fields = set(campaign.OUTER_PAYLOAD_FIELDS)
    child_fields = set(campaign.CHILD_PAYLOAD_FIELDS)
    candidate_field = _candidate_metadata_field()
    expected_outer = _expected_grid()
    expected_children = _expected_child_grid()
    outer: dict[tuple[str, int, int, int, int, str], dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    for value in raw_outer:
        row = dict(_as_mapping(value, "outer row"))
        if set(row) != outer_fields:
            raise RuntimeError("safe outer row fields are not exact")
        key = _outer_key(row)
        source = row.get("source")
        if key not in expected_outer or key in outer:
            raise RuntimeError(f"safe outer row is duplicate or outside grid: {key}")
        if not isinstance(source, str) or source not in artifacts:
            raise RuntimeError("safe outer row source is not attested")
        if source != campaign.expected_result_relative_path(*key):
            raise RuntimeError("safe outer row source is not its canonical result path")
        if (
            row.get("num_cpus") != 18
            or row.get("num_cpus_child") != 18
            or row.get("num_gpus") != 0
            or row.get("num_gpus_child") != 0
        ):
            raise RuntimeError("safe outer row resource allocation changed")
        row["test_rmse"] = _finite_positive(row.get("test_rmse"), "test RMSE")
        row["val_rmse"] = _finite_positive(row.get("val_rmse"), "validation RMSE")
        source_counts[source] += 1
        outer[key] = row
    if (
        set(outer) != expected_outer
        or set(source_counts) != set(artifacts)
        or any(count != 1 for count in source_counts.values())
    ):
        raise RuntimeError("safe outer rows do not bind one-to-one to raw results")

    children: dict[tuple[str, int, int, int, int, str, int], dict[str, Any]] = {}
    child_counts: Counter[tuple[str, int, int, int, int, str]] = Counter()
    for value in raw_children:
        row = dict(_as_mapping(value, "child row"))
        if set(row) != child_fields:
            raise RuntimeError("safe child row fields are not exact")
        key = _child_key(row)
        outer_key = key[:-1]
        if key not in expected_children or key in children:
            raise RuntimeError(f"safe child row is duplicate or outside grid: {key}")
        if row.get("source") != outer[outer_key]["source"]:
            raise RuntimeError("safe child source does not match its outer result")
        if row.get("num_cpus") != 18 or row.get("num_gpus") != 0:
            raise RuntimeError("safe child resource allocation changed")
        if (
            row.get("deadline_hit") is not False
            or row.get("time_callback_hit") is not False
            or row.get("stop_reason") == "time_limit"
        ):
            raise RuntimeError("safe payload contains a deadline/time-limit child")
        callback_instances = _exact_int(
            row.get("time_callback_instance_count"),
            "time callback instance count",
        )
        callback_calls = _exact_int(
            row.get("time_callback_call_count"), "time callback call count"
        )
        if outer_key[-1] in {"A10", "D"}:
            callback_audit_valid = callback_instances == callback_calls == 0
        elif outer_key[-1] == "M":
            callback_audit_valid = (
                callback_instances in {1, 2} and callback_calls >= callback_instances
            )
        else:
            callback_audit_valid = callback_instances == 1 and callback_calls >= 1
        if not callback_audit_valid:
            raise RuntimeError("safe child time-callback coverage is invalid")
        for field in (
            "iterations_requested",
            "iterations_attempted",
            "rounds_completed",
            "rounds_retained",
            "best_iteration",
        ):
            if _exact_int(row.get(field), f"child {field}") < 0:
                raise RuntimeError(f"child {field} must be nonnegative")
        requested = int(row["iterations_requested"])
        attempted = int(row["iterations_attempted"])
        completed = int(row["rounds_completed"])
        retained = int(row["rounds_retained"])
        best = int(row["best_iteration"])
        if not (0 <= retained == best <= completed <= attempted <= requested):
            raise RuntimeError("safe child round counters are inconsistent")
        _finite_positive(row.get("resolved_learning_rate"), "resolved learning rate")
        _validate_arm_child_contract(row, outer_key[-1])
        if row.get("deadline_is_soft") is not True:
            raise RuntimeError("safe child deadline policy changed")
        stop_reason = row.get("stop_reason")
        _validate_child_stop_causality(
            outer_key[-1],
            stop_reason,
            requested=requested,
            attempted=attempted,
            completed=completed,
            retained=retained,
        )
        if outer_key[-1] == "A10":
            selection = _validate_a10_candidate_metadata(row.get(candidate_field))
            selected_candidate = selection["candidates"][
                selection["selected_candidate_index"]
            ]
            if (
                row.get("selected_tree_mode")
                != selected_candidate["tree_mode"]
                or row.get("stop_reason") != selected_candidate["stop_reason"]
                or row.get("selected_lane") != "boosting"
            ):
                raise RuntimeError("A10 child does not contain all frozen candidates")
        elif row.get(candidate_field) is not None:
            raise RuntimeError("non-A10 child carries candidate metadata")
        child_counts[outer_key] += 1
        children[key] = row
    if set(children) != expected_children or any(
        child_counts[key] != 8 for key in expected_outer
    ):
        raise RuntimeError("safe child grid is incomplete")
    return list(outer.values()), list(children.values())


def pair_outer_rows(
    outer_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Form exact paired coordinate rows without pooling raw errors."""
    if len(outer_rows) != 90:
        raise RuntimeError("pairing requires exactly 90 outer rows")
    index: dict[tuple[str, int, int, int, int, str], Mapping[str, Any]] = {}
    for raw in outer_rows:
        row = _as_mapping(raw, "outer row")
        key = _outer_key(row)
        if key in index:
            raise RuntimeError(f"duplicate outer row while pairing: {key}")
        index[key] = row
    if set(index) != _expected_grid():
        raise RuntimeError("outer rows do not form the frozen 90-job grid")

    coordinates = sorted(
        {key[:-1] for key in index if key[-1] in CORE_ARMS},
        key=lambda key: (key[1], key[2], key[3], key[4]),
    )
    if len(coordinates) != 27:
        raise RuntimeError("primary pairing requires exactly 27 coordinates")
    paired: list[dict[str, Any]] = []
    for coordinate in coordinates:
        rows = {arm: index[(*coordinate, arm)] for arm in CORE_ARMS}
        c = index.get((*coordinate, "C"))
        if (coordinate[3] == 0) != (c is not None):
            raise RuntimeError("CatBoost rows must exist only on r0f0")
        result: dict[str, Any] = {
            "dataset": coordinate[0],
            "task_id": coordinate[1],
            "repeat": coordinate[2],
            "fold": coordinate[3],
            "sample": coordinate[4],
            "performance_evidence_disposition": PERFORMANCE_EVIDENCE_DISPOSITION,
        }
        for arm in ALL_ARMS:
            row = rows.get(arm) if arm in rows else c
            for metric in ("test_rmse", "val_rmse"):
                result[f"{arm}_{metric}"] = (
                    None
                    if row is None
                    else _finite_positive(row.get(metric), f"{arm} {metric}")
                )
        for numerator, denominator, code in (
            ("A10", "M", "a10_over_m"),
            ("A10", "D", "a10_over_d"),
        ):
            for metric in ("test_rmse", "val_rmse"):
                ratio = result[f"{numerator}_{metric}"] / result[
                    f"{denominator}_{metric}"
                ]
                result[f"{code}_{metric}_ratio"] = ratio
        for numerator, denominator, code in (
            ("A10", "C", "a10_over_c"),
            ("M", "C", "m_over_c"),
            ("D", "C", "d_over_c"),
        ):
            for metric in ("test_rmse", "val_rmse"):
                result[f"{code}_{metric}_ratio"] = (
                    None
                    if c is None
                    else result[f"{numerator}_{metric}"]
                    / result[f"{denominator}_{metric}"]
                )
        paired.append(result)
    return paired


def pair_child_rows(
    child_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair child audit metadata while deliberately excluding performance data."""
    if len(child_rows) != 720:
        raise RuntimeError("child pairing requires exactly 720 rows")
    index: dict[tuple[str, int, int, int, int, str, int], Mapping[str, Any]] = {}
    for raw in child_rows:
        row = _as_mapping(raw, "child row")
        key = _child_key(row)
        if key in index:
            raise RuntimeError(f"duplicate child row while pairing: {key}")
        index[key] = row
    if set(index) != _expected_child_grid():
        raise RuntimeError("child rows do not form the frozen 720-row grid")

    coordinates = sorted(
        {key[:5] for key in index if key[5] in CORE_ARMS},
        key=lambda key: (key[1], key[2], key[3], key[4]),
    )
    paired: list[dict[str, Any]] = []
    candidate_field = _candidate_metadata_field()
    for coordinate in coordinates:
        for child_fold in range(8):
            rows = {
                arm: index[(*coordinate, arm, child_fold)] for arm in CORE_ARMS
            }
            c = index.get((*coordinate, "C", child_fold))
            result: dict[str, Any] = {
                "dataset": coordinate[0],
                "task_id": coordinate[1],
                "repeat": coordinate[2],
                "fold": coordinate[3],
                "sample": coordinate[4],
                "child_fold": child_fold,
                "performance_evidence_disposition": PERFORMANCE_EVIDENCE_DISPOSITION,
            }
            for arm in ALL_ARMS:
                row = rows.get(arm) if arm in rows else c
                result[f"{arm}_present"] = row is not None
                result[f"{arm}_source"] = None if row is None else row.get("source")
                result[f"{arm}_stop_reason"] = (
                    None if row is None else row.get("stop_reason")
                )
                result[f"{arm}_deadline_hit"] = (
                    None if row is None else row.get("deadline_hit")
                )
                result[f"{arm}_time_callback_hit"] = (
                    None if row is None else row.get("time_callback_hit")
                )
                result[f"{arm}_time_callback_instance_count"] = (
                    None
                    if row is None
                    else row.get("time_callback_instance_count")
                )
                result[f"{arm}_time_callback_call_count"] = (
                    None if row is None else row.get("time_callback_call_count")
                )
                result[f"{arm}_selected_tree_mode"] = (
                    None if row is None else row.get("selected_tree_mode")
                )
            selection = _as_mapping(
                rows["A10"].get(candidate_field),
                "A10 candidate metadata",
            )
            result["A10_candidate_count"] = selection.get("candidate_count")
            result["A10_fitted_candidate_count"] = selection.get(
                "fitted_candidate_count"
            )
            result["A10_candidate_order"] = json.dumps(
                selection.get("candidate_order"),
                allow_nan=False,
                separators=(",", ":"),
            )
            paired.append(result)
    if len(paired) != 216:
        raise RuntimeError("paired child export must contain 216 coordinate children")
    return paired


def _contrast_dataset_rows(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    code: str,
    folds: Sequence[int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    ratio_field = f"{code}_test_rmse_ratio"
    val_field = f"{code}_val_rmse_ratio"
    for row in paired_rows:
        if row.get("fold") in folds and row.get(ratio_field) is not None:
            grouped[(_exact_int(row.get("task_id"), "task id"), str(row["dataset"]))].append(
                row
            )
    if len(grouped) != 9:
        raise RuntimeError(f"contrast {code} does not cover all nine tasks")
    result = []
    for (task_id, dataset), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["fold"]))
        if [row["fold"] for row in rows] != list(folds):
            raise RuntimeError(f"contrast {code} task folds are incomplete")
        test_ratios = [
            _finite_positive(row[ratio_field], f"{code} test ratio") for row in rows
        ]
        val_ratios = [
            _finite_positive(row[val_field], f"{code} validation ratio")
            for row in rows
        ]
        test_log = float(np.log(test_ratios).mean())
        val_log = float(np.log(val_ratios).mean())
        test_counts = _counts(test_ratios)
        val_counts = _counts(val_ratios)
        worst_index = int(np.argmax(test_ratios))
        result.append(
            {
                "contrast": code,
                "dataset": dataset,
                "task_id": task_id,
                "coordinate_count": len(rows),
                "test_rmse_ratio": math.exp(test_log),
                "test_rmse_pct": 100.0 * math.expm1(test_log),
                "val_rmse_ratio": math.exp(val_log),
                "val_rmse_pct": 100.0 * math.expm1(val_log),
                "test_split_wins": test_counts["wins"],
                "test_split_losses": test_counts["losses"],
                "test_split_ties": test_counts["ties"],
                "val_split_wins": val_counts["wins"],
                "val_split_losses": val_counts["losses"],
                "val_split_ties": val_counts["ties"],
                "worst_test_fold": int(rows[worst_index]["fold"]),
                "worst_test_split_ratio": test_ratios[worst_index],
                "product_point_flag": (
                    code == "a10_over_d"
                    and math.exp(test_log) > PRODUCT_TASK_FLAG_LIMIT
                ),
            }
        )
    return result


def _aggregate_contrast(
    dataset_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    code: str,
) -> dict[str, Any]:
    if len(dataset_rows) != 9:
        raise RuntimeError(f"contrast {code} needs nine dataset rows")
    dataset_test = [
        _finite_positive(row["test_rmse_ratio"], f"{code} dataset test ratio")
        for row in dataset_rows
    ]
    dataset_val = [
        _finite_positive(row["val_rmse_ratio"], f"{code} dataset val ratio")
        for row in dataset_rows
    ]
    split_test = [
        _finite_positive(row[f"{code}_test_rmse_ratio"], f"{code} split ratio")
        for row in paired_rows
        if row.get(f"{code}_test_rmse_ratio") is not None
    ]
    split_val = [
        _finite_positive(row[f"{code}_val_rmse_ratio"], f"{code} split val ratio")
        for row in paired_rows
        if row.get(f"{code}_val_rmse_ratio") is not None
    ]
    return {
        "test_rmse": {
            **_ratio_summary(float(np.log(dataset_test).mean())),
            "dataset_counts": _counts(dataset_test),
            "split_counts": _counts(split_test),
            "worst_dataset_ratio": max(dataset_test),
            "worst_split_ratio": max(split_test),
        },
        "val_rmse": {
            **_ratio_summary(float(np.log(dataset_val).mean())),
            "dataset_counts": _counts(dataset_val),
            "split_counts": _counts(split_val),
            "worst_dataset_ratio": max(dataset_val),
            "worst_split_ratio": max(split_val),
        },
    }


def analyze(
    paired_rows: Sequence[Mapping[str, Any]],
    paired_children: Sequence[Mapping[str, Any]],
    *,
    execution_mode: str = "concurrent",
    swap_policy: str = "quality_only_swap_in",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the frozen estimands, intervals, and terminal decision rule."""
    if len(paired_rows) != 27 or len(paired_children) != 216:
        raise RuntimeError("analysis inputs do not match the frozen paired grids")
    if any(
        row.get("performance_evidence_disposition")
        != PERFORMANCE_EVIDENCE_DISPOSITION
        for row in (*paired_rows, *paired_children)
    ):
        raise RuntimeError("analysis input promotes inadmissible performance evidence")
    if execution_mode not in {"concurrent", "sequential_recovery"}:
        raise RuntimeError("analysis execution mode is invalid")
    if swap_policy != "quality_only_swap_in":
        raise RuntimeError("analysis swap policy is not quality-only")

    contrast_specs = (
        ("a10_over_m", (0, 1, 2)),
        ("a10_over_d", (0, 1, 2)),
        ("a10_over_c", (0,)),
        ("m_over_c", (0,)),
        ("d_over_c", (0,)),
    )
    per_dataset: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    contrasts: dict[str, Any] = {}
    for code, folds in contrast_specs:
        rows = _contrast_dataset_rows(paired_rows, code=code, folds=folds)
        grouped[code] = rows
        per_dataset.extend(rows)
        contrasts[code] = _aggregate_contrast(rows, paired_rows, code=code)

    by_task_split: dict[tuple[int, int], Mapping[str, Any]] = {
        (int(row["task_id"]), int(row["fold"])): row for row in paired_rows
    }
    task_ids = sorted({int(row["task_id"]) for row in paired_rows})
    primary_logs = [
        [
            math.log(
                _finite_positive(
                    by_task_split[(task_id, fold)]["a10_over_m_test_rmse_ratio"],
                    "primary split ratio",
                )
            )
            for fold in (0, 1, 2)
        ]
        for task_id in task_ids
    ]
    product_logs = [
        [
            math.log(
                _finite_positive(
                    by_task_split[(task_id, fold)]["a10_over_d_test_rmse_ratio"],
                    "product split ratio",
                )
            )
            for fold in (0, 1, 2)
        ]
        for task_id in task_ids
    ]
    primary_interval = _primary_bootstrap(primary_logs)
    product_interval = _product_guardrail_bootstrap(product_logs)
    for code in ("a10_over_c", "m_over_c", "d_over_c"):
        logs = [
            math.log(
                _finite_positive(
                    by_task_split[(task_id, 0)][f"{code}_test_rmse_ratio"],
                    f"{code} task ratio",
                )
            )
            for task_id in task_ids
        ]
        contrasts[code]["test_rmse"]["descriptive_interval"] = (
            _catboost_bootstrap(logs)
        )

    primary_pass = primary_interval["upper_95"] < PRIMARY_RATIO_LIMIT
    product_pass = (
        product_interval["upper_95_simultaneous_max_regret"]
        <= PRODUCT_MAX_REGRET_LIMIT
    )
    primary_point = contrasts["a10_over_m"]["test_rmse"]["ratio"]
    product_flags = [
        {
            "task_id": row["task_id"],
            "dataset": row["dataset"],
            "ratio": row["test_rmse_ratio"],
        }
        for row in grouped["a10_over_d"]
        if row["product_point_flag"]
    ]
    summary = {
        "schema_version": 1,
        "kind": "ctr23_minimal_confirmation_analysis",
        "claim_scope": (
            "external fixed-panel confirmation only; no lockbox, preset, or "
            "default-change authorization"
        ),
        "performance_evidence_disposition": PERFORMANCE_EVIDENCE_DISPOSITION,
        "execution_mode": execution_mode,
        "swap_policy": swap_policy,
        "timing_admissible": False,
        "aggregation": (
            "paired split ratios; geometric mean within task; equal-task "
            "geometric mean"
        ),
        "contrasts": contrasts,
        "primary_interval": primary_interval,
        "product_guardrail_interval": product_interval,
        "product_task_point_flags": product_flags,
        "gates": {
            "complete_grid_and_safety": True,
            "a10_over_m_one_sided_95_upper_strictly_below_1": primary_pass,
            "a10_over_d_simultaneous_max_regret_95_upper_at_most_1_02": (
                product_pass
            ),
        },
        "desired_primary_point_at_most_0_995": primary_point <= DESIRED_PRIMARY_POINT,
        "confirmation_passed": primary_pass and product_pass,
        "decision": (
            "confirmation_passed_clean_stop"
            if primary_pass and product_pass
            else "confirmation_not_established_clean_stop"
        ),
        "terminal_policy": (
            "stop regardless of outcome; do not add folds, tune, open the "
            "lockbox, or change a preset/default"
        ),
    }
    return summary, per_dataset


def _csv_bytes(rows: Sequence[Mapping[str, Any]], field: str) -> bytes:
    if not rows:
        raise RuntimeError(f"refusing to write empty {field}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise RuntimeError(f"{field} rows do not have identical ordered fields")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def render_report(
    summary: Mapping[str, Any], per_dataset: Sequence[Mapping[str, Any]]
) -> str:
    primary = summary["contrasts"]["a10_over_m"]["test_rmse"]
    product = summary["contrasts"]["a10_over_d"]["test_rmse"]
    a10_cat = summary["contrasts"]["a10_over_c"]["test_rmse"]
    m_cat = summary["contrasts"]["m_over_c"]["test_rmse"]
    d_cat = summary["contrasts"]["d_over_c"]["test_rmse"]
    lines = [
        "# Minimal CTR23 regression confirmation",
        "",
        str(summary["claim_scope"]),
        "",
        "Negative percentages favor the numerator. All quality estimates use "
        "paired outer-test RMSE ratios. Timing and memory-performance evidence "
        "is inadmissible by the frozen quality-only resource policy.",
        "",
        "## Primary result and product guardrail",
        "",
        "| Measure | Point | Registered interval statistic | Gate |",
        "| --- | ---: | ---: | --- |",
        f"| A10 / ChimeraBoost | {primary['ratio']:.6f} "
        f"({primary['pct']:+.3f}%) | one-sided 95% upper "
        f"{summary['primary_interval']['upper_95']:.6f} | "
        f"{'PASS' if summary['gates']['a10_over_m_one_sided_95_upper_strictly_below_1'] else 'FAIL'} |",
        f"| A10 / product default | {product['ratio']:.6f} "
        f"({product['pct']:+.3f}%) | simultaneous max-regret 95% upper "
        f"{summary['product_guardrail_interval']['upper_95_simultaneous_max_regret']:.6f} | "
        f"{'PASS' if summary['gates']['a10_over_d_simultaneous_max_regret_95_upper_at_most_1_02'] else 'FAIL'} |",
        "",
        "The requested A10/M point estimate of at most 0.995 is report-only: "
        f"**{'met' if summary['desired_primary_point_at_most_0_995'] else 'not met'}**.",
        "",
        "## Secondary CatBoost context (r0f0 only)",
        "",
        "These contrasts are descriptive and have no advancement gate.",
        "",
        "| Contrast | Point | Two-sided task-bootstrap 95% interval |",
        "| --- | ---: | ---: |",
        f"| A10 / CatBoost | {a10_cat['ratio']:.6f} ({a10_cat['pct']:+.3f}%) | "
        f"[{a10_cat['descriptive_interval']['lower_95']:.6f}, "
        f"{a10_cat['descriptive_interval']['upper_95']:.6f}] |",
        f"| ChimeraBoost / CatBoost | {m_cat['ratio']:.6f} ({m_cat['pct']:+.3f}%) | "
        f"[{m_cat['descriptive_interval']['lower_95']:.6f}, "
        f"{m_cat['descriptive_interval']['upper_95']:.6f}] |",
        f"| product default / CatBoost | {d_cat['ratio']:.6f} ({d_cat['pct']:+.3f}%) | "
        f"[{d_cat['descriptive_interval']['lower_95']:.6f}, "
        f"{d_cat['descriptive_interval']['upper_95']:.6f}] |",
        "",
        "## A10 / ChimeraBoost by task",
        "",
        "| Task | Dataset | Ratio | Change | Split W/L/T | Worst fold |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in per_dataset:
        if row["contrast"] != "a10_over_m":
            continue
        lines.append(
            f"| {row['task_id']} | {row['dataset']} | "
            f"{row['test_rmse_ratio']:.6f} | {row['test_rmse_pct']:+.3f}% | "
            f"{row['test_split_wins']}/{row['test_split_losses']}/"
            f"{row['test_split_ties']} | f{row['worst_test_fold']} "
            f"({row['worst_test_split_ratio']:.6f}) |"
        )
    lines.extend(["", "## Product-default task flags", ""])
    flags = summary["product_task_point_flags"]
    if flags:
        for flag in flags:
            lines.append(
                f"- Task {flag['task_id']} (`{flag['dataset']}`): "
                f"A10/default = {flag['ratio']:.6f}, above 1.01."
            )
    else:
        lines.append("- None.")
    integrity = summary.get("integrity")
    if isinstance(integrity, Mapping):
        if integrity.get("swap_in_audit_evidence_retained") is True:
            lines.extend(
                [
                    "",
                    "## Operational integrity",
                    "",
                    "Swap-in host counters were retained for the complete preflight "
                    "and production worker lifecycles and for every production "
                    f"dispatch ({integrity['production_dispatches_with_swap_in_telemetry']} "
                    f"dispatches across {integrity['production_waves_with_swap_in_telemetry']} "
                    "waves). Swap-out remained zero. These counters are integrity "
                    "evidence only and remain inadmissible for timing or memory-"
                    "performance claims.",
                ]
            )
        unresolved = integrity.get("unresolved_comparator_stop_count")
        if isinstance(unresolved, int) and unresolved:
            lines.extend(
                [
                    "",
                    "## Comparator stop-state qualification",
                    "",
                    f"{unresolved} ChimeraBoost/CatBoost child stop reasons were "
                    "semantically unresolved by their official adapters. Direct "
                    "callback instrumentation proves the time callback did not "
                    "fire, and CatBoost's memory callback was ineligible. The "
                    "unresolved label therefore distinguishes early stopping from "
                    "iteration/no-split termination and does not weaken budget "
                    "integrity.",
                ]
            )
    lines.extend(
        [
            "",
            "## Terminal decision",
            "",
            f"Decision: **{summary['decision']}**.",
            "",
            str(summary["terminal_policy"]),
            "",
        ]
    )
    return "\n".join(lines)


def _build_output_payloads(
    paired_rows: Sequence[Mapping[str, Any]],
    per_dataset: Sequence[Mapping[str, Any]],
    paired_children: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, bytes]:
    context = {
        "execution_mode": summary["execution_mode"],
        "swap_policy": summary["swap_policy"],
        "timing_admissible": False,
        "performance_evidence_disposition": PERFORMANCE_EVIDENCE_DISPOSITION,
    }
    integrity = summary.get("integrity")
    if isinstance(integrity, Mapping):
        for name in (
            "swap_in_audit_evidence_retained",
            "preflight_worker_lifecycle_swap_in_bytes",
            "production_worker_lifecycle_swap_in_bytes",
            "production_measured_phase_swap_in_bytes",
            "production_dispatches_with_swap_in_telemetry",
            "production_waves_with_swap_in_telemetry",
            "swap_out_bytes",
        ):
            context[name] = integrity.get(name)

    def labeled(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
        result = []
        for raw in rows:
            row = dict(raw)
            for key, value in context.items():
                if key in row and row[key] != value:
                    raise RuntimeError(f"{field} context conflicts with the run")
                row.pop(key, None)
            result.append({**context, **row})
        return result

    return {
        "split_csv": _csv_bytes(labeled(paired_rows, "paired split"), "paired split CSV"),
        "dataset_csv": _csv_bytes(labeled(per_dataset, "per-dataset"), "per-dataset CSV"),
        "child_csv": _csv_bytes(labeled(paired_children, "paired child"), "paired child CSV"),
        "summary_json": (
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "report_md": render_report(summary, per_dataset).encode("utf-8"),
    }


def _validate_sequential_recovery(
    manifest: Mapping[str, Any], *, campaign_root: Path
) -> None:
    """Independently authenticate the one permitted fresh recovery source."""
    mode = manifest.get("execution_mode")
    value = manifest.get("sequential_recovery")
    if mode == "concurrent":
        if value is not None:
            raise RuntimeError("concurrent campaign carries recovery state")
        return
    if mode != "sequential_recovery":
        raise RuntimeError("campaign execution mode is invalid")

    record = dict(_as_mapping(value, "sequential recovery record"))
    if set(record) != {
        "source_output_dir",
        "invalid_attempt_artifact",
        "source_manifest_artifact",
        "reuse_policy",
    } or record.get("reuse_policy") != "no_results_reused_fresh_wave_zero":
        raise RuntimeError("sequential recovery record fields changed")
    source_raw = record.get("source_output_dir")
    if not isinstance(source_raw, str) or not Path(source_raw).is_absolute():
        raise RuntimeError("sequential recovery source path is not absolute")
    source_path = Path(source_raw)
    source_parts = tuple(part.casefold() for part in source_path.parts)
    campaign_parts = tuple(part.casefold() for part in campaign_root.parts)
    overlaps_campaign = (
        source_parts == campaign_parts[: len(source_parts)]
        or campaign_parts == source_parts[: len(campaign_parts)]
    )
    if (
        str(source_path) != source_raw
        or any(part in {".", ".."} for part in source_parts)
        or "results.pkl" in source_parts
        or overlaps_campaign
    ):
        raise RuntimeError("sequential recovery source path is unsafe")
    _reject_symlink_components(source_path, "sequential recovery source")
    try:
        source_root = source_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("sequential recovery source does not exist") from exc
    if not source_root.is_dir() or source_raw != str(source_root):
        raise RuntimeError("sequential recovery source path is not canonical")
    if source_root != source_path:
        raise RuntimeError("sequential recovery source path is not canonical")
    if (
        source_root == campaign_root
        or source_root in campaign_root.parents
        or campaign_root in source_root.parents
    ):
        raise RuntimeError("sequential recovery source overlaps its destination")
    _validate_recovery_namespace_policy(source_root)
    if _exists_including_broken_symlink(
        source_root / campaign.COMPLETION_ATTESTATION_FILENAME,
        "recovery-source completion attestation",
    ):
        raise RuntimeError("cannot recover from a completed campaign")

    marker_path = source_root / campaign.INVALID_ATTEMPT_FILENAME
    source_manifest_path = source_root / campaign.MANIFEST_FILENAME
    marker_bytes, _ = _absolute_artifact_bytes(
        marker_path,
        record.get("invalid_attempt_artifact"),
        field="recovery invalid-attempt artifact",
    )
    source_manifest_bytes, source_manifest_digest = _absolute_artifact_bytes(
        source_manifest_path,
        record.get("source_manifest_artifact"),
        field="recovery source-manifest artifact",
    )
    marker = dict(
        _as_mapping(
            _strict_json_loads(marker_bytes, "recovery invalid-attempt artifact"),
            "recovery invalid-attempt artifact",
        )
    )
    source_manifest = dict(
        _as_mapping(
            _strict_json_loads(
                source_manifest_bytes, "recovery source-manifest artifact"
            ),
            "recovery source-manifest artifact",
        )
    )
    marker_fields = {
        "schema_version",
        "kind",
        "invalidated_at_utc",
        "execution_mode",
        "stage",
        "reuse_allowed",
        "recovery_policy",
        "manifest_sha256",
        "worker_shutdown_confirmed",
        "failure_swap_telemetry",
        "error_type",
        "error",
    }
    if (
        set(marker) != marker_fields
        or type(marker.get("schema_version")) is not int
        or marker.get("schema_version") != campaign.HARNESS_SCHEMA_VERSION
        or marker.get("kind") != campaign.CAMPAIGN_KIND + "_invalid_attempt"
        or not isinstance(marker.get("invalidated_at_utc"), str)
        or not marker["invalidated_at_utc"]
        or marker.get("execution_mode") != "concurrent"
        or marker.get("stage") != "production"
        or marker.get("reuse_allowed") is not False
        or marker.get("recovery_policy")
        != "fresh_sequential_namespace_from_wave_zero_only"
        or marker.get("manifest_sha256") != source_manifest_digest
        or marker.get("worker_shutdown_confirmed") is not True
        or not isinstance(marker.get("error_type"), str)
        or not 0 < len(marker["error_type"]) <= 256
        or not isinstance(marker.get("error"), str)
        or not 0 < len(marker["error"]) <= 4_096
    ):
        raise RuntimeError("recovery invalid-attempt marker is not canonical")
    _validate_recovery_failure_swap_telemetry(
        marker.get("failure_swap_telemetry")
    )

    frozen_fields = (
        "protocol_sha256",
        "frozen_protocol_sha256",
        "coordinate_manifest_sha256",
        "schedule_sha256",
        "schedule",
        "expected_jobs",
        "expected_child_fits",
        "time_limit_seconds",
        "resolved_child_num_cpus",
        "swap_policy",
        "timing_admissible",
        "source_freeze",
        "source",
        "runtime",
    )
    if (
        set(source_manifest) != MANIFEST_FIELDS
        or type(source_manifest.get("schema_version")) is not int
        or source_manifest.get("schema_version") != 1
        or source_manifest.get("kind") != campaign.CAMPAIGN_KIND
        or source_manifest.get("output_dir") != str(source_root)
        or not isinstance(source_manifest.get("created_at_utc"), str)
        or not source_manifest["created_at_utc"]
        or source_manifest.get("execution_mode") != "concurrent"
        or source_manifest.get("sequential_recovery") is not None
        or any(
            _canonical_json(source_manifest.get(field))
            != _canonical_json(manifest.get(field))
            for field in frozen_fields
        )
    ):
        raise RuntimeError("recovery source manifest is foreign or changed")

def verify_campaign_integrity(
    input_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    list[Path],
]:
    """Authenticate the complete campaign without decoding a raw result."""
    input_parts = tuple(part.casefold() for part in input_dir.parts)
    if ".." in input_parts or "results.pkl" in input_parts:
        # Reject the raw-result artifact itself before lstat/resolve/is_dir can
        # inspect it, including malformed descendants and parent traversals.
        # An unrelated ``experiments`` ancestor is not raw authority: valid
        # campaign roots may live under such a namespace.
        raise RuntimeError("campaign directory path is unsafe")
    _reject_symlink_components(input_dir, "campaign directory")
    try:
        root = input_dir.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("campaign directory does not exist") from exc
    if not root.is_dir():
        raise RuntimeError("campaign path is not a directory")

    manifest_path = _campaign_json_path(
        root, campaign.MANIFEST_FILENAME, "run manifest"
    )
    manifest, manifest_bytes = _read_json_file(manifest_path, "run manifest")
    if set(manifest) != MANIFEST_FIELDS:
        raise RuntimeError("run manifest fields are not exact")
    execution_mode = manifest.get("execution_mode")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != campaign.CAMPAIGN_KIND
        or not isinstance(manifest.get("created_at_utc"), str)
        or not manifest.get("created_at_utc")
        or manifest.get("output_dir") != str(root)
        or manifest.get("protocol_sha256") != campaign.protocol_sha256()
        or manifest.get("frozen_protocol_sha256")
        != campaign.frozen_protocol_sha256()
        or manifest.get("coordinate_manifest_sha256")
        != campaign.COORDINATE_MANIFEST_SHA256
        or manifest.get("schedule_sha256") != campaign.schedule_sha256()
        or _canonical_json(manifest.get("schedule"))
        != _canonical_json(campaign.expected_schedule())
        or type(manifest.get("expected_jobs")) is not int
        or manifest.get("expected_jobs") != 90
        or type(manifest.get("expected_child_fits")) is not int
        or manifest.get("expected_child_fits") != 720
        or type(manifest.get("time_limit_seconds")) is not float
        or manifest.get("time_limit_seconds") != 3_600.0
        or type(manifest.get("resolved_child_num_cpus")) is not int
        or manifest.get("resolved_child_num_cpus") != 18
        or execution_mode not in {"concurrent", "sequential_recovery"}
        or manifest.get("swap_policy") != "quality_only_swap_in"
        or manifest.get("timing_admissible") is not False
    ):
        raise RuntimeError("run manifest does not match the frozen campaign")
    current_freeze = campaign.validate_source_freeze()
    current_source = campaign.collect_source_provenance(output_dir=root)
    current_runtime = campaign.collect_runtime_provenance()
    if (
        _canonical_json(manifest.get("source_freeze"))
        != _canonical_json(current_freeze)
        or _canonical_json(manifest.get("source"))
        != _canonical_json(current_source)
        or _canonical_json(manifest.get("runtime"))
        != _canonical_json(current_runtime)
    ):
        raise RuntimeError("source, registry, or runtime provenance changed")
    _validate_sequential_recovery(manifest, campaign_root=root)
    source_diagnostics = {
        "source_freeze_sha256": _sha256(_canonical_json(current_freeze)),
        "source_provenance_sha256": _sha256(_canonical_json(current_source)),
        "runtime_provenance_sha256": _sha256(_canonical_json(current_runtime)),
    }

    attestation_path = _campaign_json_path(
        root, campaign.COMPLETION_ATTESTATION_FILENAME, "completion attestation"
    )
    attestation, attestation_bytes = _read_json_file(
        attestation_path, "completion attestation"
    )
    attestation_fields = {
        "schema_version",
        "kind",
        "completed_at_utc",
        "pid",
        "execution_mode",
        "swap_policy",
        "timing_admissible",
        "protocol_sha256",
        "frozen_protocol_sha256",
        "coordinate_manifest_sha256",
        "schedule_sha256",
        "manifest_sha256",
        "result_count",
        "expected_result_count",
        "expected_child_fits",
        "result_artifacts",
        "raw_result_verification",
        "analysis_boundary",
        "swap_audit",
        "analysis_payload_artifact",
        "schedule_artifact",
        "preflight_report_artifact",
        "concurrency_history_artifact",
        "warmup_history_artifact",
        "validation",
    }
    if set(attestation) != attestation_fields:
        raise RuntimeError("completion attestation fields are not exact")
    if (
        type(attestation.get("schema_version")) is not int
        or attestation.get("schema_version") != campaign.HARNESS_SCHEMA_VERSION
        or attestation.get("kind") != campaign.COMPLETION_KIND
        or not isinstance(attestation.get("completed_at_utc"), str)
        or not attestation.get("completed_at_utc")
        or type(attestation.get("pid")) is not int
        or attestation.get("pid") <= 0
        or attestation.get("execution_mode") != execution_mode
        or attestation.get("swap_policy") != "quality_only_swap_in"
        or attestation.get("timing_admissible") is not False
        or attestation.get("protocol_sha256") != campaign.protocol_sha256()
        or attestation.get("frozen_protocol_sha256")
        != campaign.frozen_protocol_sha256()
        or attestation.get("coordinate_manifest_sha256")
        != campaign.COORDINATE_MANIFEST_SHA256
        or attestation.get("schedule_sha256") != campaign.schedule_sha256()
        or attestation.get("manifest_sha256") != _sha256(manifest_bytes)
        or type(attestation.get("result_count")) is not int
        or attestation.get("result_count") != 90
        or type(attestation.get("expected_result_count")) is not int
        or attestation.get("expected_result_count") != 90
        or type(attestation.get("expected_child_fits")) is not int
        or attestation.get("expected_child_fits") != 720
    ):
        raise RuntimeError("completion attestation does not match the campaign")

    artifacts = _validate_runner_attested_result_manifest(
        attestation.get("result_artifacts")
    )
    boundary = campaign.analysis_boundary()
    raw_verification = _as_mapping(
        attestation.get("raw_result_verification"),
        "completion raw-result verification",
    )
    if (
        set(raw_verification)
        != {"authority", "count", "method", "analyzer_access"}
        or raw_verification.get("authority") != "runner"
        or _exact_int(raw_verification.get("count"), "raw result count") != 90
        or raw_verification.get("method")
        != "sha256_size_and_safe_extraction"
        or raw_verification.get("analyzer_access") != "forbidden"
        or _canonical_json(attestation.get("analysis_boundary"))
        != _canonical_json(boundary)
    ):
        raise RuntimeError("completion attestation analysis boundary changed")

    schedule_bytes, schedule_digest = _singleton_artifact_bytes(
        root,
        attestation.get("schedule_artifact"),
        expected_name=campaign.SCHEDULE_FILENAME,
        field="schedule artifact",
    )
    schedule = _strict_json_loads(schedule_bytes, "schedule artifact")
    if (
        schedule != campaign.expected_schedule()
        or _sha256(_canonical_json(schedule)) != campaign.schedule_sha256()
    ):
        raise RuntimeError("schedule artifact does not match the frozen schedule")

    singleton_specs = (
        (
            "preflight_report_artifact",
            campaign.PREFLIGHT_REPORT_FILENAME,
            "preflight report",
        ),
        (
            "concurrency_history_artifact",
            campaign.CONCURRENCY_HISTORY_FILENAME,
            "concurrency history",
        ),
        (
            "warmup_history_artifact",
            campaign.WARMUP_HISTORY_FILENAME,
            "warmup history",
        ),
    )
    operational: dict[str, Any] = {}
    singleton_digests: dict[str, str] = {}
    for key, filename, field in singleton_specs:
        raw, digest = _singleton_artifact_bytes(
            root,
            attestation.get(key),
            expected_name=filename,
            field=field,
        )
        operational[key] = _strict_json_loads(raw, field)
        singleton_digests[key] = digest
    operational_validator = getattr(
        campaign, "validate_operational_artifacts_for_analysis", None
    )
    if not callable(operational_validator):
        raise RuntimeError("runner lacks operational artifact validation support")
    operational_validator(
        operational,
        manifest=manifest,
        attestation=attestation,
        output_dir=root,
    )
    rebuilt_swap_audit = campaign.build_swap_audit(
        operational["preflight_report_artifact"],
        operational["concurrency_history_artifact"],
        execution_mode=str(execution_mode),
    )

    payload_bytes, payload_digest = _singleton_artifact_bytes(
        root,
        attestation.get("analysis_payload_artifact"),
        expected_name=campaign.ANALYSIS_PAYLOAD_FILENAME,
        field="safe analysis payload",
    )
    payload = dict(
        _as_mapping(
            _strict_json_loads(payload_bytes, "safe analysis payload"),
            "safe analysis payload",
        )
    )
    payload_fields = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "frozen_protocol_sha256",
        "coordinate_manifest_sha256",
        "schedule_sha256",
        "manifest_sha256",
        "result_artifacts_sha256",
        "analysis_boundary_sha256",
        "swap_policy",
        "timing_admissible",
        "swap_audit",
        "outer_rows",
        "child_rows",
    }
    if set(payload) != payload_fields:
        raise RuntimeError("safe analysis payload fields are not exact")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != campaign.HARNESS_SCHEMA_VERSION
        or payload.get("kind") != campaign.PAYLOAD_KIND
        or payload.get("protocol_sha256") != campaign.protocol_sha256()
        or payload.get("frozen_protocol_sha256")
        != campaign.frozen_protocol_sha256()
        or payload.get("coordinate_manifest_sha256")
        != campaign.COORDINATE_MANIFEST_SHA256
        or payload.get("schedule_sha256") != campaign.schedule_sha256()
        or payload.get("manifest_sha256") != _sha256(manifest_bytes)
        or payload.get("result_artifacts_sha256")
        != _sha256(_canonical_json(artifacts))
        or payload.get("analysis_boundary_sha256")
        != _sha256(_canonical_json(boundary))
        or payload.get("swap_policy") != "quality_only_swap_in"
        or payload.get("timing_admissible") is not False
        or _canonical_json(payload.get("swap_audit"))
        != _canonical_json(attestation.get("swap_audit"))
        or _canonical_json(payload.get("swap_audit"))
        != _canonical_json(rebuilt_swap_audit)
    ):
        raise RuntimeError("safe analysis payload does not bind the campaign")
    outer_rows, child_rows = _validate_payload_rows(payload, artifacts)
    payload["outer_rows"] = outer_rows
    payload["child_rows"] = child_rows

    completion_validator = getattr(
        campaign, "validate_completion_for_analysis", None
    )
    if not callable(completion_validator):
        raise RuntimeError("runner lacks completion validation support")
    completion_validator(
        attestation.get("validation"),
        manifest=manifest,
        outer_rows=outer_rows,
        child_rows=child_rows,
        swap_audit=rebuilt_swap_audit,
    )

    digests = {
        "manifest_sha256": _sha256(manifest_bytes),
        "completion_attestation_sha256": _sha256(attestation_bytes),
        "analysis_payload_sha256": payload_digest,
        "schedule_artifact_sha256": schedule_digest,
        **singleton_digests,
        "source_validation_sha256": _sha256(_canonical_json(source_diagnostics)),
    }
    protected = [
        _campaign_json_path(root, name, f"protected analyzer input {name}")
        for name in campaign.ANALYZER_CAMPAIGN_JSON_FILENAMES
    ]
    return manifest, attestation, payload, digests, protected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, attestation, payload, digests, protected = verify_campaign_integrity(
        args.input_dir
    )
    paired = pair_outer_rows(payload["outer_rows"])
    paired_children = pair_child_rows(payload["child_rows"])
    summary, per_dataset = analyze(
        paired,
        paired_children,
        execution_mode=str(manifest["execution_mode"]),
        swap_policy=str(manifest["swap_policy"]),
    )
    swap_audit = _as_mapping(payload.get("swap_audit"), "safe swap audit")
    preflight_swap = _as_mapping(swap_audit.get("preflight"), "preflight swap audit")
    production_swap = _as_mapping(
        swap_audit.get("production"), "production swap audit"
    )
    summary["integrity"] = {
        "outer_jobs_verified": 90,
        "selected_children_verified": 720,
        "primary_coordinates_verified": 27,
        "catboost_coordinates_verified": 9,
        "failure_count": 0,
        "imputation_count": 0,
        "deadline_hit_count": 0,
        "known_time_limit_stop_count": 0,
        "time_callback_hit_count": 0,
        "unresolved_comparator_stop_count": sum(
            row["arm"] in {"M", "C"} and row["stop_reason"] is None
            for row in payload["child_rows"]
        ),
        "worker_failure_count": 0,
        "recovery_mixing_count": 0,
        "swap_in_audit_evidence_retained": True,
        "preflight_worker_lifecycle_swap_in_bytes": preflight_swap[
            "worker_lifecycle_swap_in_bytes"
        ],
        "production_worker_lifecycle_swap_in_bytes": production_swap[
            "worker_lifecycle_swap_in_bytes"
        ],
        "production_measured_phase_swap_in_bytes": production_swap[
            "measured_phase_swap_in_bytes"
        ],
        "production_dispatches_with_swap_in_telemetry": production_swap[
            "measured_dispatch_count"
        ],
        "production_waves_with_swap_in_telemetry": production_swap["wave_count"],
        "swap_out_bytes": 0,
        "raw_results_attested_by_runner": True,
        "raw_results_read_by_analyzer": False,
        "raw_results_deserialized_by_analyzer": False,
    }
    summary["provenance"] = {
        **digests,
        "completed_at_utc": attestation.get("completed_at_utc"),
        "protocol_sha256": campaign.protocol_sha256(),
        "frozen_protocol_sha256": campaign.frozen_protocol_sha256(),
        "coordinate_manifest_sha256": campaign.COORDINATE_MANIFEST_SHA256,
        "schedule_sha256": campaign.schedule_sha256(),
    }
    output_root = args.input_dir.resolve(strict=True)
    outputs = {
        key: output_root / name for key, name in zip(OUTPUT_KEYS, OUTPUT_NAMES)
    }
    outputs = hardened._canonical_output_targets(
        output_root,
        outputs,
        protected_paths=protected,
        target_names=OUTPUT_KEYS,
    )
    output_payloads = _build_output_payloads(
        paired, per_dataset, paired_children, summary
    )
    # A second pure pass catches accidental RNG or row-order state before write.
    check_paired = pair_outer_rows(payload["outer_rows"])
    check_children = pair_child_rows(payload["child_rows"])
    check_summary, check_dataset = analyze(
        check_paired,
        check_children,
        execution_mode=str(manifest["execution_mode"]),
        swap_policy=str(manifest["swap_policy"]),
    )
    check_summary["integrity"] = summary["integrity"]
    check_summary["provenance"] = summary["provenance"]
    if output_payloads != _build_output_payloads(
        check_paired, check_dataset, check_children, check_summary
    ):
        raise RuntimeError("analysis is not byte deterministic")
    hardened._atomic_write_group(
        [(outputs[key], output_payloads[key]) for key in OUTPUT_KEYS],
        post_write_check=lambda: verify_campaign_integrity(args.input_dir),
    )
    print(
        "analyzed 90 jobs and 720 selected children; "
        f"decision={summary['decision']}; wrote {outputs['summary_json']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
