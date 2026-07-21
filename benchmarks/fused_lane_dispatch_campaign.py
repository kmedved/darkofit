"""Frozen mechanics and analyzers for fused-lane dispatch evidence.

This module contains no execution side effects.  The runner owns fresh-process
orchestration; this module owns the prospectively declared cases, deterministic
generators, canonical exactness projection, and conjunctive decision rules.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
CAMPAIGN_NAME = "fused_lane_dispatch_calibration_v1_20260721"
CALIBRATION_SEED = 20_260_721
VALIDATION_SEED = 20_260_722
CALIBRATION_ROWS = (8_192, 32_768, 131_072, 524_288, 1_048_576)
CALIBRATION_SHAPES = ((15, 4), (24, 9), (48, 14))
CALIBRATION_HESSIANS = ("unit", "variable")
CALIBRATION_DEPTH = 6
CALIBRATION_BINS = 128
CALIBRATION_WARMUPS = 2
CALIBRATION_REPEATS = 7
LANES = ("fused", "unfused")

VALIDATION_CELLS = (
    {
        "cell_id": "small_unit",
        "task": "rmse",
        "rows": 12_000,
        "features": 15,
        "threads": 14,
        "depth": 6,
        "max_bins": 254,
        "rounds": 200,
    },
    {
        "cell_id": "mid_weighted",
        "task": "weighted_rmse",
        "rows": 75_000,
        "features": 37,
        "threads": 6,
        "depth": 4,
        "max_bins": 64,
        "rounds": 80,
    },
    {
        "cell_id": "mid_binary",
        "task": "binary_logloss",
        "rows": 280_000,
        "features": 19,
        "threads": 11,
        "depth": 8,
        "max_bins": 254,
        "rounds": 60,
    },
    {
        "cell_id": "large_unit",
        "task": "rmse",
        "rows": 650_000,
        "features": 24,
        "threads": 14,
        "depth": 6,
        "max_bins": 128,
        "rounds": 40,
    },
    {
        "cell_id": "large_binary",
        "task": "binary_logloss",
        "rows": 900_000,
        "features": 47,
        "threads": 9,
        "depth": 5,
        "max_bins": 192,
        "rounds": 40,
    },
    {
        "cell_id": "large_weighted",
        "task": "weighted_rmse",
        "rows": 1_100_000,
        "features": 63,
        "threads": 14,
        "depth": 7,
        "max_bins": 254,
        "rounds": 30,
    },
)
VALIDATION_BLOCK_ORDERS = (
    ("fused", "auto"),
    ("auto", "fused"),
    ("fused", "auto"),
)

STABILITY_LIMIT = 0.10
CALIBRATION_GEOMEAN_LIMIT = 0.97
CALIBRATION_WORST_LIMIT = 1.02
VALIDATION_ALL_FIT_LIMIT = 0.98
VALIDATION_UNFUSED_FIT_LIMIT = 0.97
VALIDATION_ALL_TREE_LIMIT = 0.98
VALIDATION_UNFUSED_TREE_LIMIT = 0.95
VALIDATION_WORST_LIMIT = 1.02
VALIDATION_RSS_LIMIT = 1.05


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(name: str, value: np.ndarray | None) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    if value is None:
        digest.update(b"<none>")
        return digest.hexdigest()
    array = np.ascontiguousarray(value)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_json_bytes(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def named_arrays_sha256(values: Mapping[str, np.ndarray | None]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(array_sha256(name, values[name])))
    return digest.hexdigest()


def scan_work(rows: int, features: int, threads: int) -> int:
    active_threads = min(int(threads), int(features))
    return int(rows) * ((int(features) + active_threads - 1) // active_threads)


def calibration_specs() -> tuple[dict[str, Any], ...]:
    specs = []
    for rows in CALIBRATION_ROWS:
        for features, threads in CALIBRATION_SHAPES:
            for hessian in CALIBRATION_HESSIANS:
                specs.append(
                    {
                        "coordinate_id": (
                            f"r{rows}_f{features}_t{threads}_{hessian}"
                        ),
                        "rows": rows,
                        "features": features,
                        "threads": threads,
                        "depth": CALIBRATION_DEPTH,
                        "bins": CALIBRATION_BINS,
                        "hessian": hessian,
                        "scan_work": scan_work(rows, features, threads),
                    }
                )
    return tuple(specs)


def validation_specs() -> tuple[dict[str, Any], ...]:
    return tuple(dict(cell) for cell in VALIDATION_CELLS)


def calibration_order(repeat: int) -> tuple[str, str]:
    if not 0 <= int(repeat) < CALIBRATION_REPEATS:
        raise ValueError("calibration repeat is outside the frozen range")
    return LANES if repeat % 2 == 0 else tuple(reversed(LANES))


def _seed_sequence(base: int, *parts: Any) -> np.random.SeedSequence:
    token = "/".join(str(part) for part in parts).encode("utf-8")
    words = np.frombuffer(hashlib.sha256(token).digest()[:16], dtype=np.uint32)
    return np.random.SeedSequence([int(base), *(int(word) for word in words)])


def generate_calibration_case(
    spec: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    rows = int(spec["rows"])
    features = int(spec["features"])
    bins = int(spec["bins"])
    base_rng = np.random.default_rng(
        _seed_sequence(CALIBRATION_SEED, rows, features, bins)
    )
    X = base_rng.integers(
        0, bins, size=(rows, features), dtype=np.uint8
    )
    grad = base_rng.standard_normal(rows)
    grad += (X[:, 0].astype(np.float64) - (bins - 1) / 2.0) / bins
    grad += 0.25 * (
        X[:, min(1, features - 1)].astype(np.float64) - (bins - 1) / 2.0
    ) / bins
    if spec["hessian"] == "unit":
        hess = np.ones(rows, dtype=np.float64)
    elif spec["hessian"] == "variable":
        hess_rng = np.random.default_rng(
            _seed_sequence(CALIBRATION_SEED, rows, features, bins, "hessian")
        )
        hess = 0.25 + 1.75 * hess_rng.random(rows)
    else:
        raise ValueError("unknown calibration Hessian case")
    values = {
        "X": np.ascontiguousarray(X),
        "grad": np.ascontiguousarray(grad),
        "hess": np.ascontiguousarray(hess),
        "n_bins": np.full(features, bins, dtype=np.int64),
    }
    fingerprints = {
        name: array_sha256(name, value) for name, value in values.items()
    }
    fingerprints["dataset_sha256"] = named_arrays_sha256(values)
    return values, fingerprints


def generate_validation_case(
    spec: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray | None], dict[str, str]]:
    rows = int(spec["rows"])
    features = int(spec["features"])
    rng = np.random.default_rng(
        _seed_sequence(VALIDATION_SEED, spec["cell_id"])
    )
    X = rng.standard_normal((rows, features), dtype=np.float32)
    latent = (
        1.15 * X[:, 0].astype(np.float64)
        - 0.75 * X[:, 1].astype(np.float64)
        + 0.30 * X[:, 2].astype(np.float64) * X[:, 3].astype(np.float64)
        + 0.15 * np.sin(X[:, 4].astype(np.float64))
    )
    noise = rng.normal(0.0, 0.35, size=rows)
    if spec["task"] == "binary_logloss":
        y = (latent + noise > 0.0).astype(np.float64)
        sample_weight = None
    else:
        y = latent + noise
        sample_weight = (
            0.5 + 1.5 * rng.random(rows)
            if spec["task"] == "weighted_rmse"
            else None
        )
    values = {
        "X": np.ascontiguousarray(X),
        "y": np.ascontiguousarray(y, dtype=np.float64),
        "sample_weight": (
            None
            if sample_weight is None
            else np.ascontiguousarray(sample_weight, dtype=np.float64)
        ),
    }
    fingerprints = {
        name: array_sha256(name, value) for name, value in values.items()
    }
    fingerprints["dataset_sha256"] = named_arrays_sha256(values)
    return values, fingerprints


def _project_dispatch_header(header: dict[str, Any]) -> dict[str, Any]:
    projected = json.loads(json.dumps(header))
    params = projected.get("params")
    if isinstance(params, dict):
        params.pop("oblivious_kernel", None)
    auto_params = projected.get("auto_params")
    if isinstance(auto_params, dict):
        auto_params.pop("oblivious_kernel_dispatch", None)
    wrapper = projected.get("wrapper")
    if isinstance(wrapper, dict) and isinstance(wrapper.get("params"), dict):
        wrapper["params"].pop("oblivious_kernel", None)
    return projected


def canonical_archive_sha256(path: Path, *, project_dispatch: bool) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        names = sorted(archive.files)
        if "header" not in names:
            raise RuntimeError("DarkoFit archive has no header")
        header = json.loads(str(archive["header"]))
        if project_dispatch:
            header = _project_dispatch_header(header)
        digest.update(b"header\0")
        digest.update(_json_bytes(header))
        for name in names:
            if name == "header":
                continue
            value = np.ascontiguousarray(archive[name])
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(value.dtype.str.encode("ascii") + b"\0")
            digest.update(_json_bytes(list(value.shape)))
            digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"{label} is not positive and finite")
    return result


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _geomean(values: Sequence[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise RuntimeError("geometric mean received invalid values")
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _iqr_over_median(values: Sequence[float]) -> float:
    if not values:
        raise RuntimeError("stability statistic received no values")
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    if center <= 0.0 or not math.isfinite(center):
        raise RuntimeError("stability statistic has an invalid median")
    q1, q3 = np.percentile(array, (25, 75))
    return float((q3 - q1) / center)


def threshold_candidates(rows: Sequence[Mapping[str, Any]]) -> tuple[int | None, ...]:
    works = sorted({int(row["scan_work"]) for row in rows})
    thresholds: list[int | None] = []
    for lower, upper in zip(works, works[1:]):
        midpoint = (lower + upper) / 2.0
        if not midpoint.is_integer():
            raise RuntimeError("frozen scan-work midpoint is not integral")
        thresholds.append(int(midpoint))
    thresholds.append(None)
    return tuple(thresholds)


def analyze_calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {spec["coordinate_id"]: spec for spec in calibration_specs()}
    observed = {str(row.get("coordinate_id")): row for row in rows}
    if set(observed) != set(expected) or len(observed) != len(rows):
        raise RuntimeError("calibration coordinates are incomplete or duplicated")
    cells = []
    all_exact = True
    all_stable = True
    case_fingerprints = {}
    for coordinate_id, spec in expected.items():
        row = observed[coordinate_id]
        if any(row.get(name) != spec[name] for name in spec):
            raise RuntimeError(f"calibration identity drifted: {coordinate_id}")
        fingerprints = row.get("fingerprints")
        runtime_before = row.get("runtime_before")
        runtime_after = row.get("runtime_after")
        coordinate_exact = bool(
            row.get("seed") == CALIBRATION_SEED
            and row.get("warmups_per_lane") == CALIBRATION_WARMUPS
            and row.get("thread_mask_restored") is True
            and isinstance(fingerprints, Mapping)
            and set(fingerprints)
            == {"X", "grad", "hess", "n_bins", "dataset_sha256"}
            and all(_is_sha256(value) for value in fingerprints.values())
            and isinstance(runtime_before, Mapping)
            and runtime_before.get("ceiling") == spec["threads"]
            and runtime_before.get("current") == spec["threads"]
            and isinstance(runtime_after, Mapping)
            and runtime_after.get("ceiling") == spec["threads"]
            and runtime_after.get("current") == spec["threads"]
        )
        if isinstance(fingerprints, Mapping):
            case_fingerprints[
                (spec["rows"], spec["features"], spec["bins"], spec["hessian"])
            ] = dict(fingerprints)
        repetitions = row.get("repetitions")
        if not isinstance(repetitions, list) or len(repetitions) != CALIBRATION_REPEATS:
            raise RuntimeError(f"calibration repetitions changed: {coordinate_id}")
        ratios = []
        exact = coordinate_exact
        for repeat, record in enumerate(repetitions):
            if (
                record.get("repeat") != repeat
                or tuple(record.get("order", ())) != calibration_order(repeat)
            ):
                raise RuntimeError(f"calibration order drifted: {coordinate_id}")
            fused = _finite_positive(
                record.get("fused_seconds"), f"{coordinate_id}/fused"
            )
            unfused = _finite_positive(
                record.get("unfused_seconds"), f"{coordinate_id}/unfused"
            )
            ratios.append(unfused / fused)
            exact = bool(exact and record.get("exact") is True)
            depth = record.get("tree_depth")
            fused_levels = record.get("fused_level_count")
            unfused_levels = record.get("unfused_level_count")
            if (
                isinstance(depth, bool)
                or not isinstance(depth, int)
                or not 0 <= depth <= int(spec["depth"])
                or isinstance(fused_levels, bool)
                or not isinstance(fused_levels, int)
                or not 1 <= fused_levels <= int(spec["depth"])
                or unfused_levels != fused_levels
                or fused_levels < depth
                or record.get("fused_opposite_level_count") != 0
                or record.get("unfused_opposite_level_count") != 0
                or not _is_sha256(record.get("state_sha256"))
                or not _is_sha256(record.get("prediction_sha256"))
            ):
                exact = False
        stability = _iqr_over_median(ratios)
        all_exact = bool(all_exact and exact)
        all_stable = bool(all_stable and stability <= STABILITY_LIMIT)
        cells.append(
            {
                "coordinate_id": coordinate_id,
                "scan_work": int(spec["scan_work"]),
                "median_unfused_fused_ratio": float(median(ratios)),
                "iqr_over_median": stability,
                "exact": exact,
            }
        )

    for rows_value in CALIBRATION_ROWS:
        for features, _threads in CALIBRATION_SHAPES:
            unit = case_fingerprints.get(
                (rows_value, features, CALIBRATION_BINS, "unit")
            )
            variable = case_fingerprints.get(
                (rows_value, features, CALIBRATION_BINS, "variable")
            )
            if (
                unit is None
                or variable is None
                or any(unit[name] != variable[name] for name in ("X", "grad", "n_bins"))
                or unit["hess"] == variable["hess"]
                or unit["dataset_sha256"] == variable["dataset_sha256"]
            ):
                all_exact = False

    candidates = []
    for threshold in threshold_candidates(cells):
        selected_ratios = []
        regrets = []
        selected_lanes = []
        for cell in cells:
            unfused_ratio = cell["median_unfused_fused_ratio"]
            use_unfused = threshold is not None and cell["scan_work"] >= threshold
            selected_ratio = unfused_ratio if use_unfused else 1.0
            selected_ratios.append(selected_ratio)
            regrets.append(selected_ratio / min(1.0, unfused_ratio))
            selected_lanes.append("unfused" if use_unfused else "fused")
        candidates.append(
            {
                "threshold": threshold,
                "selected_fused_cells": selected_lanes.count("fused"),
                "selected_unfused_cells": selected_lanes.count("unfused"),
                "selected_fused_geomean_ratio": _geomean(selected_ratios),
                "worst_selected_fused_ratio": max(selected_ratios),
                "geomean_regret": _geomean(regrets),
            }
        )
    best_regret = min(candidate["geomean_regret"] for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if candidate["geomean_regret"] - best_regret <= 0.001
    ]
    selected = max(
        tied,
        key=lambda candidate: (
            math.inf if candidate["threshold"] is None else candidate["threshold"]
        ),
    )
    qualifies = bool(
        all_exact
        and all_stable
        and selected["selected_fused_cells"] > 0
        and selected["selected_unfused_cells"] > 0
        and selected["selected_fused_geomean_ratio"]
        <= CALIBRATION_GEOMEAN_LIMIT
        and selected["worst_selected_fused_ratio"] <= CALIBRATION_WORST_LIMIT
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN_NAME,
        "phase": "calibration",
        "cell_count": len(cells),
        "all_exact": all_exact,
        "all_stable": all_stable,
        "cells": cells,
        "candidates": candidates,
        "selected": selected,
        "qualifies": qualifies,
        "disposition": (
            "freeze_threshold_before_validation"
            if qualifies
            else "close_dispatch_campaign"
        ),
    }


def _validation_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row.get("cell_id")), int(row.get("block", -1)), str(row.get("arm"))


def _validation_dispatch_exact(
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    threshold: int,
    expected_lane: str,
    expected_reason: str,
    requested: str,
) -> bool:
    dispatch = row.get("dispatch_metadata")
    if not isinstance(dispatch, Mapping):
        return False
    expected_fields = {
        "schema_version",
        "requested",
        "resolved",
        "reason",
        "functional_eligible",
        "automatic_eligible",
        "threshold",
        "scan_work",
        "engaged",
        "fused_level_count",
        "unfused_level_count",
        "inputs",
    }
    inputs = dispatch.get("inputs")
    fingerprints = row.get("fingerprints")
    thread_counts = row.get("thread_counts")
    runtime_before = row.get("runtime_before")
    runtime_after = row.get("runtime_after")
    selected = dispatch.get(f"{expected_lane}_level_count")
    opposite = "unfused" if expected_lane == "fused" else "fused"
    expected_work = scan_work(
        spec["rows"], spec["features"], spec["threads"]
    )
    return bool(
        set(dispatch) == expected_fields
        and dispatch.get("schema_version") == 1
        and dispatch.get("requested") == requested
        and dispatch.get("resolved") == expected_lane
        and dispatch.get("reason") == expected_reason
        and dispatch.get("functional_eligible") is True
        and dispatch.get("automatic_eligible") is True
        and dispatch.get("threshold") == threshold
        and dispatch.get("scan_work") == expected_work
        and dispatch.get("engaged") is True
        and isinstance(selected, int)
        and not isinstance(selected, bool)
        and selected > 0
        and dispatch.get(f"{opposite}_level_count") == 0
        and row.get("selected_level_count") == selected
        and row.get("opposite_level_count") == 0
        and isinstance(inputs, Mapping)
        and set(inputs)
        == {
            "platform_system",
            "platform_machine",
            "logical_cpu_count",
            "n_rows",
            "n_active_features",
            "n_threads",
            "depth",
            "max_realized_bins",
        }
        and inputs.get("platform_system") == "Darwin"
        and inputs.get("platform_machine") == "arm64"
        and isinstance(inputs.get("logical_cpu_count"), int)
        and not isinstance(inputs.get("logical_cpu_count"), bool)
        and inputs["logical_cpu_count"] >= int(spec["threads"])
        and inputs.get("n_rows") == spec["rows"]
        and inputs.get("n_active_features") == spec["features"]
        and inputs.get("n_threads") == spec["threads"]
        and inputs.get("depth") == spec["depth"]
        and inputs.get("max_realized_bins") == int(spec["max_bins"]) + 1
        and row.get("threshold") == threshold
        and row.get("seed") == VALIDATION_SEED
        and isinstance(fingerprints, Mapping)
        and set(fingerprints)
        == {"X", "y", "sample_weight", "dataset_sha256"}
        and all(_is_sha256(value) for value in fingerprints.values())
        and row.get("dataset_sha256") == fingerprints.get("dataset_sha256")
        and _is_sha256(row.get("projected_archive_sha256"))
        and _is_sha256(row.get("archive_sha256"))
        and _is_sha256(row.get("prediction_sha256"))
        and _is_sha256(row.get("feature_importance_sha256"))
        and (
            _is_sha256(row.get("probability_sha256"))
            if spec["task"] == "binary_logloss"
            else row.get("probability_sha256") is None
        )
        and isinstance(thread_counts, Mapping)
        and set(thread_counts)
        == {
            "ambient",
            "after_warmup",
            "after_fit",
            "after_predict",
            "after_roundtrip",
        }
        and all(value == spec["threads"] for value in thread_counts.values())
        and isinstance(runtime_before, Mapping)
        and runtime_before.get("ceiling") == spec["threads"]
        and runtime_before.get("current") == spec["threads"]
        and isinstance(runtime_after, Mapping)
        and runtime_after.get("ceiling") == spec["threads"]
        and runtime_after.get("current") == spec["threads"]
    )


def analyze_validation(
    rows: Sequence[Mapping[str, Any]], *, threshold: int
) -> dict[str, Any]:
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        raise RuntimeError("validation threshold is invalid")
    specs = {spec["cell_id"]: spec for spec in validation_specs()}
    expected_sequence = [
        (cell_id, block, arm)
        for cell_id in specs
        for block, order in enumerate(VALIDATION_BLOCK_ORDERS)
        for arm in order
    ]
    expected_keys = set(expected_sequence)
    observed_sequence = [_validation_key(row) for row in rows]
    observed = {_validation_key(row): row for row in rows}
    if (
        observed_sequence != expected_sequence
        or set(observed) != expected_keys
        or len(observed) != len(rows)
    ):
        raise RuntimeError("validation rows are incomplete or duplicated")
    for (cell_id, _block, _arm), row in zip(expected_sequence, rows):
        spec = specs[cell_id]
        if any(row.get(name) != spec[name] for name in spec):
            raise RuntimeError(f"validation identity drifted: {cell_id}")

    summaries = []
    all_exact = True
    for cell_id, spec in specs.items():
        fit_ratios = []
        tree_ratios = []
        rss_ratios = []
        expected_lane = (
            "unfused"
            if scan_work(spec["rows"], spec["features"], spec["threads"])
            >= int(threshold)
            else "fused"
        )
        expected_reason = (
            "at_or_above_threshold"
            if expected_lane == "unfused"
            else "below_threshold"
        )
        cell_exact = True
        for block in range(len(VALIDATION_BLOCK_ORDERS)):
            control = observed[(cell_id, block, "fused")]
            candidate = observed[(cell_id, block, "auto")]
            if (
                not _validation_dispatch_exact(
                    control,
                    spec,
                    threshold=threshold,
                    expected_lane="fused",
                    expected_reason="user_forced_fused",
                    requested="fused",
                )
                or not _validation_dispatch_exact(
                    candidate,
                    spec,
                    threshold=threshold,
                    expected_lane=expected_lane,
                    expected_reason=expected_reason,
                    requested="auto",
                )
                or control.get("dataset_sha256")
                != candidate.get("dataset_sha256")
                or control.get("projected_archive_sha256")
                != candidate.get("projected_archive_sha256")
                or control.get("prediction_sha256")
                != candidate.get("prediction_sha256")
                or control.get("probability_sha256")
                != candidate.get("probability_sha256")
                or control.get("feature_importance_sha256")
                != candidate.get("feature_importance_sha256")
                or control.get("safe_roundtrip_exact") is not True
                or candidate.get("safe_roundtrip_exact") is not True
                or candidate.get("resolved_lane") != expected_lane
                or control.get("resolved_lane") != "fused"
                or candidate.get("requested_lane") != "auto"
                or control.get("requested_lane") != "fused"
                or candidate.get("dispatch_reason") != expected_reason
                or control.get("dispatch_reason") != "user_forced_fused"
                or candidate.get("thread_mask_restored") is not True
                or control.get("thread_mask_restored") is not True
            ):
                cell_exact = False
            for row in (control, candidate):
                if row.get("selected_level_count", 0) < 1 or row.get(
                    "opposite_level_count"
                ) != 0:
                    cell_exact = False
            fit_ratios.append(
                _finite_positive(candidate.get("fit_seconds"), "candidate fit")
                / _finite_positive(control.get("fit_seconds"), "control fit")
            )
            tree_ratios.append(
                _finite_positive(candidate.get("tree_seconds"), "candidate tree")
                / _finite_positive(control.get("tree_seconds"), "control tree")
            )
            rss_ratios.append(
                _finite_positive(candidate.get("peak_rss_bytes"), "candidate RSS")
                / _finite_positive(control.get("peak_rss_bytes"), "control RSS")
            )
        fit_stability = _iqr_over_median(fit_ratios)
        tree_stability = _iqr_over_median(tree_ratios)
        rss_stability = _iqr_over_median(rss_ratios)
        all_exact = bool(all_exact and cell_exact)
        summaries.append(
            {
                "cell_id": cell_id,
                "selected_lane": expected_lane,
                "exact": cell_exact,
                "fit_ratio": float(median(fit_ratios)),
                "tree_ratio": float(median(tree_ratios)),
                "rss_ratio": float(median(rss_ratios)),
                "fit_iqr_over_median": fit_stability,
                "tree_iqr_over_median": tree_stability,
                "rss_iqr_over_median": rss_stability,
                "stable": bool(
                    fit_stability <= STABILITY_LIMIT
                    and tree_stability <= STABILITY_LIMIT
                    and rss_stability <= STABILITY_LIMIT
                ),
            }
        )
    fit_ratios = [summary["fit_ratio"] for summary in summaries]
    tree_ratios = [summary["tree_ratio"] for summary in summaries]
    unfused = [summary for summary in summaries if summary["selected_lane"] == "unfused"]
    stable = all(summary["stable"] for summary in summaries)
    mixed_dispatch = 2 <= len(unfused) <= len(summaries) - 2
    qualifies = bool(
        all_exact
        and stable
        and mixed_dispatch
        and _geomean(fit_ratios) <= VALIDATION_ALL_FIT_LIMIT
        and _geomean([summary["fit_ratio"] for summary in unfused])
        <= VALIDATION_UNFUSED_FIT_LIMIT
        and _geomean(tree_ratios) <= VALIDATION_ALL_TREE_LIMIT
        and _geomean([summary["tree_ratio"] for summary in unfused])
        <= VALIDATION_UNFUSED_TREE_LIMIT
        and max(fit_ratios + tree_ratios) <= VALIDATION_WORST_LIMIT
        and max(summary["rss_ratio"] for summary in summaries)
        <= VALIDATION_RSS_LIMIT
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN_NAME,
        "phase": "validation",
        "threshold": int(threshold),
        "all_exact": all_exact,
        "all_stable": stable,
        "mixed_dispatch": mixed_dispatch,
        "cells": summaries,
        "all_fit_geomean_ratio": _geomean(fit_ratios),
        "unfused_fit_geomean_ratio": (
            _geomean([summary["fit_ratio"] for summary in unfused])
            if unfused
            else None
        ),
        "all_tree_geomean_ratio": _geomean(tree_ratios),
        "unfused_tree_geomean_ratio": (
            _geomean([summary["tree_ratio"] for summary in unfused])
            if unfused
            else None
        ),
        "qualifies": qualifies,
        "disposition": (
            "retain_measured_dispatch" if qualifies else "close_dispatch_campaign"
        ),
    }
