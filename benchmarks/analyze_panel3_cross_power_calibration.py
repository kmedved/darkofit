#!/usr/bin/env python3
"""Analyze the complete spent Panel 3 exact-policy calibration census."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import freeze_panel3_cross_power_calibration as freeze  # noqa: E402
from benchmarks import panel3_registry_common as common  # noqa: E402
from benchmarks import run_panel3_cross_power_calibration as runner  # noqa: E402


DEFAULT_RAW = (
    ROOT / "benchmarks" / "panel3_cross_power_calibration_raw.json"
)
RAW_RELATIVE = str(DEFAULT_RAW.relative_to(ROOT))
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "panel3_cross_power_calibration_summary.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"{label} is not finite and positive")
    return result


def _geomean(values: Sequence[float]) -> float:
    if not values:
        raise RuntimeError("calibration geometric mean is empty")
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _direction(ratio: float) -> str:
    if ratio < 1.0:
        return "win"
    if ratio > 1.0:
        return "loss"
    return "tie"


def _expected_result_keys() -> list[str]:
    return [
        runner.worker_key(coordinate, arm)
        for coordinate in runner.expected_coordinates()
        for arm in runner.ARM_ORDER
    ]


def _validate_result(
    result: Any,
    *,
    expected_key: str,
    expected_coordinate: dict[str, int],
    expected_arm: str,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise RuntimeError("calibration result is not an object")
    required = {
        "worker_key",
        "coordinate",
        "arm",
        "task",
        "split",
        "t5_size_gate_applicable",
        "rmse",
        "fit_seconds",
        "wall_seconds",
        "prediction_timing",
        "prediction_sha256",
        "test_target_sha256",
        "metadata",
        "peak_rss_bytes",
        "behavior_fingerprint_sha256",
    }
    if (
        set(result) != required
        or result["worker_key"] != expected_key
        or result["coordinate"] != expected_coordinate
        or result["arm"] != expected_arm
        or not isinstance(result["task"], dict)
        or not isinstance(result["split"], dict)
        or type(result["t5_size_gate_applicable"]) is not bool
        or not isinstance(result["metadata"], dict)
        or type(result["peak_rss_bytes"]) is not int
        or result["peak_rss_bytes"] <= 0
    ):
        raise RuntimeError(f"calibration result contract changed: {expected_key}")
    _positive_finite(result["rmse"], f"{expected_key} RMSE")
    if (
        isinstance(result["fit_seconds"], bool)
        or not isinstance(result["fit_seconds"], (int, float))
        or isinstance(result["wall_seconds"], bool)
        or not isinstance(result["wall_seconds"], (int, float))
        or not math.isfinite(float(result["fit_seconds"]))
        or float(result["fit_seconds"]) < 0.0
        or not math.isfinite(float(result["wall_seconds"]))
        or float(result["wall_seconds"]) < 0.0
        or not _is_sha256(result["prediction_sha256"])
        or not _is_sha256(result["test_target_sha256"])
    ):
        raise RuntimeError(f"calibration result measurements changed: {expected_key}")
    split = result["split"]
    if (
        set(split)
        != {
            "repeat",
            "fold",
            "sample",
            "train_rows",
            "test_rows",
            "train_index_sha256",
            "test_index_sha256",
        }
        or split["repeat"] != expected_coordinate["repeat"]
        or split["fold"] != expected_coordinate["fold"]
        or split["sample"] != expected_coordinate["sample"]
        or type(split["train_rows"]) is not int
        or split["train_rows"] <= 0
        or type(split["test_rows"]) is not int
        or split["test_rows"] <= 0
        or result["t5_size_gate_applicable"]
        is not (split["train_rows"] >= runner.T5_SIZE_GATE)
    ):
        raise RuntimeError(f"calibration split contract changed: {expected_key}")
    metadata = result["metadata"]
    if metadata.get("kind") != expected_arm:
        raise RuntimeError(f"calibration metadata arm changed: {expected_key}")
    from benchmarks import analyze_panel3_confirmation as confirmation_analyzer

    confirmation_analyzer._validate_prediction_timing(
        result["prediction_timing"]
    )
    runner.validate_arm_metadata(
        metadata,
        arm=expected_arm,
        t5_size_gate_applicable=result["t5_size_gate_applicable"],
        fit_seconds=float(result["fit_seconds"]),
        train_rows=int(result["split"]["train_rows"]),
        feature_count=int(result["task"]["n_features"]),
        categorical_indices=result["task"][
            "categorical_feature_indices"
        ],
    )
    if expected_arm == runner.CONTROL_ARM:
        if (
            metadata.get("engaged") is not False
            or metadata.get("selected_configuration") != "product_default"
        ):
            raise RuntimeError("calibration control metadata changed")
    elif expected_arm == "t5_composite_policy":
        if (
            type(metadata.get("engaged")) is not bool
            or metadata.get("size_gate") != runner.T5_SIZE_GATE
        ):
            raise RuntimeError("calibration T5 metadata changed")
        if not result["t5_size_gate_applicable"] and (
            metadata["engaged"] is not False
            or metadata.get("decline_reason") != "below_size_gate"
        ):
            raise RuntimeError("calibration T5 size-gate decline changed")
    else:
        if (
            type(metadata.get("engaged")) is not bool
            or metadata.get("cross_guard_ratio")
            != runner.panel3.GUARDED_CROSS_RATIO
            or type(metadata.get("selected_crosses")) is not bool
            or metadata["selected_crosses"] is not metadata["engaged"]
        ):
            raise RuntimeError("calibration guarded-cross metadata changed")
    behavior = {
        "coordinate": expected_coordinate,
        "arm": expected_arm,
        "rmse": result["rmse"],
        "prediction_sha256": result["prediction_sha256"],
        "metadata": metadata,
    }
    if result["behavior_fingerprint_sha256"] != _json_sha256(behavior):
        raise RuntimeError(f"calibration behavior hash changed: {expected_key}")
    return result


def validate_raw(
    raw: dict[str, Any],
    *,
    raw_path: Path = DEFAULT_RAW,
    verify_source: bool = True,
    verify_spool: bool = True,
) -> list[dict[str, Any]]:
    if verify_spool and not verify_source:
        raise ValueError(
            "calibration spool verification requires source verification"
        )
    common.verify_artifact_sha256(raw, "raw_artifact_sha256")
    required = {
        "schema_version",
        "name",
        "created_at",
        "source_freeze_path",
        "source_freeze_file_sha256",
        "source_freeze_sha256",
        "runtime",
        "tasks",
        "coordinates",
        "arms",
        "execution",
        "spool",
        "results",
        "result_count",
        "all_results_preserved_without_filtering",
        "outcomes_scored",
        "analysis_performed",
        "development_only",
        "panel3_authorized",
        "default_promotion_authorized",
        "product_claim_authorized",
        "raw_artifact_sha256",
    }
    if (
        set(raw) != required
        or raw["schema_version"] != 1
        or raw["name"]
        != "darkofit_panel3_cross_power_calibration_raw_v1"
        or raw["source_freeze_path"]
        != str(runner.DEFAULT_FREEZE.relative_to(ROOT))
        or raw["tasks"] != freeze.TASKS
        or raw["coordinates"] != runner.expected_coordinates()
        or raw["arms"] != list(runner.ARM_ORDER)
        or raw["execution"]
        != {
            "kind": (
                "coordinate_waves_three_concurrent_isolated_arm_processes"
            ),
            "concurrent_processes": 3,
            "worker_thread_count": runner.THREAD_COUNT,
            "random_state": runner.RANDOM_STATE,
            "timing_and_memory_claim_eligible": False,
        }
        or raw["result_count"] != 117
        or raw["all_results_preserved_without_filtering"] is not True
        or raw["outcomes_scored"] is not True
        or raw["analysis_performed"] is not False
        or raw["development_only"] is not True
        or raw["panel3_authorized"] is not False
        or raw["default_promotion_authorized"] is not False
        or raw["product_claim_authorized"] is not False
        or not isinstance(raw["results"], list)
        or len(raw["results"]) != 117
    ):
        raise RuntimeError("calibration raw contract changed")
    if verify_source:
        freeze_path = ROOT / raw["source_freeze_path"]
        source_freeze, source_freeze_file_sha256 = (
            common.secure_load_json(freeze_path)
        )
        if (
            source_freeze_file_sha256
            != raw["source_freeze_file_sha256"]
        ):
            raise RuntimeError("calibration source-freeze file changed")
        if (
            not isinstance(source_freeze, dict)
            or source_freeze.get("source_freeze_sha256")
            != raw["source_freeze_sha256"]
        ):
            raise RuntimeError("calibration source-freeze binding changed")
        campaign_sources = (
            Path(__file__).resolve(),
            freeze.RUNNER,
            Path(freeze.__file__).resolve(),
            freeze.PROTOCOL,
        )
        for path in campaign_sources:
            relative = str(path.relative_to(ROOT))
            if (
                source_freeze["source_file_sha256"].get(relative)
                != _sha256(path)
            ):
                raise RuntimeError(
                    "calibration analysis source changed: "
                    f"{relative}"
                )
        runtime = runner.validate_source_freeze(
            source_freeze,
            freeze_path=freeze_path,
            require_repository_state=False,
        )
        if raw["runtime"] != runtime:
            raise RuntimeError("calibration runtime record changed")
    expected = [
        (runner.worker_key(coordinate, arm), coordinate, arm)
        for coordinate in runner.expected_coordinates()
        for arm in runner.ARM_ORDER
    ]
    results = [
        _validate_result(
            result,
            expected_key=key,
            expected_coordinate=coordinate,
            expected_arm=arm,
        )
        for result, (key, coordinate, arm) in zip(
            raw["results"], expected, strict=True
        )
    ]
    if [result["worker_key"] for result in results] != _expected_result_keys():
        raise RuntimeError("calibration raw result order changed")
    if verify_source:
        for result, (_key, coordinate, arm) in zip(
            results,
            expected,
            strict=True,
        ):
            runner.validate_worker_result(
                result,
                source_freeze,
                coordinate,
                arm,
            )
    by_coordinate: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        key = runner.coordinate_key(**result["coordinate"])
        by_coordinate.setdefault(key, []).append(result)
    for coordinate_results in by_coordinate.values():
        if [row["arm"] for row in coordinate_results] != list(runner.ARM_ORDER):
            raise RuntimeError("calibration coordinate arm grid changed")
        first = coordinate_results[0]
        for row in coordinate_results[1:]:
            if (
                row["task"] != first["task"]
                or row["split"] != first["split"]
                or row["test_target_sha256"] != first["test_target_sha256"]
            ):
                raise RuntimeError("calibration arm data boundary changed")
        control = first
        t5 = coordinate_results[1]
        if not t5["metadata"]["engaged"] and (
            t5["prediction_sha256"] != control["prediction_sha256"]
            or t5["rmse"] != control["rmse"]
        ):
            raise RuntimeError("calibration T5 decline is not exact")
    spool = raw["spool"]
    if (
        not isinstance(spool, dict)
        or set(spool)
        != {
            "directory",
            "binding",
            "record_count",
            "resumed_record_count",
            "records",
        }
        or spool["record_count"] != 117
        or spool["directory"]
        != str(runner.DEFAULT_SPOOL_DIRECTORY.relative_to(ROOT))
        or type(spool["resumed_record_count"]) is not int
        or not 0 <= spool["resumed_record_count"] <= 117
        or not isinstance(spool["records"], list)
        or len(spool["records"]) != 117
    ):
        raise RuntimeError("calibration spool ledger changed")
    if verify_source and spool["binding"] != runner.spool_binding(
        source_freeze,
        freeze_path,
        raw["source_freeze_file_sha256"],
    ):
        raise RuntimeError("calibration spool source binding changed")
    if verify_spool:
        for result, record, (key, coordinate, arm) in zip(
            results, spool["records"], expected, strict=True
        ):
            if (
                not isinstance(record, dict)
                or set(record)
                != {
                    "worker_key",
                    "path",
                    "file_sha256",
                    "spool_record_sha256",
                    "result_sha256",
                    "resumed",
                }
                or record["worker_key"] != key
                or record["result_sha256"] != _json_sha256(result)
                or type(record["resumed"]) is not bool
            ):
                raise RuntimeError("calibration spool ledger record changed")
            path = ROOT / record["path"]
            expected_path = runner.spool_path(
                runner.DEFAULT_SPOOL_DIRECTORY,
                coordinate,
                arm,
            )
            if (
                path.expanduser().absolute()
                != expected_path.expanduser().absolute()
            ):
                raise RuntimeError(f"calibration spool file changed: {key}")
            reopened, digest, file_sha256 = runner.load_spool(
                path,
                spool["binding"],
                coordinate,
                arm,
            )
            if (
                reopened != result
                or digest != record["spool_record_sha256"]
                or file_sha256 != record["file_sha256"]
            ):
                raise RuntimeError(f"calibration spool result changed: {key}")
    return results


def analyze(
    raw: dict[str, Any],
    *,
    raw_path: Path = DEFAULT_RAW,
    raw_file_sha256: str | None = None,
    verify_source: bool = True,
    verify_spool: bool = True,
) -> dict[str, Any]:
    if raw_file_sha256 is not None and not _is_sha256(raw_file_sha256):
        raise RuntimeError("calibration raw-file hash is invalid")
    results = validate_raw(
        raw,
        raw_path=raw_path,
        verify_source=verify_source,
        verify_spool=verify_spool,
    )
    by_key = {result["worker_key"]: result for result in results}
    candidate_results: dict[str, Any] = {}
    fixed_panel_inputs: dict[str, list[dict[str, Any]]] = {}
    for candidate in runner.CANDIDATE_ARMS:
        coordinate_rows = []
        dataset_rows = []
        for dataset_name, task_id in freeze.TASKS.items():
            split_rows = []
            for coordinate_part in freeze.COORDINATES:
                coordinate = {"task_id": task_id, **coordinate_part}
                control = by_key[runner.worker_key(coordinate, runner.CONTROL_ARM)]
                result = by_key[runner.worker_key(coordinate, candidate)]
                ratio = float(result["rmse"] / control["rmse"])
                if not math.isfinite(ratio) or ratio <= 0.0:
                    raise RuntimeError("calibration candidate ratio is invalid")
                metadata = result["metadata"]
                row = {
                    "dataset_name": dataset_name,
                    "task_id": task_id,
                    "repeat": coordinate_part["repeat"],
                    "fold": coordinate_part["fold"],
                    "sample": coordinate_part["sample"],
                    "train_rows": result["split"]["train_rows"],
                    "t5_size_gate_applicable": result[
                        "t5_size_gate_applicable"
                    ],
                    "engaged": bool(metadata["engaged"]),
                    "decline_reason": metadata.get("decline_reason"),
                    "candidate_rmse": result["rmse"],
                    "current_default_rmse": control["rmse"],
                    "ratio": ratio,
                    "log_ratio": float(math.log(ratio)),
                    "direction": _direction(ratio),
                    "validation_ratio": (
                        metadata.get("relative_challenger_validation_ratio")
                        if candidate == "t5_composite_policy"
                        else metadata.get(
                            "relative_crossed_validation_ratio"
                        )
                    ),
                }
                split_rows.append(row)
                coordinate_rows.append(row)
            dataset_ratio = _geomean([row["ratio"] for row in split_rows])
            dataset_rows.append(
                {
                    "dataset_name": dataset_name,
                    "task_id": task_id,
                    "three_coordinate_geomean_ratio": dataset_ratio,
                    "log_ratio": float(math.log(dataset_ratio)),
                    "direction": _direction(dataset_ratio),
                    "coordinate_ratios": [
                        row["ratio"] for row in split_rows
                    ],
                    "t5_size_gate_applicable_coordinates": sum(
                        bool(row["t5_size_gate_applicable"])
                        for row in split_rows
                    ),
                    "engaged_coordinates": sum(
                        bool(row["engaged"]) for row in split_rows
                    ),
                }
            )
        coordinate_directions = Counter(
            row["direction"] for row in coordinate_rows
        )
        dataset_directions = Counter(
            row["direction"] for row in dataset_rows
        )
        equal_dataset_ratio = _geomean(
            [row["three_coordinate_geomean_ratio"] for row in dataset_rows]
        )
        candidate_results[candidate] = {
            "estimand": f"{candidate}/current_default",
            "coordinate_count": 39,
            "dataset_count": 13,
            "equal_dataset_geomean_ratio": equal_dataset_ratio,
            "worst_dataset_ratio": max(
                row["three_coordinate_geomean_ratio"]
                for row in dataset_rows
            ),
            "coordinate_wins_losses_ties": {
                name: coordinate_directions[name]
                for name in ("win", "loss", "tie")
            },
            "dataset_wins_losses_ties": {
                name: dataset_directions[name]
                for name in ("win", "loss", "tie")
            },
            "t5_size_gate_applicable_coordinates": sum(
                bool(row["t5_size_gate_applicable"])
                for row in coordinate_rows
            ),
            "engaged_coordinates": sum(
                bool(row["engaged"]) for row in coordinate_rows
            ),
            "declined_coordinates": sum(
                not bool(row["engaged"]) for row in coordinate_rows
            ),
            "coordinates": coordinate_rows,
            "datasets": dataset_rows,
        }
        fixed_panel_inputs[candidate] = [
            {
                "source": (
                    "spent_tabarena_13x3_exact_policy_complete_census"
                ),
                "dataset_name": row["dataset_name"],
                "task_id": row["task_id"],
                "ratio": row["three_coordinate_geomean_ratio"],
                "coordinate_ratios": row["coordinate_ratios"],
                "t5_size_gate_applicable_coordinates": row[
                    "t5_size_gate_applicable_coordinates"
                ],
                "engaged_coordinates": row["engaged_coordinates"],
            }
            for row in dataset_rows
        ]
    return common.bind_artifact_sha256(
        {
            "schema_version": 1,
            "name": "darkofit_panel3_cross_power_calibration_summary_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_path": RAW_RELATIVE,
            "raw_file_sha256": raw_file_sha256,
            "raw_artifact_sha256": raw["raw_artifact_sha256"],
            "source_freeze_sha256": raw["source_freeze_sha256"],
            "estimand": "exact_candidate/current_default",
            "candidate_results": candidate_results,
            "fixed_panel_power_inputs": fixed_panel_inputs,
            "complete_unfiltered_coordinate_census": True,
            "ties_and_losses_preserved": True,
            "development_only": True,
            "may_inform_separately_frozen_power_design": True,
            "independent_confirmation": False,
            "panel3_authorized": False,
            "default_promotion_authorized": False,
            "product_claim_authorized": False,
        },
        "summary_sha256",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.raw.expanduser().absolute() != DEFAULT_RAW
        or args.output.expanduser().absolute() != DEFAULT_OUTPUT
    ):
        raise RuntimeError("calibration analysis path changed")
    common.validate_create_path(args.output)
    raw, raw_file_sha256 = common.secure_load_json(args.raw)
    if not isinstance(raw, dict):
        raise RuntimeError("calibration raw artifact is not an object")
    summary = analyze(
        raw,
        raw_path=args.raw,
        raw_file_sha256=raw_file_sha256,
    )
    encoded = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    final_raw, final_raw_file_sha256 = common.secure_load_json(args.raw)
    if (
        final_raw != raw
        or final_raw_file_sha256 != raw_file_sha256
    ):
        raise RuntimeError(
            "calibration raw artifact changed before summary publish"
        )
    validate_raw(
        final_raw,
        raw_path=args.raw,
        verify_source=True,
        verify_spool=True,
    )
    common.atomic_create(args.output, encoded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary_sha256": summary["summary_sha256"],
                "panel3_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
