#!/usr/bin/env python3
"""Preflight and execute the P1-v3 automatic-depth fresh Tier-D one-shot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_t7b_automatic_depth_fresh_tier_d as helpers


CONTRACT_ID = "t7b-automatic-depth-fresh-tier-d-v3-execution-v1-20260723"
CONTROL_HEAD = "e23d2b164f10374b1c0e02521c33fc96d48980da"
CANDIDATE_HEAD = "41e948f0c53b1d124e16071a7fa66eba47d084d3"
CONTRACT = (
    ROOT / "benchmarks" / "t7b_automatic_depth_fresh_tier_d_v3_execution_contract.json"
)
ENUMERATION = (
    ROOT
    / "benchmarks"
    / "t7b_automatic_depth_fresh_tier_d_v3_enumeration_v2_20260723.json"
)
POWER_CONTRACT = (
    ROOT
    / "benchmarks"
    / "t7b_automatic_depth_fresh_tier_d_v3_power_design_contract.json"
)
POWER_RESULT = (
    ROOT
    / "benchmarks"
    / "t7b_automatic_depth_fresh_tier_d_v3_power_design_result_20260723.json"
)
PROTOCOL = (
    ROOT / "benchmarks" / "t7b_automatic_depth_fresh_tier_d_v3_execution_protocol.md"
)
ANALYZER = ROOT / "benchmarks" / "analyze_t7b_automatic_depth_fresh_tier_d_v3.py"
HELPER_RUNNER = ROOT / "benchmarks" / "run_t7b_automatic_depth_fresh_tier_d.py"
HELPER_ANALYZER = ROOT / "benchmarks" / "analyze_t7b_automatic_depth_fresh_tier_d.py"
THREADS = 14


canonical_json_bytes = helpers.canonical_json_bytes
file_sha256 = helpers.file_sha256
source_state = helpers.source_state
_load_json = helpers._load_json
_write_create_only = helpers._write_create_only
_git = helpers._git


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != 1:
        raise RuntimeError("unsupported execution contract")
    if contract.get("contract_id") != CONTRACT_ID:
        raise RuntimeError("execution contract identity changed")
    candidate = contract["candidate"]
    if (
        candidate["source_commit"] != CANDIDATE_HEAD
        or candidate["control_commit"] != CONTROL_HEAD
        or candidate["candidate_must_remain_byte_identical"] is not True
    ):
        raise RuntimeError("candidate/control contract changed")
    bindings = {
        "protocol": PROTOCOL,
        "enumeration": ENUMERATION,
        "power_contract": POWER_CONTRACT,
        "power_result": POWER_RESULT,
        "runner": Path(__file__),
        "analyzer": ANALYZER,
        "helper_runner": HELPER_RUNNER,
        "helper_analyzer": HELPER_ANALYZER,
    }
    for name, path in bindings.items():
        if contract["source_hashes"][name] != file_sha256(path):
            raise RuntimeError(f"execution source hash changed: {name}")
    enumeration = _load_json(ENUMERATION)
    eligible = [row for row in enumeration["identities"] if row["status"] == "eligible"]
    expected_ids = contract["verified_registry"]["eligible_lineage_ids"]
    if [row["lineage_id"] for row in eligible] != expected_ids:
        raise RuntimeError("eligible registry changed")
    if len(eligible) != 32 or any(not row["resource_loaded"] for row in eligible):
        raise RuntimeError("execution registry is not fully fillable")
    power = _load_json(POWER_RESULT)
    if (
        power["disposition"] != "design_power_qualified"
        or power["power_qualified"] is not True
        or power["primary_scenario"]["pass_probability"] != 0.998
        or power["primary_scenario"]["wilson_lower_bound"] != 0.99665735839545
    ):
        raise RuntimeError("bound power result changed")
    if contract["quality_gates"] != _load_json(POWER_CONTRACT)["quality_gates"]:
        raise RuntimeError("quality gates differ from power contract")
    execution = contract["execution"]
    expected_execution = {
        "logical_cpu_count": THREADS,
        "threads_per_worker": THREADS,
        "iterations": 600,
        "early_stopping_rounds": 30,
        "validation_fraction": 0.15,
        "use_best_model": True,
        "refit": False,
        "random_state": 20260723,
        "depth_input": None,
        "prediction_rows_per_repeat": 50_000,
        "prediction_repeats": 3,
        "same_source_warmup_iterations": 2,
        "high_density_cap_rows_per_input_feature": 3_250,
        "coordinate_folds": [0, 1, 2],
        "ordinary_folds": [0, 2],
        "nonuniform_weight_fold": 1,
        "nonuniform_weight_values": [1.0, 1.25],
    }
    for key, value in expected_execution.items():
        if execution.get(key) != value:
            raise RuntimeError(f"execution contract field changed: {key}")
    if contract["authorization"] != {
        "freeze_package_authorized_by_r2": True,
        "execution_preflight_authorized": True,
        "fresh_access_authorized": False,
        "confirmation_run_authorized": False,
        "candidate_modification": False,
        "gate_relaxation": False,
        "second_attempt": False,
        "partial_read": False,
        "tabarena": False,
        "ctr23": False,
        "lockbox": False,
        "v0_12_release_publication": False,
    }:
        raise RuntimeError("execution authorization boundary changed")


def build_preflight() -> dict[str, Any]:
    contract = _load_json(CONTRACT)
    validate_contract(contract)
    enumeration = _load_json(ENUMERATION)
    active = [row for row in enumeration["identities"] if row["status"] == "eligible"]
    if [row["lineage_id"] for row in active] != contract["verified_registry"][
        "eligible_lineage_ids"
    ]:
        raise RuntimeError("preflight lineage order changed")
    if len(active) != 32 or any(len(row["coordinates"]) != 3 for row in active):
        raise RuntimeError("preflight lineage/coordinate census changed")
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "preflight_passed",
        "execution_contract_sha256": file_sha256(CONTRACT),
        "enumeration_sha256": file_sha256(ENUMERATION),
        "power_result_sha256": file_sha256(POWER_RESULT),
        "active_lineage_count": 32,
        "active_stratum_counts": enumeration["eligible_stratum_counts"],
        "active_branch_counts": enumeration["eligible_branch_counts"],
        "active_group_safe_count": enumeration["eligible_group_safe_count"],
        "active_lineages": active,
        "attestations": {
            "all_resources_loaded_before_design_freeze": True,
            "no_execution_time_resource_discovery": True,
            "no_model_fit": True,
            "no_new_data_access": True,
            "no_target_statistics_computed": True,
            "no_quality_outcomes_inspected": True,
            "exact_and_near_lineage_fingerprints_bound": True,
            "all_realized_branches_bound": True,
            "all_group_splits_disjoint": True,
            "fresh_inspection_spent": False,
            "lockbox_data_used": False,
        },
    }


def validate_owner_authorization(
    authorization: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": 1,
        "authorization_id": (
            "t7b-automatic-depth-fresh-tier-d-v3-owner-run-authorization-v1"
        ),
        "contract_id": CONTRACT_ID,
        "execution_contract_sha256": file_sha256(CONTRACT),
        "enumeration_sha256": file_sha256(ENUMERATION),
        "power_result_sha256": file_sha256(POWER_RESULT),
        "confirmation_run_authorized": True,
        "candidate_modification_authorized": False,
        "panel_change_authorized": False,
        "gate_change_authorized": False,
        "rerun_authorized": False,
        "partial_read_authorized": False,
        "tabarena_authorized": False,
        "ctr23_authorized": False,
        "lockbox_authorized": False,
        "release_publication_authorized": False,
    }
    if dict(authorization) != expected:
        raise RuntimeError("owner run authorization is absent or changed")
    if contract["authorization"]["confirmation_run_authorized"] is not False:
        raise RuntimeError("frozen contract must remain non-executable alone")


def _worker_command(
    lineage_path: Path,
    coordinate: int,
    arm: str,
    source: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--lineage",
        str(lineage_path),
        "--coordinate",
        str(coordinate),
        "--arm",
        arm,
        "--source",
        str(source),
    ]


def execute(
    *,
    preflight_path: Path,
    owner_authorization_path: Path,
    control: Path,
    candidate: Path,
    prefix: Path,
) -> dict[str, Any]:
    contract = _load_json(CONTRACT)
    validate_contract(contract)
    authorization = _load_json(owner_authorization_path)
    validate_owner_authorization(authorization, contract)
    preflight = _load_json(preflight_path)
    if (
        preflight.get("status") != "preflight_passed"
        or preflight.get("contract_id") != CONTRACT_ID
        or preflight.get("execution_contract_sha256") != file_sha256(CONTRACT)
        or preflight.get("enumeration_sha256") != file_sha256(ENUMERATION)
    ):
        raise RuntimeError("execution preflight changed")
    paths = helpers.output_paths(prefix)
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise RuntimeError(f"one-shot output collision: {collisions}")
    harness = source_state(ROOT)
    control_state = source_state(control)
    candidate_state = source_state(candidate)
    if not all(state["clean"] for state in (harness, control_state, candidate_state)):
        raise RuntimeError("one-shot requires clean source trees")
    if control_state["head"] != CONTROL_HEAD:
        raise RuntimeError("control source pin changed")
    if candidate_state["head"] != CANDIDATE_HEAD:
        raise RuntimeError("candidate source pin changed")
    audit = helpers._exclusive_machine_audit()
    environment = helpers._environment()
    if environment["logical_cpu_count"] != THREADS:
        raise RuntimeError(f"execution requires exactly {THREADS} logical CPUs")
    published = sorted(
        line.strip()
        for line in _git(
            ROOT, "branch", "-r", "--contains", harness["head"]
        ).splitlines()
        if line.strip()
    )
    if not published:
        raise RuntimeError("execution harness commit is not published")

    launch = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sole_inspection_spent": True,
        "sources": {
            "harness": harness,
            "control": control_state,
            "candidate": candidate_state,
            "published_harness_refs": published,
        },
        "source_hashes": {
            "contract": file_sha256(CONTRACT),
            "enumeration": file_sha256(ENUMERATION),
            "power_result": file_sha256(POWER_RESULT),
            "preflight": file_sha256(preflight_path),
            "owner_authorization": file_sha256(owner_authorization_path),
            "runner": file_sha256(Path(__file__)),
            "analyzer": file_sha256(ANALYZER),
            "helper_runner": file_sha256(HELPER_RUNNER),
            "helper_analyzer": file_sha256(HELPER_ANALYZER),
        },
        "environment": environment,
        "exclusive_machine_audit": audit,
        "output_paths": {key: str(value) for key, value in paths.items()},
        "planned_arm_rows": 192,
        "active_lineages": [
            {
                "slot": row["slot"],
                "lineage_id": row["lineage_id"],
                "task_id": row["task_id"],
                "dataset_id": row["dataset_id"],
            }
            for row in preflight["active_lineages"]
        ],
        "no_rerun": True,
        "partial_reads_forbidden": True,
    }
    _write_create_only(paths["launch"], launch)

    rows: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="t7b-fresh-v3-one-shot-") as temp:
            temp_path = Path(temp)
            caches = {
                "control": temp_path / "numba-control",
                "candidate": temp_path / "numba-candidate",
            }
            for cache in caches.values():
                cache.mkdir()
            for lineage_index, lineage in enumerate(preflight["active_lineages"]):
                lineage_path = temp_path / f"lineage-{lineage_index:02d}.json"
                lineage_path.write_bytes(canonical_json_bytes(lineage))
                for coordinate in (0, 1, 2):
                    arms = (
                        ("control", "candidate")
                        if (lineage_index + coordinate) % 2 == 0
                        else ("candidate", "control")
                    )
                    for arm in arms:
                        source = control if arm == "control" else candidate
                        completed = subprocess.run(
                            _worker_command(lineage_path, coordinate, arm, source),
                            cwd=ROOT,
                            env=helpers._worker_env(source, caches[arm]),
                            check=False,
                            capture_output=True,
                            text=True,
                        )
                        if completed.returncode:
                            raise RuntimeError(
                                f"worker failed for {lineage['lineage_id']}/"
                                f"{coordinate}/{arm}: {completed.stderr[-4000:]}"
                            )
                        output_lines = [
                            line
                            for line in completed.stdout.splitlines()
                            if line.strip()
                        ]
                        if not output_lines:
                            raise RuntimeError("worker returned no JSON row")
                        row = json.loads(output_lines[-1])
                        if row.get("status") != "ok":
                            raise RuntimeError(f"worker integrity failed: {row}")
                        rows.append(row)
        if len(rows) != 192:
            raise RuntimeError("one-shot row census changed")
        raw = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "complete": True,
            "launch_manifest_sha256": file_sha256(paths["launch"]),
            "preflight_sha256": file_sha256(preflight_path),
            "owner_authorization_sha256": file_sha256(owner_authorization_path),
            "environment": environment,
            "rows": rows,
        }
        _write_create_only(paths["raw"], raw)
        completed = subprocess.run(
            [
                sys.executable,
                str(ANALYZER),
                "--raw",
                str(paths["raw"]),
                "--output",
                str(paths["result"]),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError("terminal analyzer failed: " + completed.stderr[-4000:])
        result = _load_json(paths["result"])
        terminal = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "terminal_complete",
            "disposition": result["disposition"],
            "go": result["go"],
            "artifact_hashes": {
                "launch": file_sha256(paths["launch"]),
                "raw": file_sha256(paths["raw"]),
                "result": file_sha256(paths["result"]),
            },
            "rerun_authorized": False,
        }
        _write_create_only(paths["terminal"], terminal)
        return result
    except BaseException as exc:
        terminal = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "terminal_failed_after_launch",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "completed_rows_unpublished_and_unread": len(rows),
            "rerun_authorized": False,
            "launch_sha256": file_sha256(paths["launch"]),
        }
        if not paths["terminal"].exists():
            _write_create_only(paths["terminal"], terminal)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--lineage", type=Path, required=True)
    worker.add_argument("--coordinate", type=int, choices=(0, 1, 2), required=True)
    worker.add_argument("--arm", choices=("control", "candidate"), required=True)
    worker.add_argument("--source", type=Path, required=True)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--preflight", type=Path, required=True)
    execute_parser.add_argument("--owner-authorization", type=Path, required=True)
    execute_parser.add_argument("--control", type=Path, required=True)
    execute_parser.add_argument("--candidate", type=Path, required=True)
    execute_parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    helpers.CONTRACT_ID = CONTRACT_ID
    if args.command == "preflight":
        if args.output.exists() or args.output.is_symlink():
            raise RuntimeError(f"refusing existing output: {args.output}")
        artifact = build_preflight()
        _write_create_only(args.output, artifact)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": artifact["status"],
                    "active_lineages": artifact["active_lineage_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "worker":
        lineage = _load_json(args.lineage)
        row = helpers.run_worker(
            lineage,
            coordinate=args.coordinate,
            arm=args.arm,
            source=args.source,
        )
        print(json.dumps(row, sort_keys=True, allow_nan=False))
        return 0
    result = execute(
        preflight_path=args.preflight,
        owner_authorization_path=args.owner_authorization,
        control=args.control,
        candidate=args.candidate,
        prefix=args.output_prefix,
    )
    print(
        json.dumps(
            {"disposition": result["disposition"], "go": result["go"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
