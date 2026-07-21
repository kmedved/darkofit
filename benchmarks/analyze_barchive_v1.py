#!/usr/bin/env python3
"""Validate and decide the frozen Wave-3 B-archive feasibility screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from . import paired_evidence_contract as paired
    from . import run_barchive_v1 as runner
except ImportError:  # direct script execution
    import paired_evidence_contract as paired
    import run_barchive_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "benchmarks" / "barchive_v1_result.json"
NOTE_PATH = ROOT / "benchmarks" / "barchive_v1_result.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"B-archive {label} must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"B-archive {label} must be a JSON object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"B-archive {label} must be a positive integer")
    return value


def _validate_runtime(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"B-archive {label} runtime is invalid")
    environment = value.get("environment")
    expected = paired._expected_environment(runner.THREADS)
    if (
        value.get("ceiling") != runner.THREADS
        or value.get("current") != runner.THREADS
        or not isinstance(value.get("threading_layer"), str)
        or not value["threading_layer"]
        or not isinstance(environment, Mapping)
        or not environment.get("NUMBA_CACHE_DIR")
        or any(
            environment.get(name) != expected_value
            for name, expected_value in expected.items()
            if name != "NUMBA_CACHE_DIR"
        )
    ):
        raise RuntimeError(f"B-archive {label} runtime drifted")


def _validate_archive_record(value: Any, *, task: str, label: str) -> dict[str, Any]:
    expected_keys = {
        "archive_bytes",
        "archive_sha256",
        "prediction_sha256",
        "probability_sha256",
        "feature_schema_sha256",
        "metadata_sha256",
        "checks",
    }
    expected_checks = {
        "prediction_exact": True,
        "probability_exact": True,
        "feature_schema_exact": True,
        "metadata_exact": True,
        "resave_bytes_exact": True,
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("checks") != expected_checks
        or not _is_hex(value.get("archive_sha256"))
        or not _is_hex(value.get("prediction_sha256"))
        or not _is_hex(value.get("feature_schema_sha256"))
        or not _is_hex(value.get("metadata_sha256"))
        or (task == "regression" and value.get("probability_sha256") is not None)
        or (task != "regression" and not _is_hex(value.get("probability_sha256")))
    ):
        raise RuntimeError(f"B-archive {label} safe-roundtrip record is invalid")
    _positive_int(value.get("archive_bytes"), f"{label} archive bytes")
    return value


def _validate_component(value: Any, *, case_id: str, name: str) -> None:
    expected_keys = {
        "name",
        "section",
        "present_in_all_members",
        "byte_identical_across_members",
        "member_fingerprints",
    }
    fingerprints = (
        value.get("member_fingerprints") if isinstance(value, Mapping) else None
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("name") != name
        or value.get("section") != "preprocessor"
        or value.get("present_in_all_members") is not True
        or value.get("byte_identical_across_members") is not True
        or not isinstance(fingerprints, list)
        or len(fingerprints) != runner.m3b.MEMBERS
    ):
        raise RuntimeError(
            f"B-archive canonical component proof is invalid: {case_id}/{name}"
        )
    hashes = set()
    for fingerprint in fingerprints:
        if (
            not isinstance(fingerprint, Mapping)
            or set(fingerprint)
            != {
                "sha256",
                "npy_bytes",
                "standalone_npz_bytes",
                "dtype",
                "shape",
            }
            or not _is_hex(fingerprint.get("sha256"))
            or not isinstance(fingerprint.get("dtype"), str)
            or not fingerprint["dtype"]
            or not isinstance(fingerprint.get("shape"), list)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in fingerprint["shape"]
            )
        ):
            raise RuntimeError(
                f"B-archive canonical fingerprint is invalid: {case_id}/{name}"
            )
        _positive_int(fingerprint.get("npy_bytes"), f"{case_id}/{name} NPY bytes")
        _positive_int(
            fingerprint.get("standalone_npz_bytes"),
            f"{case_id}/{name} standalone NPZ bytes",
        )
        hashes.add(fingerprint["sha256"])
    if len(hashes) != 1:
        raise RuntimeError(
            f"B-archive canonical component is not byte-identical: {case_id}/{name}"
        )


def _git_payload(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(f"B-archive Git provenance is missing {commit}:{relative}")
    return completed.stdout


def _assert_harness_provenance(
    raw: Mapping[str, Any], contract: Mapping[str, Any], contract_path: Path
) -> None:
    source_state = raw.get("source_state")
    harness_state = raw.get("harness_state")
    if (
        not isinstance(source_state, Mapping)
        or source_state.get("head") != runner.MODEL_SOURCE_HEAD
        or source_state.get("status") != ""
        or not isinstance(source_state.get("path"), str)
        or not isinstance(harness_state, Mapping)
        or not _is_hex(harness_state.get("head"), 40)
        or harness_state.get("status") != ""
        or not isinstance(harness_state.get("path"), str)
    ):
        raise RuntimeError("B-archive source or harness provenance is invalid")
    completed = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            contract["sources"]["harness"],
            harness_state["head"],
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError("B-archive execution harness is outside frozen lineage")
    for name, record in contract["bound_files"].items():
        commit = (
            runner.MODEL_SOURCE_HEAD
            if record["path"] in runner.PINNED_SOURCE_PATHS
            else harness_state["head"]
        )
        payload = _git_payload(commit, record["path"])
        if (
            len(payload) != record["bytes"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise RuntimeError(f"B-archive bound file drifted in execution: {name}")
    contract_payload = _git_payload(
        harness_state["head"], str(contract_path.relative_to(ROOT))
    )
    if hashlib.sha256(contract_payload).hexdigest() != _sha256(contract_path):
        raise RuntimeError("B-archive execution did not use its committed contract")


def validate_raw(
    raw_path: Path = runner.RAW_PATH,
    contract_path: Path = runner.CONTRACT_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_path = raw_path.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    contract = runner.load_contract(contract_path)
    raw = _load_object(raw_path, "raw artifact")
    if runner.TERMINAL_PATH.exists() or runner.TERMINAL_PATH.is_symlink():
        raise RuntimeError("B-archive terminal artifact conflicts with raw evidence")
    expected_top_keys = {
        "schema_version",
        "name",
        "status",
        "created_at",
        "contract_path",
        "contract_sha256",
        "source_state",
        "harness_state",
        "runtime_versions",
        "panel_cache",
        "case_fingerprints",
        "rows",
    }
    if (
        set(raw) != expected_top_keys
        or raw.get("schema_version") != runner.SCHEMA_VERSION
        or raw.get("name") != runner.CONTRACT_NAME
        or raw.get("status") != "complete"
        or not isinstance(raw.get("created_at"), str)
        or raw.get("contract_path") != str(contract_path.relative_to(ROOT))
        or raw.get("contract_sha256") != _sha256(contract_path)
        or raw.get("runtime_versions") != runner.FROZEN_RUNTIME
        or raw.get("case_fingerprints") != contract["case_fingerprints"]
        or raw.get("panel_cache") != contract["panel_cache"]
        or not isinstance(raw.get("rows"), list)
        or len(raw["rows"]) != runner.decision_rules()["expected_case_count"]
    ):
        raise RuntimeError("B-archive raw artifact identity is invalid")
    try:
        created_at = datetime.fromisoformat(raw["created_at"])
    except ValueError as exc:
        raise RuntimeError("B-archive raw creation timestamp is invalid") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise RuntimeError("B-archive raw creation timestamp lacks a timezone")
    _assert_harness_provenance(raw, contract, contract_path)

    specs = contract["cases"]
    expected_provenance = contract["expected_shared_preprocessing"]
    expected_ids = [spec["case_id"] for spec in specs]
    observed_ids = [
        row.get("case_id") if isinstance(row, Mapping) else None for row in raw["rows"]
    ]
    if observed_ids != expected_ids or len(set(observed_ids)) != len(expected_ids):
        raise RuntimeError("B-archive raw rows are missing, duplicated, or reordered")

    numeric_count = 0
    member_local_count = 0
    source_root = Path(raw["source_state"]["path"]).resolve()
    fingerprint_names = (
        "case_sha256",
        "dataset_sha256",
        "split_sha256",
        "weight_sha256",
    )
    for spec, row in zip(specs, raw["rows"]):
        case_id = spec["case_id"]
        if not isinstance(row, dict):
            raise RuntimeError(f"B-archive row is invalid: {case_id}")
        fingerprints = {name: row.get(name) for name in fingerprint_names}
        if (
            row.get("domain") != spec["domain"]
            or row.get("task") != spec["task"]
            or fingerprints != contract["case_fingerprints"][case_id]
            or row.get("shared_preprocessing") != expected_provenance[case_id]
        ):
            raise RuntimeError(f"B-archive row identity drifted: {case_id}")
        implementation = Path(str(row.get("implementation_path", ""))).resolve()
        try:
            relative_implementation = implementation.relative_to(source_root)
        except ValueError as exc:
            raise RuntimeError(
                f"B-archive implementation escaped pinned source: {case_id}"
            ) from exc
        if relative_implementation != Path("darkofit/sklearn_api.py"):
            raise RuntimeError(f"B-archive implementation path drifted: {case_id}")

        single = _validate_archive_record(
            row.get("single"), task=spec["task"], label=f"{case_id}/single"
        )
        combined = _validate_archive_record(
            row.get("combined"), task=spec["task"], label=f"{case_id}/combined"
        )
        expected_current_ratio = combined["archive_bytes"] / single["archive_bytes"]
        if row.get("current_archive_to_single") != expected_current_ratio:
            raise RuntimeError(f"B-archive current size ratio drifted: {case_id}")

        canonical = row.get("canonical_preprocessor")
        optimistic = row.get("optimistic_all_exact_entries")
        components = row.get("components")
        if (
            not isinstance(canonical, Mapping)
            or set(canonical)
            != {
                "numeric_target_free_provenance",
                "array_schema_identical",
                "arrays_byte_identical",
                "headers_byte_identical",
                "eligible",
                "array_names",
                "header_fields",
                "simulated_archive_bytes",
            }
            or not isinstance(optimistic, Mapping)
            or set(optimistic)
            != {
                "array_names",
                "simulated_archive_bytes",
                "includes_out_of_scope_sections",
            }
            or not isinstance(optimistic.get("array_names"), list)
            or len(set(optimistic["array_names"])) != len(optimistic["array_names"])
            or not all(isinstance(name, str) for name in optimistic["array_names"])
            or optimistic.get("includes_out_of_scope_sections")
            is not bool(
                set(optimistic["array_names"]).difference(
                    canonical.get("array_names", ())
                )
            )
            or not isinstance(components, list)
        ):
            raise RuntimeError(f"B-archive component census is invalid: {case_id}")
        optimistic_bytes = _positive_int(
            optimistic.get("simulated_archive_bytes"),
            f"{case_id} optimistic bytes",
        )
        if optimistic_bytes > combined["archive_bytes"]:
            raise RuntimeError(
                f"B-archive optimistic simulation grew archive: {case_id}"
            )
        provenance = expected_provenance[case_id]
        eligible = provenance == "numeric_target_free"
        if eligible:
            numeric_count += 1
            required_flags = {
                "numeric_target_free_provenance": True,
                "array_schema_identical": True,
                "arrays_byte_identical": True,
                "headers_byte_identical": True,
                "eligible": True,
            }
            if (
                any(
                    canonical.get(name) is not value
                    for name, value in required_flags.items()
                )
                or canonical.get("array_names")
                != sorted(runner.ALLOWED_CANONICAL_ARRAYS)
                or canonical.get("header_fields")
                != sorted(runner.ALLOWED_CANONICAL_HEADER_FIELDS)
            ):
                raise RuntimeError(
                    f"B-archive canonical section is incomplete: {case_id}"
                )
            simulated = _positive_int(
                canonical.get("simulated_archive_bytes"),
                f"{case_id} canonical bytes",
            )
            if simulated > combined["archive_bytes"]:
                raise RuntimeError(
                    f"B-archive canonical simulation grew archive: {case_id}"
                )
            effective = simulated
        else:
            member_local_count += 1
            if (
                canonical.get("numeric_target_free_provenance") is not False
                or canonical.get("eligible") is not False
                or canonical.get("simulated_archive_bytes") is not None
            ):
                raise RuntimeError(
                    f"B-archive member-local case became canonical: {case_id}"
                )
            effective = combined["archive_bytes"]

        component_map = {
            item.get("name"): item
            for item in components
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
        if len(component_map) != len(components):
            raise RuntimeError(f"B-archive component census is malformed: {case_id}")
        if eligible:
            for name in runner.ALLOWED_CANONICAL_ARRAYS:
                if name not in component_map:
                    raise RuntimeError(
                        "B-archive canonical component proof is incomplete: "
                        f"{case_id}"
                    )
                _validate_component(component_map[name], case_id=case_id, name=name)
        expected_uses_canonical = eligible
        expected_effective_ratio = effective / single["archive_bytes"]
        if (
            row.get("effective_uses_only_canonical_preprocessor")
            is not expected_uses_canonical
            or row.get("effective_candidate_archive_bytes") != effective
            or row.get("effective_candidate_archive_to_single")
            != expected_effective_ratio
        ):
            raise RuntimeError(f"B-archive effective size drifted: {case_id}")
        _validate_runtime(row.get("runtime_before"), f"{case_id}/before")
        _validate_runtime(row.get("runtime_after"), f"{case_id}/after")

    rules = runner.decision_rules()
    if (
        numeric_count != rules["expected_numeric_target_free_cases"]
        or member_local_count != rules["expected_case_count"] - numeric_count
    ):
        raise RuntimeError("B-archive preprocessing case counts drifted")
    return raw, contract


def build_result(
    raw_path: Path = runner.RAW_PATH,
    contract_path: Path = runner.CONTRACT_PATH,
) -> dict[str, Any]:
    raw, contract = validate_raw(raw_path, contract_path)
    rows = raw["rows"]
    current_ratios = [float(row["current_archive_to_single"]) for row in rows]
    effective_ratios = [
        float(row["effective_candidate_archive_to_single"]) for row in rows
    ]
    if not all(math.isfinite(value) and value > 0.0 for value in effective_ratios):
        raise RuntimeError("B-archive effective ratios are not finite and positive")
    current_median = float(np.median(np.asarray(current_ratios)))
    effective_median = float(np.median(np.asarray(effective_ratios)))
    limit = float(
        contract["decision_rules"]["median_effective_archive_to_single_at_most"]
    )
    advances = effective_median <= limit
    disposition = (
        contract["decision_rules"]["advance_disposition"]
        if advances
        else contract["decision_rules"]["close_disposition"]
    )
    case_readout = [
        {
            "case_id": row["case_id"],
            "domain": row["domain"],
            "task": row["task"],
            "shared_preprocessing": row["shared_preprocessing"],
            "single_archive_bytes": row["single"]["archive_bytes"],
            "current_ensemble_archive_bytes": row["combined"]["archive_bytes"],
            "effective_candidate_archive_bytes": row[
                "effective_candidate_archive_bytes"
            ],
            "current_archive_to_single": row["current_archive_to_single"],
            "effective_candidate_archive_to_single": row[
                "effective_candidate_archive_to_single"
            ],
        }
        for row in rows
    ]
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "name": runner.CONTRACT_NAME,
        "analysis": "canonical_preprocessor_archive_size_feasibility",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(Path(contract_path).resolve().relative_to(ROOT)),
            "sha256": _sha256(Path(contract_path).resolve()),
        },
        "raw_artifact": {
            "path": str(Path(raw_path).resolve().relative_to(ROOT)),
            "bytes": Path(raw_path).resolve().stat().st_size,
            "sha256": _sha256(Path(raw_path).resolve()),
        },
        "m3b_r3_amended": False,
        "spent_development_evidence": True,
        "case_count": len(rows),
        "numeric_target_free_case_count": sum(
            row["shared_preprocessing"] == "numeric_target_free" for row in rows
        ),
        "member_local_case_count": sum(
            row["shared_preprocessing"] == "member_local" for row in rows
        ),
        "case_readout": case_readout,
        "gate": {
            "median_effective_archive_to_single_at_most": limit,
            "median_current_archive_to_single": current_median,
            "median_effective_archive_to_single": effective_median,
            "clears": advances,
        },
        "disposition": disposition,
        "canonical_serializer_prototype_authorized": advances,
        "serializer_retention_authorized": False,
        "fused_lane_dispatch_nominated_next": not advances,
        "claims": contract["claims"],
    }


def render_note(result: Mapping[str, Any]) -> str:
    gate = result["gate"]
    if result["canonical_serializer_prototype_authorized"]:
        finding = (
            "The complete canonical numeric-preprocessor size model clears the "
            "frozen feasibility gate. This authorizes only a separately pinned "
            "serializer prototype and behavior/resource verification."
        )
    else:
        finding = (
            "The complete canonical numeric-preprocessor size model does not "
            "clear the frozen feasibility gate. B-archive closes and "
            "behavior-exact fused-lane dispatch is nominated next."
        )
    return (
        "# Wave 3 B-archive v1 result\n\n"
        f"- Disposition: `{result['disposition']}`\n"
        f"- Cases: {result['case_count']} "
        f"({result['numeric_target_free_case_count']} numeric target-free, "
        f"{result['member_local_case_count']} member-local)\n"
        f"- Median current archive / single: "
        f"`{gate['median_current_archive_to_single']:.6f}`\n"
        f"- Median effective archive / single: "
        f"`{gate['median_effective_archive_to_single']:.6f}`\n"
        f"- Frozen limit: `<= {gate['median_effective_archive_to_single_at_most']:.1f}`\n\n"
        f"{finding}\n\n"
        "This is spent Tier-E size-feasibility evidence. It does not amend "
        "M3b r3, retain a serializer, change a public/default surface, or "
        "authorize B3, M2, TabArena/M4, fresh confirmation, or lockbox access.\n"
    )


def _publish_pair(result: Mapping[str, Any], output: Path, note: Path) -> None:
    result_payload = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    note_payload = render_note(result).encode("utf-8")
    output = output.expanduser().resolve()
    note = note.expanduser().resolve()
    if output != RESULT_PATH.resolve() or note != NOTE_PATH.resolve():
        raise RuntimeError("B-archive analysis requires canonical output paths")
    if output.exists() or output.is_symlink() or note.exists() or note.is_symlink():
        raise RuntimeError("B-archive result or note already exists")
    wrote_output = False
    try:
        paired.write_create_only(output, result_payload)
        wrote_output = True
        paired.write_create_only(note, note_payload)
    except BaseException:
        if (
            wrote_output
            and output.is_file()
            and _sha256(output) == hashlib.sha256(result_payload).hexdigest()
        ):
            output.unlink()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=runner.RAW_PATH)
    parser.add_argument("--contract", type=Path, default=runner.CONTRACT_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--note", type=Path, default=NOTE_PATH)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    state = runner.m3b.git_state(ROOT)
    if state["status"]:
        raise RuntimeError("B-archive analysis requires a clean harness tree")
    result = build_result(args.raw, args.contract)
    if runner.m3b.git_state(ROOT) != state:
        raise RuntimeError("B-archive harness changed during analysis")
    _publish_pair(result, args.output, args.note)
    print(
        json.dumps(
            {
                "result": str(RESULT_PATH),
                "sha256": _sha256(RESULT_PATH),
                "disposition": result["disposition"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
