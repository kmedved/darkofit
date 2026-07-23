#!/usr/bin/env python3
"""Launch the one permitted group-centered-cross M6 v3 inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import m6_quality_rule_v3 as m6_rule
except ImportError:  # pragma: no cover
    from benchmarks import m6_quality_rule_v3 as m6_rule


CONTRACT_ID = "group-centered-categorical-crosses-v1-development-20260722"
MECHANISM_ID = "group_centered_categorical_crosses_v1"
INSPECTION_INDEX = 1
CONTROL_HEAD = "01ae675bcebdf435988ce9e0d493d0fc0017f54a"
CANDIDATE_HEAD = "c3f2608cd3033cfc00aa0737897a92ed868b5865"
M5_CONTROL_HEAD = "726e5d8e6131c580bce948db833a5007d0692dca"
RUNNER_PATH = Path(__file__).resolve()
ROOT = RUNNER_PATH.parents[1]
CONTRACT_PATH = RUNNER_PATH.with_name(
    "group_centered_categorical_crosses_v1_development_contract.md"
)
INVARIANT_RUNNER_PATH = RUNNER_PATH.with_name(
    "run_group_centered_categorical_crosses_v1_invariants.py"
)
ENGAGEMENT_RUNNER_PATH = RUNNER_PATH.with_name(
    "run_group_centered_categorical_crosses_v1_engagement.py"
)
ENGAGEMENT_PROTOCOL_PATH = RUNNER_PATH.with_name(
    "group_centered_categorical_crosses_v1_m6_engagement_companion.md"
)
M5_RUNNER_PATH = RUNNER_PATH.with_name("run_m5_sentinels.py")
M6_RUNNER_PATH = RUNNER_PATH.with_name("run_m6_quality_successor_v3.py")
M6_RULE_PATH = RUNNER_PATH.with_name("m6_quality_rule_v3.py")
M6_CONTRACT_PATH = RUNNER_PATH.with_name("m6_quality_successor_v3_contract.md")
INVARIANT_IDENTITY = (
    "group-centered-categorical-crosses-v1-invariants-20260722"
)
ENGAGEMENT_IDENTITY = (
    "group-centered-categorical-crosses-v1-m6-engagement-20260722"
)
HARNESS_FILES = {
    "benchmarks/group_centered_categorical_crosses_v1_m6_engagement_companion.md",
    "benchmarks/run_group_centered_categorical_crosses_v1.py",
    "benchmarks/run_group_centered_categorical_crosses_v1_engagement.py",
    "benchmarks/run_group_centered_categorical_crosses_v1_invariants.py",
    "tests/test_group_centered_campaign_harness.py",
}
CANDIDATE_FILES = {
    "darkofit/booster.py",
    "darkofit/preprocessing.py",
    "darkofit/serialization.py",
    "darkofit/sklearn_api.py",
    "tests/test_darkofit.py",
    "tests/test_group_centered_preprocessing.py",
    "tests/test_group_centered_selector.py",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip())
    return process.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    if Path(_git(repository, "rev-parse", "--show-toplevel")).resolve() != repository:
        raise RuntimeError("campaign source must name its Git root")
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


def _changed_files(repository: Path, before: str, after: str) -> set[str]:
    return {
        line
        for line in _git(
            repository, "diff", "--name-only", f"{before}..{after}"
        ).splitlines()
        if line
    }


def validate_sources(control: Path, candidate: Path) -> dict[str, Any]:
    states = {
        "harness": source_state(ROOT),
        "control": source_state(control),
        "candidate": source_state(candidate),
    }
    if not all(state["clean"] for state in states.values()):
        raise RuntimeError("campaign execution requires clean sources")
    if states["control"]["head"] != CONTROL_HEAD:
        raise RuntimeError("campaign control commit is wrong")
    if states["candidate"]["head"] != CANDIDATE_HEAD:
        raise RuntimeError("campaign candidate commit is wrong")
    harness_parent = _git(ROOT, "rev-parse", f"{states['harness']['head']}^")
    if harness_parent != CONTROL_HEAD:
        raise RuntimeError("campaign harness must directly descend from control")
    if _changed_files(ROOT, CONTROL_HEAD, states["harness"]["head"]) != HARNESS_FILES:
        raise RuntimeError("campaign harness file allowlist drifted")
    if _changed_files(candidate, CONTROL_HEAD, CANDIDATE_HEAD) != CANDIDATE_FILES:
        raise RuntimeError("campaign candidate file allowlist drifted")
    if _git(ROOT, "rev-parse", "origin/codex/catcross-evidence-20260722") != states[
        "harness"
    ]["head"]:
        raise RuntimeError("campaign harness is not published")
    if _git(ROOT, "rev-parse", "origin/codex/catcross-v1-20260722") != CANDIDATE_HEAD:
        raise RuntimeError("campaign candidate is not published")
    return states


def _source_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(
        actual.get(key) == expected.get(key)
        for key in ("path", "head", "tree", "clean", "status")
    )


def _changed_m5_cells(m5: Mapping[str, Any]) -> set[tuple[str, int]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in m5.get("rows", []):
        key = (str(row.get("domain_id")), int(row.get("seed", -1)))
        grouped.setdefault(key, {})[str(row.get("arm"))] = row
    changed = set()
    for key, arms in grouped.items():
        if set(arms) != {"control", "candidate"}:
            raise RuntimeError("M5 paired rows are incomplete")
        if (
            arms["control"].get("behavior_fingerprint_sha256")
            != arms["candidate"].get("behavior_fingerprint_sha256")
        ):
            changed.add(key)
    return changed


def _validate_engagement_records(engagement: Mapping[str, Any]) -> bool:
    records = engagement.get("records")
    if not isinstance(records, list):
        return False
    expected = {
        (dataset, size, seed, weight_mode)
        for size in m6_rule.SIZES
        for dataset in m6_rule.DATASETS
        for seed in m6_rule.SEEDS
        for weight_mode in m6_rule.WEIGHT_MODES
    }
    observed = set()
    for record in records:
        if not isinstance(record, Mapping):
            return False
        identity = (
            record.get("dataset"),
            record.get("size"),
            record.get("seed"),
            record.get("weight_mode"),
        )
        if identity in observed:
            return False
        observed.add(identity)
        selector = record.get("selector")
        task = record.get("task")
        if (
            not isinstance(selector, Mapping)
            or any(
                not isinstance(record.get(name), str)
                or len(record[name]) != 64
                for name in (
                    "case_sha256",
                    "dataset_sha256",
                    "split_sha256",
                    "weight_sha256",
                )
            )
        ):
            return False
        if task == "regression":
            if (
                not isinstance(selector.get("eligible"), bool)
                or not isinstance(selector.get("selected"), bool)
                or not isinstance(selector.get("reason"), str)
                or not selector["reason"]
            ):
                return False
        elif selector != {
            "eligible": False,
            "selected": False,
            "reason": "classification_not_applicable",
        }:
            return False
    return observed == expected


def validate_preconditions(
    invariants_path: Path,
    m5_path: Path,
    engagement_path: Path,
    *,
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    paths = {
        "invariants": invariants_path.expanduser().resolve(),
        "m5": m5_path.expanduser().resolve(),
        "engagement": engagement_path.expanduser().resolve(),
    }
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError("all campaign precondition artifacts are required")
    invariants = json.loads(paths["invariants"].read_text())
    m5 = json.loads(paths["m5"].read_text())
    engagement = json.loads(paths["engagement"].read_text())
    floor_checks = m5.get("analysis", {}).get("known_floor_checks", {})
    m5_sources = m5.get("sources", {})
    expected_m5_changed = {
        ("categorical_missing_regression", 0),
        ("categorical_missing_regression", 1),
    }
    if (
        invariants.get("identity") != INVARIANT_IDENTITY
        or invariants.get("contract_id") != CONTRACT_ID
        or invariants.get("quality_outcomes_inspected") is not False
        or invariants.get("analysis", {}).get("focused_invariants_passed")
        is not True
        or invariants.get("bindings", {}).get("contract_sha256")
        != file_sha256(CONTRACT_PATH)
        or invariants.get("bindings", {}).get("runner_sha256")
        != file_sha256(INVARIANT_RUNNER_PATH)
        or not _source_matches(
            invariants.get("sources", {}).get("harness", {}),
            sources["harness"],
        )
        or not _source_matches(
            invariants.get("sources", {}).get("candidate", {}),
            sources["candidate"],
        )
        or m5.get("runner_version") != "m5-sentinels-v1"
        or m5.get("evidence_status") != "sentinel_check"
        or m5.get("non_ranking") is not True
        or m5.get("shipping_or_default_claim_authorized") is not False
        or m5.get("runner_sha256") != file_sha256(M5_RUNNER_PATH)
        or m5.get("analysis", {}).get("baseline_drift") != []
        or m5.get("analysis", {}).get("advancement_blocked_for_drift") is not False
        or not floor_checks
        or not all(check.get("passed") is True for check in floor_checks.values())
        or m5_sources.get("control", {}).get("head") != M5_CONTROL_HEAD
        or m5_sources.get("candidate", {}).get("head") != CANDIDATE_HEAD
        or m5_sources.get("candidate", {}).get("clean") is not True
        or m5_sources.get("harness", {}).get("head") != CANDIDATE_HEAD
        or m5_sources.get("harness", {}).get("clean") is not True
        or _changed_m5_cells(m5) != expected_m5_changed
        or engagement.get("identity") != ENGAGEMENT_IDENTITY
        or engagement.get("mechanism_id") != MECHANISM_ID
        or engagement.get("inspection_index") != INSPECTION_INDEX
        or engagement.get("companion_only") is not True
        or engagement.get("quality_metrics_recorded") is not False
        or not _validate_engagement_records(engagement)
        or not _source_matches(
            engagement.get("sources", {}).get("harness", {}),
            sources["harness"],
        )
        or not _source_matches(
            engagement.get("sources", {}).get("candidate", {}),
            sources["candidate"],
        )
        or engagement.get("bindings", {}).get("protocol_sha256")
        != file_sha256(ENGAGEMENT_PROTOCOL_PATH)
        or engagement.get("bindings", {}).get("selector_contract_sha256")
        != file_sha256(CONTRACT_PATH)
        or engagement.get("bindings", {}).get("runner_sha256")
        != file_sha256(ENGAGEMENT_RUNNER_PATH)
        or engagement.get("bindings", {}).get("m6_rule_sha256")
        != file_sha256(M6_RULE_PATH)
        or engagement.get("bindings", {}).get("comparison_runner_sha256")
        != file_sha256(RUNNER_PATH.with_name("bench_compare_revisions.py"))
        or engagement.get("bindings", {}).get("paired_evidence_sha256")
        != file_sha256(RUNNER_PATH.with_name("paired_evidence_contract.py"))
    ):
        raise RuntimeError("campaign precondition artifact is invalid")
    return {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }


def _exclusive_machine_audit() -> dict[str, Any]:
    import psutil

    own_chain = {os.getpid()}
    ancestor = psutil.Process().parent()
    while ancestor is not None:
        own_chain.add(ancestor.pid)
        try:
            ancestor = ancestor.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            break
    markers = (
        "run_m6_quality_successor",
        "run_group_centered_categorical_crosses",
        "run_v011_compute_ladder",
        "run_v011_m2_broad_panel",
        "run_tabarena",
    )
    conflicts = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid not in own_chain and any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"another benchmark process is active: {conflicts}")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "conflicting_benchmark_processes": [],
        "load_average": [float(value) for value in os.getloadavg()],
    }


def output_paths(prefix: Path) -> dict[str, Path]:
    prefix = prefix.expanduser().resolve()
    try:
        prefix.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("campaign outputs must be outside the harness")
    result = Path(str(prefix) + "_result.json")
    return {
        "launch_manifest": Path(str(prefix) + "_launch_manifest.json"),
        "raw": Path(str(prefix) + "_raw.csv"),
        "result": result,
        "m6_manifest": result.with_suffix(result.suffix + ".manifest.json"),
        "terminal_attestation": Path(str(prefix) + "_terminal_attestation.json"),
    }


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def m6_command(control: Path, candidate: Path, paths: Mapping[str, Path]):
    return [
        sys.executable,
        str(M6_RUNNER_PATH),
        "--control",
        str(control),
        "--candidate",
        str(candidate),
        "--mechanism-id",
        MECHANISM_ID,
        "--inspection-index",
        str(INSPECTION_INDEX),
        "--raw-csv",
        str(paths["raw"]),
        "--output",
        str(paths["result"]),
    ]


def run(args: argparse.Namespace) -> Path:
    control = args.control.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    paths = output_paths(args.output_prefix)
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise FileExistsError("campaign output is create-only")
    sources = validate_sources(control, candidate)
    preconditions = validate_preconditions(
        args.invariants,
        args.m5_result,
        args.engagement,
        sources=sources,
    )
    command = m6_command(control, candidate, paths)
    launch = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "mechanism_id": MECHANISM_ID,
        "inspection_index": INSPECTION_INDEX,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inspection_spent_on_manifest_creation": True,
        "rerun_authorized": False,
        "sources": sources,
        "preconditions": preconditions,
        "harness_file_allowlist": sorted(HARNESS_FILES),
        "candidate_file_allowlist": sorted(CANDIDATE_FILES),
        "exclusive_machine": _exclusive_machine_audit(),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "bindings": {
            "contract_sha256": file_sha256(CONTRACT_PATH),
            "wrapper_sha256": file_sha256(RUNNER_PATH),
            "engagement_runner_sha256": file_sha256(ENGAGEMENT_RUNNER_PATH),
            "engagement_protocol_sha256": file_sha256(ENGAGEMENT_PROTOCOL_PATH),
            "invariant_runner_sha256": file_sha256(INVARIANT_RUNNER_PATH),
            "m5_runner_sha256": file_sha256(M5_RUNNER_PATH),
            "m6_runner_sha256": file_sha256(M6_RUNNER_PATH),
            "m6_rule_sha256": file_sha256(M6_RULE_PATH),
            "m6_contract_sha256": file_sha256(M6_CONTRACT_PATH),
        },
        "command": command,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    _write_create_only_json(paths["launch_manifest"], launch)
    subprocess.run(command, cwd=ROOT, check=True)
    for name in ("raw", "result", "m6_manifest"):
        if not paths[name].is_file():
            raise RuntimeError(f"M6 execution did not create {name}")
    result = json.loads(paths["result"].read_text())
    manifest = json.loads(paths["m6_manifest"].read_text())
    disposition = result.get("analysis", {}).get("disposition")
    expected_generic_sources = {
        "control_default": sources["control"],
        "candidate_default": sources["candidate"],
    }
    if (
        result.get("contract_id") != "m6-quality-successor-v3"
        or result.get("mechanism_id") != MECHANISM_ID
        or result.get("inspection_index") != INSPECTION_INDEX
        or result.get("candidate_ranking_eligible") is not True
        or result.get("shipping_or_default_claim_eligible") is not False
        or disposition not in {"advance", "kill"}
        or manifest.get("contract_id") != "m6-quality-successor-v3"
        or manifest.get("mechanism_id") != MECHANISM_ID
        or manifest.get("inspection_index") != INSPECTION_INDEX
        or manifest.get("inspection_spent") is not True
        or manifest.get("sources_before_and_after") != expected_generic_sources
        or manifest.get("raw_csv", {}).get("sha256") != file_sha256(paths["raw"])
    ):
        raise RuntimeError("terminal M6 artifacts are invalid")
    attestation = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "mechanism_id": MECHANISM_ID,
        "inspection_index": INSPECTION_INDEX,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "disposition": disposition,
        "launch_manifest_sha256": file_sha256(paths["launch_manifest"]),
        "raw_sha256": file_sha256(paths["raw"]),
        "result_sha256": file_sha256(paths["result"]),
        "m6_manifest_sha256": file_sha256(paths["m6_manifest"]),
        "shipping_or_default_claim_eligible": False,
        "rerun_authorized": False,
    }
    _write_create_only_json(paths["terminal_attestation"], attestation)
    return paths["terminal_attestation"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--invariants", type=Path, required=True)
    parser.add_argument("--m5-result", type=Path, required=True)
    parser.add_argument("--engagement", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(0 if run(parse_args()) else 1)
