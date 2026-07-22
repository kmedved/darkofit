#!/usr/bin/env python3
"""Execute the one-shot T7b automatic scalar-RMSE depth M6 inspection."""

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


CONTRACT_ID = "t7b-automatic-scalar-rmse-depth-v1-20260722"
MECHANISM_ID = "t7b_automatic_scalar_rmse_depth_v1"
INSPECTION_INDEX = 1
CONTROL_HEAD = "e23d2b164f10374b1c0e02521c33fc96d48980da"
RUNNER_PATH = Path(__file__).resolve()
ROOT = RUNNER_PATH.parents[1]
CONTRACT_PATH = RUNNER_PATH.with_name(
    "t7b_automatic_depth_development_contract.md"
)
M6_RUNNER_PATH = RUNNER_PATH.with_name("run_m6_quality_successor_v3.py")
M6_RULE_PATH = RUNNER_PATH.with_name("m6_quality_rule_v3.py")
M6_CONTRACT_PATH = RUNNER_PATH.with_name("m6_quality_successor_v3_contract.md")
INVARIANT_RUNNER_PATH = RUNNER_PATH.with_name(
    "check_t7b_automatic_depth_invariants.py"
)
M5_RUNNER_PATH = RUNNER_PATH.with_name("run_m5_sentinels.py")
M5_CONTROL_HEAD = "726e5d8e6131c580bce948db833a5007d0692dca"
INVARIANT_IDENTITY = (
    "t7b-automatic-scalar-rmse-depth-v1-invariants-20260722"
)
HARNESS_FILES = {
    "benchmarks/check_t7b_automatic_depth_invariants.py",
    "benchmarks/run_t7b_automatic_depth_v1.py",
    "benchmarks/t7b_automatic_depth_development_contract.md",
    "tests/test_t7b_automatic_depth_contract.py",
    "tests/test_t7b_automatic_depth_invariants.py",
}
CANDIDATE_FILES = {
    "darkofit/booster.py",
    "tests/test_darkofit.py",
    "tests/test_t7b_automatic_depth_policy.py",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    root = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if root != repository or not (root / "darkofit").is_dir():
        raise RuntimeError(f"not a DarkoFit Git root: {repository}")
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


def _is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.returncode == 0


def _tracked_bytes(repository: Path, head: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{head}:{relative}"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def _candidate_changed_files(
    candidate: Path, *, harness_head: str, candidate_head: str
) -> set[str]:
    output = _git(
        candidate,
        "diff",
        "--name-only",
        f"{harness_head}..{candidate_head}",
    )
    return {line for line in output.splitlines() if line}


def validate_sources(control: Path, candidate: Path) -> dict[str, Any]:
    harness = source_state(ROOT)
    control_state = source_state(control)
    candidate_state = source_state(candidate)
    if any(
        not state["clean"]
        for state in (harness, control_state, candidate_state)
    ):
        raise RuntimeError("T7b depth execution requires clean source trees")
    if control_state["head"] != CONTROL_HEAD:
        raise RuntimeError(
            f"T7b depth control is {control_state['head']}, expected {CONTROL_HEAD}"
        )
    harness_parent = _git(ROOT, "rev-parse", f"{harness['head']}^")
    if harness_parent != CONTROL_HEAD:
        raise RuntimeError(
            "T7b depth harness must be the single frozen contract commit "
            "directly above the control"
        )
    harness_changed = _candidate_changed_files(
        ROOT,
        harness_head=CONTROL_HEAD,
        candidate_head=harness["head"],
    )
    if harness_changed != HARNESS_FILES:
        raise RuntimeError(
            "harness changed files differ from the frozen allowlist: "
            f"{sorted(harness_changed)}"
        )
    if not _is_ancestor(candidate, harness["head"], candidate_state["head"]):
        raise RuntimeError("candidate does not descend from the frozen harness")
    changed = _candidate_changed_files(
        candidate,
        harness_head=harness["head"],
        candidate_head=candidate_state["head"],
    )
    if changed != CANDIDATE_FILES:
        raise RuntimeError(
            f"candidate changed files differ from the frozen allowlist: {sorted(changed)}"
        )
    relative_contract = str(CONTRACT_PATH.relative_to(ROOT))
    if CONTRACT_PATH.read_bytes() != _tracked_bytes(
        ROOT, harness["head"], relative_contract
    ):
        raise RuntimeError("T7b depth contract differs from harness HEAD")
    return {
        "harness": harness,
        "control": control_state,
        "candidate": candidate_state,
        "harness_changed_files": sorted(harness_changed),
        "candidate_changed_files": sorted(changed),
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
        "run_t7b_automatic_depth_v1",
        "run_m6_quality_successor",
        "run_automatic_linear_selector",
        "run_v011_compute_ladder",
        "run_v011_m2_broad_panel",
        "run_v011_ensemble_evidence",
        "run_m3",
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


def _external_prefix(prefix: Path) -> Path:
    resolved = prefix.expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return resolved
    raise ValueError("T7b depth outputs must be outside the harness checkout")


def output_paths(prefix: Path) -> dict[str, Path]:
    prefix = _external_prefix(prefix)
    result = Path(str(prefix) + "_result.json")
    return {
        "launch_manifest": Path(str(prefix) + "_launch_manifest.json"),
        "raw": Path(str(prefix) + "_raw.csv"),
        "result": result,
        "m6_manifest": result.with_suffix(result.suffix + ".manifest.json"),
        "terminal_attestation": Path(
            str(prefix) + "_terminal_attestation.json"
        ),
    }


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode()
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _source_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    keys = ("path", "head", "tree", "clean", "status")
    return all(actual.get(key) == expected.get(key) for key in keys)


def validate_preconditions(
    invariant_path: Path,
    m5_path: Path,
    *,
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    invariant_path = invariant_path.expanduser().resolve()
    m5_path = m5_path.expanduser().resolve()
    if not invariant_path.is_file() or not m5_path.is_file():
        raise RuntimeError("T7b depth invariant and M5 artifacts are required")
    invariant = json.loads(invariant_path.read_text())
    m5 = json.loads(m5_path.read_text())
    invariant_bindings = invariant.get("bindings", {})
    m5_analysis = m5.get("analysis", {})
    floor_checks = m5_analysis.get("known_floor_checks", {})
    m5_sources = m5.get("sources", {})
    if (
        invariant.get("schema_version") != 1
        or invariant.get("identity") != INVARIANT_IDENTITY
        or invariant.get("contract_id") != CONTRACT_ID
        or invariant.get("quality_outcomes_inspected") is not False
        or invariant.get("sources") != sources
        or invariant.get("analysis", {}).get("all_noop_cases_exact") is not True
        or invariant.get("analysis", {}).get("all_depth_branches_engaged")
        is not True
        or invariant_bindings.get("contract_sha256") != file_sha256(CONTRACT_PATH)
        or invariant_bindings.get("campaign_runner_sha256")
        != file_sha256(RUNNER_PATH)
        or invariant_bindings.get("invariant_runner_sha256")
        != file_sha256(INVARIANT_RUNNER_PATH)
        or m5.get("schema_version") != 1
        or m5.get("runner_version") != "m5-sentinels-v1"
        or m5.get("contract", {}).get("m5", {}).get("contract_frozen")
        is not True
        or m5.get("contract", {}).get("m5", {}).get("control_source")
        != M5_CONTROL_HEAD
        or m5.get("evidence_status") != "sentinel_check"
        or m5.get("non_ranking") is not True
        or m5.get("shipping_or_default_claim_authorized") is not False
        or m5.get("runner_sha256") != file_sha256(M5_RUNNER_PATH)
        or not _source_matches(m5_sources.get("harness", {}), sources["harness"])
        or not _source_matches(m5_sources.get("candidate", {}), sources["candidate"])
        or m5_sources.get("control", {}).get("head") != M5_CONTROL_HEAD
        or m5_sources.get("control", {}).get("clean") is not True
        or m5_analysis.get("behavior_fingerprints_equal_between_arms") is not True
        or m5_analysis.get("baseline_drift") != []
        or m5_analysis.get("advancement_blocked_for_drift") is not False
        or not floor_checks
        or not all(check.get("passed") is True for check in floor_checks.values())
    ):
        raise RuntimeError("T7b depth invariant or M5 precondition is invalid")
    return {
        "invariant": {
            "path": str(invariant_path),
            "sha256": file_sha256(invariant_path),
        },
        "m5": {"path": str(m5_path), "sha256": file_sha256(m5_path)},
    }


def m6_command(
    *, control: Path, candidate: Path, paths: Mapping[str, Path]
) -> list[str]:
    return [
        sys.executable,
        str(M6_RUNNER_PATH),
        "--control",
        str(control.resolve()),
        "--candidate",
        str(candidate.resolve()),
        "--mechanism-id",
        MECHANISM_ID,
        "--inspection-index",
        str(INSPECTION_INDEX),
        "--raw-csv",
        str(paths["raw"]),
        "--output",
        str(paths["result"]),
    ]


def _validate_terminal(
    paths: Mapping[str, Path], *, sources: Mapping[str, Any]
) -> dict[str, Any]:
    for name in ("raw", "result", "m6_manifest"):
        if not paths[name].is_file():
            raise RuntimeError(f"T7b depth execution did not create {name}")
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
        or manifest.get("raw_csv", {}).get("sha256")
        != file_sha256(paths["raw"])
    ):
        raise RuntimeError("T7b depth terminal M6 artifacts are invalid")
    return {
        "disposition": disposition,
        "raw_sha256": file_sha256(paths["raw"]),
        "result_sha256": file_sha256(paths["result"]),
        "m6_manifest_sha256": file_sha256(paths["m6_manifest"]),
    }


def run(args: argparse.Namespace) -> Path:
    control = args.control.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    paths = output_paths(args.output_prefix)
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        existing = [
            name
            for name, path in paths.items()
            if path.exists() or path.is_symlink()
        ]
        raise FileExistsError(f"T7b depth output is create-only: {existing}")
    sources = validate_sources(control, candidate)
    preconditions = validate_preconditions(
        args.invariants,
        args.m5_result,
        sources=sources,
    )
    command = m6_command(control=control, candidate=candidate, paths=paths)
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
        "bindings": {
            "contract_sha256": file_sha256(CONTRACT_PATH),
            "wrapper_sha256": file_sha256(RUNNER_PATH),
            "m6_runner_sha256": file_sha256(M6_RUNNER_PATH),
            "m6_rule_sha256": file_sha256(M6_RULE_PATH),
            "m6_contract_sha256": file_sha256(M6_CONTRACT_PATH),
            "invariant_runner_sha256": file_sha256(INVARIANT_RUNNER_PATH),
            "m5_runner_sha256": file_sha256(M5_RUNNER_PATH),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "exclusive_machine": _exclusive_machine_audit(),
        "command": command,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    _write_create_only_json(paths["launch_manifest"], launch)
    subprocess.run(command, cwd=ROOT, check=True)
    terminal = _validate_terminal(paths, sources=sources)
    attestation = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "mechanism_id": MECHANISM_ID,
        "inspection_index": INSPECTION_INDEX,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "launch_manifest_sha256": file_sha256(paths["launch_manifest"]),
        **terminal,
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
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(0 if run(parse_args()) else 1)
