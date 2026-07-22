#!/usr/bin/env python3
"""Execute the one-shot T7b automatic scalar-RMSE L2 M6 inspection."""

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


CONTRACT_ID = "t7b-automatic-scalar-rmse-l2-v1-20260722"
MECHANISM_ID = "t7b_automatic_scalar_rmse_l2_v1"
INSPECTION_INDEX = 1
CONTROL_HEAD = "370b8924c034de0332a4b990817972cf0e876f3e"
RUNNER_PATH = Path(__file__).resolve()
ROOT = RUNNER_PATH.parents[1]
CONTRACT_PATH = RUNNER_PATH.with_name(
    "t7b_automatic_l2_development_contract.md"
)
M6_RUNNER_PATH = RUNNER_PATH.with_name("run_m6_quality_successor_v3.py")
M6_RULE_PATH = RUNNER_PATH.with_name("m6_quality_rule_v3.py")
M6_CONTRACT_PATH = RUNNER_PATH.with_name("m6_quality_successor_v3_contract.md")
CANDIDATE_FILES = {
    "darkofit/booster.py",
    "tests/test_t7b_automatic_l2_policy.py",
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
        raise RuntimeError("T7b L2 execution requires clean source trees")
    if control_state["head"] != CONTROL_HEAD:
        raise RuntimeError(
            f"T7b L2 control is {control_state['head']}, expected {CONTROL_HEAD}"
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
        raise RuntimeError("T7b L2 contract differs from harness HEAD")
    return {
        "harness": harness,
        "control": control_state,
        "candidate": candidate_state,
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
        "run_t7b_automatic_l2_v1",
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
    raise ValueError("T7b L2 outputs must be outside the harness checkout")


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
            raise RuntimeError(f"T7b L2 execution did not create {name}")
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
        raise RuntimeError("T7b L2 terminal M6 artifacts are invalid")
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
        raise FileExistsError(f"T7b L2 output is create-only: {existing}")
    sources = validate_sources(control, candidate)
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
        "candidate_file_allowlist": sorted(CANDIDATE_FILES),
        "bindings": {
            "contract_sha256": file_sha256(CONTRACT_PATH),
            "wrapper_sha256": file_sha256(RUNNER_PATH),
            "m6_runner_sha256": file_sha256(M6_RUNNER_PATH),
            "m6_rule_sha256": file_sha256(M6_RULE_PATH),
            "m6_contract_sha256": file_sha256(M6_CONTRACT_PATH),
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
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(0 if run(parse_args()) else 1)
