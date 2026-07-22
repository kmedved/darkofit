"""Run the warmup-corrected 14-CPU v0.11 M2 successor."""

from __future__ import annotations

import copy
import hashlib
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

if __package__ in {None, ""}:  # Direct execution from a clean checkout.
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from benchmarks import run_v011_m2_broad_panel_v2 as _v2


ROOT = Path(__file__).resolve().parents[1]
_v1 = _v2._v1
_V2_CONTRACT_ID = _v2.CONTRACT_ID
_V2_PROTOCOL = _v2.frozen_protocol()
CONTRACT_ID = "v011-m2-broad-panel-20260722-v3"
CONTRACT_PATH = ROOT / "benchmarks/v011_m2_broad_panel_contract_v3_20260722.json"
EXPECTED_CHILD_CPUS = _v2.EXPECTED_CHILD_CPUS
DEFAULT_OUTPUT_DIR = Path(".cache/v011-m2-broad-panel-v3-20260722")
WORKER_ENVIRONMENT = dict(_v2.WORKER_ENVIRONMENT)
CAMPAIGN_KIND = "darkofit_v011_m2_broad_panel_v3_20260722"
COMPLETION_KIND = CAMPAIGN_KIND + "_completion"
PAYLOAD_KIND = CAMPAIGN_KIND + "_analysis_payload"

SOURCE_FILES = (
    *_v2.SOURCE_FILES,
    Path("benchmarks/v011_m2_broad_panel_v2_preflight_failure_20260722.md"),
    Path("benchmarks/run_v011_m2_broad_panel_v3.py"),
    Path("benchmarks/analyze_v011_m2_broad_panel_v3.py"),
    Path("benchmarks/freeze_v011_m2_broad_panel_v3.py"),
    Path("benchmarks/v011_m2_broad_panel_protocol_v3_20260722.md"),
    Path("benchmarks/v011_m2_broad_panel_contract_v3_20260722.json"),
)
BOUND_PATHS = {
    "authorization": Path("benchmarks/v011_evidence_phase_instruction_20260721.md"),
    "v2_preflight_record": Path(
        "benchmarks/v011_m2_broad_panel_v2_preflight_failure_20260722.md"
    ),
    "protocol": Path("benchmarks/v011_m2_broad_panel_protocol_v3_20260722.md"),
    "runner": Path("benchmarks/run_v011_m2_broad_panel_v3.py"),
    "analyzer": Path("benchmarks/analyze_v011_m2_broad_panel_v3.py"),
    "freezer": Path("benchmarks/freeze_v011_m2_broad_panel_v3.py"),
    "tests": Path("tests/test_v011_m2_broad_panel_v3.py"),
}


def _bound_record(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not inspect bound source: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"bound source is not a regular file: {path}")
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def frozen_protocol() -> dict[str, Any]:
    protocol = copy.deepcopy(_V2_PROTOCOL)
    protocol["successor"] = {
        "supersedes_contract_id": _V2_CONTRACT_ID,
        "reason": "v2_pre_execution_warmup_constant_remained_18",
        "v2_fit_count": 0,
        "v2_completed_worker_count": 0,
        "v2_campaign_artifact_count": 2,
        "scientific_protocol_change": "none",
        "only_harness_change": "bind_warmup_thread_constant_to_frozen_14",
    }
    protocol["darkofit_execution_source_pin"] = {
        "policy": "published_contract_commit_only",
        "required_parent": "harness_freeze_git_head",
        "only_path_added_after_harness_freeze": str(CONTRACT_PATH.relative_to(ROOT)),
        "required_remote_ref": "origin/main",
    }
    return protocol


def protocol_sha256() -> str:
    return hashlib.sha256(
        _v1._base.hardened._canonical_json(frozen_protocol())
    ).hexdigest()


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract_path = Path(os.path.abspath(path.expanduser()))
    payload = _v1._read_finite_json(contract_path)
    if (
        set(payload)
        != {
            "schema_version",
            "contract_id",
            "created_at_utc",
            "contract_frozen",
            "outcome_blind",
            "authorization",
            "harness_freeze_git_head",
            "bindings",
            "protocol_sha256",
            "protocol",
        }
        or payload.get("schema_version") != 1
        or payload.get("contract_id") != CONTRACT_ID
        or payload.get("contract_frozen") is not True
        or payload.get("outcome_blind") is not True
        or payload.get("authorization")
        != "Phase 2 of v011_evidence_phase_instruction_20260721.md"
        or payload.get("protocol") != frozen_protocol()
        or payload.get("protocol_sha256") != protocol_sha256()
        or set(payload.get("bindings", {})) != set(BOUND_PATHS)
    ):
        raise RuntimeError("v0.11 M2 v3 contract is invalid")
    for name, relative in BOUND_PATHS.items():
        if payload["bindings"][name] != _bound_record(ROOT / relative):
            raise RuntimeError(f"v0.11 M2 v3 contract binding drifted: {name}")
    freeze_head = str(payload.get("harness_freeze_git_head", ""))
    if len(freeze_head) != 40:
        raise RuntimeError("v0.11 M2 v3 harness freeze commit is invalid")
    current = _v1._base._run_git(["rev-parse", "HEAD"], cwd=ROOT)
    try:
        _v1._base._run_git(
            ["merge-base", "--is-ancestor", freeze_head, current], cwd=ROOT
        )
    except RuntimeError as exc:
        raise RuntimeError("current source does not descend from the M2 v3 freeze") from exc
    return payload


def validate_execution_source_pin(contract: Mapping[str, Any]) -> str:
    freeze_head = str(contract["harness_freeze_git_head"])
    current = _v1._base._run_git(["rev-parse", "HEAD"], cwd=ROOT)
    revision = _v1._base._run_git(
        ["rev-list", "--parents", "-n", "1", current], cwd=ROOT
    ).split()
    if revision != [current, freeze_head]:
        raise RuntimeError("M2 v3 source must be the direct contract-commit child")
    changed = _v1._base._run_git(
        ["diff", "--name-only", freeze_head, current], cwd=ROOT
    ).splitlines()
    if changed != [str(CONTRACT_PATH.relative_to(ROOT))]:
        raise RuntimeError("M2 v3 contract commit changed more than its contract")
    if _v1._base._run_git(["rev-parse", "origin/main"], cwd=ROOT) != current:
        raise RuntimeError("M2 v3 execution source is not published at origin/main")
    return current


_SUCCESSOR_GLOBALS = {
    "CONTRACT_ID": CONTRACT_ID,
    "CONTRACT_PATH": CONTRACT_PATH,
    "EXPECTED_CHILD_CPUS": EXPECTED_CHILD_CPUS,
    "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
    "WORKER_ENVIRONMENT": WORKER_ENVIRONMENT,
    "CAMPAIGN_KIND": CAMPAIGN_KIND,
    "COMPLETION_KIND": COMPLETION_KIND,
    "PAYLOAD_KIND": PAYLOAD_KIND,
    "SOURCE_FILES": SOURCE_FILES,
    "frozen_protocol": frozen_protocol,
    "protocol_sha256": protocol_sha256,
    "load_contract": load_contract,
    "validate_execution_source_pin": validate_execution_source_pin,
}


@contextmanager
def configured_successor() -> Iterator[None]:
    with _v2.configured_successor():
        saved = {name: getattr(_v1, name) for name in _SUCCESSOR_GLOBALS}
        saved_file = _v1.__file__
        saved_overrides = _v1._OVERRIDES
        saved_import_warmup = _v1._import_warmup_module
        warmup_state: dict[str, Any] = {}

        def import_warmup_module():
            warmup = saved_import_warmup()
            if "module" not in warmup_state:
                warmup_state["module"] = warmup
                warmup_state["thread_count"] = warmup.THREAD_COUNT
            elif warmup is not warmup_state["module"]:
                raise RuntimeError("M2 v3 warmup module identity changed")
            warmup.THREAD_COUNT = EXPECTED_CHILD_CPUS
            return warmup

        try:
            for name, value in _SUCCESSOR_GLOBALS.items():
                setattr(_v1, name, value)
            _v1._import_warmup_module = import_warmup_module
            _v1.__file__ = __file__
            _v1._OVERRIDES = {
                **saved_overrides,
                "EXPECTED_CHILD_CPUS": EXPECTED_CHILD_CPUS,
                "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
                "CAMPAIGN_KIND": CAMPAIGN_KIND,
                "COMPLETION_KIND": COMPLETION_KIND,
                "PAYLOAD_KIND": PAYLOAD_KIND,
                "SOURCE_FILES": SOURCE_FILES,
                "frozen_protocol": frozen_protocol,
                "protocol_sha256": protocol_sha256,
            }
            yield
        finally:
            if "module" in warmup_state:
                warmup_state["module"].THREAD_COUNT = warmup_state["thread_count"]
            _v1._import_warmup_module = saved_import_warmup
            _v1._OVERRIDES = saved_overrides
            _v1.__file__ = saved_file
            for name, value in saved.items():
                setattr(_v1, name, value)


def main(argv: list[str] | None = None) -> int:
    with configured_successor():
        args = _v1.parse_args(argv)
        contract = load_contract()
        validate_execution_source_pin(contract)
        with _v1.configured_base():
            if args.worker_index is not None:
                return _v1._run_worker(args)
            return _v1._run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
