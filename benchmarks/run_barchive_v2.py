#!/usr/bin/env python3
"""Run B-archive attempt 2 after the optional feature-name correction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
FOUNDATION_PATH = BENCH_DIR / "run_barchive_v1.py"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def _load_foundation():
    name = "_darkofit_barchive_v2_foundation"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, FOUNDATION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the bound B-archive v1 foundation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_foundation = _load_foundation()
_foundation.__file__ = str(Path(__file__).resolve())
m3b = _foundation.m3b
paired = _foundation.paired

CONTRACT_NAME = "wave3_barchive_v2_20260721"
SCHEMA_VERSION = _foundation.SCHEMA_VERSION
MODEL_SOURCE_HEAD = _foundation.MODEL_SOURCE_HEAD
FROZEN_RUNTIME = _foundation.FROZEN_RUNTIME
THREADS = _foundation.THREADS
SINGLE = _foundation.SINGLE
COMBINED = _foundation.COMBINED
DEFAULT_PANEL_CACHE = _foundation.DEFAULT_PANEL_CACHE
ALLOWED_CANONICAL_ARRAYS = _foundation.ALLOWED_CANONICAL_ARRAYS
ALLOWED_CANONICAL_HEADER_FIELDS = frozenset(
    {"n_input_features", "prep", "random_state"}
)
PINNED_SOURCE_PATHS = _foundation.PINNED_SOURCE_PATHS

PROTOCOL_PATH = BENCH_DIR / "barchive_v2_protocol.md"
CONTRACT_PATH = BENCH_DIR / "barchive_v2_contract.json"
RAW_PATH = BENCH_DIR / "barchive_v2_raw.json"
TERMINAL_PATH = BENCH_DIR / "barchive_v2_terminal.json"
ANALYZER_PATH = BENCH_DIR / "analyze_barchive_v2.py"
FREEZER_PATH = BENCH_DIR / "freeze_barchive_v2.py"
V1_CONTRACT_PATH = BENCH_DIR / "barchive_v1_contract.json"
V1_TERMINAL_PATH = BENCH_DIR / "barchive_v1_terminal.json"
V1_FAILURE_PATH = BENCH_DIR / "barchive_v1_failure_record.json"

BOUND_PATHS = {
    "protocol": "benchmarks/barchive_v2_protocol.md",
    "runner": "benchmarks/run_barchive_v2.py",
    "analyzer": "benchmarks/analyze_barchive_v2.py",
    "freezer": "benchmarks/freeze_barchive_v2.py",
    "tests": "tests/test_barchive_v2.py",
    "v1_foundation_runner": "benchmarks/run_barchive_v1.py",
    "v1_foundation_analyzer": "benchmarks/analyze_barchive_v1.py",
    "v1_protocol": "benchmarks/barchive_v1_protocol.md",
    "v1_contract": "benchmarks/barchive_v1_contract.json",
    "v1_terminal": "benchmarks/barchive_v1_terminal.json",
    "v1_failure_record": "benchmarks/barchive_v1_failure_record.json",
    "component_analyzer": "benchmarks/analyze_ensemble_archive_components.py",
    "paired_execution": "benchmarks/paired_evidence_contract.py",
    "m6_adapter": "benchmarks/benchmark_adapters.py",
    "m3a_runner": "benchmarks/run_m3a_wave1.py",
    "m3a_contract": "benchmarks/m3a_wave1_contract.json",
    "sports_manifest": "benchmarks/basketball_sports_panel_v2_manifest.json",
    "m3b_runner": "benchmarks/run_m3b_ensemble_v3.py",
    "m3b_r3_contract": "benchmarks/m3b_ensemble_v3_r3_contract.json",
    "m3b_r3_quality": "benchmarks/m3b_ensemble_v3_r3_quality.json",
    "m3b_r3_result": "benchmarks/m3b_ensemble_v3_r3_result.json",
    "m3b_vs_single_readout": (
        "benchmarks/m3b_ensemble_v3_r3_vs_single_readout_20260721.json"
    ),
    "serialization": "darkofit/serialization.py",
    "sklearn_api": "darkofit/sklearn_api.py",
    "private_ensemble_tests": "tests/test_private_ensemble_v3.py",
}
WORKER_PREFIX = "BARCHIVE_V2_RESULT="


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"B-archive v2 lineage input is invalid: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"B-archive v2 lineage input is not an object: {path}")
    return value


def v1_attempt_lineage() -> dict[str, Any]:
    contract = _load_object(V1_CONTRACT_PATH)
    terminal = _load_object(V1_TERMINAL_PATH)
    failure = _load_object(V1_FAILURE_PATH)
    valid = (
        terminal.get("name") == "wave3_barchive_v1_20260721"
        and terminal.get("status") == "terminal_failure"
        and terminal.get("contract_sha256") == _sha256(V1_CONTRACT_PATH)
        and terminal.get("completed_rows_discarded") == 0
        and terminal.get("partial_rows_published") is False
        and terminal.get("rerun_allowed") is False
        and "canonical header set drifted" in str(terminal.get("error", ""))
        and failure.get("failed_contract") == _artifact(V1_CONTRACT_PATH)
        and failure.get("terminal_artifact") == _artifact(V1_TERMINAL_PATH)
        and failure.get("failure", {}).get("completed_rows_discarded") == 0
        and failure.get("failure", {}).get("raw_artifact_published") is False
        and failure.get("failure", {}).get("size_outcomes_published") is False
        and failure.get("scientific_disposition", {}).get("decision_threshold_changed")
        is False
        and failure.get("scientific_disposition", {}).get("cases_changed") is False
        and failure.get("scientific_disposition", {}).get("model_source_changed")
        is False
        and failure.get("scientific_disposition", {}).get("v1_rerun_allowed") is False
        and failure.get("verified_cause", {}).get("actual_numpy_input_header_fields")
        == ["n_input_features", "prep", "random_state"]
        and contract.get("decision_rules", {}).get(
            "median_effective_archive_to_single_at_most"
        )
        == 4.0
    )
    return {
        "v1_contract": _artifact(V1_CONTRACT_PATH),
        "v1_terminal": _artifact(V1_TERMINAL_PATH),
        "v1_failure_record": _artifact(V1_FAILURE_PATH),
        "v1_completed_rows_discarded": 0,
        "v1_partial_rows_published": False,
        "v1_size_outcomes_published": False,
        "v1_rerun": False,
        "successor_identity": CONTRACT_NAME,
        "source_changed": False,
        "cases_changed": False,
        "threshold_changed": False,
        "correction": (
            "preserve optional feature_names_in absence for frozen NumPy inputs"
        ),
        "lineage_valid": valid,
    }


for name, value in {
    "CONTRACT_NAME": CONTRACT_NAME,
    "PROTOCOL_PATH": PROTOCOL_PATH,
    "CONTRACT_PATH": CONTRACT_PATH,
    "RAW_PATH": RAW_PATH,
    "TERMINAL_PATH": TERMINAL_PATH,
    "ANALYZER_PATH": ANALYZER_PATH,
    "FREEZER_PATH": FREEZER_PATH,
    "BOUND_PATHS": BOUND_PATHS,
    "ALLOWED_CANONICAL_HEADER_FIELDS": ALLOWED_CANONICAL_HEADER_FIELDS,
    "WORKER_PREFIX": WORKER_PREFIX,
}.items():
    setattr(_foundation, name, value)

_foundation_load_contract = _foundation.load_contract


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = _foundation_load_contract(path)
    lineage = v1_attempt_lineage()
    if (
        lineage["lineage_valid"] is not True
        or contract.get("attempt_lineage") != lineage
    ):
        raise RuntimeError("B-archive v2 predecessor lineage is invalid")
    return contract


_foundation.load_contract = load_contract
_foundation_parser = _foundation._parser


def _parser():
    parser = _foundation_parser()
    parser.set_defaults(cache_dir=Path("/tmp/darkofit-barchive-v2-cache"))
    return parser


def _publish_terminal(
    *,
    error: BaseException,
    contract: Path,
    source_state: Mapping[str, Any],
    harness_state: Mapping[str, Any],
    completed_rows: int,
):
    if RAW_PATH.exists() or RAW_PATH.is_symlink():
        raise RuntimeError("cannot publish terminal after B-archive v2 raw artifact")
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "name": CONTRACT_NAME,
        "status": "terminal_failure",
        "failed_at": _foundation.datetime.now(_foundation.timezone.utc).isoformat(),
        "contract_sha256": _foundation._sha256(contract),
        "source_state": source_state,
        "harness_state": harness_state,
        "completed_rows_discarded": int(completed_rows),
        "partial_rows_published": False,
        "rerun_allowed": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    _foundation._write_json_create_only(TERMINAL_PATH, terminal)
    raise RuntimeError(f"B-archive v2 terminated; see {TERMINAL_PATH}") from error


_foundation._parser = _parser
_foundation._publish_terminal = _publish_terminal


def __getattr__(name: str):
    return getattr(_foundation, name)


def main(argv=None) -> int:
    return _foundation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
