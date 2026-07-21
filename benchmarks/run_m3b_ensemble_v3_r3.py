#!/usr/bin/env python3
"""Run M3b attempt 3 after the group-bootstrap loader correction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
FOUNDATION_PATH = BENCH_DIR / "run_m3b_ensemble_v3_r2.py"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def _load_foundation():
    name = "_darkofit_m3b_r3_foundation"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, FOUNDATION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the bound M3b attempt-2 foundation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_foundation = _load_foundation()
_base = _foundation._base

CONTRACT_NAME = "wave2_m3b_ensemble_v3_r3_20260720"
PROTOCOL_PATH = BENCH_DIR / "m3b_ensemble_v3_r3_protocol.md"
CONTRACT_PATH = BENCH_DIR / "m3b_ensemble_v3_r3_contract.json"
ANALYZER_PATH = BENCH_DIR / "analyze_m3b_ensemble_v3_r3.py"
FREEZER_PATH = BENCH_DIR / "freeze_m3b_ensemble_v3_r3.py"
RSS_SCOPE = _foundation.RSS_SCOPE
MODEL_SOURCE_HEAD = "6d063f98128d457f8b8bbf610c7aec46e675d844"

BOUND_PATHS = {
    "attempt1_protocol": "benchmarks/m3b_ensemble_v3_protocol.md",
    "attempt1_runner": "benchmarks/run_m3b_ensemble_v3.py",
    "attempt1_analyzer": "benchmarks/analyze_m3b_ensemble_v3.py",
    "attempt1_freezer": "benchmarks/freeze_m3b_ensemble_v3.py",
    "attempt1_tests": "tests/test_m3b_ensemble_v3.py",
    "attempt1_contract": "benchmarks/m3b_ensemble_v3_contract.json",
    "attempt1_failure": "benchmarks/m3b_ensemble_v3_attempt1_failure_record.json",
    "attempt2_protocol": "benchmarks/m3b_ensemble_v3_r2_protocol.md",
    "attempt2_runner": "benchmarks/run_m3b_ensemble_v3_r2.py",
    "attempt2_analyzer": "benchmarks/analyze_m3b_ensemble_v3_r2.py",
    "attempt2_freezer": "benchmarks/freeze_m3b_ensemble_v3_r2.py",
    "attempt2_tests": "tests/test_m3b_ensemble_v3_r2.py",
    "attempt2_contract": "benchmarks/m3b_ensemble_v3_r2_contract.json",
    "attempt2_failure": "benchmarks/m3b_ensemble_v3_attempt2_failure_record.json",
    "attempt3_protocol": "benchmarks/m3b_ensemble_v3_r3_protocol.md",
    "attempt3_runner": "benchmarks/run_m3b_ensemble_v3_r3.py",
    "attempt3_analyzer": "benchmarks/analyze_m3b_ensemble_v3_r3.py",
    "attempt3_freezer": "benchmarks/freeze_m3b_ensemble_v3_r3.py",
    "attempt3_tests": "tests/test_m3b_ensemble_v3_r3.py",
    "b0_contract": "benchmarks/b0_ensemble_v3_contract.md",
    "implementation": "darkofit/sklearn_api.py",
    "implementation_tests": "tests/test_private_ensemble_v3.py",
    "paired_execution": "benchmarks/paired_evidence_contract.py",
    "m6_adapter": "benchmarks/benchmark_adapters.py",
    "m3a_runner": "benchmarks/run_m3a_wave1.py",
    "m3a_contract": "benchmarks/m3a_wave1_contract.json",
    "sports_manifest": "benchmarks/basketball_sports_panel_v2_manifest.json",
}

for module in (_foundation, _base):
    module.CONTRACT_NAME = CONTRACT_NAME
    module.PROTOCOL_PATH = PROTOCOL_PATH
    module.CONTRACT_PATH = CONTRACT_PATH
    module.ANALYZER_PATH = ANALYZER_PATH
    module.FREEZER_PATH = FREEZER_PATH
    module.BOUND_PATHS = BOUND_PATHS
_base.__file__ = str(Path(__file__).resolve())

_foundation_execution_contract = _foundation.execution_contract
_foundation_load_contract = _foundation.load_contract


def execution_contract():
    contract = _foundation_execution_contract()
    contract["attempt2_terminal_failure_bound"] = True
    contract["group_bootstrap_loader_fix_bound"] = True
    return contract


def load_contract(path: Path = CONTRACT_PATH):
    contract = _foundation_load_contract(path)
    lineage = contract.get("attempt_lineage", {})
    if (
        lineage.get("attempt2_completed_rows_discarded") != 1
        or lineage.get("attempt2_results_published") is not False
        or lineage.get("attempt2_results_inspected") is not False
        or lineage.get("attempt2_rerun") is not False
        or lineage.get("source_fix")
        != "allow variable sampled row counts for group bootstrap"
        or contract.get("sources", {}).get("darkofit") != MODEL_SOURCE_HEAD
    ):
        raise RuntimeError("M3b attempt-3 source or failure lineage changed")
    return contract


_foundation.execution_contract = execution_contract
_base.execution_contract = execution_contract
_foundation.load_contract = load_contract
_base.load_contract = load_contract


def __getattr__(name: str):
    return getattr(_foundation, name)


def main() -> int:
    return _foundation.main()


if __name__ == "__main__":
    raise SystemExit(main())
