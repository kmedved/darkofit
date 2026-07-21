#!/usr/bin/env python3
"""Run the attempt-2 M3b attribution with self-worker RSS telemetry."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
BASE_RUNNER_PATH = BENCH_DIR / "run_m3b_ensemble_v3.py"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def _load_isolated_base():
    name = f"{__name__.replace('.', '_')}_base_runner"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the bound attempt-1 M3b runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_isolated_base()
_base_m3a = _base.m3a

CONTRACT_NAME = "wave2_m3b_ensemble_v3_r2_20260720"
PROTOCOL_PATH = BENCH_DIR / "m3b_ensemble_v3_r2_protocol.md"
CONTRACT_PATH = BENCH_DIR / "m3b_ensemble_v3_r2_contract.json"
ANALYZER_PATH = BENCH_DIR / "analyze_m3b_ensemble_v3_r2.py"
FREEZER_PATH = BENCH_DIR / "freeze_m3b_ensemble_v3_r2.py"
RSS_SCOPE = "self_worker_process"

BOUND_PATHS = {
    "base_protocol": "benchmarks/m3b_ensemble_v3_protocol.md",
    "base_runner": "benchmarks/run_m3b_ensemble_v3.py",
    "base_analyzer": "benchmarks/analyze_m3b_ensemble_v3.py",
    "base_freezer": "benchmarks/freeze_m3b_ensemble_v3.py",
    "base_harness_tests": "tests/test_m3b_ensemble_v3.py",
    "attempt1_contract": "benchmarks/m3b_ensemble_v3_contract.json",
    "attempt1_failure_record": (
        "benchmarks/m3b_ensemble_v3_attempt1_failure_record.json"
    ),
    "successor_protocol": "benchmarks/m3b_ensemble_v3_r2_protocol.md",
    "successor_runner": "benchmarks/run_m3b_ensemble_v3_r2.py",
    "successor_analyzer": "benchmarks/analyze_m3b_ensemble_v3_r2.py",
    "successor_freezer": "benchmarks/freeze_m3b_ensemble_v3_r2.py",
    "successor_tests": "tests/test_m3b_ensemble_v3_r2.py",
    "b0_contract": "benchmarks/b0_ensemble_v3_contract.md",
    "implementation": "darkofit/sklearn_api.py",
    "implementation_tests": "tests/test_private_ensemble_v3.py",
    "paired_execution": "benchmarks/paired_evidence_contract.py",
    "m6_adapter": "benchmarks/benchmark_adapters.py",
    "m3a_runner": "benchmarks/run_m3a_wave1.py",
    "m3a_contract": "benchmarks/m3a_wave1_contract.json",
    "sports_manifest": "benchmarks/basketball_sports_panel_v2_manifest.json",
}


def assert_rss_capability() -> int:
    import psutil

    rss = int(psutil.Process().memory_info().rss)
    if rss <= 0:
        raise RuntimeError("M3b attempt-2 self-worker RSS probe is nonpositive")
    return rss


class SelfWorkerRSSSampler(_base_m3a.AggregateRSSSampler):
    """Sample only this sequential worker and its native threads."""

    def _sample_once(self) -> None:
        import psutil

        rss = int(psutil.Process().memory_info().rss)
        if rss <= 0:
            raise RuntimeError("M3b attempt-2 worker RSS is nonpositive")
        self.peak_bytes = max(self.peak_bytes, rss)
        self.samples += 1


_m3a_proxy = types.ModuleType("_darkofit_m3b_r2_m3a_proxy")
_m3a_proxy.__dict__.update(_base_m3a.__dict__)
_m3a_proxy.AggregateRSSSampler = SelfWorkerRSSSampler
_base.m3a = _m3a_proxy

_base.CONTRACT_NAME = CONTRACT_NAME
_base.PROTOCOL_PATH = PROTOCOL_PATH
_base.CONTRACT_PATH = CONTRACT_PATH
_base.ANALYZER_PATH = ANALYZER_PATH
_base.FREEZER_PATH = FREEZER_PATH
_base.BOUND_PATHS = BOUND_PATHS
_base.__file__ = str(Path(__file__).resolve())

_base_execution_contract = _base.execution_contract
_base_run_worker = _base.run_worker
_base_write_json_create_only = _base._write_json_create_only
_base_load_gate = _base._load_gate
_base_load_contract = _base.load_contract


def execution_contract() -> dict[str, Any]:
    contract = _base_execution_contract()
    contract.update(
        {
            "rss_scope": RSS_SCOPE,
            "rss_capability_preflight": True,
            "attempt1_terminal_failure_bound": True,
        }
    )
    return contract


def run_worker(args):
    assert_rss_capability()
    row = _base_run_worker(args)
    row["rss_scope"] = RSS_SCOPE
    return row


def _write_json_create_only(path: Path, value):
    payload = dict(value)
    if payload.get("name") == CONTRACT_NAME and "status" in payload:
        payload["rss_scope"] = RSS_SCOPE
    return _base_write_json_create_only(path, payload)


def _load_gate(path: Path, contract_sha256: str):
    gate = _base_load_gate(path, contract_sha256)
    if gate.get("rss_scope") != RSS_SCOPE:
        raise RuntimeError("M3b attempt-2 timing gate has the wrong RSS scope")
    return gate


_base.execution_contract = execution_contract
_base.run_worker = run_worker
_base._write_json_create_only = _write_json_create_only
_base._load_gate = _load_gate


def load_contract(path: Path = CONTRACT_PATH):
    contract = _base_load_contract(path)
    lineage = contract.get("attempt_lineage")
    if (
        contract.get("rss_scope") != RSS_SCOPE
        or not isinstance(lineage, dict)
        or lineage.get("attempt1_completed_rows") != 0
        or lineage.get("attempt1_model_outcomes_opened") is not False
        or lineage.get("attempt1_rerun") is not False
        or lineage.get("sole_amendment") != "self-worker RSS with capability preflight"
        or not _base._is_hex_digest(contract.get("sources", {}).get("harness"), 40)
    ):
        raise RuntimeError("M3b attempt-2 lineage or RSS scope changed")
    return contract


_base.load_contract = load_contract


def __getattr__(name: str):
    return getattr(_base, name)


def main() -> int:
    assert_rss_capability()
    return _base.main()


if __name__ == "__main__":
    raise SystemExit(main())
