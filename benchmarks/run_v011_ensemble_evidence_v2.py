#!/usr/bin/env python3
"""Run v0.11 private ensemble evidence successor v2."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
BASE_PATH = BENCH / "run_v011_ensemble_evidence.py"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))


def _load_base():
    name = "_darkofit_v011_ensemble_evidence_v2_base"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v1 ensemble evidence harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base()

CONTRACT_ID = "v011-private-ensemble-evidence-v2"
PROTOCOL_PATH = BENCH / "v011_ensemble_evidence_v2_protocol.md"
CONTRACT_PATH = BENCH / "v011_ensemble_evidence_v2_contract.json"
ANALYZER_PATH = BENCH / "analyze_v011_ensemble_evidence_v2.py"
FREEZER_PATH = BENCH / "freeze_v011_ensemble_evidence_v2.py"
DEFAULT_OUTPUT = BENCH / "v011_ensemble_evidence_v2_raw.json"
DEFAULT_TERMINAL = BENCH / "v011_ensemble_evidence_v2_terminal.json"

BOUND_PATHS = {
    "v1_protocol": "benchmarks/v011_ensemble_evidence_protocol.md",
    "v1_runner": "benchmarks/run_v011_ensemble_evidence.py",
    "v1_analyzer": "benchmarks/analyze_v011_ensemble_evidence.py",
    "v1_freezer": "benchmarks/freeze_v011_ensemble_evidence.py",
    "v1_tests": "tests/test_v011_ensemble_evidence.py",
    "v1_contract": "benchmarks/v011_ensemble_evidence_contract.json",
    "v2_protocol": "benchmarks/v011_ensemble_evidence_v2_protocol.md",
    "v2_runner": "benchmarks/run_v011_ensemble_evidence_v2.py",
    "v2_analyzer": "benchmarks/analyze_v011_ensemble_evidence_v2.py",
    "v2_freezer": "benchmarks/freeze_v011_ensemble_evidence_v2.py",
    "v2_tests": "tests/test_v011_ensemble_evidence_v2.py",
    **{
        name: path
        for name, path in _base.BOUND_PATHS.items()
        if name not in {"protocol", "runner", "analyzer", "freezer", "tests"}
    },
}

for name, value in {
    "CONTRACT_ID": CONTRACT_ID,
    "PROTOCOL_PATH": PROTOCOL_PATH,
    "CONTRACT_PATH": CONTRACT_PATH,
    "ANALYZER_PATH": ANALYZER_PATH,
    "FREEZER_PATH": FREEZER_PATH,
    "DEFAULT_OUTPUT": DEFAULT_OUTPUT,
    "DEFAULT_TERMINAL": DEFAULT_TERMINAL,
    "BOUND_PATHS": BOUND_PATHS,
}.items():
    setattr(_base, name, value)
_base.__file__ = str(Path(__file__).resolve())

_v1_warmup = _base._warmup


def _warmup(*args, **kwargs):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return _v1_warmup(*args, **kwargs)


_base._warmup = _warmup

_base_load_contract = _base.load_contract


def load_contract(path: Path = CONTRACT_PATH):
    contract = _base_load_contract(path)
    if contract.get("attempt_lineage") != {
        "v1_formal_workers_started": 0,
        "v1_outcomes_opened": False,
        "v1_raw_or_terminal_published": False,
        "sole_amendment": "capture and discard unmeasured warmup warnings",
    }:
        raise RuntimeError("v0.11 ensemble evidence v2 attempt lineage drifted")
    return contract


_base.load_contract = load_contract


def __getattr__(name: str):
    return getattr(_base, name)


def main(argv=None) -> int:
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
