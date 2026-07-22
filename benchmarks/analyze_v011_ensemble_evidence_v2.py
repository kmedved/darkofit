#!/usr/bin/env python3
"""Analyze v0.11 private ensemble evidence successor v2."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    from . import run_v011_ensemble_evidence_v2 as campaign
except ImportError:
    import run_v011_ensemble_evidence_v2 as campaign


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
BASE_PATH = BENCH / "analyze_v011_ensemble_evidence.py"


def _load_base():
    name = "_darkofit_v011_ensemble_evidence_v2_analyzer_base"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v1 ensemble evidence analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base()
_base.campaign = campaign
_base.DEFAULT_RAW = BENCH / "v011_ensemble_evidence_v2_raw.json"
_base.DEFAULT_RESULT = BENCH / "v011_ensemble_evidence_v2_result.json"
_base.DEFAULT_NOTE = BENCH / "v011_ensemble_evidence_v2_result.md"


def __getattr__(name: str):
    return getattr(_base, name)


def main(argv=None) -> int:
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
