#!/usr/bin/env python3
"""Analyze M3b attempt-3 evidence under its new source lineage."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    from . import run_m3b_ensemble_v3_r3 as runner
except ImportError:  # direct script execution
    import run_m3b_ensemble_v3_r3 as runner


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
FOUNDATION_PATH = BENCH_DIR / "analyze_m3b_ensemble_v3_r2.py"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def _load_foundation():
    name = "_darkofit_m3b_r3_analyzer_foundation"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, FOUNDATION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the bound M3b attempt-2 analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.runner = runner
    module._base.runner = runner
    return module


_foundation = _load_foundation()


def __getattr__(name: str):
    return getattr(_foundation, name)


def main() -> int:
    return _foundation.main()


if __name__ == "__main__":
    raise SystemExit(main())
