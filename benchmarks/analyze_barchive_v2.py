#!/usr/bin/env python3
"""Validate and decide the frozen B-archive attempt-2 screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

try:
    from . import run_barchive_v2 as runner
except ImportError:  # direct script execution
    import run_barchive_v2 as runner


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
FOUNDATION_PATH = BENCH_DIR / "analyze_barchive_v1.py"
RESULT_PATH = BENCH_DIR / "barchive_v2_result.json"
NOTE_PATH = BENCH_DIR / "barchive_v2_result.md"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def _load_foundation():
    name = "_darkofit_barchive_v2_analyzer_foundation"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, FOUNDATION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the bound B-archive v1 analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_foundation = _load_foundation()
_foundation.runner = runner
_foundation.RESULT_PATH = RESULT_PATH
_foundation.NOTE_PATH = NOTE_PATH
_foundation_validate_raw = _foundation.validate_raw
_foundation_build_result = _foundation.build_result
render_note = _foundation.render_note


def validate_raw(
    raw_path: Path = runner.RAW_PATH,
    contract_path: Path = runner.CONTRACT_PATH,
):
    return _foundation_validate_raw(raw_path, contract_path)


_foundation.validate_raw = validate_raw


def build_result(
    raw_path: Path = runner.RAW_PATH,
    contract_path: Path = runner.CONTRACT_PATH,
):
    return _foundation_build_result(raw_path, contract_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=runner.RAW_PATH)
    parser.add_argument("--contract", type=Path, default=runner.CONTRACT_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--note", type=Path, default=NOTE_PATH)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    state = runner.m3b.git_state(ROOT)
    if state["status"]:
        raise RuntimeError("B-archive v2 analysis requires a clean harness tree")
    result = build_result(args.raw, args.contract)
    if runner.m3b.git_state(ROOT) != state:
        raise RuntimeError("B-archive v2 harness changed during analysis")
    _foundation._publish_pair(result, args.output, args.note)
    print(
        json.dumps(
            {
                "result": str(RESULT_PATH),
                "sha256": _foundation._sha256(RESULT_PATH),
                "disposition": result["disposition"],
            },
            sort_keys=True,
        )
    )
    return 0


def __getattr__(name: str):
    return getattr(_foundation, name)


if __name__ == "__main__":
    raise SystemExit(main())
