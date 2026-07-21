#!/usr/bin/env python3
"""Create the prospective B-archive attempt-2 contract."""

from __future__ import annotations

import hashlib
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
FOUNDATION_PATH = BENCH_DIR / "freeze_barchive_v1.py"
OUTPUT = runner.CONTRACT_PATH
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def _load_foundation():
    name = "_darkofit_barchive_v2_freezer_foundation"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, FOUNDATION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the bound B-archive v1 freezer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_foundation = _load_foundation()
_foundation.runner = runner
_foundation.OUTPUT = OUTPUT
_foundation_build_contract = _foundation.build_contract


def build_contract():
    contract = _foundation_build_contract()
    lineage = runner.v1_attempt_lineage()
    if lineage["lineage_valid"] is not True:
        raise RuntimeError("B-archive v2 predecessor lineage is invalid")
    contract["attempt_lineage"] = lineage
    return contract


def main() -> int:
    state = runner.m3b.git_state(ROOT)
    if state["status"]:
        raise RuntimeError("B-archive v2 freeze requires a clean harness tree")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError(f"refusing existing B-archive v2 contract: {OUTPUT}")
    contract = build_contract()
    if (
        runner.m3b.git_state(ROOT) != state
        or contract["sources"]["harness"] != state["head"]
    ):
        raise RuntimeError("B-archive v2 harness changed during freeze")
    payload = (
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    runner.paired.write_create_only(OUTPUT, payload)
    print(
        json.dumps(
            {
                "contract": str(OUTPUT),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source": runner.MODEL_SOURCE_HEAD,
                "harness": state["head"],
                "predecessor": "wave3_barchive_v1_20260721",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
