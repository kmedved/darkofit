#!/usr/bin/env python3
"""Analyze the P1-v3 automatic-depth fresh Tier-D one-shot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(ROOT))

from benchmarks import analyze_t7b_automatic_depth_fresh_tier_d as engine


CONTRACT = (
    ROOT / "benchmarks" / "t7b_automatic_depth_fresh_tier_d_v3_execution_contract.json"
)
POWER_CONTRACT = (
    ROOT
    / "benchmarks"
    / "t7b_automatic_depth_fresh_tier_d_v3_power_design_contract.json"
)


def _load_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_create_only(path: Path, value) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    raw = _load_json(args.raw)
    contract = _load_json(CONTRACT)
    power_contract = _load_json(POWER_CONTRACT)
    result = engine.analyze(raw, contract, power_contract)
    result["source_hashes"] = {
        "raw": file_sha256(args.raw),
        "execution_contract": file_sha256(CONTRACT),
        "power_contract": file_sha256(POWER_CONTRACT),
        "analyzer": file_sha256(Path(__file__)),
        "analysis_engine": file_sha256(Path(engine.__file__)),
    }
    _write_create_only(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "disposition": result["disposition"],
                "quality_ratio": result["quality"]["equal_lineage_geomean_ratio"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
