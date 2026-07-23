#!/usr/bin/env python3
"""Analyze the 32-lineage automatic-depth development benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(ROOT))

from benchmarks import analyze_t7b_automatic_depth_fresh_tier_d as legacy
from benchmarks import run_t7b_automatic_depth_development_v1 as runner
from benchmarks import run_t7b_automatic_depth_fresh_tier_d as helpers


def analyze(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("benchmark_id") != runner.BENCHMARK_ID:
        raise RuntimeError("raw development benchmark identity changed")
    historical_contract = helpers._load_json(legacy.CONTRACT)
    power_contract = helpers._load_json(legacy.POWER_CONTRACT)
    compatible_raw = {**raw, "contract_id": historical_contract["contract_id"]}
    historical = legacy.analyze(
        compatible_raw,
        historical_contract,
        power_contract,
    )

    quality = dict(historical["quality"])
    quality_reference = quality.pop("component_passes")
    quality.pop("passes")
    quality["panel_branch_geomean_ratio"] = quality.pop("branch_geomean_ratio")
    costs = dict(historical["costs"])
    cost_reference = costs.pop("component_passes")
    costs.pop("passes")

    resolution_counts: Counter[str] = Counter()
    by_panel_branch: dict[str, Counter[str]] = defaultdict(Counter)
    for row in raw["rows"]:
        if row["arm"] != "candidate":
            continue
        depth = str(int(row["resolved_depth"]))
        resolution_counts[depth] += 1
        by_panel_branch[str(row["panel_branch"])][depth] += 1

    return {
        "schema_version": 1,
        "benchmark_id": runner.BENCHMARK_ID,
        "kind": "paired_development",
        "quality": quality,
        "costs": costs,
        "integrity": historical["integrity"],
        "candidate_policy_resolutions": {
            "all_coordinates": dict(sorted(resolution_counts.items())),
            "by_panel_branch": {
                branch: dict(sorted(counts.items()))
                for branch, counts in sorted(by_panel_branch.items())
            },
        },
        "lineages": historical["lineages"],
        "historical_reference_gate_diagnostics": {
            "quality": quality_reference,
            "costs": cost_reference,
            "note": (
                "Telemetry only. The retired Tier-D gates do not decide this "
                "SHIP_RULES development benchmark."
            ),
        },
        "interpretation": [
            "Consult the holdout only if the development quality effect is clear.",
            "Costs are reported as product-frontier telemetry, not old campaign gates.",
            "This result is development evidence and says nothing about the holdout.",
        ],
    }


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(helpers.canonical_json_bytes(value))
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
    result = analyze(helpers._load_json(args.raw))
    result["source_hashes"] = {
        "raw": helpers.file_sha256(args.raw),
        "analyzer": helpers.file_sha256(Path(__file__)),
        "legacy_analysis_engine": helpers.file_sha256(Path(legacy.__file__)),
    }
    _write_create_only(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "quality_ratio": result["quality"][
                    "equal_lineage_geomean_ratio"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
