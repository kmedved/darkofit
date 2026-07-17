#!/usr/bin/env python3
"""Confirm P2 after removing redundant contiguous-block binning copies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks import run_predict_throughput as baseline


ROOT = baseline.ROOT
PROTOCOL = (
    ROOT / "benchmarks" / "predict_throughput_contiguous_blocks_protocol.md"
)
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "predict_throughput_contiguous_blocks.json"
)
MAX_PEAK_RSS_RATIO = 1.05
STRETCH_PUBLIC_RATIO = 1.0


def analyze_successor(
    canonical: dict[str, dict[str, Any]],
    block_results: list[dict[str, Any]],
    rss: dict[str, list[int]],
) -> dict[str, Any]:
    analysis = baseline.analyze(canonical, block_results)
    paired_rss = baseline.campaign.paired_ratio_summary(
        rss[baseline.DARKOFIT],
        rss[baseline.CHIMERABOOST],
    )
    public_summaries = [
        analysis["paired_ratios"][dataset][str(rows)]["warm_public"]
        for dataset in baseline.DATASETS
        for rows in baseline.BATCH_SIZES
    ]
    gates = {
        "predecessor_public_target": analysis["meets_public_target"],
        "peak_rss_ratio_at_most_1_05": (
            paired_rss["median_ratio"] <= MAX_PEAK_RSS_RATIO
        ),
    }
    analysis["paired_peak_rss"] = paired_rss
    analysis["successor_gates"] = gates
    analysis["meets_successor_target"] = all(gates.values())
    analysis["stretch_public_cases_at_or_below_chimera"] = int(
        sum(
            summary["stable"]
            and summary["median_ratio"] <= STRETCH_PUBLIC_RATIO
            for summary in public_summaries
        )
    )
    analysis["stretch_public_case_count"] = len(public_summaries)
    analysis["successor_recommendation"] = (
        "close_p2_matched_prediction_target"
        if analysis["meets_successor_target"]
        else analysis["recommendation"]
    )
    return analysis


def run_parent(args) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    if args.threads != baseline.EXPECTED_THREADS:
        raise RuntimeError("throughput protocol requires exactly 18 threads")
    sources = baseline._source_states(args)
    canonical = {}
    block_results = []
    rss = {arm: [] for arm in baseline.ARMS}
    for block, order in enumerate(baseline.BLOCK_ORDERS):
        for position, arm in enumerate(order):
            baseline._assert_sources_unchanged(
                sources,
                baseline._source_states(args),
                f"before block {block} {arm}",
            )
            print(
                f"running block {block + 1}/{len(baseline.BLOCK_ORDERS)} "
                f"position {position + 1}: {arm}",
                flush=True,
            )
            result = baseline._run_worker_process(args, arm)
            result["block"] = int(block)
            result["position"] = int(position)
            canonical.setdefault(arm, result)
            block_results.append(result)
            rss[arm].append(int(result["peak_rss_bytes"]))
    baseline._assert_sources_unchanged(
        sources,
        baseline._source_states(args),
        "during throughput campaign",
    )
    analysis = analyze_successor(canonical, block_results, rss)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "contiguous_layout_prediction_throughput_confirmation",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": baseline._sha256(PROTOCOL),
            "parent_runner_sha256": baseline._sha256(
                Path(baseline.__file__).resolve()
            ),
            "runner_sha256": baseline._sha256(Path(__file__).resolve()),
            "predecessor_artifact_sha256": (
                "cf25311f3364f4f939cf0324eb97f08641e92b68a0e82eac54c390d0b64e71c9"
            ),
            "chimeraboost_head": baseline.EXPECTED_CHIMERA_HEAD,
            "threads": baseline.EXPECTED_THREADS,
            "batch_sizes": list(baseline.BATCH_SIZES),
            "warm_repeats": baseline.WARM_REPEATS,
            "block_orders": [list(order) for order in baseline.BLOCK_ORDERS],
            "target_public_ratio": baseline.TARGET_PUBLIC_RATIO,
            "stretch_public_ratio": STRETCH_PUBLIC_RATIO,
            "max_peak_rss_ratio": MAX_PEAK_RSS_RATIO,
            "paired_ratio_max_iqr_over_median": (
                baseline.campaign.MAX_PAIRED_RATIO_IQR_OVER_MEDIAN
            ),
            "default_promotion_authorized": False,
            "lockbox_data_used": False,
        },
        "sources": sources,
        "environment": {
            "machine": baseline.creator._machine_details(),
            "dependencies": baseline.creator._dependency_versions(),
        },
        "canonical_results": canonical,
        "block_results": block_results,
        "analysis": analysis,
    }
    baseline.creator._atomic_write_bytes(
        args.output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    print(f"decision: {analysis['successor_recommendation']}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv=None):
    args = baseline.parse_args(argv)
    if args.output == baseline.DEFAULT_OUTPUT.resolve():
        args.output = DEFAULT_OUTPUT.resolve()
    if args.worker_arm is not None:
        raise ValueError("successor parent does not expose worker mode")
    return args


def main(argv=None) -> int:
    run_parent(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
