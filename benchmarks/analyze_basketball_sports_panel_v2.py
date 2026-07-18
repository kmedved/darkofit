#!/usr/bin/env python3
"""Analyze the frozen T10 basketball sports panel 2 artifact."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import build_basketball_sports_panel_v2 as panel_builder
from benchmarks import run_basketball_sports_panel_v2 as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_BAR = 1.0
BOOTSTRAP_UPPER_BAR = 1.002
LEAVE_ONE_OUT_BAR = 1.003
WORST_LINEAGE_BAR = 1.02
GUARDRAIL_AGGREGATE_BAR = 1.005
GUARDRAIL_WORST_BAR = 1.02
MAX_COST_RATIO = 3.0
MAX_PAIRED_RATIO_IQR_OVER_MEDIAN = 0.20
BOOTSTRAP_SEED = 20_260_718
BOOTSTRAP_RESAMPLES = 100_000


def _sha256(path: Path) -> str:
    return runner._sha256(path)


def _atomic_create(path: Path, value: bytes) -> None:
    runner._atomic_create(path, value)


def _geometric_mean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or np.any(array <= 0.0):
        raise RuntimeError("RMSE ratios must be positive and one-dimensional")
    return float(np.exp(np.mean(np.log(array))))


def _cell_key(cell: dict[str, Any]) -> tuple[int, str]:
    return int(cell["season"]), str(cell["target"])


def _canonical_results(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if raw.get("name") != "darkofit_basketball_sports_panel_raw_v2":
        raise RuntimeError("raw artifact is not sports panel 2")
    if not raw.get("panel_spent_by_this_run"):
        raise RuntimeError("raw sports panel 2 artifact is not marked spent")
    expected_positions = {
        (block, position): arm
        for block, order in enumerate(runner.BLOCK_ORDERS)
        for position, arm in enumerate(order)
    }
    observed_positions: set[tuple[int, int]] = set()
    grouped = {arm: [] for arm in runner.ARM_ORDER}
    for record in raw["repeats"]:
        coordinate = (int(record["block"]), int(record["position"]))
        if coordinate in observed_positions:
            raise RuntimeError("raw sports panel 2 repeats a worker coordinate")
        observed_positions.add(coordinate)
        arm = str(record["arm"])
        if expected_positions.get(coordinate) != arm:
            raise RuntimeError("raw sports panel 2 worker order changed")
        if record["result"]["arm"] != arm:
            raise RuntimeError("raw sports panel 2 worker arm is inconsistent")
        grouped[arm].append(record["result"])
    if observed_positions != set(expected_positions):
        raise RuntimeError("raw sports panel 2 is missing worker coordinates")

    expected_keys = [
        (season, target)
        for season in panel_builder.SEASONS
        for target in panel_builder.TARGET_COLUMNS
    ]
    canonical: dict[str, dict[str, Any]] = {}
    for arm, results in grouped.items():
        if len(results) != len(runner.BLOCK_ORDERS):
            raise RuntimeError(f"raw sports panel 2 has wrong repeats for {arm}")
        fingerprints = {row["behavior_fingerprint_sha256"] for row in results}
        if fingerprints != {raw["behavior_fingerprints"].get(arm)}:
            raise RuntimeError(f"raw sports panel 2 behavior changed for {arm}")
        keys = [_cell_key(cell) for cell in results[0]["cells"]]
        if keys != expected_keys or len(set(keys)) != len(keys):
            raise RuntimeError(f"raw sports panel 2 cells changed for {arm}")
        canonical[arm] = results[0]
    return canonical


def _cell_map(
    result: dict[str, Any],
) -> dict[tuple[int, str], dict[str, Any]]:
    return {_cell_key(cell): cell for cell in result["cells"]}


def _bootstrap_upper(ratios: np.ndarray) -> float:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(ratios),
        size=(BOOTSTRAP_RESAMPLES, len(ratios)),
    )
    aggregates = np.exp(np.mean(np.log(ratios[indices]), axis=1))
    return float(np.quantile(aggregates, 0.95))


def _quality_comparison(
    candidate: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    candidate_cells = _cell_map(candidate)
    control_cells = _cell_map(control)
    if candidate_cells.keys() != control_cells.keys():
        raise RuntimeError("sports panel 2 arms do not share cells")
    rows = []
    for key in candidate_cells:
        left = candidate_cells[key]
        right = control_cells[key]
        ratio = float(left["primary"]["rmse"] / right["primary"]["rmse"])
        rows.append(
            {
                "season": key[0],
                "target": key[1],
                "candidate_rmse": float(left["primary"]["rmse"]),
                "control_rmse": float(right["primary"]["rmse"]),
                "ratio": ratio,
            }
        )
    ratios = np.asarray([row["ratio"] for row in rows], dtype=np.float64)
    aggregate = _geometric_mean(ratios)
    bootstrap_upper = _bootstrap_upper(ratios)
    leave_one_out = [
        _geometric_mean(np.delete(ratios, index)) for index in range(len(ratios))
    ]
    primary_gates = {
        "aggregate_ratio_at_most_1_000": aggregate <= AGGREGATE_BAR,
        "bootstrap_upper_at_most_1_002": bootstrap_upper <= BOOTSTRAP_UPPER_BAR,
        "remove_most_favorable_at_most_1_003": max(leave_one_out) <= LEAVE_ONE_OUT_BAR,
        "worst_lineage_at_most_1_020": float(np.max(ratios)) <= WORST_LINEAGE_BAR,
    }

    guardrails = {}
    for view in ("held_team", "seen_player", "cold_player"):
        view_ratios = np.asarray(
            [
                candidate_cells[key]["guardrail"]["scores"][view]["rmse"]
                / control_cells[key]["guardrail"]["scores"][view]["rmse"]
                for key in candidate_cells
            ],
            dtype=np.float64,
        )
        guardrails[view] = {
            "ratios": view_ratios.tolist(),
            "aggregate_ratio": _geometric_mean(view_ratios),
            "worst_ratio": float(np.max(view_ratios)),
        }
    guardrail_gates = {
        "held_team_aggregate_at_most_1_005": guardrails["held_team"]["aggregate_ratio"]
        <= GUARDRAIL_AGGREGATE_BAR,
        "cold_player_aggregate_at_most_1_005": guardrails["cold_player"][
            "aggregate_ratio"
        ]
        <= GUARDRAIL_AGGREGATE_BAR,
        "held_team_worst_at_most_1_020": guardrails["held_team"]["worst_ratio"]
        <= GUARDRAIL_WORST_BAR,
        "cold_player_worst_at_most_1_020": guardrails["cold_player"]["worst_ratio"]
        <= GUARDRAIL_WORST_BAR,
    }
    return {
        "candidate_arm": candidate["arm"],
        "control_arm": control["arm"],
        "cells": rows,
        "aggregate_rmse_ratio": aggregate,
        "bootstrap_95_upper": bootstrap_upper,
        "leave_one_out_ratios": leave_one_out,
        "remove_most_favorable_ratio": max(leave_one_out),
        "worst_lineage_ratio": float(np.max(ratios)),
        "wins_ties_losses": {
            "wins": int(np.sum(ratios < 1.0)),
            "ties": int(np.sum(ratios == 1.0)),
            "losses": int(np.sum(ratios > 1.0)),
        },
        "primary_gates": primary_gates,
        "guardrails": guardrails,
        "guardrail_gates": guardrail_gates,
        "passes_quality": all(primary_gates.values()) and all(guardrail_gates.values()),
    }


def _group_repeats(
    raw: dict[str, Any],
) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in raw["repeats"]:
        grouped.setdefault(int(record["block"]), {})[str(record["arm"])] = record[
            "result"
        ]
    for block, arms in grouped.items():
        if set(arms) != set(runner.ARM_ORDER):
            raise RuntimeError(f"sports panel 2 block {block} is incomplete")
    return grouped


def _ratio_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(runner.BLOCK_ORDERS),) or np.any(array <= 0.0):
        raise RuntimeError("sports panel 2 paired ratios are invalid")
    median = float(np.median(array))
    q25, q75 = np.percentile(array, [25.0, 75.0])
    relative = float((q75 - q25) / median)
    return {
        "values": array.tolist(),
        "median": median,
        "q25": float(q25),
        "q75": float(q75),
        "iqr_over_median": relative,
        "stable": relative <= MAX_PAIRED_RATIO_IQR_OVER_MEDIAN,
    }


def _timing_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    blocks = _group_repeats(raw)
    summaries: dict[str, Any] = {}
    metrics = (
        "total_fit_seconds",
        "total_predict_seconds",
        "steady_wall_seconds",
        "peak_rss_bytes",
    )
    for arm in runner.ARM_ORDER:
        results = [blocks[block][arm] for block in sorted(blocks)]
        summaries[arm] = {
            metric: {
                "values": [float(row[metric]) for row in results],
                "median": float(statistics.median(row[metric] for row in results)),
            }
            for metric in metrics
        }
    ratios = {}
    for name in metrics:
        ratios[name] = _ratio_summary(
            [
                blocks[block][runner.CANDIDATE][name]
                / blocks[block][runner.CONTROL][name]
                for block in sorted(blocks)
            ]
        )
    return {"arms": summaries, "candidate_over_control": ratios}


def analyze(raw: dict[str, Any], raw_sha256: str) -> dict[str, Any]:
    canonical = _canonical_results(raw)
    candidate = _quality_comparison(
        canonical[runner.CANDIDATE], canonical[runner.CONTROL]
    )
    timing = _timing_analysis(raw)
    paired = timing["candidate_over_control"]
    cost_gates = {
        "fit_ratio_at_most_3": paired["total_fit_seconds"]["median"] <= MAX_COST_RATIO,
        "predict_ratio_at_most_3": paired["total_predict_seconds"]["median"]
        <= MAX_COST_RATIO,
        "rss_ratio_at_most_3": paired["peak_rss_bytes"]["median"] <= MAX_COST_RATIO,
        "fit_ratio_stable": paired["total_fit_seconds"]["stable"],
        "predict_ratio_stable": paired["total_predict_seconds"]["stable"],
        "behavior_reproduced": True,
    }
    passed = candidate["passes_quality"] and all(cost_gates.values())
    external = {
        arm: _quality_comparison(
            canonical[runner.CANDIDATE if passed else runner.CONTROL],
            canonical[arm],
        )
        for arm in (runner.CHIMERABOOST, runner.CATBOOST)
    }
    arm_summary = {}
    for arm, result in canonical.items():
        cells = result["cells"]
        arm_summary[arm] = {
            "geometric_mean_primary_rmse": _geometric_mean(
                [cell["primary"]["rmse"] for cell in cells]
            ),
            "geometric_mean_held_team_rmse": _geometric_mean(
                [cell["guardrail"]["scores"]["held_team"]["rmse"] for cell in cells]
            ),
            "geometric_mean_cold_player_rmse": _geometric_mean(
                [cell["guardrail"]["scores"]["cold_player"]["rmse"] for cell in cells]
            ),
            "median_total_fit_seconds": timing["arms"][arm]["total_fit_seconds"][
                "median"
            ],
            "median_total_predict_seconds": timing["arms"][arm][
                "total_predict_seconds"
            ]["median"],
            "median_peak_rss_bytes": timing["arms"][arm]["peak_rss_bytes"]["median"],
        }
    return {
        "schema_version": 1,
        "name": "darkofit_basketball_sports_panel_result_v2",
        "raw": {
            "sha256": raw_sha256,
            "runner_sha256": raw["runner"]["sha256"],
            "protocol_sha256": raw["protocol"]["sha256"],
            "panel_sha256": raw["panel_manifest"]["processed_panel_sha256"],
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
        },
        "candidate": {
            "comparison": candidate,
            "cost_gates": cost_gates,
            "passes": passed,
            "decision": (
                "advance_oob_ensemble5_as_sports_automatic_policy"
                if passed
                else "close_oob_ensemble5_as_sports_automatic_policy"
            ),
            "global_default_change_authorized": False,
            "sports_profile_change_authorized": passed,
        },
        "eligible_darkofit_arm": (runner.CANDIDATE if passed else runner.CONTROL),
        "external_context": external,
        "timing": timing,
        "arm_summary": arm_summary,
        "panel_spent": True,
        "retuning_on_panel_authorized": False,
    }


def render_report(result: dict[str, Any]) -> str:
    candidate = result["candidate"]
    comparison = candidate["comparison"]
    cost = result["timing"]["candidate_over_control"]
    lines = [
        "# Basketball sports automatic-policy confirmation, panel 2",
        "",
        "## Decision",
        "",
        (
            "The five-member row-OOB ensemble **passed** the frozen Tier-D "
            "gate and may advance as the named sports-profile automatic policy. "
            "The global default remains unchanged."
            if candidate["passes"]
            else "The five-member row-OOB ensemble **failed** the frozen Tier-D "
            "gate. Close it as a sports automatic policy without retuning on "
            "this now-spent panel."
        ),
        "",
        f"Decision code: `{candidate['decision']}`.",
        "",
        "## Candidate versus control",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Equal-lineage RMSE ratio | {comparison['aggregate_rmse_ratio']:.6f}× |",
        f"| 95% bootstrap upper | {comparison['bootstrap_95_upper']:.6f}× |",
        (
            "| Remove-best-lineage ratio | "
            f"{comparison['remove_most_favorable_ratio']:.6f}× |"
        ),
        f"| Worst lineage ratio | {comparison['worst_lineage_ratio']:.6f}× |",
        (
            "| Held-team aggregate ratio | "
            f"{comparison['guardrails']['held_team']['aggregate_ratio']:.6f}× |"
        ),
        (
            "| Cold-player aggregate ratio | "
            f"{comparison['guardrails']['cold_player']['aggregate_ratio']:.6f}× |"
        ),
        (f"| Median total-fit ratio | {cost['total_fit_seconds']['median']:.3f}× |"),
        (
            "| Median total-predict ratio | "
            f"{cost['total_predict_seconds']['median']:.3f}× |"
        ),
        f"| Median peak-RSS ratio | {cost['peak_rss_bytes']['median']:.3f}× |",
        "",
        "## Same-machine context",
        "",
        "| Arm | Primary RMSE | Cold-player RMSE | Median fit |",
        "|---|---:|---:|---:|",
    ]
    ordered = sorted(
        result["arm_summary"],
        key=lambda arm: result["arm_summary"][arm]["geometric_mean_primary_rmse"],
    )
    for arm in ordered:
        row = result["arm_summary"][arm]
        lines.append(
            f"| `{arm}` | {row['geometric_mean_primary_rmse']:.6f} | "
            f"{row['geometric_mean_cold_player_rmse']:.6f} | "
            f"{row['median_total_fit_seconds']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "The nine target-season lineages receive equal weight. Primary "
            "folds are player-disjoint. External comparisons are descriptive "
            "and cannot rescue the candidate decision. Panel 2 is spent and "
            "may not be used for retuning.",
            "",
            f"Raw artifact SHA-256: `{result['raw']['sha256']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _validate_paths(raw: Path, output: Path, report: Path) -> None:
    resolved = [path.resolve() for path in (raw, output, report)]
    if len(set(resolved)) != 3:
        raise RuntimeError("raw, JSON output, and report paths must be distinct")
    if not raw.is_file() or raw.is_symlink():
        raise RuntimeError(f"raw sports panel 2 artifact is unavailable: {raw}")
    for path in (output, report):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to replace analyzer output: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_paths(args.raw, args.output, args.report)
    raw_sha256 = _sha256(args.raw)
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    result = analyze(raw, raw_sha256)
    _atomic_create(
        args.output,
        (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    _atomic_create(args.report, render_report(result).encode("utf-8"))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(args.report),
                "decision": result["candidate"]["decision"],
                "passed": result["candidate"]["passes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
