#!/usr/bin/env python3
"""Analyze an immutable S4 basketball sports-panel raw artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from benchmarks import build_basketball_sports_panel as panel_builder  # noqa: E402
from benchmarks import run_basketball_sports_panel as runner  # noqa: E402


MIN_PRIMARY_DELTA = panel_builder.MIN_PRIMARY_MEAN_DELTA
MAX_FIT_RATIO = 1.75
MAX_RSS_RATIO = 1.10
MAX_PAIRED_RATIO_IQR_OVER_MEDIAN = 0.15


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_create(path: Path, value: bytes) -> None:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cell_key(cell: dict[str, Any]) -> tuple[int, str]:
    return int(cell["season"]), str(cell["target"])


def _canonical_results(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if raw.get("schema_version") != 1 or raw.get("name") != (
        "darkofit_basketball_sports_panel_raw_v1"
    ):
        raise RuntimeError("raw sports-panel artifact has an unknown schema")
    repeats = raw.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != (
        len(runner.BLOCK_ORDERS) * len(runner.ARM_ORDER)
    ):
        raise RuntimeError("raw sports-panel artifact has an invalid worker count")
    grouped = {arm: [] for arm in runner.ARM_ORDER}
    expected_positions = {
        (block, position): arm
        for block, order in enumerate(runner.BLOCK_ORDERS)
        for position, arm in enumerate(order)
    }
    observed_positions = set()
    for record in repeats:
        coordinate = (int(record["block"]), int(record["position"]))
        if coordinate in observed_positions:
            raise RuntimeError("raw sports-panel artifact repeats a worker coordinate")
        observed_positions.add(coordinate)
        arm = str(record["arm"])
        if expected_positions.get(coordinate) != arm:
            raise RuntimeError("raw sports-panel worker order differs from protocol")
        result = record["result"]
        if result["arm"] != arm:
            raise RuntimeError("raw sports-panel worker arm is inconsistent")
        grouped[arm].append(result)
    if observed_positions != set(expected_positions):
        raise RuntimeError("raw sports-panel artifact is missing worker coordinates")

    canonical: dict[str, dict[str, Any]] = {}
    for arm, results in grouped.items():
        if len(results) != len(runner.BLOCK_ORDERS):
            raise RuntimeError(f"raw sports-panel artifact has wrong repeats for {arm}")
        fingerprints = {row["behavior_fingerprint_sha256"] for row in results}
        if len(fingerprints) != 1:
            raise RuntimeError(f"raw sports-panel behavior changed for {arm}")
        expected_fingerprint = raw["behavior_fingerprints"].get(arm)
        if fingerprints != {expected_fingerprint}:
            raise RuntimeError(f"raw sports-panel fingerprint ledger changed for {arm}")
        keys = [_cell_key(cell) for cell in results[0]["cells"]]
        expected_keys = [
            (season, target)
            for season in panel_builder.SEASONS
            for target in panel_builder.TARGET_COLUMNS
        ]
        if keys != expected_keys or len(set(keys)) != len(keys):
            raise RuntimeError(f"raw sports-panel cells changed for {arm}")
        canonical[arm] = results[0]
    return canonical


def _cell_map(result: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {_cell_key(cell): cell for cell in result["cells"]}


def _view_mean(result: dict[str, Any], view: str) -> float:
    return float(
        np.mean([cell["guardrail"]["scores"][view]["r2"] for cell in result["cells"]])
    )


def _quality_comparison(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_cells = _cell_map(left)
    right_cells = _cell_map(right)
    if left_cells.keys() != right_cells.keys():
        raise RuntimeError("sports-panel arms do not contain identical cells")
    rows = []
    for key in left_cells:
        left_cell = left_cells[key]
        right_cell = right_cells[key]
        rows.append(
            {
                "season": key[0],
                "target": key[1],
                "left_r2": float(left_cell["primary_mean_r2"]),
                "right_r2": float(right_cell["primary_mean_r2"]),
                "delta": float(
                    left_cell["primary_mean_r2"] - right_cell["primary_mean_r2"]
                ),
                "left_fold_r2": [float(row["r2"]) for row in left_cell["folds"]],
                "right_fold_r2": [float(row["r2"]) for row in right_cell["folds"]],
            }
        )
    deltas = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    leave_one_out = [
        float(np.mean(np.delete(deltas, index))) for index in range(len(deltas))
    ]
    views = (
        "overlap_exposed_team_holdout",
        "seen_player_subset",
        "cold_player_subset",
    )
    view_deltas = {
        view: float(_view_mean(left, view) - _view_mean(right, view)) for view in views
    }
    gates = {
        "equal_cell_mean_delta_at_least_0_0005": float(np.mean(deltas))
        >= MIN_PRIMARY_DELTA,
        "leave_one_cell_out_no_regression": min(leave_one_out) >= 0.0,
        "overlap_exposed_team_no_regression": view_deltas[
            "overlap_exposed_team_holdout"
        ]
        >= 0.0,
        "cold_player_no_regression": view_deltas["cold_player_subset"] >= 0.0,
    }
    return {
        "left_arm": left["arm"],
        "right_arm": right["arm"],
        "cells": rows,
        "equal_cell_mean_r2": {
            "left": float(np.mean([row["left_r2"] for row in rows])),
            "right": float(np.mean([row["right_r2"] for row in rows])),
            "delta": float(np.mean(deltas)),
        },
        "cell_wins_ties_losses": {
            "wins": int(np.sum(deltas > 0.0)),
            "ties": int(np.sum(deltas == 0.0)),
            "losses": int(np.sum(deltas < 0.0)),
        },
        "leave_one_cell_out_deltas": leave_one_out,
        "minimum_leave_one_cell_out_delta": float(min(leave_one_out)),
        "guardrail_equal_cell_r2": {
            view: {
                "left": _view_mean(left, view),
                "right": _view_mean(right, view),
                "delta": view_deltas[view],
            }
            for view in views
        },
        "quality_gates": gates,
        "passes_quality": all(gates.values()),
    }


def _group_repeats(raw: dict[str, Any]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in raw["repeats"]:
        grouped.setdefault(int(record["block"]), {})[record["arm"]] = record["result"]
    for block, arms in grouped.items():
        if set(arms) != set(runner.ARM_ORDER):
            raise RuntimeError(f"sports-panel timing block {block} is incomplete")
    return grouped


def _ratio_stability(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(runner.BLOCK_ORDERS),) or np.any(array <= 0.0):
        raise RuntimeError("sports-panel paired ratios are invalid")
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
    for arm in runner.ARM_ORDER:
        results = [blocks[block][arm] for block in sorted(blocks)]
        summaries[arm] = {
            metric: {
                "values": [float(row[metric]) for row in results],
                "median": float(statistics.median(row[metric] for row in results)),
            }
            for metric in (
                "total_fit_seconds",
                "total_predict_seconds",
                "steady_wall_seconds",
                "peak_rss_bytes",
            )
        }
    candidate_control_fit = _ratio_stability(
        [
            blocks[block][runner.CANDIDATE]["total_fit_seconds"]
            / blocks[block][runner.CONTROL]["total_fit_seconds"]
            for block in sorted(blocks)
        ]
    )
    candidate_control_rss = _ratio_stability(
        [
            blocks[block][runner.CANDIDATE]["peak_rss_bytes"]
            / blocks[block][runner.CONTROL]["peak_rss_bytes"]
            for block in sorted(blocks)
        ]
    )
    return {
        "arms": summaries,
        "candidate_over_control": {
            "fit": candidate_control_fit,
            "peak_rss": candidate_control_rss,
        },
    }


def analyze(raw: dict[str, Any], raw_sha256: str) -> dict[str, Any]:
    canonical = _canonical_results(raw)
    candidate_quality = _quality_comparison(
        canonical[runner.CANDIDATE], canonical[runner.CONTROL]
    )
    timing = _timing_analysis(raw)
    operating_gates = {
        "median_fit_ratio_at_most_1_75": timing["candidate_over_control"]["fit"][
            "median"
        ]
        <= MAX_FIT_RATIO,
        "median_peak_rss_ratio_at_most_1_10": timing["candidate_over_control"][
            "peak_rss"
        ]["median"]
        <= MAX_RSS_RATIO,
        "fit_paired_ratio_stable": timing["candidate_over_control"]["fit"]["stable"],
        "behavior_reproduced": True,
    }
    candidate_passes = candidate_quality["passes_quality"] and all(
        operating_gates.values()
    )
    eligible_arm = runner.CANDIDATE if candidate_passes else runner.CONTROL
    external = {
        comparator: _quality_comparison(canonical[eligible_arm], canonical[comparator])
        for comparator in (runner.CHIMERABOOST, runner.CATBOOST)
    }

    arm_summary = {}
    for arm, result in canonical.items():
        arm_summary[arm] = {
            "equal_cell_mean_r2": float(
                np.mean([cell["primary_mean_r2"] for cell in result["cells"]])
            ),
            "held_team_equal_cell_mean_r2": _view_mean(
                result, "overlap_exposed_team_holdout"
            ),
            "seen_player_equal_cell_mean_r2": _view_mean(result, "seen_player_subset"),
            "cold_player_equal_cell_mean_r2": _view_mean(result, "cold_player_subset"),
            "median_total_fit_seconds": timing["arms"][arm]["total_fit_seconds"][
                "median"
            ],
            "median_total_predict_seconds": timing["arms"][arm][
                "total_predict_seconds"
            ]["median"],
            "median_peak_rss_bytes": timing["arms"][arm]["peak_rss_bytes"]["median"],
        }
    quality_ranking = sorted(
        runner.ARM_ORDER,
        key=lambda arm: arm_summary[arm]["equal_cell_mean_r2"],
        reverse=True,
    )
    return {
        "schema_version": 1,
        "name": "darkofit_basketball_sports_panel_result_v1",
        "raw": {
            "sha256": raw_sha256,
            "runner_sha256": raw["runner"]["sha256"],
            "protocol_sha256": raw["protocol"]["sha256"],
            "panel_sha256": raw["panel_manifest"]["processed_panel_sha256"],
        },
        "candidate": {
            "comparison": candidate_quality,
            "operating_gates": operating_gates,
            "passes": candidate_passes,
            "decision": (
                "confirm_opt_in_random_strength_0_5_for_noisy_basketball"
                if candidate_passes
                else "close_random_strength_0_5_without_s4_confirmation"
            ),
            "global_default_change_authorized": False,
        },
        "eligible_darkofit_arm": eligible_arm,
        "external_comparisons": external,
        "timing": timing,
        "arm_summary": arm_summary,
        "quality_ranking": quality_ranking,
        "claims": {
            "beats_chimeraboost_on_s4": external[runner.CHIMERABOOST]["passes_quality"],
            "beats_catboost_on_s4": external[runner.CATBOOST]["passes_quality"],
        },
        "panel_spent": True,
        "retuning_on_panel_authorized": False,
    }


def render_report(result: dict[str, Any]) -> str:
    candidate = result["candidate"]
    comparison = candidate["comparison"]
    summary = result["arm_summary"]
    lines = [
        "# Basketball multi-season sports confirmation result",
        "",
        "## Decision",
        "",
        (
            "`random_strength=0.5` passed the frozen S4 confirmation gate and is "
            "confirmed as an opt-in noisy-basketball setting. The global default "
            "remains `0.0`."
            if candidate["passes"]
            else "`random_strength=0.5` failed the frozen S4 confirmation gate. "
            "Close it without retuning on this now-spent panel; the global default "
            "remains `0.0`."
        ),
        "",
        f"Decision code: `{candidate['decision']}`.",
        "",
        "## Candidate versus control",
        "",
        "| Measure | Result |",
        "|---|---:|",
        (
            "| Equal-cell mean R² delta | "
            f"{comparison['equal_cell_mean_r2']['delta']:+.6f} |"
        ),
        (
            "| Minimum leave-one-cell-out delta | "
            f"{comparison['minimum_leave_one_cell_out_delta']:+.6f} |"
        ),
        (
            "| Held-team equal-cell R² delta | "
            f"{comparison['guardrail_equal_cell_r2']['overlap_exposed_team_holdout']['delta']:+.6f} |"
        ),
        (
            "| Cold-player equal-cell R² delta | "
            f"{comparison['guardrail_equal_cell_r2']['cold_player_subset']['delta']:+.6f} |"
        ),
        (
            "| Median fit-time ratio | "
            f"{result['timing']['candidate_over_control']['fit']['median']:.3f}× |"
        ),
        (
            "| Fit paired-ratio IQR/median | "
            f"{result['timing']['candidate_over_control']['fit']['iqr_over_median']:.3f} |"
        ),
        "",
        "## Same-machine comparison",
        "",
        "| Arm | Equal-cell R² | Cold-player R² | Median total fit |",
        "|---|---:|---:|---:|",
    ]
    for arm in result["quality_ranking"]:
        row = summary[arm]
        lines.append(
            f"| `{arm}` | {row['equal_cell_mean_r2']:.6f} | "
            f"{row['cold_player_equal_cell_mean_r2']:.6f} | "
            f"{row['median_total_fit_seconds']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "Formal quality claims:",
            "",
            f"- Beats ChimeraBoost on S4: **{result['claims']['beats_chimeraboost_on_s4']}**.",
            f"- Beats CatBoost on S4: **{result['claims']['beats_catboost_on_s4']}**.",
            "",
            "The nine target-season cells receive equal weight. The result is "
            "specific to this preregistered basketball panel; it does not authorize "
            "a global default change. The panel is now spent and may not be used "
            "for retuning.",
            "",
            f"Raw artifact SHA-256: `{result['raw']['sha256']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _validate_paths(raw: Path, output: Path, report: Path) -> None:
    paths = [path.resolve() for path in (raw, output, report)]
    if len(set(paths)) != 3:
        raise RuntimeError("raw, JSON output, and report paths must be distinct")
    if not raw.is_file() or raw.is_symlink():
        raise RuntimeError(f"raw sports-panel artifact is unavailable: {raw}")
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
                "raw": str(args.raw),
                "raw_sha256": raw_sha256,
                "output": str(args.output),
                "report": str(args.report),
                "candidate_passes": result["candidate"]["passes"],
                "claims": result["claims"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
