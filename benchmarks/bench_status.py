"""Render the release-level benchmark status from frozen same-machine evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
TABARENA_SUMMARY = BENCHMARKS / "tabarena_regression_same_machine_summary.json"
SPORTS_SUMMARY = BENCHMARKS / "basketball_sports_panel_result.json"
LARGE_N_SUMMARY = BENCHMARKS / "large_n_engine.json"
PREDICT_SUMMARY = BENCHMARKS / "predict_throughput_integrated.json"
C2_SUMMARY = BENCHMARKS / "native_ordinal_c2_development_result.json"
FUSED_SUBSET_SUMMARY = BENCHMARKS / "fused_subset_oblivious.json"
OUTPUT_JSON = BENCHMARKS / "benchmark_status.json"
OUTPUT_MARKDOWN = BENCHMARKS / "benchmark_status.md"
OUTPUT_MEASUREMENTS = ROOT / "docs" / "measurements.md"
EXPECTED_SOURCE_SHA256 = {
    "general_panel": "ca23618bdc3d9e0ab38557e7738c66e95827945ad34e3eb63005f253c92ccf01",
    "sports_panel": "4f20aed49ef0936a9442111b106aa5004b342c1924837a53c46e8010b2ae7189",
    "large_n": "ac9e6e9f136117b7b1db7488b38f660561195f86b29ae2f87868a5d293c62508",
    "prediction": "5ec81511e3026f5efadd8623228920da8d154a2f99719ca8f4116cd2c5b3653b",
    "native_ordinal_c2": "7aeb83131bb7604a3eaabc2789f048d40dabb58791b6ab6aad0ac26f0f0f566f",
    "fused_subset": "ed45820d74733ebcc6fca3ed1524a49eb9d73ae7ef22925bec41e7dea22d9d01",
}


def _read_json_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_positive(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _find_contrast(
    summary: dict[str, Any], numerator: str, denominator: str
) -> dict[str, Any]:
    matches = [
        row
        for row in summary.get("primary", [])
        if row.get("numerator") == numerator
        and row.get("denominator") == denominator
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {numerator}/{denominator} primary contrast, "
            f"found {len(matches)}"
        )
    row = matches[0]
    if row.get("paired_splits") != 39:
        raise ValueError("same-machine TabArena contrast must bind 39 splits")
    return row


def _metric_ratio(row: dict[str, Any], metric: str) -> float:
    value = row.get("metrics", {}).get(metric, {})
    if value.get("available") is False:
        raise ValueError(f"{metric} is unavailable")
    return _finite_positive(value.get("ratio"), metric)


def _pareto_flags(
    rows: list[dict[str, Any]],
    *,
    minimize: tuple[str, ...],
    maximize: tuple[str, ...] = (),
) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            weakly_better = all(
                float(other[key]) <= float(candidate[key]) for key in minimize
            ) and all(
                float(other[key]) >= float(candidate[key]) for key in maximize
            )
            strictly_better = any(
                float(other[key]) < float(candidate[key]) for key in minimize
            ) or any(
                float(other[key]) > float(candidate[key]) for key in maximize
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        flags[str(candidate["engine"])] = not dominated
    return flags


def _general_panel(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("decision") != "descriptive_only":
        raise ValueError("same-machine TabArena result must remain descriptive")
    if len(summary.get("primary", [])) != 3:
        raise ValueError("same-machine TabArena result must contain 3 contrasts")
    darko = _find_contrast(
        summary,
        "darkofit_product_default",
        "catboost_1_2_10_default",
    )
    chimera = _find_contrast(
        summary,
        "chimeraboost_0_14_1_default",
        "catboost_1_2_10_default",
    )
    rows = [
        {
            "engine": "DarkoFit 0.9.0",
            "test_rmse_ratio": _metric_ratio(darko, "test_rmse"),
            "fit_time_ratio": _metric_ratio(darko, "train_time_s"),
            "predict_time_ratio": _metric_ratio(darko, "infer_time_s"),
            "incremental_memory_ratio": _metric_ratio(
                darko, "incremental_memory_bytes"
            ),
        },
        {
            "engine": "ChimeraBoost 0.14.1",
            "test_rmse_ratio": _metric_ratio(chimera, "test_rmse"),
            "fit_time_ratio": _metric_ratio(chimera, "train_time_s"),
            "predict_time_ratio": _metric_ratio(chimera, "infer_time_s"),
            "incremental_memory_ratio": _metric_ratio(
                chimera, "incremental_memory_bytes"
            ),
        },
        {
            "engine": "CatBoost 1.2.10",
            "test_rmse_ratio": 1.0,
            "fit_time_ratio": 1.0,
            "predict_time_ratio": 1.0,
            "incremental_memory_ratio": 1.0,
        },
    ]
    flags = _pareto_flags(
        rows,
        minimize=(
            "test_rmse_ratio",
            "fit_time_ratio",
            "predict_time_ratio",
            "incremental_memory_ratio",
        ),
    )
    for row in rows:
        row["pareto"] = flags[row["engine"]]
    return {
        "scope": "13 TabArena regression datasets, r0f0/r1f1/r2f2",
        "baseline": "CatBoost 1.2.10",
        "decision": "descriptive_only",
        "rows": rows,
    }


def _sports_panel(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("panel_spent") is not True:
        raise ValueError("sports panel must remain marked spent")
    if summary.get("eligible_darkofit_arm") != "darkofit_control":
        raise ValueError("sports product frontier must use darkofit_control")
    candidate = summary.get("candidate", {})
    if (
        candidate.get("decision")
        != "close_random_strength_0_5_without_s4_confirmation"
        or candidate.get("passes") is not False
    ):
        raise ValueError("sports status requires the closed S4 candidate")
    arms = summary.get("arm_summary", {})
    identities = (
        ("DarkoFit 0.9.0", "darkofit_control"),
        ("ChimeraBoost 0.15.0", "chimeraboost_0_15_0"),
        ("CatBoost 1.2.10", "catboost_1_2_10"),
    )
    chimera = arms.get("chimeraboost_0_15_0", {})
    baseline_fit = _finite_positive(
        chimera.get("median_total_fit_seconds"), "sports ChimeraBoost fit"
    )
    baseline_predict = _finite_positive(
        chimera.get("median_total_predict_seconds"),
        "sports ChimeraBoost prediction",
    )
    rows = []
    for engine, arm in identities:
        values = arms.get(arm, {})
        rows.append(
            {
                "engine": engine,
                "equal_cell_r2": _finite(
                    values["equal_cell_mean_r2"], f"{engine} equal-cell R2"
                ),
                "cold_player_r2": _finite(
                    values["cold_player_equal_cell_mean_r2"],
                    f"{engine} cold-player R2",
                ),
                "fit_time_ratio": _finite_positive(
                    values["median_total_fit_seconds"], f"{engine} fit"
                )
                / baseline_fit,
                "predict_time_ratio": _finite_positive(
                    values["median_total_predict_seconds"], f"{engine} predict"
                )
                / baseline_predict,
            }
        )
    flags = _pareto_flags(
        rows,
        minimize=("fit_time_ratio", "predict_time_ratio"),
        maximize=("equal_cell_r2", "cold_player_r2"),
    )
    for row in rows:
        row["pareto"] = flags[row["engine"]]
    return {
        "scope": (
            "historical S4 2017–2019 panel: nine target-season basketball "
            "cells plus cold-player guardrail"
        ),
        "baseline": "ChimeraBoost 0.15.0",
        "decision": summary.get("candidate", {}).get("decision"),
        "rows": rows,
    }


def _large_n_status(summary: dict[str, Any]) -> dict[str, Any]:
    analysis = summary.get("analysis", {})
    if (
        analysis.get("passes_all_gates") is not False
        or analysis.get("recommendation")
        != "do_not_claim_large_n_engine_advantage"
    ):
        raise ValueError("large-n release status requires the closed result")
    sizes = analysis.get("sizes", {})
    if set(sizes) != {"500000", "1000000"}:
        raise ValueError("large-n status must bind the 500k and 1M rows")
    size_rows = []
    for size, values in sorted(
        sizes.items(), key=lambda item: int(item[0])
    ):
        size_rows.append({
            "rows": int(size),
            "fit_ratio": _finite_positive(
                values.get("fit_ratio", {}).get("median_ratio"),
                f"large-n {size} fit",
            ),
            "fit_iqr_over_median": _finite(
                values.get("fit_ratio", {}).get("iqr_over_median"),
                f"large-n {size} fit dispersion",
            ),
            "fit_q1_ratio": _finite_positive(
                values.get("fit_ratio", {}).get("q1_ratio"),
                f"large-n {size} fit Q1",
            ),
            "fit_q3_ratio": _finite_positive(
                values.get("fit_ratio", {}).get("q3_ratio"),
                f"large-n {size} fit Q3",
            ),
            "rmse_ratio": _finite_positive(
                values.get("rmse_ratio"), f"large-n {size} RMSE"
            ),
            "peak_rss_ratio": _finite_positive(
                values.get("rss_ratio", {}).get("median_ratio"),
                f"large-n {size} RSS",
            ),
        })
    rmse_ratios = [row["rmse_ratio"] for row in size_rows]
    return {
        "scope": "matched numeric core at 500k and 1M training rows",
        "fit_ratio": _finite_positive(
            analysis.get("fit_geomean_ratio"), "large-n fit ratio"
        ),
        "fit_speedup": _finite_positive(
            analysis.get("fit_geomean_speedup"), "large-n fit speedup"
        ),
        "rmse_ratio_min": min(rmse_ratios),
        "rmse_ratio_max": max(rmse_ratios),
        "sizes": size_rows,
        "certified": bool(analysis.get("passes_all_gates")),
        "decision": analysis.get("recommendation"),
    }


def _prediction_status(summary: dict[str, Any]) -> dict[str, Any]:
    analysis = summary.get("analysis", {})
    if (
        analysis.get("passes_all_gates") is not False
        or analysis.get("recommendation") != "p2_target_remains_open"
    ):
        raise ValueError("prediction release status requires the failed gate")
    cases = []
    for dataset, rows in sorted(analysis.get("paired_ratios", {}).items()):
        for row_count, values in sorted(rows.items(), key=lambda item: int(item[0])):
            cases.append({
                "input": str(dataset).replace("_", " "),
                "rows": int(row_count),
                "median_ratio": _finite_positive(
                    values.get("median_ratio"), "prediction median ratio"
                ),
                "iqr_over_median": _finite(
                    values.get("iqr_over_median"), "prediction dispersion"
                ),
                "q1_ratio": _finite_positive(
                    values.get("q1_ratio"), "prediction Q1 ratio"
                ),
                "q3_ratio": _finite_positive(
                    values.get("q3_ratio"), "prediction Q3 ratio"
                ),
                "stable": bool(values.get("stable")),
            })
    ratios = [
        {
            "median_ratio": case["median_ratio"],
            "stable": case["stable"],
        }
        for case in cases
    ]
    medians = [
        _finite_positive(values.get("median_ratio"), "prediction median ratio")
        for values in ratios
    ]
    median_wins = sum(value <= 1.0 for value in medians)
    stable_wins = sum(
        bool(values.get("stable"))
        and _finite_positive(values.get("median_ratio"), "prediction ratio")
        <= 1.0
        for values in ratios
    )
    recorded = int(
        analysis.get("stretch_public_cases_at_or_below_chimera", -1)
    )
    recorded_count = int(analysis.get("stretch_public_case_count", -1))
    if len(ratios) != 8 or recorded_count != len(ratios):
        raise ValueError("integrated prediction status must bind eight cases")
    if recorded != stable_wins:
        raise ValueError(
            "legacy integrated-prediction counter no longer matches its "
            "historical stable-and-at-or-below-Chimera semantics"
        )
    return {
        "scope": "eight numeric/mixed public prediction cases",
        "case_count": len(ratios),
        "median_at_or_below_chimera_count": median_wins,
        "stable_and_at_or_below_chimera_count": stable_wins,
        "legacy_counter_name_is_ambiguous": True,
        "median_ratio_min": min(medians),
        "median_ratio_max": max(medians),
        "cases": cases,
        "certified": bool(analysis.get("passes_all_gates")),
        "decision": analysis.get("recommendation"),
    }


def _c2_status(summary: dict[str, Any]) -> dict[str, Any]:
    if (
        summary.get("tier") != "development"
        or summary.get("decision") != "close_native_ordinal_c2_development"
        or summary.get("passes") is not False
        or summary.get("confirmation_outcomes_inspected") is not False
        or summary.get("confirmation_run_authorized") is not False
        or summary.get("lockbox_touched") is not False
    ):
        raise ValueError(
            "C2 status requires closed development with confirmation sealed"
        )
    aggregate = summary.get("aggregate", {})
    return {
        "scope": "native-ordinal C2 development panel",
        "candidate_over_default_rmse": _finite_positive(
            aggregate["equal_task_test_rmse_ratio"], "C2 RMSE ratio"
        ),
        "candidate_over_default_fit": _finite_positive(
            aggregate["fit_ratio"]["median"], "C2 fit ratio"
        ),
        "candidate_over_default_predict": _finite_positive(
            aggregate["predict_ratio"]["median"], "C2 prediction ratio"
        ),
        "decision": summary.get("decision"),
        "confirmation_run": False,
    }


def _fused_subset_status(summary: dict[str, Any]) -> dict[str, Any]:
    analysis = summary.get("analysis", {})
    if (
        analysis.get("passes_all_gates") is not False
        or analysis.get("recommendation")
        != "restore_full_lane_only_dispatch"
    ):
        raise ValueError("fused-subset status requires the immutable old decision")
    cells = analysis.get("cells", {})
    if len(cells) != 8 or not all(
        bool(cell.get("exact")) for cell in cells.values()
    ):
        raise ValueError("fused-subset status requires eight exact cells")
    subset_cells = [
        cell for cell in cells.values()
        if cell.get("sampling_lane") != "full"
    ]
    fit_rows = [
        cell["paired_ratios"]["fit_seconds"] for cell in subset_cells
    ]
    tree_rows = [
        cell["paired_ratios"]["tree_build_seconds"] for cell in subset_cells
    ]
    return {
        "scope": (
            "5,241-row x 15-feature basketball workload; selected rows, "
            "features, and both; unit and variable Hessians; 18 threads"
        ),
        "cell_count": len(cells),
        "all_exact": True,
        "fit_geomean_ratio": _finite_positive(
            analysis.get("subset_fit_geomean_ratio"), "subset fit ratio"
        ),
        "tree_build_geomean_ratio": _finite_positive(
            analysis.get("subset_tree_build_geomean_ratio"),
            "subset tree-build ratio",
        ),
        "fit_ratio_min": min(
            _finite_positive(row["median_ratio"], "subset fit cell")
            for row in fit_rows
        ),
        "fit_ratio_max": max(
            _finite_positive(row["median_ratio"], "subset fit cell")
            for row in fit_rows
        ),
        "tree_build_ratio_min": min(
            _finite_positive(row["median_ratio"], "subset tree cell")
            for row in tree_rows
        ),
        "tree_build_ratio_max": max(
            _finite_positive(row["median_ratio"], "subset tree cell")
            for row in tree_rows
        ),
        "fit_iqr_over_median_max": max(
            _finite(row["iqr_over_median"], "subset fit dispersion")
            for row in fit_rows
        ),
        "tree_build_iqr_over_median_max": max(
            _finite(row["iqr_over_median"], "subset tree dispersion")
            for row in tree_rows
        ),
        "prior_frozen_decision": analysis.get("recommendation"),
        "tier_e_dispatch_shipped": True,
    }


def build_status() -> dict[str, Any]:
    sources = {
        "general_panel": TABARENA_SUMMARY,
        "sports_panel": SPORTS_SUMMARY,
        "large_n": LARGE_N_SUMMARY,
        "prediction": PREDICT_SUMMARY,
        "native_ordinal_c2": C2_SUMMARY,
        "fused_subset": FUSED_SUBSET_SUMMARY,
    }
    loaded = {}
    actual_hashes = {}
    for name, path in sources.items():
        loaded[name], actual_hashes[name] = _read_json_with_sha256(path)
    for name, actual in actual_hashes.items():
        expected = EXPECTED_SOURCE_SHA256[name]
        if actual != expected:
            raise ValueError(
                f"{name} frozen source hash changed: expected {expected}, "
                f"found {actual}"
            )
    return {
        "schema_version": 1,
        "evidence_policy": (
            "same-machine panels remain separate; no cross-panel composite "
            "score or unmatched timing is used"
        ),
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": actual_hashes[name],
            }
            for name, path in sources.items()
        },
        "general_pareto": _general_panel(loaded["general_panel"]),
        "sports_pareto": _sports_panel(loaded["sports_panel"]),
        "large_n": _large_n_status(loaded["large_n"]),
        "prediction": _prediction_status(loaded["prediction"]),
        "native_ordinal_c2": _c2_status(loaded["native_ordinal_c2"]),
        "fused_subset": _fused_subset_status(loaded["fused_subset"]),
        "release_conclusion": (
            "Ship exact Tier-E engine work, opt-in product surfaces, and "
            "descriptive measurements; do not promote a Tier-D quality "
            "default without fresh preregistered confirmation."
        ),
    }


def render_markdown(status: dict[str, Any]) -> str:
    general = status["general_pareto"]
    sports = status["sports_pareto"]
    large_n = status["large_n"]
    prediction = status["prediction"]
    c2 = status["native_ordinal_c2"]
    fused_subset = status["fused_subset"]
    lines = [
        "# Benchmark status",
        "",
        "This release status is generated from frozen same-machine artifacts. "
        "Panels remain separate; it does not average unrelated datasets or "
        "reuse timings from another machine.",
        "",
        "## General regression Pareto",
        "",
        f"Scope: {general['scope']}. Ratios use {general['baseline']} as 1.0; "
        "lower is better.",
        "",
        "| Engine | Test RMSE | Fit | Predict | Incremental memory | Pareto |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in general["rows"]:
        lines.append(
            f"| {row['engine']} | {row['test_rmse_ratio']:.4f}× | "
            f"{row['fit_time_ratio']:.4f}× | "
            f"{row['predict_time_ratio']:.4f}× | "
            f"{row['incremental_memory_ratio']:.4f}× | "
            f"{'yes' if row['pareto'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "All three engines remain on this four-axis frontier: CatBoost has "
            "the best quality, DarkoFit the lowest fit time and incremental "
            "memory, and ChimeraBoost the best prediction time.",
            "",
            "## Sports Pareto (historical S4)",
            "",
            f"Scope: {sports['scope']}. Timing ratios use "
            f"{sports['baseline']} as 1.0.",
            "",
            "| Engine | Equal-cell R² | Cold-player R² | Fit | Predict | Pareto |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in sports["rows"]:
        lines.append(
            f"| {row['engine']} | {row['equal_cell_r2']:.6f} | "
            f"{row['cold_player_r2']:.6f} | {row['fit_time_ratio']:.3f}× | "
            f"{row['predict_time_ratio']:.3f}× | "
            f"{'yes' if row['pareto'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "On historical S4, DarkoFit beats ChimeraBoost on sports quality, "
            "but CatBoost is both more accurate and faster than DarkoFit. The "
            "failed `random_strength=0.5` candidate is excluded from this "
            "panel's product frontier. The later "
            "[player-disjoint T10 panel]"
            "(basketball_sports_panel_v2_result.md) is reported separately "
            "and is not blended into S4.",
            "",
            "## Engine tracks",
            "",
            "| Track | Observed result | Formal status |",
            "|---|---|---|",
            f"| Large-n matched core | Darko/Chimera fit "
            f"{large_n['fit_ratio']:.4f}× "
            f"({large_n['fit_speedup']:.4f}× speedup); RMSE "
            f"{large_n['rmse_ratio_min']:.5f}–"
            f"{large_n['rmse_ratio_max']:.5f}× | Tier-E measurement; the "
            "old frozen 1.30× certification remains closed |",
            f"| Public prediction | "
            f"{prediction['median_at_or_below_chimera_count']}/"
            f"{prediction['case_count']} median wins, "
            f"{prediction['stable_and_at_or_below_chimera_count']}/"
            f"{prediction['case_count']} also stable; ratios "
            f"{prediction['median_ratio_min']:.3f}–"
            f"{prediction['median_ratio_max']:.3f}× | Tier-E measurement; "
            "the old all-case certification remains closed |",
            f"| Sampled fused training | Exact in "
            f"{fused_subset['cell_count']}/{fused_subset['cell_count']} "
            f"cells; fit {fused_subset['fit_geomean_ratio']:.4f}× and tree "
            f"build {fused_subset['tree_build_geomean_ratio']:.4f}× | "
            "Shipped as behavior-exact Tier-E engine work; no external "
            "universal speed claim |",
            f"| Native ordinal C2 | Candidate/default RMSE "
            f"{c2['candidate_over_default_rmse']:.4f}×, fit "
            f"{c2['candidate_over_default_fit']:.4f}×, predict "
            f"{c2['candidate_over_default_predict']:.4f}× | Closed in "
            "development; confirmation remained sealed |",
            "",
            "The historical integrated-prediction JSON field "
            "`stretch_public_cases_at_or_below_chimera` counts cases that "
            "were both stable and no slower (6), despite its broader name. "
            "The eight raw median ratios show 8/8 no-slower medians. This "
            "report preserves the immutable artifact and labels both counts "
            "explicitly.",
            "",
            "## Release conclusion",
            "",
            status["release_conclusion"],
            "",
        ]
    )
    return "\n".join(lines)


def render_measurements(status: dict[str, Any]) -> str:
    """Render hash-bound Tier-E engineering measurements for user docs."""
    large_n = status["large_n"]
    prediction = status["prediction"]
    fused_subset = status["fused_subset"]
    lines = [
        "# Engineering measurements",
        "",
        "These are descriptive Tier-E measurements, not universal performance "
        "certifications. Each table is generated from immutable same-machine "
        "artifacts whose hashes are checked before rendering. Ratios below "
        "1.0 favor DarkoFit.",
        "",
        "## Matched large-n fitting",
        "",
        "DarkoFit / ChimeraBoost 0.15.0 on the matched numeric core.",
        "",
        "| Training rows | Fit ratio | Paired Q1–Q3 | IQR / median | Speedup | "
        "RMSE ratio | Peak RSS ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in large_n["sizes"]:
        lines.append(
            f"| {row['rows']:,} | {row['fit_ratio']:.4f}× | "
            f"{row['fit_q1_ratio']:.4f}–{row['fit_q3_ratio']:.4f}× | "
            f"{row['fit_iqr_over_median']:.5f} | "
            f"{1.0 / row['fit_ratio']:.4f}× | {row['rmse_ratio']:.5f}× | "
            f"{row['peak_rss_ratio']:.4f}× |"
        )
    lines.extend([
        "",
        f"The equal-size geometric mean was {large_n['fit_ratio']:.4f}× "
        f"({large_n['fit_speedup']:.4f}× faster). The immutable campaign's "
        "old 1.30× binary certification remains closed; the measured number "
        "is reported without rounding it into a pass.",
        "",
        "## Public prediction throughput",
        "",
        "Seconds-integrated public prediction, DarkoFit / ChimeraBoost 0.15.0.",
        "",
        "| Input | Rows | Median ratio | Paired Q1–Q3 | IQR / median |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in prediction["cases"]:
        lines.append(
            f"| {row['input']} | {row['rows']:,} | "
            f"{row['median_ratio']:.3f}× | "
            f"{row['q1_ratio']:.3f}–{row['q3_ratio']:.3f}× | "
            f"{row['iqr_over_median']:.5f} |"
        )
    lines.extend([
        "",
        f"All {prediction['case_count']} medians were no slower; "
        f"{prediction['stable_and_at_or_below_chimera_count']} also met the "
        "old campaign's stability threshold. The two facts are stated "
        "separately so a noisy ratio is not mislabeled as certified.",
        "",
        "## Sampled fused training",
        "",
        f"Scope: {fused_subset['scope']}.",
        "",
        "| Measure | Geometric-mean ratio | Cell-median range | "
        "Maximum paired IQR / median |",
        "|---|---:|---:|---:|",
        f"| Fit | {fused_subset['fit_geomean_ratio']:.4f}× | "
        f"{fused_subset['fit_ratio_min']:.4f}–"
        f"{fused_subset['fit_ratio_max']:.4f}× | "
        f"{fused_subset['fit_iqr_over_median_max']:.4f} |",
        f"| Tree build | {fused_subset['tree_build_geomean_ratio']:.4f}× | "
        f"{fused_subset['tree_build_ratio_min']:.4f}–"
        f"{fused_subset['tree_build_ratio_max']:.4f}× | "
        f"{fused_subset['tree_build_iqr_over_median_max']:.4f} |",
        "",
        "All eight reference/candidate cells had identical predictions, "
        "behavior fingerprints, and canonical serialized model state. The "
        "sampled dispatch ships as exact internal engine work. These workload "
        "measurements do not imply the same speedup on every dataset or "
        "machine.",
        "",
        "## Evidence policy",
        "",
        "See the [shipping policy](https://github.com/kmedved/darkofit/blob/"
        "main/benchmarks/SHIPPING_POLICY.md). Defaults and automatic modeling "
        "policies remain Tier-D and require fresh, power-checked, "
        "preregistered confirmation with uncertainty, concentration, harm, "
        "and cost controls.",
        "",
    ])
    return "\n".join(lines)


def _serialized(status: dict[str, Any]) -> tuple[str, str, str]:
    return (
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        render_markdown(status),
        render_measurements(status),
    )


def write_outputs() -> None:
    status = build_status()
    json_text, markdown_text, measurements_text = _serialized(status)
    OUTPUT_JSON.write_text(json_text, encoding="utf-8")
    OUTPUT_MARKDOWN.write_text(markdown_text, encoding="utf-8")
    OUTPUT_MEASUREMENTS.write_text(measurements_text, encoding="utf-8")


def check_outputs() -> None:
    status = build_status()
    expected_json, expected_markdown, expected_measurements = _serialized(status)
    actual_json = OUTPUT_JSON.read_text(encoding="utf-8")
    actual_markdown = OUTPUT_MARKDOWN.read_text(encoding="utf-8")
    actual_measurements = OUTPUT_MEASUREMENTS.read_text(encoding="utf-8")
    if (
        actual_json != expected_json
        or actual_markdown != expected_markdown
        or actual_measurements != expected_measurements
    ):
        raise SystemExit(
            "benchmark status outputs are stale; run "
            "`python benchmarks/bench_status.py --write`"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_outputs()
        return
    if args.check:
        check_outputs()
        return
    print(render_markdown(build_status()), end="")


if __name__ == "__main__":
    main()
