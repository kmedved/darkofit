"""Reusable design-time power simulation for prospective Tier-D panels.

This module never loads prospective targets.  It sizes a panel from a frozen
template and an explicitly declared effect scenario, then applies the same
quality gates that a later fresh confirmation analyzer must implement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from benchmarks.campaign_lib import provenance


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "t7b_automatic_depth_fresh_tier_d_power_design_contract.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks"
    / "t7b_automatic_depth_fresh_tier_d_power_design_result_20260723.json"
)
PROTOCOL = (
    ROOT
    / "benchmarks"
    / "t7b_automatic_depth_fresh_tier_d_power_design_protocol.md"
)
TESTS = ROOT / "tests" / "test_tier_d_fresh_power_design.py"


def _load_json(path: Path) -> dict[str, Any]:
    value = provenance.strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def load_contract(path: Path = CONTRACT) -> dict[str, Any]:
    contract = _load_json(path)
    validate_contract(contract)
    return contract


def _require_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], *, name: str
) -> None:
    observed = set(value)
    expected_set = set(expected)
    if observed != expected_set:
        raise RuntimeError(
            f"{name} keys changed: expected {sorted(expected_set)}, "
            f"observed {sorted(observed)}"
        )


def validate_contract(contract: Mapping[str, Any]) -> None:
    _require_exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "authority",
            "candidate",
            "spent_sizing_inputs",
            "panel_template",
            "effect_scenario",
            "quality_gates",
            "simulation",
            "execution_contract_requirements",
            "authorization",
        },
        name="contract",
    )
    if contract["schema_version"] != 1:
        raise RuntimeError("unsupported contract schema")
    if contract["contract_id"] != (
        "t7b-automatic-depth-shared-tier-d-power-v1-20260723"
    ):
        raise RuntimeError("contract identity changed")

    candidate = contract["candidate"]
    if candidate["source_commit"] != (
        "41e948f0c53b1d124e16071a7fa66eba47d084d3"
    ):
        raise RuntimeError("candidate source changed")
    if candidate["mechanism_id"] != (
        "t7b_automatic_scalar_rmse_depth_v1"
    ):
        raise RuntimeError("candidate mechanism changed")
    if candidate["branches"] != ["depth_4", "depth_8"]:
        raise RuntimeError("candidate branches changed")

    inputs = contract["spent_sizing_inputs"]
    if len(inputs) != 2:
        raise RuntimeError("spent sizing input count changed")
    for row in inputs:
        path = ROOT / row["path"]
        if not path.is_file():
            raise RuntimeError(f"missing spent sizing input: {path}")
        observed = provenance.file_sha256(path)
        if observed != row["file_sha256"]:
            raise RuntimeError(f"spent sizing input drift: {row['path']}")

    template = contract["panel_template"]
    strata = template["strata"]
    if len(strata) != 4:
        raise RuntimeError("panel must retain four coverage strata")
    if sum(row["lineages"] for row in strata) != 32:
        raise RuntimeError("panel must retain 32 independent lineages")
    if template["lineage_count"] != 32:
        raise RuntimeError("declared lineage count changed")
    if {row["branch"] for row in strata} != {"depth_4", "depth_8"}:
        raise RuntimeError("panel must cover both changed depth branches")
    if {
        row["feature_family"] for row in strata
    } != {"numeric", "categorical_or_grouped"}:
        raise RuntimeError("panel feature-family coverage changed")
    if any(row["lineages"] != 8 for row in strata):
        raise RuntimeError("every coverage stratum must retain eight lineages")
    if template["coordinates_per_lineage"] != 3:
        raise RuntimeError("coordinate count changed")
    if template["paired_model_fits"] != 2 * 32 * 3:
        raise RuntimeError("paired model-fit count changed")
    if template["analysis_unit"] != "independent_lineage_cluster":
        raise RuntimeError("analysis unit changed")
    if template["middle_density_policy"] != (
        "exact_noop_invariants_not_quality_power"
    ):
        raise RuntimeError("middle-density policy changed")

    scenario = contract["effect_scenario"]
    if scenario["primary_retained_log_effect_fraction"] != 0.20:
        raise RuntimeError("primary effect scenario changed")
    if scenario["sensitivity_retained_log_effect_fractions"] != [
        0.10,
        0.15,
        0.25,
    ]:
        raise RuntimeError("sensitivity scenarios changed")
    if scenario["true_lineage_ratio_cap"] != 1.015:
        raise RuntimeError("harm-compatible true-ratio cap changed")

    gates = contract["quality_gates"]
    expected_gates = {
        "equal_lineage_geomean_ratio_at_most": 0.995,
        "bootstrap_upper_ratio_at_most": 1.002,
        "leave_one_favorable_lineage_out_ratio_at_most": 0.998,
        "worst_lineage_ratio_at_most": 1.02,
        "each_branch_geomean_ratio_at_most": 1.0,
    }
    if gates != expected_gates:
        raise RuntimeError("Tier-D quality gates changed")

    simulation = contract["simulation"]
    if simulation != {
        "outer_panel_simulations": 5000,
        "outer_seed": 20260723,
        "lineage_bootstrap_replicates": 5000,
        "lineage_bootstrap_seed": 20260724,
        "lineage_bootstrap_percentile": 95.0,
        "outer_batch": 100,
        "power_floor": 0.8,
        "power_wilson_one_sided_confidence": 0.95,
        "power_decision_rule": (
            "point_estimate_and_one_sided_wilson_lower_bound_at_least_floor"
        ),
    }:
        raise RuntimeError("simulation design changed")

    authorization = contract["authorization"]
    if not authorization or any(
        value is not False for value in authorization.values()
    ):
        raise RuntimeError("design contract must not authorize downstream work")


def _group_centered_residuals(
    groups: Sequence[Sequence[float]],
) -> list[float]:
    residuals: list[float] = []
    for values in groups:
        logs = [math.log(float(value)) for value in values]
        center = sum(logs) / len(logs)
        residuals.extend(value - center for value in logs)
    return residuals


def derive_effect_model(contract: Mapping[str, Any]) -> dict[str, Any]:
    general = _load_json(ROOT / contract["spent_sizing_inputs"][0]["path"])
    sports = _load_json(ROOT / contract["spent_sizing_inputs"][1]["path"])

    general_analysis = general["analysis"]
    general_group = general_analysis["group_geometric_mean_ratio"]
    general_cells = general_analysis["ratios"]
    selected_general_groups = ["diabetes_resampled", "wide_numeric_reg"]
    source_lineage_ratios = [
        float(general_group[group]) for group in selected_general_groups
    ]
    coordinate_groups = [
        [
            float(value)
            for key, value in general_cells.items()
            if key.startswith(f"{group}/")
        ]
        for group in selected_general_groups
    ]

    sports_analysis = sports["analysis"]["cold_player"]
    for season in ("2014", "2015", "2016"):
        source_lineage_ratios.append(
            float(sports_analysis["season_ratio"][season])
        )
        coordinate_groups.append(
            [
                float(value)
                for key, value in sports_analysis[
                    "per_lineage_ratio"
                ].items()
                if f"_{season}_" in key
            ]
        )

    if len(source_lineage_ratios) != 5:
        raise RuntimeError("spent sizing lineage census changed")
    if [len(values) for values in coordinate_groups] != [6, 6, 3, 3, 3]:
        raise RuntimeError("spent sizing coordinate census changed")

    source_logs = np.log(np.asarray(source_lineage_ratios, dtype=np.float64))
    residuals = np.asarray(
        _group_centered_residuals(coordinate_groups), dtype=np.float64
    )
    source_mean_log = float(np.mean(source_logs))
    source_between_sd_log = float(np.std(source_logs, ddof=0))
    source_coordinate_sd_log = float(np.std(residuals, ddof=0))
    return {
        "source_lineage_ratios": source_lineage_ratios,
        "source_lineage_count": len(source_lineage_ratios),
        "source_coordinate_count": int(
            sum(len(values) for values in coordinate_groups)
        ),
        "source_geometric_mean_ratio": math.exp(source_mean_log),
        "source_mean_log_ratio": source_mean_log,
        "source_between_lineage_sd_log": source_between_sd_log,
        "source_within_lineage_coordinate_sd_log": source_coordinate_sd_log,
        "selection_bias_treatment": (
            "retain_only_declared_fraction_of_spent_log_effect_and_log_dispersion"
        ),
    }


def _wilson_lower(successes: int, trials: int, confidence: float) -> float:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid Wilson inputs")
    z = NormalDist().inv_cdf(confidence)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    )
    return max(0.0, (center - radius) / denominator)


def evaluate_panel_logs(
    lineage_log_ratios: np.ndarray,
    branch_labels: Sequence[str],
    bootstrap_counts: np.ndarray,
    gates: Mapping[str, float],
    *,
    bootstrap_percentile: float = 95.0,
) -> dict[str, np.ndarray]:
    if lineage_log_ratios.ndim != 2:
        raise ValueError("lineage_log_ratios must be a matrix")
    panels, lineages = lineage_log_ratios.shape
    if len(branch_labels) != lineages:
        raise ValueError("branch label count differs from lineages")
    if bootstrap_counts.ndim != 2 or bootstrap_counts.shape[1] != lineages:
        raise ValueError("bootstrap count matrix shape differs from lineages")

    point = np.exp(np.mean(lineage_log_ratios, axis=1))
    bootstrap_means = (
        np.einsum(
            "pl,bl->pb",
            lineage_log_ratios,
            bootstrap_counts,
            optimize=True,
        )
        / float(lineages)
    )
    bootstrap_upper = np.exp(
        np.percentile(
            bootstrap_means,
            bootstrap_percentile,
            axis=1,
            method="linear",
        )
    )
    favorable = np.min(lineage_log_ratios, axis=1)
    loo = np.exp(
        (np.sum(lineage_log_ratios, axis=1) - favorable)
        / float(lineages - 1)
    )
    worst = np.exp(np.max(lineage_log_ratios, axis=1))

    branch_points: dict[str, np.ndarray] = {}
    for branch in sorted(set(branch_labels)):
        indices = np.asarray(
            [index for index, label in enumerate(branch_labels) if label == branch]
        )
        branch_points[branch] = np.exp(
            np.mean(lineage_log_ratios[:, indices], axis=1)
        )

    components = {
        "point": point
        <= gates["equal_lineage_geomean_ratio_at_most"],
        "bootstrap_upper": bootstrap_upper
        <= gates["bootstrap_upper_ratio_at_most"],
        "leave_one_favorable_out": loo
        <= gates["leave_one_favorable_lineage_out_ratio_at_most"],
        "worst_lineage": worst
        <= gates["worst_lineage_ratio_at_most"],
        "each_branch_direction": np.logical_and.reduce(
            [
                values <= gates["each_branch_geomean_ratio_at_most"]
                for values in branch_points.values()
            ]
        ),
    }
    passes = np.logical_and.reduce(list(components.values()))
    if passes.shape != (panels,):
        raise RuntimeError("panel decision shape changed")
    return {
        "point": point,
        "bootstrap_upper": bootstrap_upper,
        "leave_one_favorable_out": loo,
        "worst_lineage": worst,
        "passes": passes,
        **{
            f"branch_{branch}": values
            for branch, values in branch_points.items()
        },
        **{
            f"component_{name}": values
            for name, values in components.items()
        },
    }


def _branch_labels(contract: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for row in contract["panel_template"]["strata"]:
        result.extend([row["branch"]] * int(row["lineages"]))
    return result


def simulate_scenario(
    contract: Mapping[str, Any],
    effect_model: Mapping[str, Any],
    *,
    retained_fraction: float,
) -> dict[str, Any]:
    simulation = contract["simulation"]
    lineages = sum(
        row["lineages"] for row in contract["panel_template"]["strata"]
    )
    coordinates = contract["panel_template"]["coordinates_per_lineage"]
    outer = simulation["outer_panel_simulations"]
    branch_labels = _branch_labels(contract)

    outer_rng = np.random.default_rng(simulation["outer_seed"])
    bootstrap_rng = np.random.default_rng(
        simulation["lineage_bootstrap_seed"]
    )
    bootstrap_counts = bootstrap_rng.multinomial(
        lineages,
        np.full(lineages, 1.0 / lineages, dtype=np.float64),
        size=simulation["lineage_bootstrap_replicates"],
    ).astype(np.float64)

    mean_log = retained_fraction * effect_model["source_mean_log_ratio"]
    between_sd = (
        retained_fraction
        * effect_model["source_between_lineage_sd_log"]
    )
    coordinate_sd = (
        retained_fraction
        * effect_model["source_within_lineage_coordinate_sd_log"]
    )
    true_cap = math.log(
        contract["effect_scenario"]["true_lineage_ratio_cap"]
    )

    component_counts = {
        "point": 0,
        "bootstrap_upper": 0,
        "leave_one_favorable_out": 0,
        "worst_lineage": 0,
        "each_branch_direction": 0,
        "all": 0,
    }
    metric_values = {
        "point": [],
        "bootstrap_upper": [],
        "leave_one_favorable_out": [],
        "worst_lineage": [],
        "branch_depth_4": [],
        "branch_depth_8": [],
    }
    remaining = outer
    while remaining:
        batch = min(simulation["outer_batch"], remaining)
        true_logs = outer_rng.normal(
            mean_log,
            between_sd,
            size=(batch, lineages),
        )
        true_logs = np.minimum(true_logs, true_cap)
        coordinate_noise = outer_rng.normal(
            0.0,
            coordinate_sd,
            size=(batch, lineages, coordinates),
        )
        observed_logs = true_logs + np.mean(coordinate_noise, axis=2)
        evaluated = evaluate_panel_logs(
            observed_logs,
            branch_labels,
            bootstrap_counts,
            contract["quality_gates"],
            bootstrap_percentile=simulation[
                "lineage_bootstrap_percentile"
            ],
        )
        for name in component_counts:
            key = "passes" if name == "all" else f"component_{name}"
            component_counts[name] += int(np.count_nonzero(evaluated[key]))
        for name in metric_values:
            metric_values[name].extend(evaluated[name].tolist())
        remaining -= batch

    passing = component_counts["all"]
    power = passing / outer
    confidence = simulation["power_wilson_one_sided_confidence"]
    lower = _wilson_lower(passing, outer, confidence)
    floor = simulation["power_floor"]
    return {
        "retained_log_effect_fraction": retained_fraction,
        "implied_geometric_mean_true_ratio": math.exp(mean_log),
        "implied_between_lineage_sd_log": between_sd,
        "implied_within_lineage_coordinate_sd_log": coordinate_sd,
        "true_lineage_ratio_cap": math.exp(true_cap),
        "outer_panel_simulations": outer,
        "passing_simulations": passing,
        "pass_probability": power,
        "wilson_one_sided_confidence": confidence,
        "wilson_lower_bound": lower,
        "minimum_required_probability": floor,
        "component_pass_probability": {
            name: count / outer
            for name, count in component_counts.items()
            if name != "all"
        },
        "metric_percentiles": {
            name: {
                "p05": float(np.percentile(values, 5.0, method="linear")),
                "p50": float(np.percentile(values, 50.0, method="linear")),
                "p95": float(np.percentile(values, 95.0, method="linear")),
            }
            for name, values in metric_values.items()
        },
        "power_floor_passes": power >= floor and lower >= floor,
    }


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _published_refs(head: str) -> list[str]:
    return sorted(
        line.strip()
        for line in _git("branch", "-r", "--contains", head).splitlines()
        if line.strip()
    )


def build(contract: Mapping[str, Any]) -> dict[str, Any]:
    effect_model = derive_effect_model(contract)
    primary_fraction = contract["effect_scenario"][
        "primary_retained_log_effect_fraction"
    ]
    primary = simulate_scenario(
        contract,
        effect_model,
        retained_fraction=primary_fraction,
    )
    sensitivity = [
        simulate_scenario(
            contract,
            effect_model,
            retained_fraction=fraction,
        )
        for fraction in contract["effect_scenario"][
            "sensitivity_retained_log_effect_fractions"
        ]
    ]
    qualified = bool(primary["power_floor_passes"])
    source_head = _git("rev-parse", "HEAD")
    result = {
        "schema_version": 1,
        "result_name": "t7b_automatic_depth_shared_tier_d_power_design_v1",
        "contract_id": contract["contract_id"],
        "source_head": source_head,
        "published_refs": _published_refs(source_head),
        "source_sha256": {
            "contract": provenance.file_sha256(CONTRACT),
            "builder": provenance.file_sha256(Path(__file__)),
            "protocol": provenance.file_sha256(PROTOCOL),
            "tests": provenance.file_sha256(TESTS),
            "shipping_policy": provenance.file_sha256(
                ROOT / "benchmarks" / "SHIPPING_POLICY.md"
            ),
        },
        "candidate": contract["candidate"],
        "panel_template": contract["panel_template"],
        "effect_model": effect_model,
        "quality_gates": contract["quality_gates"],
        "simulation": contract["simulation"],
        "primary_scenario": primary,
        "sensitivity_scenarios": sensitivity,
        "disposition": (
            "design_power_qualified" if qualified else "design_lacks_power"
        ),
        "power_qualified": qualified,
        "fresh_access_authorized": False,
        "registry_build_authorized": False,
        "confirmation_run_authorized": False,
        "candidate_merge_authorized": False,
        "default_change_authorized": False,
        "release_authorized": False,
        "lockbox_access_authorized": False,
    }
    result["result_sha256"] = provenance.canonical_json_sha256(result)
    return result


def validate_result(result: Mapping[str, Any]) -> None:
    value = dict(result)
    claimed = value.pop("result_sha256", None)
    if not provenance.is_sha256(claimed):
        raise RuntimeError("invalid result self-hash")
    if provenance.canonical_json_sha256(value) != claimed:
        raise RuntimeError("result self-hash mismatch")
    forbidden = [
        "fresh_access_authorized",
        "registry_build_authorized",
        "confirmation_run_authorized",
        "candidate_merge_authorized",
        "default_change_authorized",
        "release_authorized",
        "lockbox_access_authorized",
    ]
    if any(result[name] for name in forbidden):
        raise RuntimeError("power result granted forbidden authority")
    if result["power_qualified"] != result["primary_scenario"][
        "power_floor_passes"
    ]:
        raise RuntimeError("power disposition differs from primary scenario")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.contract.resolve() != CONTRACT.resolve():
        raise RuntimeError("contract path changed")
    if args.output.resolve() != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("output path changed")
    if args.output.exists():
        raise FileExistsError(f"create-only output exists: {args.output}")
    if _git("status", "--porcelain"):
        raise RuntimeError("power design requires a clean source tree")
    head = _git("rev-parse", "HEAD")
    if not _published_refs(head):
        raise RuntimeError("power design source head is not published")

    contract = load_contract(args.contract)
    result = build(contract)
    validate_result(result)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.write("\n")
    print(
        f"{result['disposition']}: power={result['primary_scenario']['pass_probability']:.6f}, "
        f"one-sided Wilson lower={result['primary_scenario']['wilson_lower_bound']:.6f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
