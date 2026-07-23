#!/usr/bin/env python3
"""Power simulation for the verified P1-v3 automatic-depth registry."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(ROOT))

from benchmarks import tier_d_fresh_power_design as engine
from benchmarks.campaign_lib import provenance


CONTRACT = (
    ROOT
    / "benchmarks"
    / "t7b_automatic_depth_fresh_tier_d_v3_power_design_contract.json"
)
PROTOCOL = (
    ROOT / "benchmarks" / "t7b_automatic_depth_fresh_tier_d_v3_power_design_protocol.md"
)
ENUMERATION = (
    ROOT
    / "benchmarks"
    / "t7b_automatic_depth_fresh_tier_d_v3_enumeration_v2_20260723.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks"
    / "t7b_automatic_depth_fresh_tier_d_v3_power_design_result_20260723.json"
)
TESTS = ROOT / "tests" / "test_t7b_automatic_depth_fresh_tier_d_v3_power_design.py"
CONTRACT_ID = "t7b-automatic-depth-fresh-tier-d-v3-power-v1-20260723"


def _load_json(path: Path) -> dict[str, Any]:
    value = provenance.strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


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
            "verified_registry",
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
    if contract["schema_version"] != 1 or contract["contract_id"] != CONTRACT_ID:
        raise RuntimeError("power contract identity changed")
    candidate = contract["candidate"]
    if candidate["source_commit"] != ("41e948f0c53b1d124e16071a7fa66eba47d084d3"):
        raise RuntimeError("candidate source changed")
    if candidate["control_commit"] != ("e23d2b164f10374b1c0e02521c33fc96d48980da"):
        raise RuntimeError("control source changed")
    if candidate["candidate_must_remain_byte_identical"] is not True:
        raise RuntimeError("candidate identity is not immutable")

    registry_binding = contract["verified_registry"]
    if provenance.file_sha256(ENUMERATION) != registry_binding["artifact_sha256"]:
        raise RuntimeError("verified enumeration artifact drifted")
    enumeration = _load_json(ENUMERATION)
    if enumeration["enumeration_id"] != registry_binding["enumeration_id"]:
        raise RuntimeError("enumeration identity changed")
    eligible = [row for row in enumeration["identities"] if row["status"] == "eligible"]
    eligible_ids = [row["lineage_id"] for row in eligible]
    if eligible_ids != registry_binding["eligible_lineage_ids"]:
        raise RuntimeError("verified eligible lineage list changed")
    if len(eligible_ids) != 32:
        raise RuntimeError("as-built panel must contain 32 verified lineages")
    if any(not row["resource_loaded"] for row in eligible):
        raise RuntimeError("as-built panel contains an unloaded resource")

    template = contract["panel_template"]
    if template["lineage_count"] != len(eligible_ids):
        raise RuntimeError("panel lineage count differs from registry")
    observed_strata = {
        stratum: sum(row["stratum"] == stratum for row in eligible)
        for stratum in enumeration["eligible_stratum_counts"]
    }
    declared_strata = {
        row["stratum"]: int(row["lineages"]) for row in template["strata"]
    }
    if declared_strata != observed_strata:
        raise RuntimeError("panel stratum counts differ from registry")
    observed_branches = {
        branch: sum(row["branch"] == branch for row in eligible)
        for branch in ("depth_4", "depth_8")
    }
    if template["branch_counts"] != observed_branches:
        raise RuntimeError("panel branch counts differ from registry")
    if template["group_safe_lineages"] != sum(
        row["split_kind"] == "group_hash_3fold" for row in eligible
    ):
        raise RuntimeError("panel group-safe count differs from registry")
    if template["coordinates_per_lineage"] != 3:
        raise RuntimeError("coordinate count changed")
    if template["paired_model_fits"] != 2 * len(eligible) * 3:
        raise RuntimeError("paired-fit count changed")

    for row in contract["spent_sizing_inputs"]:
        if provenance.file_sha256(ROOT / row["path"]) != row["file_sha256"]:
            raise RuntimeError(f"spent sizing input drifted: {row['path']}")
    if (
        contract["quality_gates"]
        != _load_json(
            ROOT
            / "benchmarks"
            / "t7b_automatic_depth_fresh_tier_d_power_design_contract.json"
        )["quality_gates"]
    ):
        raise RuntimeError("Tier-D quality gates changed")
    if contract["effect_scenario"]["primary_retained_log_effect_fraction"] != 0.2:
        raise RuntimeError("primary effect scenario changed")
    if contract["effect_scenario"]["sensitivity_retained_log_effect_fractions"] != [
        0.1,
        0.15,
        0.25,
    ]:
        raise RuntimeError("sensitivity scenarios changed")
    expected_simulation = {
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
    }
    if contract["simulation"] != expected_simulation:
        raise RuntimeError("simulation method changed")
    if not contract["authorization"] or any(
        value is not False for value in contract["authorization"].values()
    ):
        raise RuntimeError("power contract grants downstream authority")


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
    effect_model = engine.derive_effect_model(contract)
    primary = engine.simulate_scenario(
        contract,
        effect_model,
        retained_fraction=contract["effect_scenario"][
            "primary_retained_log_effect_fraction"
        ],
    )
    sensitivity = [
        engine.simulate_scenario(
            contract,
            effect_model,
            retained_fraction=fraction,
        )
        for fraction in contract["effect_scenario"][
            "sensitivity_retained_log_effect_fractions"
        ]
    ]
    qualified = bool(primary["power_floor_passes"])
    head = _git("rev-parse", "HEAD")
    result = {
        "schema_version": 1,
        "result_name": "t7b_automatic_depth_fresh_tier_d_v3_power_design_v1",
        "contract_id": contract["contract_id"],
        "source_head": head,
        "published_refs": _published_refs(head),
        "source_sha256": {
            "contract": provenance.file_sha256(CONTRACT),
            "builder": provenance.file_sha256(Path(__file__)),
            "protocol": provenance.file_sha256(PROTOCOL),
            "tests": provenance.file_sha256(TESTS),
            "enumeration": provenance.file_sha256(ENUMERATION),
            "simulation_engine": provenance.file_sha256(Path(engine.__file__)),
            "shipping_policy": provenance.file_sha256(
                ROOT / "benchmarks" / "SHIPPING_POLICY.md"
            ),
        },
        "candidate": contract["candidate"],
        "verified_registry": contract["verified_registry"],
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
        "confirmation_design_freeze_authorized": False,
        "fresh_access_authorized": False,
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
    if result["power_qualified"] != result["primary_scenario"]["power_floor_passes"]:
        raise RuntimeError("power disposition differs from primary scenario")
    forbidden = [
        "confirmation_design_freeze_authorized",
        "fresh_access_authorized",
        "confirmation_run_authorized",
        "candidate_merge_authorized",
        "default_change_authorized",
        "release_authorized",
        "lockbox_access_authorized",
    ]
    if any(result[name] for name in forbidden):
        raise RuntimeError("power result granted forbidden authority")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.resolve() != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("power result output path changed")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"create-only output exists: {args.output}")
    if _git("status", "--porcelain"):
        raise RuntimeError("power design requires a clean source tree")
    head = _git("rev-parse", "HEAD")
    if not _published_refs(head):
        raise RuntimeError("power design source head is not published")

    contract = _load_json(CONTRACT)
    validate_contract(contract)
    result = build(contract)
    validate_result(result)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        f"{result['disposition']}: "
        f"power={result['primary_scenario']['pass_probability']:.6f}, "
        "one-sided Wilson lower="
        f"{result['primary_scenario']['wilson_lower_bound']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
