from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from benchmarks import analyze_panel3_confirmation as confirmation_analyzer
from benchmarks import build_panel3_power_design as power
from benchmarks import freeze_panel3_cross_power_calibration as calibration_freeze
from benchmarks import panel3_registry_common as common


ROOT = Path(__file__).resolve().parents[1]


def _profiles(ratio: float = 0.99):
    contract = power.load_contract()
    result = {}
    for candidate in power.CANDIDATES:
        result[candidate] = [
            {
                "dataset_name": row["dataset_name"],
                "task_id": row["task_id"],
                "stratum": row["stratum"],
                "t5_size_gate_applicable_coordinates": row[
                    "expected_t5_size_gate_applicable_coordinates"
                ],
                "coordinate_ratios": (
                    [1.0, 1.0, 1.0]
                    if candidate == "t5_composite_policy"
                    and row[
                        "expected_t5_size_gate_applicable_coordinates"
                    ]
                    == 0
                    else [ratio, ratio, ratio]
                ),
            }
            for row in contract["calibration"]["tasks"]
        ]
    return result


def _summary():
    contract = power.load_contract()
    profiles = _profiles()
    fixed = {}
    candidate_results = {}
    for candidate, rows in profiles.items():
        fixed[candidate] = [
            {
                "source": (
                    "spent_tabarena_13x3_exact_policy_complete_census"
                ),
                "dataset_name": row["dataset_name"],
                "task_id": row["task_id"],
                "ratio": math.prod(row["coordinate_ratios"]) ** (1.0 / 3.0),
                "coordinate_ratios": row["coordinate_ratios"],
                "t5_size_gate_applicable_coordinates": row[
                    "t5_size_gate_applicable_coordinates"
                ],
                "engaged_coordinates": (
                    row["t5_size_gate_applicable_coordinates"]
                    if candidate == "t5_composite_policy"
                    else 3
                ),
            }
            for row in rows
        ]
        candidate_results[candidate] = {
            "coordinate_count": 39,
            "dataset_count": 13,
        }
    return common.bind_artifact_sha256(
        {
            "schema_version": 1,
            "name": contract["calibration"]["summary_name"],
            "created_at": "2026-07-18T00:00:00+00:00",
            "raw_path": str(power.DEFAULT_RAW.relative_to(ROOT)),
            "raw_file_sha256": "a" * 64,
            "raw_artifact_sha256": "b" * 64,
            "source_freeze_sha256": "c" * 64,
            "estimand": "exact_candidate/current_default",
            "candidate_results": candidate_results,
            "fixed_panel_power_inputs": fixed,
            "complete_unfiltered_coordinate_census": True,
            "ties_and_losses_preserved": True,
            "development_only": True,
            "may_inform_separately_frozen_power_design": True,
            "independent_confirmation": False,
            "panel3_authorized": False,
            "default_promotion_authorized": False,
            "product_claim_authorized": False,
        },
        "summary_sha256",
    )


def _decision_artifact(*, retained=None):
    contract = power.load_contract()
    panel = contract["prospective_panel"]
    simulation = contract["simulation"]
    retained = list(power.CANDIDATES) if retained is None else list(retained)
    source_sha256 = {
        relative: "c" * 64
        for relative in power.PANEL3_V1_SOURCE_RELATIVE_PATHS
    }
    source_sha256[str(power.CONTRACT.relative_to(ROOT))] = (
        power.PANEL3_V1_POWER_CONTRACT_SHA256
    )
    source_sha256[
        str(common.CANDIDATE_CONTRACT.relative_to(ROOT))
    ] = power.PANEL3_V1_CANDIDATE_CONTRACT_SHA256
    source_sha256[
        str(common.ENVIRONMENT_CONTRACT.relative_to(ROOT))
    ] = power.PANEL3_V1_ENVIRONMENT_CONTRACT_SHA256

    def result(candidate, *, passes, percentile, confidence):
        outer = simulation["outer_panel_simulations"]
        passing = 4_500 if passes else 3_000
        probability = passing / outer
        lower = power._wilson_lower(passing, outer, confidence)
        applicability = [
            row["t5_size_gate_applicability"] for row in panel["slots"]
        ]
        return {
            "candidate": candidate,
            "outer_panel_simulations": outer,
            "outer_seed": simulation["outer_seed"],
            "complete_triplets_preserved": True,
            "prospective_tasks": 12,
            "prospective_coordinates": 36,
            "stratum_composition": {
                stratum: sum(
                    row["stratum"] == stratum for row in panel["slots"]
                )
                for stratum in power.STRATA
            },
            "fixed_t5_noop_slots": (
                sum(not any(vector) for vector in applicability)
                if candidate == "t5_composite_policy"
                else 0
            ),
            "fixed_t5_noop_coordinates": (
                sum(not value for vector in applicability for value in vector)
                if candidate == "t5_composite_policy"
                else 0
            ),
            "hierarchical_bootstrap": {
                "seed": simulation["hierarchical_bootstrap_seed"],
                "replicates": simulation[
                    "hierarchical_bootstrap_replicates"
                ],
                "batch": simulation["hierarchical_bootstrap_batch"],
                "hierarchy": (
                    "lineage_then_three_coordinates_within_lineage"
                ),
                "percentile": percentile,
                "numpy_percentile_method": "linear",
            },
            "component_passing_simulations": {
                "point": passing,
                "hierarchical_bootstrap_upper": passing,
                "leave_one_favorable_out": passing,
                "worst_dataset": passing,
            },
            "passing_simulations": passing,
            "pass_probability": probability,
            "wilson_one_sided_confidence": confidence,
            "wilson_lower_bound": lower,
            "minimum_required_probability": simulation["power_floor"],
            "point_estimate_passes": probability >= simulation["power_floor"],
            "wilson_lower_bound_passes": lower >= simulation["power_floor"],
            "passes": passes,
            "statistical_gates_only": True,
            "operational_gates_remain_required_at_confirmation": True,
        }

    initial = {
        candidate: result(
            candidate,
            passes=candidate in retained,
            percentile=simulation["initial_bootstrap_percentile"],
            confidence=simulation["initial_power_wilson_confidence"],
        )
        for candidate in power.CANDIDATES
    }
    singleton = None
    if len(retained) == 1:
        singleton = result(
            retained[0],
            passes=True,
            percentile=simulation["singleton_bootstrap_percentile"],
            confidence=simulation["singleton_power_wilson_confidence"],
        )
    checks = {
        "calibration_summary_valid": True,
        "calibration_raw_and_spool_valid": True,
        "complete_39_coordinate_triplets_preserved": True,
        "frozen_4_4_4_composition_preserved": True,
        "known_t5_size_gate_applicability_preserved": True,
        "minimum_stratum_support_preserved": True,
        "candidate_retention_rule_applied_without_discretion": True,
        "design_sources_bound_at_calibration_h1": True,
        "power_decision_computed_from_bound_calibration": True,
        "at_least_one_candidate_meets_power_floor": bool(retained),
    }
    return common.bind_artifact_sha256(
        {
            "schema_version": 1,
            "name": "darkofit_panel3_power_design_decision_v1",
            "created_at": "2026-07-18T00:00:00+00:00",
            "source_head": "a" * 40,
            "decision_execution_head": "b" * 40,
            "source_sha256": source_sha256,
            "contract": {
                "path": "benchmarks/panel3_power_design_contract.json",
                "file_sha256": power.PANEL3_V1_POWER_CONTRACT_SHA256,
                "contract_name": (
                    "darkofit_panel3_authorization_power_design_v1"
                ),
            },
            "calibration": {
                "summary_path": (
                    "benchmarks/"
                    "panel3_cross_power_calibration_summary.json"
                ),
                "summary_file_sha256": "e" * 64,
                "summary_sha256": "f" * 64,
                "raw_path": (
                    "benchmarks/panel3_cross_power_calibration_raw.json"
                ),
                "raw_file_sha256": "1" * 64,
                "raw_artifact_sha256": "2" * 64,
                "source_freeze_sha256": "3" * 64,
            },
            "runtime": copy.deepcopy(
                power.PANEL3_V1_RUNTIME_CONTRACT
            ),
            "mapping": {
                "calibration_tasks": contract["calibration"]["tasks"],
                "exchangeability": contract["exchangeability"],
            },
            "pre_h1_target_statistic_exclusions": [
                dict(row)
                for row in common.PRE_H1_TARGET_STATISTIC_EXCLUSIONS
            ],
            "prospective_panel": panel,
            "simulation": simulation,
            "initial_bonferroni_screen": initial,
            "singleton_fallback": singleton,
            "retained_candidates": retained,
            "candidate_count": len(retained),
            "familywise_one_sided_alpha": 0.05,
            "per_candidate_one_sided_alpha": (
                0.025 if len(retained) == 2 else 0.05
            ),
            "bootstrap_percentile": (
                97.5 if len(retained) == 2 else 95.0
            ),
            "power_floor": 0.8,
            "owner_decision_statement": (
                power._owner_decision_statement(
                    initial_bonferroni_screen=initial,
                    singleton_fallback=singleton,
                    retained_candidates=retained,
                    power_floor=0.8,
                )
            ),
            "checks": checks,
            "target_preflight_authorized": bool(retained),
            "registry_build_authorized": False,
            "confirmation_run_authorized": False,
            "default_promotion_authorized": False,
            "product_claim_authorized": False,
        },
        "decision_sha256",
    )


def _rebind_decision(artifact):
    artifact.pop("decision_sha256")
    return common.bind_artifact_sha256(artifact, "decision_sha256")


def test_contract_freezes_strata_triplets_and_four_t5_noops():
    contract = power.load_contract()

    assert [row["stratum"] for row in contract["calibration"]["tasks"]].count(
        "smooth_numeric"
    ) == 4
    assert [row["stratum"] for row in contract["calibration"]["tasks"]].count(
        "mixed_categorical"
    ) == 4
    assert [row["stratum"] for row in contract["calibration"]["tasks"]].count(
        "applied_noisy"
    ) == 5
    slots = contract["prospective_panel"]["slots"]
    assert len(slots) == 12
    assert {
        stratum: sum(row["stratum"] == stratum for row in slots)
        for stratum in power.STRATA
    } == {stratum: 4 for stratum in power.STRATA}
    assert sum(
        row["t5_size_gate_applicability"] == [False, False, False]
        for row in slots
    ) == 7
    assert all(
        len(row["t5_size_gate_applicability"]) == 3 for row in slots
    )
    assert (
        contract["exchangeability"]["sampling_unit"]
        == "complete_three_coordinate_dataset_triplet"
    )


def test_calibration_h1_binds_every_prospective_source_and_test():
    frozen = set(calibration_freeze.source_paths())
    required = {
        str(path.relative_to(ROOT)) for path in power.SOURCE_PATHS
    }

    assert required == frozen
    assert {
        "tests/conftest.py",
        "tests/test_campaign_partition.py",
        "benchmarks/preflight_panel3_registry.py",
        "benchmarks/build_panel3_registry.py",
        "benchmarks/panel3_registry_protocol.md",
        "benchmarks/confirmation_target_preflight.py",
        "tests/test_confirmation_target_preflight.py",
        "tests/test_panel3_power_design.py",
        "tests/test_panel3_registry.py",
        "tests/test_panel3_execution.py",
    } <= frozen
    assert {
        str(path.relative_to(ROOT))
        for path in (ROOT / "darkofit").rglob("*.py")
    } <= frozen


def test_power_design_cli_rejects_noncanonical_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        power,
        "build",
        lambda **_kwargs: pytest.fail("build must not run"),
    )

    with pytest.raises(RuntimeError, match="decision path changed"):
        power.main(["--summary", str(tmp_path / "summary.json")])
    with pytest.raises(RuntimeError, match="decision path changed"):
        power.main(["--raw", str(tmp_path / "raw.json")])
    with pytest.raises(RuntimeError, match="decision path changed"):
        power.main(["--output", str(tmp_path / "decision.json")])


def _minimal_main_decision():
    contract = power.load_contract()
    return {
        "decision_execution_head": "a" * 40,
        "decision_sha256": "b" * 64,
        "retained_candidates": ["guarded_cross_features_policy"],
        "target_preflight_authorized": True,
        "owner_decision_statement": (
            "Panel 3 decision-stage simulated pass probabilities were "
            "guarded_cross_features_policy 90.00% "
            "(one-sided Wilson lower bound 85.00%), against the required "
            "80.00%; therefore GO."
        ),
        "calibration": {
            "summary_file_sha256": "1" * 64,
            "summary_sha256": "2" * 64,
            "raw_file_sha256": "3" * 64,
            "raw_artifact_sha256": "4" * 64,
        },
        "contract": {
            "path": str(power.CONTRACT.relative_to(ROOT)),
            "file_sha256": common.sha256_file(power.CONTRACT),
            "contract_name": contract["contract_name"],
        },
        "source_sha256": {
            str(common.CANDIDATE_CONTRACT.relative_to(ROOT)): (
                common.sha256_file(common.CANDIDATE_CONTRACT)
            )
        },
    }


def test_power_design_cli_rechecks_source_head_before_publication(
    monkeypatch,
):
    monkeypatch.setattr(power, "build", lambda **_kwargs: _minimal_main_decision())
    monkeypatch.setattr(
        power,
        "_require_clean_committed_sources",
        lambda: "c" * 40,
    )
    monkeypatch.setattr(
        common,
        "atomic_create",
        lambda *_args, **_kwargs: pytest.fail("decision was published"),
    )

    with pytest.raises(RuntimeError, match="source changed"):
        power.main([])


def test_power_design_cli_rechecks_evidence_snapshot_before_publication(
    monkeypatch,
):
    contract = power.load_contract()
    candidate_contract = common.load_json(common.CANDIDATE_CONTRACT)
    monkeypatch.setattr(power, "build", lambda **_kwargs: _minimal_main_decision())
    monkeypatch.setattr(
        power,
        "_require_clean_committed_sources",
        lambda: "a" * 40,
    )

    def changed_snapshot(path):
        if path == power.DEFAULT_SUMMARY:
            return {"summary_sha256": "2" * 64}, "9" * 64
        if path == power.DEFAULT_RAW:
            return {"raw_artifact_sha256": "4" * 64}, "3" * 64
        if path == power.CONTRACT:
            return contract, common.sha256_file(power.CONTRACT)
        if path == common.CANDIDATE_CONTRACT:
            return (
                candidate_contract,
                common.sha256_file(common.CANDIDATE_CONTRACT),
            )
        raise AssertionError(path)

    monkeypatch.setattr(common, "secure_load_json", changed_snapshot)
    monkeypatch.setattr(
        common,
        "atomic_create",
        lambda *_args, **_kwargs: pytest.fail("decision was published"),
    )

    with pytest.raises(RuntimeError, match="evidence changed"):
        power.main([])


def test_power_design_cli_revalidates_spool_chain_before_publication(
    monkeypatch,
):
    contract = power.load_contract()
    candidate_contract = common.load_json(common.CANDIDATE_CONTRACT)
    summary = {"summary_sha256": "2" * 64}
    raw = {"raw_artifact_sha256": "4" * 64}
    expected_contract = contract
    monkeypatch.setattr(power, "build", lambda **_kwargs: _minimal_main_decision())
    monkeypatch.setattr(
        power,
        "_require_clean_committed_sources",
        lambda: "a" * 40,
    )

    def unchanged_snapshot(path):
        if path == power.DEFAULT_SUMMARY:
            return summary, "1" * 64
        if path == power.DEFAULT_RAW:
            return raw, "3" * 64
        if path == power.CONTRACT:
            return contract, common.sha256_file(power.CONTRACT)
        if path == common.CANDIDATE_CONTRACT:
            return (
                candidate_contract,
                common.sha256_file(common.CANDIDATE_CONTRACT),
            )
        raise AssertionError(path)

    def changed_spool_chain(
        observed_summary,
        *,
        summary_path,
        raw_path,
        verify_raw,
        contract,
    ):
        assert observed_summary is summary
        assert summary_path == power.DEFAULT_SUMMARY
        assert raw_path == power.DEFAULT_RAW
        assert verify_raw is True
        assert contract is expected_contract
        raise RuntimeError("calibration spool record changed")

    monkeypatch.setattr(common, "secure_load_json", unchanged_snapshot)
    monkeypatch.setattr(power, "validate_calibration", changed_spool_chain)
    monkeypatch.setattr(
        common,
        "atomic_create",
        lambda *_args, **_kwargs: pytest.fail("decision was published"),
    )

    with pytest.raises(RuntimeError, match="spool record changed"):
        power.main([])


def test_power_design_cli_publishes_exact_owner_decision(
    monkeypatch,
    capsys,
):
    artifact = _minimal_main_decision()
    contract = power.load_contract()
    candidate_contract = common.load_json(common.CANDIDATE_CONTRACT)
    summary = {"summary_sha256": "2" * 64}
    raw = {"raw_artifact_sha256": "4" * 64}
    published = []
    monkeypatch.setattr(power, "build", lambda **_kwargs: artifact)
    monkeypatch.setattr(
        power,
        "_require_clean_committed_sources",
        lambda: "a" * 40,
    )

    def unchanged_snapshot(path):
        if path == power.DEFAULT_SUMMARY:
            return summary, "1" * 64
        if path == power.DEFAULT_RAW:
            return raw, "3" * 64
        if path == power.CONTRACT:
            return contract, common.sha256_file(power.CONTRACT)
        if path == common.CANDIDATE_CONTRACT:
            return (
                candidate_contract,
                common.sha256_file(common.CANDIDATE_CONTRACT),
            )
        raise AssertionError(path)

    monkeypatch.setattr(common, "secure_load_json", unchanged_snapshot)
    monkeypatch.setattr(power, "validate_calibration", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        common,
        "atomic_create",
        lambda path, encoded: published.append((path, encoded)),
    )

    assert power.main([]) == 0
    assert published and published[0][0] == power.DEFAULT_OUTPUT
    assert (
        capsys.readouterr().out.splitlines()[0]
        == artifact["owner_decision_statement"]
    )


def test_summary_validation_preserves_every_ordered_triplet():
    summary = _summary()

    profiles = power.validate_calibration(summary, verify_raw=False)

    assert set(profiles) == set(power.CANDIDATES)
    assert all(len(rows) == 13 for rows in profiles.values())
    assert sum(len(row["coordinate_ratios"]) for rows in profiles.values() for row in rows) == 78


def test_summary_rejects_checkout_specific_absolute_raw_path():
    summary = _summary()
    summary["raw_path"] = (
        "/tmp/another-checkout/benchmarks/"
        "panel3_cross_power_calibration_raw.json"
    )
    summary.pop("summary_sha256")
    common.bind_artifact_sha256(summary, "summary_sha256")

    with pytest.raises(RuntimeError, match="summary boundary changed"):
        power.validate_calibration(summary, verify_raw=False)


def test_summary_rejects_coordinate_aggregation_or_reordering():
    summary = _summary()
    changed = copy.deepcopy(summary)
    changed["fixed_panel_power_inputs"]["t5_composite_policy"][0][
        "coordinate_ratios"
    ] = [1.0, 0.99]
    changed.pop("summary_sha256")
    common.bind_artifact_sha256(changed, "summary_sha256")

    with pytest.raises(RuntimeError, match="exactly three coordinates"):
        power.validate_calibration(changed, verify_raw=False)

    changed = copy.deepcopy(summary)
    rows = changed["fixed_panel_power_inputs"][
        "guarded_cross_features_policy"
    ]
    rows[0], rows[1] = rows[1], rows[0]
    changed.pop("summary_sha256")
    common.bind_artifact_sha256(changed, "summary_sha256")
    with pytest.raises(RuntimeError, match="census changed"):
        power.validate_calibration(changed, verify_raw=False)


def test_power_hierarchical_quantile_matches_confirmation_analyzer():
    ratios = {
        f"lineage-{index:02d}": [0.97, 1.0, 1.02]
        for index in range(12)
    }
    logs = power.np.log(power.np.asarray(list(ratios.values())))
    draws = power._bootstrap_draws(seed=19, replicates=500, batch=500)

    observed = power._hierarchical_upper(
        logs,
        draws,
        percentile=97.5,
    )
    expected = confirmation_analyzer.hierarchical_bootstrap_upper(
        ratios,
        seed=19,
        replicates=500,
        percentile=97.5,
    )

    assert observed == expected


def test_batched_bootstrap_weights_equal_simple_reference():
    ratios = {
        f"lineage-{index:02d}": [
            0.97 + index / 10_000,
            1.0,
            1.02 - index / 10_000,
        ]
        for index in range(12)
    }
    logs = power.np.log(power.np.asarray(list(ratios.values())))
    draws = power._bootstrap_draws(seed=29, replicates=400, batch=400)
    weights = power._bootstrap_weight_matrix(draws)

    reference = power._hierarchical_upper(
        logs,
        draws,
        percentile=97.5,
    )
    batched = power._hierarchical_uppers(
        power.np.stack([logs, logs]),
        weights,
        percentile=97.5,
        panel_batch=1,
    )

    power.np.testing.assert_allclose(
        batched,
        [reference, reference],
        rtol=2e-15,
        atol=2e-15,
    )


def test_batched_bootstrap_is_rounding_equivalent_on_random_panel():
    logs = power.np.random.default_rng(15).normal(
        0.0,
        0.5,
        size=(12, 3),
    )
    draws = power._bootstrap_draws(
        seed=20_260_717,
        replicates=10_003,
        batch=10_003,
    )
    weights = power._bootstrap_weight_matrix(draws)

    direct = power._hierarchical_upper(
        logs,
        draws,
        percentile=97.5,
    )
    batched = power._hierarchical_uppers(
        logs[None, ...],
        weights,
        percentile=97.5,
        panel_batch=1,
    )[0]

    assert abs(direct - batched) <= 4 * power.np.spacing(direct)
    assert bool(direct <= 1.002) is bool(batched <= 1.002)


def test_t5_simulation_fixes_noop_slots_and_uses_hierarchical_gate():
    result = power.simulate_candidate_power(
        _profiles()["t5_composite_policy"],
        candidate="t5_composite_policy",
        percentile=97.5,
        wilson_confidence=0.975,
        outer_simulations=20,
        bootstrap_replicates=200,
    )

    assert result["complete_triplets_preserved"] is True
    assert result["prospective_coordinates"] == 36
    assert result["stratum_composition"] == {
        stratum: 4 for stratum in power.STRATA
    }
    assert result["fixed_t5_noop_slots"] == 7
    assert result["fixed_t5_noop_coordinates"] == 21
    assert result["hierarchical_bootstrap"]["hierarchy"] == (
        "lineage_then_three_coordinates_within_lineage"
    )
    assert result["hierarchical_bootstrap"]["percentile"] == 97.5


def _fake_power(candidate, *, percentile, passes):
    return {
        "candidate": candidate,
        "hierarchical_bootstrap": {"percentile": percentile},
        "pass_probability": 0.9 if passes else 0.7,
        "wilson_lower_bound": 0.85 if passes else 0.65,
        "passes": passes,
    }


def test_retention_keeps_two_without_joint_probability(monkeypatch):
    def fake(_profiles, *, candidate, percentile, **_kwargs):
        return _fake_power(candidate, percentile=percentile, passes=True)

    monkeypatch.setattr(power, "simulate_candidate_power", fake)
    result = power.decide_retention(_profiles())

    assert result["retained_candidates"] == list(power.CANDIDATES)
    assert result["per_candidate_one_sided_alpha"] == 0.025
    assert result["bootstrap_percentile"] == 97.5
    assert "joint" not in result
    assert result["owner_decision_statement"] == (
        "Panel 3 decision-stage simulated pass probabilities were "
        "t5_composite_policy 90.00% "
        "(one-sided Wilson lower bound 85.00%) and "
        "guarded_cross_features_policy 90.00% "
        "(one-sided Wilson lower bound 85.00%), against the required "
        "80.00%; therefore GO."
    )


def test_retention_recomputes_only_existing_single_survivor(monkeypatch):
    calls = []

    def fake(_profiles, *, candidate, percentile, **_kwargs):
        calls.append((candidate, percentile))
        passes = candidate == "t5_composite_policy"
        return _fake_power(candidate, percentile=percentile, passes=passes)

    monkeypatch.setattr(power, "simulate_candidate_power", fake)
    result = power.decide_retention(_profiles())

    assert calls == [
        ("t5_composite_policy", 97.5),
        ("guarded_cross_features_policy", 97.5),
        ("t5_composite_policy", 95.0),
    ]
    assert result["retained_candidates"] == ["t5_composite_policy"]
    assert result["per_candidate_one_sided_alpha"] == 0.05
    assert result["bootstrap_percentile"] == 95.0
    assert result["owner_decision_statement"].endswith("therefore GO.")


def test_retention_does_not_rescue_zero_bonferroni_survivors(monkeypatch):
    calls = []

    def fake(_profiles, *, candidate, percentile, **_kwargs):
        calls.append((candidate, percentile))
        return _fake_power(candidate, percentile=percentile, passes=False)

    monkeypatch.setattr(power, "simulate_candidate_power", fake)
    result = power.decide_retention(_profiles())

    assert calls == [
        ("t5_composite_policy", 97.5),
        ("guarded_cross_features_policy", 97.5),
    ]
    assert result["retained_candidates"] == []
    assert result["passes"] is False
    assert result["owner_decision_statement"].endswith(
        "therefore NO-GO."
    )


def test_historical_decision_rejects_owner_statement_tampering():
    artifact = _decision_artifact()

    for mutation in ("missing", "tampered"):
        changed = copy.deepcopy(artifact)
        if mutation == "missing":
            changed.pop("owner_decision_statement")
        else:
            changed["owner_decision_statement"] = (
                changed["owner_decision_statement"].replace("GO.", "NO-GO.")
            )
        _rebind_decision(changed)

        with pytest.raises(
            RuntimeError,
            match=(
                "decision contract changed"
                if mutation == "missing"
                else "owner-facing power decision changed"
            ),
        ):
            power.validate_decision(
                changed,
                require_current_sources=False,
                recompute=False,
            )


def test_missing_decision_blocks_target_access(monkeypatch, tmp_path):
    monkeypatch.setattr(power, "DEFAULT_OUTPUT", tmp_path / "absent.json")

    with pytest.raises(RuntimeError, match="target access is blocked"):
        power.load_decision()


def test_historical_decision_validation_is_embedded_only(monkeypatch):
    artifact = _decision_artifact(
        retained=["guarded_cross_features_policy"]
    )
    monkeypatch.setattr(
        power,
        "load_contract",
        lambda: (_ for _ in ()).throw(
            AssertionError("historical validation opened live contract")
        ),
    )
    monkeypatch.setattr(
        common,
        "PRE_H1_TARGET_STATISTIC_EXCLUSIONS",
        [
            {
                **artifact["pre_h1_target_statistic_exclusions"][0],
                "replacement_task_id": 999_999,
            }
        ],
    )

    observed = power.validate_decision(
        artifact,
        require_current_sources=False,
        recompute=False,
    )

    assert observed is artifact
    with pytest.raises(RuntimeError, match="ledger changed"):
        power._validate_pre_h1_target_exclusion_slots(
            artifact["prospective_panel"],
            artifact["pre_h1_target_statistic_exclusions"],
            require_current_sources=True,
        )


def test_singleton_decision_rejects_bonferroni_alpha_mutation():
    artifact = _decision_artifact(
        retained=["guarded_cross_features_policy"]
    )
    changed = copy.deepcopy(artifact)
    changed["per_candidate_one_sided_alpha"] = 0.025
    changed.pop("decision_sha256")
    common.bind_artifact_sha256(changed, "decision_sha256")

    with pytest.raises(RuntimeError, match="retention decision changed"):
        power.validate_decision(
            changed,
            require_current_sources=False,
            recompute=False,
        )


@pytest.mark.parametrize("fallback_kind", ["missing", "wrong_candidate"])
def test_historical_decision_requires_exact_bonferroni_survivor_fallback(
    fallback_kind,
):
    survivor = "guarded_cross_features_policy"
    changed = _decision_artifact(retained=[survivor])
    changed["singleton_fallback"] = (
        None
        if fallback_kind == "missing"
        else _decision_artifact(
            retained=["t5_composite_policy"]
        )["singleton_fallback"]
    )
    changed["retained_candidates"] = []
    changed["candidate_count"] = 0
    changed["per_candidate_one_sided_alpha"] = None
    changed["bootstrap_percentile"] = None
    changed["checks"]["at_least_one_candidate_meets_power_floor"] = False
    changed["target_preflight_authorized"] = False
    changed.pop("decision_sha256")
    common.bind_artifact_sha256(changed, "decision_sha256")

    with pytest.raises(
        RuntimeError,
        match="does not match the Bonferroni survivor",
    ):
        power.validate_decision(
            changed,
            require_current_sources=False,
            recompute=False,
        )


def _mutate_power_arithmetic(artifact):
    artifact["initial_bonferroni_screen"]["t5_composite_policy"][
        "passing_simulations"
    ] -= 1


def _mutate_runtime_schema(artifact):
    artifact["runtime"]["packages"].pop("numpy")


def _mutate_quality_gate(artifact):
    artifact["simulation"]["quality_gates"] = None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_mutate_power_arithmetic, "power arithmetic changed"),
        (_mutate_runtime_schema, "runtime binding changed"),
        (
            _mutate_quality_gate,
            "historical power-design structure changed",
        ),
    ],
    ids=("power-arithmetic", "runtime-schema", "quality-gate"),
)
def test_historical_decision_rejects_bound_mutation(mutation, message):
    changed = _decision_artifact()
    mutation(changed)
    _rebind_decision(changed)

    with pytest.raises(RuntimeError, match=message):
        power.validate_decision(
            changed,
            require_current_sources=False,
            recompute=False,
        )


def test_historical_runtime_schema_is_independent_of_live_runner(
    monkeypatch,
):
    artifact = _decision_artifact()
    monkeypatch.setattr(
        power.confirmation,
        "RUNTIME_PACKAGE_NAMES",
        ("future-runtime-package",),
    )

    power.validate_decision(
        artifact,
        require_current_sources=False,
        recompute=False,
    )
    with pytest.raises(RuntimeError, match="runner runtime package schema changed"):
        power._validate_decision_runtime(
            artifact["runtime"],
            require_current_sources=True,
        )


def test_v1_archival_constants_match_the_frozen_contract():
    contract = power.load_contract()

    assert contract["calibration"]["tasks"] == (
        power.PANEL3_V1_CALIBRATION_TASKS
    )
    assert contract["exchangeability"] == (
        power.PANEL3_V1_EXCHANGEABILITY
    )
    assert contract["prospective_panel"] == (
        power.PANEL3_V1_PROSPECTIVE_PANEL
    )
    assert contract["simulation"] == power.PANEL3_V1_SIMULATION
    assert tuple(
        str(path.relative_to(ROOT)) for path in power.SOURCE_PATHS
    ) == power.PANEL3_V1_SOURCE_RELATIVE_PATHS


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: artifact["runtime"].__setitem__(
            "python_version", "3.12.99"
        ),
        lambda artifact: artifact.__setitem__("created_at", "not-a-date"),
        lambda artifact: artifact["source_sha256"].pop(
            "benchmarks/panel3_environment_contract.json"
        ),
        lambda artifact: artifact["mapping"].__setitem__(
            "calibration_tasks", [None] * 13
        ),
        lambda artifact: artifact["prospective_panel"]["slots"][
            0
        ].__setitem__("stratum", "smooth_numeric"),
    ],
)
def test_historical_decision_rejects_archival_provenance_drift(mutation):
    artifact = _decision_artifact()
    mutation(artifact)
    artifact.pop("decision_sha256")
    common.bind_artifact_sha256(artifact, "decision_sha256")

    with pytest.raises(RuntimeError):
        power.validate_decision(
            artifact,
            require_current_sources=False,
            recompute=False,
        )


def test_historical_recompute_requires_bound_calibration_files():
    with pytest.raises(
        RuntimeError,
        match="historical recomputation requires",
    ):
        power.validate_decision(
            _decision_artifact(),
            require_current_sources=False,
            recompute=True,
        )


def test_calibration_validation_consumes_captured_contract(
    monkeypatch,
):
    contract = power.load_contract()
    summary = _summary()
    monkeypatch.setattr(
        power,
        "load_contract",
        lambda: pytest.fail("captured contract was reopened"),
    )

    power.validate_calibration(
        summary,
        verify_raw=False,
        contract=contract,
    )


def test_wilson_lower_bound_is_conservative():
    lower = power._wilson_lower(800, 1000, 0.975)

    assert lower < 0.8
    assert power._wilson_lower(1000, 1000, 0.975) < 1.0
