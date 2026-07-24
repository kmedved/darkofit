"""Contract tests for the prospective Wave-2 M3b attribution."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmarks import analyze_m3b_ensemble_v3 as analyzer
from benchmarks import freeze_m3b_ensemble_v3 as freezer
from benchmarks import paired_evidence_contract as paired
from benchmarks import run_m3b_ensemble_v3 as runner


def _hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _case_manifests():
    return {
        spec["case_id"]: {
            "fingerprints": {
                name: _hex(f"{spec['case_id']}/{name}")
                for name in (
                    "case_sha256",
                    "dataset_sha256",
                    "split_sha256",
                    "weight_sha256",
                )
            },
            "fit_rows": 100,
            "test_rows": 40,
            "primary_rows": 20 if spec["domain"] == "sports" else 40,
            "feature_count": 5,
            "class_count": (
                None
                if spec["task"] == "regression"
                else (2 if spec["task"] == "binary" else 3)
            ),
        }
        for spec in runner.case_specs()
    }


def _contract():
    manifests = _case_manifests()
    return {
        "schema_version": 1,
        "name": runner.CONTRACT_NAME,
        "contract_frozen": True,
        "outcomes_opened": False,
        "paired_execution_contract": paired.CONTRACT_VERSION,
        "threads": runner.THREADS,
        "sources": {"darkofit": "a" * 40},
        "panel_cache": {"bytes": 123, "sha256": "c" * 64},
        "cases": list(runner.case_specs()),
        "case_manifests": manifests,
        "case_fingerprints": {
            case_id: record["fingerprints"] for case_id, record in manifests.items()
        },
        "arms": {arm: runner.arm_config(arm) for arm in runner.ARMS},
        "quality_orders": runner.quality_orders(),
        "decision_rules": runner.decision_rules(),
    }


def _environment(source_root: Path):
    expected = paired._expected_environment(runner.THREADS)
    environment = {name: expected.get(name) for name in paired.CONTRACT_ENV_KEYS}
    environment["NUMBA_CACHE_DIR"] = str(source_root / "numba-cache")
    return environment


def _policy_metadata(arm: str):
    donor = arm in {runner.B2, runner.COMBINED}
    source = "member_policy" if donor else "base"
    return {
        "learning_rate": {
            "base": None,
            "resolved": 0.15 if donor else None,
            "source": source,
        },
        "colsample": {
            "base": None,
            "resolved": 0.85 if donor else None,
            "source": source,
        },
    }


def _ensemble_metadata(spec, arm: str, manifest):
    config = runner.arm_config(arm)
    without_replacement = config["sampling"] == "without_replacement"
    groups = spec["sampling_unit"] == "groups"
    policy = _policy_metadata(arm)
    members = []
    for index in range(runner.MEMBERS):
        sampled_rows = 80 if without_replacement else (120 if groups else 100)
        sampled_unique_rows = 80 if without_replacement else 60
        oob_rows = 20 if without_replacement else 40
        members.append(
            {
                "member": index,
                "seed": index + 10,
                "sampling_attempts": 1,
                "sampled_rows": sampled_rows,
                "sampled_unique_rows": sampled_unique_rows,
                "sampled_indices_sha256": _hex(f"sample/{arm}/{index}"),
                "oob_rows": oob_rows,
                "oob_indices_sha256": _hex(f"oob/{arm}/{index}"),
                "sampled_group_draws": 8 if groups else None,
                "sampled_unique_groups": 8 if groups else None,
                "oob_groups": 2 if groups else None,
                "group_disjoint": True if groups else None,
                "requested_sample_fraction": 0.8 if without_replacement else None,
                "realized_row_fraction": sampled_unique_rows / 100.0,
                "policy_resolutions": copy.deepcopy(policy),
                "constructor_learning_rate": (
                    0.15 if arm in {runner.B2, runner.COMBINED} else None
                ),
                "constructor_colsample": (
                    0.85 if arm in {runner.B2, runner.COMBINED} else None
                ),
                "fitted_thread_count": runner.THREADS,
                "best_iteration": 5,
                "learning_rate": 0.1,
                "stop_reason": "max_iterations",
                "validation_source": "explicit_eval_set",
            }
        )
    return {
        "version": 2,
        "private_prototype": "ensemble_v3_b1_b2",
        "claim_tier": "E",
        "default_changed": False,
        "public_fit_surface": False,
        "sequential": True,
        "member_count": runner.MEMBERS,
        "member_seeds": [member["seed"] for member in members],
        "fit_random_state_seed": runner.RANDOM_STATE,
        "sampling": config["sampling"],
        "sampling_unit": spec["sampling_unit"],
        "sample_fraction": config["sample_fraction"],
        "bootstrap": spec["sampling_unit"],
        "member_policy": config["member_policy"],
        "explicit_user_params": [],
        "policy_resolutions": policy,
        "aggregation": ("mean" if spec["task"] == "regression" else "soft_vote"),
        "oob_early_stopping": True,
        "shared_preprocessing_requested": True,
        "shared_preprocessing": "numeric_target_free",
        "shared_preprocessing_fallback_reason": None,
        "input_row_count": manifest["fit_rows"],
        "input_feature_count": manifest["feature_count"],
        "members": members,
    }


def _row(
    contract,
    source_root: Path,
    spec,
    arm: str,
    *,
    phase: str,
    repeat: int,
    quality_ratios,
):
    manifest = contract["case_manifests"][spec["case_id"]]
    base_loss = 1.0 + list(runner.case_specs()).index(spec) / 100.0
    loss_ratio = quality_ratios.get(arm, 1.0)
    fit_ratio = {
        runner.SINGLE: 0.2,
        runner.CONTROL: 1.0,
        runner.B1: 0.85,
        runner.B2: 1.0,
        runner.COMBINED: 1.0,
    }[arm]
    archive_bytes = 100 if arm == runner.SINGLE else 300
    peak_rss = 1_000 if arm == runner.SINGLE else 1_500
    expected_metrics = (
        ("cold_player_rmse", "held_team_rmse")
        if spec["domain"] == "sports"
        else (
            ("weighted_rmse", "rmse")
            if spec["task"] == "regression"
            else ("weighted_log_loss", "log_loss")
        )
    )
    runtime = {
        "ceiling": runner.THREADS,
        "current": runner.THREADS,
        "threading_layer": "omp",
        "environment": _environment(source_root),
    }
    member_count = 1 if arm == runner.SINGLE else runner.MEMBERS
    return {
        "phase": phase,
        "repeat": repeat,
        "case_id": spec["case_id"],
        "domain": spec["domain"],
        "task": spec["task"],
        "arm": arm,
        "arm_config": runner.arm_config(arm),
        **manifest["fingerprints"],
        "primary_metric": expected_metrics[0],
        "primary_loss": base_loss * loss_ratio,
        "secondary_metric": expected_metrics[1],
        "secondary_loss": base_loss * loss_ratio,
        "fit_rows": manifest["fit_rows"],
        "primary_rows": manifest["primary_rows"],
        "test_rows": manifest["test_rows"],
        "feature_count": manifest["feature_count"],
        "class_count": manifest["class_count"],
        "fit_seconds": 10.0 * fit_ratio,
        "predict_seconds": 1.0,
        "peak_rss_bytes": peak_rss,
        "rss_samples": 10,
        "rss_errors": [],
        "archive_bytes": archive_bytes,
        "prediction_sha256": _hex(f"prediction/{spec['case_id']}/{arm}"),
        "probability_sha256": (
            None
            if spec["task"] == "regression"
            else _hex(f"probability/{spec['case_id']}/{arm}")
        ),
        "safe_roundtrip_exact": True,
        "implementation_path": str(source_root / "darkofit/sklearn_api.py"),
        "fitted_model_metadata": {
            "member_count": member_count,
            "tree_count": 5 * member_count,
            "tree_counts": [5] * member_count,
            "tree_modes": ["lightgbm"],
            "resolved_thread_counts": [runner.THREADS],
            "best_iterations": [5] * member_count,
        },
        "ensemble_metadata": (
            None if arm == runner.SINGLE else _ensemble_metadata(spec, arm, manifest)
        ),
        "oob_member_scores": (None if arm == runner.SINGLE else [1.0] * runner.MEMBERS),
        "runtime_before": copy.deepcopy(runtime),
        "runtime_after": copy.deepcopy(runtime),
        "warnings": [],
        "python": "3.11.0",
        "numpy": "2.2.0",
    }


def _artifact(
    contract,
    contract_path: Path,
    source_root: Path,
    *,
    phase: str,
    quality_ratios,
    eligible=(),
    quality_sha256=None,
    gate_sha256=None,
):
    rows = []
    if phase == "quality":
        for spec in runner.case_specs():
            for arm in contract["quality_orders"][spec["case_id"]]:
                rows.append(
                    _row(
                        contract,
                        source_root,
                        spec,
                        arm,
                        phase=phase,
                        repeat=0,
                        quality_ratios=quality_ratios,
                    )
                )
    elif eligible:
        timing_arms = (runner.SINGLE, runner.CONTROL, *eligible)
        for repeat in contract["decision_rules"]["timing_repeats"]:
            for spec in runner.case_specs():
                for arm in runner.timing_order(spec["case_id"], repeat, timing_arms):
                    rows.append(
                        _row(
                            contract,
                            source_root,
                            spec,
                            arm,
                            phase=phase,
                            repeat=repeat,
                            quality_ratios=quality_ratios,
                        )
                    )
    return {
        "schema_version": 1,
        "name": runner.CONTRACT_NAME,
        "phase": phase,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_path),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "quality_artifact_sha256": quality_sha256,
        "gate_sha256": gate_sha256,
        "source_state": {
            "path": str(source_root),
            "head": contract["sources"]["darkofit"],
            "status": "",
        },
        "harness_state": {
            "path": str(contract_path.parent),
            "head": "b" * 40,
            "status": "",
        },
        "panel_cache": {
            "path": str(source_root / "panel.csv"),
            **contract["panel_cache"],
        },
        "case_fingerprints": contract["case_fingerprints"],
        "rows": rows,
    }


@pytest.fixture
def synthetic_campaign(tmp_path, monkeypatch):
    contract = _contract()
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    source_root = tmp_path / "source"
    monkeypatch.setattr(
        analyzer.runner,
        "load_contract",
        lambda _path: copy.deepcopy(contract),
    )
    return contract, contract_path, source_root


def test_m3b_grid_closes_medium_classification_and_weight_blindspots():
    specs = runner.case_specs()
    assert len(specs) == 13
    assert sum(spec["domain"] == "sports" for spec in specs) == 9
    general = [spec for spec in specs if spec["domain"] == "general"]
    assert {spec["task"] for spec in general} == {
        "regression",
        "binary",
        "multiclass",
    }
    assert {spec["size"] for spec in general} == {"medium"}
    assert {spec["weight_mode"] for spec in general} == {"stress"}


def test_m3b_arms_and_rules_preserve_causal_attribution():
    assert runner.arm_config(runner.B1) == {
        "kind": "private_ensemble_v3",
        "sampling": "without_replacement",
        "sample_fraction": 0.8,
        "member_policy": "none",
    }
    assert runner.arm_config(runner.B2)["sampling"] == "bootstrap"
    assert runner.arm_config(runner.B2)["member_policy"] == "donor_balanced_v1"
    assert runner.arm_config(runner.COMBINED)["sample_fraction"] == 0.8
    rules = runner.decision_rules()
    assert rules["quality"]["worst_primary_at_most"] == 1.03
    assert rules["timing_repeats"] == [1, 2]


def test_m3b_gate_is_conjunctive_and_kills_known_negative(synthetic_campaign):
    contract, contract_path, source_root = synthetic_campaign
    ratios = {runner.B1: 1.0, runner.B2: 1.01, runner.COMBINED: 0.99}
    quality_path = contract_path.parent / "quality.json"
    quality_path.write_text(
        json.dumps(
            _artifact(
                contract,
                contract_path,
                source_root,
                phase="quality",
                quality_ratios=ratios,
            )
        ),
        encoding="utf-8",
    )

    gate = analyzer.build_gate(quality_path, contract_path)

    assert gate["eligible_candidates"] == [runner.B1, runner.COMBINED]
    assert gate["quality_summaries"][runner.B2]["eligible"] is False
    assert (
        gate["quality_summaries"][runner.B2]["checks"]["all_primary_geomean"] is False
    )


def test_m3b_artifact_rejects_reordering_and_escaped_implementation(
    synthetic_campaign,
):
    contract, contract_path, source_root = synthetic_campaign
    artifact = _artifact(
        contract,
        contract_path,
        source_root,
        phase="quality",
        quality_ratios={},
    )
    path = contract_path.parent / "quality.json"
    reordered = copy.deepcopy(artifact)
    reordered["rows"][:2] = reversed(reordered["rows"][:2])
    path.write_text(json.dumps(reordered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete or reordered"):
        analyzer.validate_artifact(path, contract_path, phase="quality")

    escaped = copy.deepcopy(artifact)
    escaped["rows"][0]["implementation_path"] = "/tmp/forged/sklearn_api.py"
    path.write_text(json.dumps(escaped), encoding="utf-8")
    with pytest.raises(RuntimeError, match="escaped source"):
        analyzer.validate_artifact(path, contract_path, phase="quality")


def test_m3b_final_prefers_surviving_combined_arm(synthetic_campaign):
    contract, contract_path, source_root = synthetic_campaign
    ratios = {runner.B1: 1.0, runner.B2: 1.01, runner.COMBINED: 0.99}
    quality_path = contract_path.parent / "quality.json"
    quality_path.write_text(
        json.dumps(
            _artifact(
                contract,
                contract_path,
                source_root,
                phase="quality",
                quality_ratios=ratios,
            )
        ),
        encoding="utf-8",
    )
    gate = analyzer.build_gate(quality_path, contract_path)
    gate_path = contract_path.parent / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    timing_path = contract_path.parent / "timing.json"
    timing_path.write_text(
        json.dumps(
            _artifact(
                contract,
                contract_path,
                source_root,
                phase="timing",
                quality_ratios=ratios,
                eligible=tuple(gate["eligible_candidates"]),
                quality_sha256=hashlib.sha256(quality_path.read_bytes()).hexdigest(),
                gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            )
        ),
        encoding="utf-8",
    )

    result = analyzer.build_final_result(
        quality_path,
        gate_path,
        timing_path,
        contract_path,
    )

    assert result["disposition"] == "continue_private_combined"
    assert result["retained_private_arms"] == [runner.COMBINED]
    assert result["candidates"][runner.B2]["resources"] is None
    assert result["public_or_default_change_authorized"] is False


def test_m3b_timing_artifact_rejects_forged_gate_hash(synthetic_campaign):
    contract, contract_path, source_root = synthetic_campaign
    quality_path = contract_path.parent / "quality.json"
    quality_path.write_text(
        json.dumps(
            _artifact(
                contract,
                contract_path,
                source_root,
                phase="quality",
                quality_ratios={},
            )
        ),
        encoding="utf-8",
    )
    gate = analyzer.build_gate(quality_path, contract_path)
    gate_path = contract_path.parent / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    timing_path = contract_path.parent / "timing.json"
    timing_path.write_text(
        json.dumps(
            _artifact(
                contract,
                contract_path,
                source_root,
                phase="timing",
                quality_ratios={},
                eligible=tuple(gate["eligible_candidates"]),
                quality_sha256=gate["quality_artifact_sha256"],
                gate_sha256="0" * 64,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not bound to its gate"):
        analyzer.validate_artifact(
            timing_path,
            contract_path,
            phase="timing",
            gate=gate,
            gate_path=gate_path,
        )


def test_m3b_no_eligible_candidate_closes_without_timing(synthetic_campaign):
    contract, contract_path, source_root = synthetic_campaign
    ratios = {
        runner.B1: 1.04,
        runner.B2: 1.04,
        runner.COMBINED: 1.04,
    }
    quality_path = contract_path.parent / "quality.json"
    quality_path.write_text(
        json.dumps(
            _artifact(
                contract,
                contract_path,
                source_root,
                phase="quality",
                quality_ratios=ratios,
            )
        ),
        encoding="utf-8",
    )
    gate = analyzer.build_gate(quality_path, contract_path)
    assert gate["eligible_candidates"] == []
    gate_path = contract_path.parent / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    result = analyzer.build_final_result(
        quality_path,
        gate_path,
        None,
        contract_path,
    )

    assert result["disposition"] == "close_b1_b2_preserve_existing_opt_in"
    assert result["timing_artifact_sha256"] is None


def test_m3b_terminal_failure_artifact_discards_partial_rows(tmp_path):
    state = {"path": str(tmp_path), "head": "a" * 40, "status": ""}
    artifact = runner.terminal_failure_artifact(
        phase="quality",
        contract_path=tmp_path / "contract.json",
        contract_sha256="b" * 64,
        source_before=state,
        source_after=state,
        harness_before=state,
        harness_after=state,
        case_fingerprints_value={"case": {"case_sha256": "c" * 64}},
        completed_rows=7,
        error=RuntimeError("worker failed"),
    )

    assert artifact["status"] == "failed"
    assert artifact["rows"] is None
    assert artifact["completed_rows_discarded"] == 7
    assert artifact["error"] == {
        "type": "RuntimeError",
        "message": "worker failed",
    }


def test_m3b_freezer_binds_source_cases_and_nonshipping_claims():
    if (
        not runner.DEFAULT_PANEL_CACHE.is_file()
        or runner.DEFAULT_PANEL_CACHE.is_symlink()
    ):
        pytest.skip(
            "historical M3b freezer requires its local sports-panel cache"
        )
    contract = freezer.build_contract()

    assert contract["contract_frozen"] is True
    assert contract["outcomes_opened"] is False
    assert contract["sources"]["darkofit"] == runner.git_state(runner.ROOT)["head"]
    assert len(contract["case_manifests"]) == 13
    assert contract["case_manifests"]["general_numeric_binary"]["class_count"] == 2
    assert contract["claims"]["public_or_default_change_authorized"] is False
    assert contract["claims"]["lockbox_access_authorized"] is False
    assert set(contract["bound_files"]) == set(freezer.BOUND_PATHS)
