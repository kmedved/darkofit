from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import analyze_t7b_automatic_depth_sports_v1 as analyzer
from benchmarks import freeze_t7b_automatic_depth_sports_v1 as freezer
from benchmarks import run_t7b_automatic_depth_sports_v1 as runner


def _hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _manifests():
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
            "fit_rows": 200,
            "test_rows": 100,
            "primary_rows": 80,
            "feature_count": 15,
            "class_count": None,
        }
        for spec in runner.case_specs()
    }


def _contract():
    return {
        "schema_version": 1,
        "contract_id": runner.CONTRACT_ID,
        "contract_frozen": True,
        "outcomes_opened": False,
        "sources": {
            runner.CONTROL: {
                "head": runner.CONTROL_HEAD,
                "tree": runner.CONTROL_TREE,
            },
            runner.CANDIDATE: {
                "head": runner.CANDIDATE_HEAD,
                "tree": runner.CANDIDATE_TREE,
            },
            "harness": "h" * 40,
        },
        "cases": list(runner.case_specs()),
        "case_manifests": _manifests(),
        "quality_orders": runner.quality_orders(),
        "execution": runner.execution_spec(),
        "decision_rules": runner.decision_rules(),
        "claims": runner.claim_spec(),
    }


def _structure(arm: str):
    if arm == runner.CANDIDATE:
        return {
            "resolved": {
                "depth": {"input": None, "resolved": 4, "source": "auto"}
            },
            "candidates": {
                "depth": {
                    "rule": runner.DEPTH_RULE,
                    "n_eff": 170.0,
                    "input_feature_count": 15,
                    "effective_rows_per_feature": 170.0 / 15.0,
                    "low_threshold": 100.0,
                    "high_threshold": 2_500.0,
                    "branch": "low_density",
                }
            },
        }
    return {
        "resolved": {
            "depth": {"input": None, "resolved": 6, "source": "default"}
        },
        "candidates": {},
    }


def _row(contract, spec, arm: str, *, primary_ratio: float, secondary_ratio: float):
    manifest = contract["case_manifests"][spec["case_id"]]
    control_primary = 2.0 + (spec["season"] - 2014) / 10.0
    control_secondary = control_primary * 0.95
    candidate = arm == runner.CANDIDATE
    return {
        "case_id": spec["case_id"],
        "season": spec["season"],
        "target": spec["target"],
        "arm": arm,
        "source_head": contract["sources"][arm]["head"],
        "source_tree": contract["sources"][arm]["tree"],
        "fingerprints": manifest["fingerprints"],
        "primary_metric": "cold_player_rmse",
        "primary_loss": control_primary * (primary_ratio if candidate else 1.0),
        "secondary_metric": "held_team_rmse",
        "secondary_loss": control_secondary * (secondary_ratio if candidate else 1.0),
        "fit_rows": manifest["fit_rows"],
        "primary_rows": manifest["primary_rows"],
        "test_rows": manifest["test_rows"],
        "feature_count": manifest["feature_count"],
        "fit_seconds": 0.8 if candidate else 1.0,
        "predict_seconds": 0.9 if candidate else 1.0,
        "peak_rss_bytes": 95 if candidate else 100,
        "rss_samples": 3,
        "rss_errors": [],
        "archive_bytes": 90 if candidate else 100,
        "prediction_sha256": _hex(f"{spec['case_id']}/{arm}"),
        "safe_roundtrip_exact": True,
        "requested_depth": None,
        "resolved_depth": 4 if candidate else 6,
        "l2_leaf_reg": 3.0,
        "auto_structure": _structure(arm),
        "requested_threads": runner.THREADS,
        "fitted_thread_counts": [runner.THREADS],
        "ambient_thread_count_before_fit": runner.THREADS,
        "ambient_thread_count_after_predict": runner.THREADS,
        "runtime_before": {"current": runner.THREADS},
        "runtime_after": {"current": runner.THREADS},
        "implementation_path": "/source/darkofit/sklearn_api.py",
        "warnings": [],
        "python": "3.11.8",
        "numpy": "2.2.6",
    }


def _rows(contract, primary_by_season=None, secondary=0.999):
    primary_by_season = primary_by_season or {
        2014: 0.990,
        2015: 0.995,
        2016: 0.998,
    }
    rows = []
    for spec in runner.case_specs():
        for arm in runner.ARMS:
            rows.append(
                _row(
                    contract,
                    spec,
                    arm,
                    primary_ratio=primary_by_season[spec["season"]],
                    secondary_ratio=secondary,
                )
            )
    return rows


def test_spent_sports_grid_and_claim_boundary_are_frozen():
    assert len(runner.case_specs()) == 9
    assert {spec["season"] for spec in runner.case_specs()} == set(runner.SEASONS)
    assert {spec["target"] for spec in runner.case_specs()} == set(runner.TARGETS)
    assert {tuple(order) for order in runner.quality_orders().values()} == {
        tuple(runner.ARMS),
        tuple(reversed(runner.ARMS)),
    }
    claims = runner.claim_spec()
    assert claims["spent_player_disjoint_sports_development"] is True
    assert all(
        claims[name] is False
        for name in (
            "shipping_or_default_change_authorized",
            "fresh_confirmation_authorized",
            "tabarena_or_m2_authorized",
            "release_authorized",
            "lockbox_access_authorized",
        )
    )


def test_analyzer_advances_only_when_all_frozen_gates_pass():
    contract = _contract()
    result = analyzer.analyze_rows(_rows(contract))

    assert result["disposition"] == "eligible_for_fresh_tier_d_design"
    assert all(result["gates"].values())
    assert result["case_count"] == 9
    assert result["season_cluster_count"] == 3
    assert result["cold_player"]["cluster_bootstrap"]["cluster_count"] == 3
    assert set(result["cold_player"]["leave_one_season_out"]) == {
        "omit_2014",
        "omit_2015",
        "omit_2016",
    }
    assert result["shipping_or_default_claim_eligible"] is False
    assert result["fresh_confirmation_authorized"] is False


@pytest.mark.parametrize(
    ("season_values", "failed_gate"),
    [
        (
            {2014: 1.021, 2015: 0.990, 2016: 0.990},
            "worst_season_at_most_1_020",
        ),
        (
            {2014: 0.970, 2015: 1.004, 2016: 1.004},
            "worst_loo_at_most_1_003",
        ),
    ],
)
def test_analyzer_closes_on_season_harm_or_concentration(season_values, failed_gate):
    result = analyzer.analyze_rows(_rows(_contract(), season_values))
    assert result["disposition"] == "closed_after_spent_sports"
    assert result["gates"][failed_gate] is False


def test_raw_validator_rejects_candidate_metadata_and_l2_drift(monkeypatch):
    contract = _contract()
    raw = {
        "schema_version": 1,
        "contract_id": runner.CONTRACT_ID,
        "status": "complete",
        "contract_sha256": "contract",
        "execution": runner.execution_spec(),
        "claims": runner.claim_spec(),
        "sources": contract["sources"],
        "case_manifests": contract["case_manifests"],
        "rows": _rows(contract),
    }
    monkeypatch.setattr(runner, "file_sha256", lambda _path: "contract")
    analyzer.validate_raw(raw, contract)

    bad = copy.deepcopy(raw)
    candidate = next(row for row in bad["rows"] if row["arm"] == runner.CANDIDATE)
    candidate["auto_structure"]["candidates"]["depth"]["branch"] = "middle_density"
    with pytest.raises(RuntimeError, match="did not engage"):
        analyzer.validate_raw(bad, contract)

    bad = copy.deepcopy(raw)
    candidate = next(row for row in bad["rows"] if row["arm"] == runner.CANDIDATE)
    candidate["l2_leaf_reg"] = 1.0
    validated = analyzer.validate_raw(bad, contract)
    with pytest.raises(RuntimeError, match="L2 drifted"):
        analyzer.analyze_rows(validated)


def test_prelaunch_failure_does_not_spend_output(tmp_path, monkeypatch):
    paths = runner.output_paths(tmp_path / "sports")
    monkeypatch.setattr(
        runner,
        "load_contract",
        lambda _path: (_ for _ in ()).throw(RuntimeError("bad contract")),
    )
    args = SimpleNamespace(
        output_prefix=tmp_path / "sports",
        contract=tmp_path / "contract.json",
        panel_cache=tmp_path / "panel.csv",
        control=tmp_path / "control",
        candidate=tmp_path / "candidate",
        cache_dir=tmp_path / "cache",
    )
    with pytest.raises(RuntimeError, match="bad contract"):
        runner.run_parent(args)
    assert not any(path.exists() for path in paths.values())


def test_failure_after_launch_is_terminal_and_discards_rows(tmp_path, monkeypatch):
    paths = runner.output_paths(tmp_path / "sports")
    contract = _contract()
    contract["panel_cache"] = {"bytes": 1, "sha256": "p" * 64}
    contract["general_preconditions"] = {"ok": True}
    harness = {"head": "h", "clean": True}
    source_states = {
        arm: {
            "head": contract["sources"][arm]["head"],
            "tree": contract["sources"][arm]["tree"],
            "clean": True,
        }
        for arm in runner.ARMS
    }
    monkeypatch.setattr(runner, "load_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "validate_harness", lambda _contract: harness)
    monkeypatch.setattr(runner, "panel_record", lambda _path: contract["panel_cache"])
    monkeypatch.setattr(runner, "case_manifests", lambda _path: contract["case_manifests"])
    monkeypatch.setattr(
        runner,
        "validate_source",
        lambda path, expected: source_states[
            runner.CONTROL if "control" in str(path) else runner.CANDIDATE
        ],
    )
    monkeypatch.setattr(runner, "validate_general_preconditions", lambda: {"ok": True})
    monkeypatch.setattr(
        runner,
        "_exclusive_machine_audit",
        lambda: {"conflicting_benchmark_processes": []},
    )

    def fail_worker(**_kwargs):
        assert paths["launch"].is_file()
        raise RuntimeError("worker stopped")

    monkeypatch.setattr(runner, "_run_one_worker", fail_worker)
    args = SimpleNamespace(
        output_prefix=tmp_path / "sports",
        contract=tmp_path / "contract.json",
        panel_cache=tmp_path / "panel.csv",
        control=tmp_path / "control",
        candidate=tmp_path / "candidate",
        cache_dir=tmp_path / "cache",
    )
    monkeypatch.setattr(runner, "file_sha256", lambda path: _hex(str(path)))
    with pytest.raises(RuntimeError, match="failed terminally"):
        runner.run_parent(args)
    terminal = json.loads(paths["terminal"].read_text())
    assert terminal["disposition"] == "terminal_failure"
    assert terminal["completed_rows_discarded"] == 0
    assert terminal["rerun_authorized"] is False
    assert not paths["raw"].exists()
    assert not paths["result"].exists()


def test_freezer_contract_is_outcome_blind(monkeypatch, tmp_path):
    harness = {"head": "h" * 40, "clean": True}
    source = {
        "head": runner.CONTROL_HEAD,
        "tree": runner.CONTROL_TREE,
        "clean": True,
    }
    candidate = {
        "head": runner.CANDIDATE_HEAD,
        "tree": runner.CANDIDATE_TREE,
        "clean": True,
    }
    manifests = _manifests()
    historical = {
        "panel_cache": {"bytes": 10, "sha256": "p" * 64},
        "case_manifests": manifests,
    }
    historical_path = tmp_path / "historical.json"
    historical_path.write_text(json.dumps(historical))
    monkeypatch.setattr(freezer, "ROOT", tmp_path)
    monkeypatch.setattr(freezer, "HISTORICAL_M3B_CONTRACT", historical_path)
    monkeypatch.setattr(runner, "source_state", lambda _path: harness)
    monkeypatch.setattr(
        runner,
        "validate_source",
        lambda path, _expected: candidate if "candidate" in str(path) else source,
    )
    monkeypatch.setattr(
        runner, "panel_record", lambda _path: historical["panel_cache"]
    )
    monkeypatch.setattr(runner, "case_manifests", lambda _path: manifests)
    monkeypatch.setattr(runner, "validate_general_preconditions", lambda: {"ok": True})
    monkeypatch.setattr(freezer, "_bound", lambda relative: {"path": relative, "bytes": 1, "sha256": "b" * 64})

    contract = freezer.build_contract(
        control=Path("/control"),
        candidate=Path("/candidate"),
        panel_cache=Path("/panel"),
    )
    assert contract["outcomes_opened"] is False
    assert contract["contract_frozen"] is True
    assert "rows" not in contract
    assert contract["decision_rules"] == runner.decision_rules()
