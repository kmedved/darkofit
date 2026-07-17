import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks import run_basketball_oob_ensemble as original
from benchmarks import run_basketball_oob_ensemble_confirmation as confirmation


def _canonical():
    artifact = json.loads(
        (confirmation.ROOT / "benchmarks/basketball_oob_ensemble.json").read_text()
    )
    return {
        result["config"]: result for result in artifact["canonical_results"]
    }


def _summary(default=10.0, candidate=25.0):
    wall = {
        original.DEFAULT_CONFIG: confirmation.timing_summary(
            [default + value for value in (0.0, 0.1, -0.1, 0.05, -0.05, 0.0)]
        ),
        original.CANDIDATE_CONFIG: confirmation.timing_summary(
            [candidate + value for value in (0.0, 0.2, -0.2, 0.1, -0.1, 0.0)]
        ),
    }
    prediction = {
        original.DEFAULT_CONFIG: confirmation.timing_summary(
            [1.0, 1.01, 0.99, 1.0, 1.01, 0.99]
        ),
        original.CANDIDATE_CONFIG: confirmation.timing_summary(
            [2.5, 2.51, 2.49, 2.5, 2.51, 2.49]
        ),
    }
    paired = confirmation.timing_summary(
        [candidate / default] * confirmation.TIMING_BLOCKS
    )
    return wall, prediction, paired


def test_schedule_is_six_block_position_balanced():
    schedule = confirmation.schedule()
    assert len(schedule) == 6
    assert schedule[0] == (
        original.DEFAULT_CONFIG,
        original.CANDIDATE_CONFIG,
    )
    assert schedule[1] == tuple(reversed(schedule[0]))
    for config in original.CONFIG_ORDER:
        assert sum(order[0] == config for order in schedule) == 3
        assert sum(order[1] == config for order in schedule) == 3


def test_worker_result_parser_allows_chatter_without_duplicate_results():
    payload = {"config": original.DEFAULT_CONFIG}
    encoded = original.WORKER_RESULT_PREFIX + json.dumps(payload)
    result, chatter = confirmation._worker_result(f"before\n{encoded}\nafter\n")
    assert result == payload
    assert chatter == "before\nafter"
    with pytest.raises(RuntimeError, match="exactly one"):
        confirmation._worker_result(f"{encoded}\n{encoded}\n")


def test_original_artifact_reproduces_all_frozen_prediction_goldens():
    for result in _canonical().values():
        confirmation.validate_golden(result)


def test_golden_validation_fails_closed_on_fold_or_cold_player_drift():
    canonical = _canonical()
    changed = deepcopy(canonical[original.CANDIDATE_CONFIG])
    changed["folds"][0]["prediction_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="fold prediction"):
        confirmation.validate_golden(changed)
    changed = deepcopy(canonical[original.CANDIDATE_CONFIG])
    changed["holdout"]["scores"]["cold_player_subset"][
        "prediction_sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="cold-player"):
        confirmation.validate_golden(changed)


def test_analysis_advances_only_to_opt_in_api_when_every_gate_passes():
    canonical = _canonical()
    wall, prediction, paired = _summary()
    fingerprints = {name: {name + "-exact"} for name in original.CONFIG_ORDER}
    decision = confirmation.analyze(
        canonical, fingerprints, wall, prediction, paired
    )
    assert decision["passed"] is True
    assert decision["candidate_scope"] == "opt_in_only"
    assert decision["default_promotion_authorized"] is False
    assert decision["recommendation"] == "advance_to_opt_in_api_implementation"


def test_analysis_fails_on_instability_cost_or_behavior_drift():
    canonical = _canonical()
    wall, prediction, paired = _summary()
    wall[original.CANDIDATE_CONFIG] = confirmation.timing_summary(
        [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    )
    fingerprints = {
        original.DEFAULT_CONFIG: {"exact"},
        original.CANDIDATE_CONFIG: {"first", "second"},
    }
    decision = confirmation.analyze(
        canonical, fingerprints, wall, prediction, paired
    )
    assert decision["gates"]["wall_timing_stable"] is False
    assert decision["gates"]["behavior_repeat_exact"] is False
    assert decision["passed"] is False
    assert decision["recommendation"] == "close_oob_ensemble_attempt"


def test_protocol_and_support_hashes_remain_bound_to_frozen_artifact():
    assert confirmation.EXPECTED_PROTOCOL_SHA256 != "TO_BE_BOUND"
    assert confirmation._sha256_file(confirmation.PROTOCOL_PATH) == (
        confirmation.EXPECTED_PROTOCOL_SHA256
    )
    for relative, expected in confirmation.EXPECTED_SUPPORT_SHA256.items():
        assert confirmation._sha256_file(confirmation.ROOT / relative) == expected
    artifact = json.loads(confirmation.DEFAULT_OUTPUT.read_text())
    assert artifact["source"]["package_manifest_sha256"] == (
        confirmation.EXPECTED_PACKAGE_MANIFEST
    )


def test_run_primes_cache_before_any_worker(monkeypatch, tmp_path):
    primed = False
    source = {"frozen": True}
    canonical = _canonical()

    def load_dataset(cache):
        nonlocal primed
        assert cache == tmp_path / "data.csv"
        primed = True

    def run_worker(config, *, threads, data_cache):
        assert primed is True
        assert threads == confirmation.EXPECTED_THREADS
        assert data_cache == tmp_path / "data.csv"
        return deepcopy(canonical[config])

    monkeypatch.setattr(
        confirmation.harness, "load_basketball_dataset", load_dataset
    )
    monkeypatch.setattr(confirmation, "require_clean_frozen_source", lambda: source)
    monkeypatch.setattr(confirmation, "_source_state", lambda: source)
    monkeypatch.setattr(confirmation, "run_worker_process", run_worker)
    monkeypatch.setattr(confirmation.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    args = confirmation.parse_args(
        [
            "--data-cache",
            str(tmp_path / "data.csv"),
            "--output",
            str(tmp_path / "result.json"),
        ]
    )
    payload = confirmation.run(args)
    assert primed is True
    assert payload["decision"]["gates"]["behavior_repeat_exact"] is True


def test_parse_args_keeps_frozen_thread_count_and_lexical_paths(tmp_path):
    output = tmp_path / "result.json"
    args = confirmation.parse_args(["--output", str(output)])
    assert args.threads == confirmation.EXPECTED_THREADS
    assert args.output == output.absolute()


def test_parse_args_preserves_symlinks_for_refusal_checks(tmp_path):
    output_target = tmp_path / "output-target.json"
    output_link = tmp_path / "output-link.json"
    output_link.symlink_to(output_target)
    cache_target = tmp_path / "cache-target.csv"
    cache_link = tmp_path / "cache-link.csv"
    cache_link.symlink_to(cache_target)
    args = confirmation.parse_args(
        ["--output", str(output_link), "--data-cache", str(cache_link)]
    )
    assert args.output == output_link.absolute()
    assert args.output.is_symlink()
    assert args.data_cache == cache_link.absolute()
    assert args.data_cache.is_symlink()
