import json

from benchmarks import run_automatic_linear_selector_v2_protein_attribution as attempt1
from benchmarks import run_automatic_linear_selector_v2_protein_attribution_attempt2 as attempt2


def test_attempt2_is_new_identity_without_mutating_attempt1_module():
    assert attempt2.ATTEMPT_INDEX == 2
    assert attempt2.CONTRACT_ID.endswith("attempt2-20260722")
    assert attempt1.ATTEMPT_INDEX == 1
    assert attempt1.CONTRACT_ID == attempt2.ATTEMPT1_CONTRACT_ID
    assert attempt1.RUNNER_PATH != attempt2.RUNNER_PATH
    assert attempt1.PROTOCOL_PATH != attempt2.PROTOCOL_PATH


def test_attempt2_binds_terminal_attempt1_and_r1_authority():
    with attempt2._configured_base():
        bindings = attempt2.validate_bound_evidence()
    lineage = bindings["attempt1_terminal_lineage"]
    assert lineage == {
        "contract_id": attempt2.ATTEMPT1_CONTRACT_ID,
        "attempt_index": 1,
        "attempt_spent": True,
        "completed_worker_count": 0,
        "scientific_outcome_observed": False,
        "disposition": "terminal_execution_failure",
        "candidate_commit": attempt1.CANDIDATE_COMMIT,
    }
    assert bindings["r1_authorization_commit"] == attempt2.R1_COMMIT


def test_attempt2_keeps_grid_policy_and_harm_rule_exact():
    assert attempt1.COORDINATES == (
        {"coordinate": 0, "repeat": 0, "fold": 0, "seed": 0},
        {"coordinate": 1, "repeat": 1, "fold": 1, "seed": 1001},
        {"coordinate": 2, "repeat": 2, "fold": 2, "seed": 2002},
    )
    assert attempt1.ARMS == {
        "constant": False,
        "automatic": "auto",
        "explicit_linear": True,
    }
    assert attempt1.THREADS == 14
    assert attempt1.HARM_BOUND == 1.02
    assert attempt1.WORKER_ENVIRONMENT["NUMBA_NUM_THREADS"] == "14"
    assert "PYTHONNOUSERSITE" not in attempt1.WORKER_ENVIRONMENT
    assert attempt2.OPENML_VERSION == "0.15.1"
    assert attempt2.EXPECTED_SPLIT_FINGERPRINTS == {
        (0, 0): "88e9e1cb742d160a72bd4fc1977613f41f2a48bb713ebe1ec4bd17365d165e9a",
        (1, 1): "2d7f22d94f5308ee464da1eea65e64bf98923453222fdd0aee99377e8323d173",
        (2, 2): "e9e44958b94bbcb462434041a9c0df76eae8cfd6d893fbac3ce8bd9a9b6ad819",
    }


def test_attempt2_protocol_declares_only_environment_repair_and_no_rerun():
    text = " ".join(attempt2.PROTOCOL_PATH.read_text().split())
    for phrase in (
        "new execution identity, not a favorable scientific rerun",
        "before the launch manifest is created",
        "There is no minimum-effect gate",
        "ready_for_powered_fresh_design",
        "No failed or inspected attempt-2 launch may be rerun",
        "historical guardrail replay",
    ):
        assert phrase in text


def test_attempt2_configuration_is_scoped_and_restored():
    original = {
        "runner": attempt1.RUNNER_PATH,
        "protocol": attempt1.PROTOCOL_PATH,
        "test": attempt1.TEST_PATH,
        "contract": attempt1.CONTRACT_ID,
        "attempt": attempt1.ATTEMPT_INDEX,
        "hashes": attempt1.EXPECTED_HASHES,
        "validator": attempt1.validate_bound_evidence,
        "loader": attempt1._load_split,
        "preflight": attempt1._data_loader_preflight,
    }
    with attempt2._configured_base():
        assert attempt1.RUNNER_PATH == attempt2.RUNNER_PATH
        assert attempt1.PROTOCOL_PATH == attempt2.PROTOCOL_PATH
        assert attempt1.TEST_PATH == attempt2.TEST_PATH
        assert attempt1.CONTRACT_ID == attempt2.CONTRACT_ID
        assert attempt1.ATTEMPT_INDEX == 2
        assert attempt1.EXPECTED_HASHES is attempt2.EXPECTED_HASHES
        assert attempt1.validate_bound_evidence is attempt2.validate_bound_evidence
        assert attempt1._load_split is attempt2._load_split
        assert attempt1._data_loader_preflight is attempt2._data_loader_preflight
    assert attempt1.RUNNER_PATH == original["runner"]
    assert attempt1.PROTOCOL_PATH == original["protocol"]
    assert attempt1.TEST_PATH == original["test"]
    assert attempt1.CONTRACT_ID == original["contract"]
    assert attempt1.ATTEMPT_INDEX == original["attempt"]
    assert attempt1.EXPECTED_HASHES is original["hashes"]
    assert attempt1.validate_bound_evidence is original["validator"]
    assert attempt1._load_split is original["loader"]
    assert attempt1._data_loader_preflight is original["preflight"]


def test_attempt1_result_has_no_scientific_rows():
    result = json.loads(attempt2.ATTEMPT1_RESULT_PATH.read_text())
    assert result["analysis"]["completed_worker_count"] == 0
    assert "raw" not in result["artifacts"]
