from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRIAGE = ROOT / "benchmarks/chimeraboost_v0210_changelog_triage_20260722.md"
CONTRACT = (
    ROOT
    / "benchmarks/group_centered_categorical_crosses_v1_development_contract.md"
)


def test_triage_binds_release_and_distinguishes_closed_mechanism() -> None:
    text = " ".join(TRIAGE.read_text(encoding="utf-8").split())

    for phrase in (
        "26fed8a715fe172518472f4fec1a663492db6f61",
        "e0d401bab9b16041f0323ae923e43dfa413532c3",
        "Apache-2.0",
        "materially different from the closed pairwise",
        "group_centered_categorical_crosses_v1",
        "worst leave-one-dataset-out ratio `<=1.003`",
    ):
        assert phrase in text


def test_contract_freezes_candidate_scope_and_one_m6_inspection() -> None:
    text = " ".join(CONTRACT.read_text(encoding="utf-8").split())

    for phrase in (
        "one non-ensemble `DarkoRegressor` with scalar `loss=\"RMSE\"`",
        "at most 12",
        "strictly lower",
        "ties select control",
        "weight-aware",
        "global weighted mean for unseen categories",
        "safe-NPZ round trips",
        "exact TreeSHAP fails loudly",
        "group_centered_categorical_crosses_v1",
        "inspection index 1",
        "eligible_for_mechanism_specific_spent_attribution",
        "No fresh, TabArena, release ladder",
    ):
        assert phrase in text


def test_binding_plan_records_expected_value_and_terminal_disposition() -> None:
    text = (ROOT / "COUNTERPUNCH_PLAN.md").read_text(encoding="utf-8")

    assert "R1 quality slot advanced — group-centered categorical crosses v1" in text
    assert "diamonds at\n  `1.386479×`" in text
    assert "immutable M6 v3 aggregate,\n  worst-dataset, and LOO gates all passed" in text
    assert "eligible only for a\n  separately frozen mechanism-specific spent attribution" in text
