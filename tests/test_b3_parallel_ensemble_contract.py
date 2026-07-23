from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "benchmarks/b3_parallel_ensemble_v1_contract.md"


def test_b3_contract_freezes_scope_topology_and_stop_rules() -> None:
    text = " ".join(CONTRACT.read_text(encoding="utf-8").split())

    for phrase in (
        "b3-parallel-ensemble-members-v1-20260723",
        "c4dae58fcf7a8d456533ba2d9b469f039adc453c",
        "26fed8a715fe172518472f4fec1a663492db6f61",
        "`W = min(K, max(1, floor(B / 2))) = 7`",
        "`T = floor(B / W) = 2`",
        "fixed same-thread equivalence control is `W=1, T=2`",
        "`cold_executor`",
        "`steady_executor`",
        "worst leave-one-case-out geometric-mean ratio is `<= 1.0`",
        "`6 GiB`",
        "`5x` the paired sequential peak",
        "more than `2 GiB`",
        "eligible_for_public_b3_contract_design",
        "no favorable rerun",
    ):
        assert phrase in text


def test_b3_contract_keeps_candidate_private_and_behavior_exact() -> None:
    text = " ".join(CONTRACT.read_text(encoding="utf-8").split())

    for phrase in (
        "adds no public constructor parameter",
        "Results are reassembled strictly by member index",
        "exact prediction and probability hashes",
        "parent ambient Numba mask is restored",
        "public fits sequential",
        "No fresh data, sports panel, M2, TabArena, lockbox",
    ):
        assert phrase in text
