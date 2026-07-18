import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks import run_t7_catboost_attribution as runner
from benchmarks.analyze_t7_catboost_attribution import depth_policy_arm


def test_arm_order_is_a_balanced_rotation():
    orders = [runner._arm_order(index) for index in range(len(runner.ARMS))]
    assert all(set(order) == set(runner.ARMS) for order in orders)
    for position in range(len(runner.ARMS)):
        assert {
            order[position] for order in orders
        } == set(runner.ARMS)


def test_catboost_frame_preserves_numeric_and_canonicalizes_categories():
    X = pd.DataFrame(
        {"number": [1.0, 2.0], "category": ["x", None]}
    )
    result = runner._catboost_frame(X, [1])
    assert result["number"].tolist() == [1.0, 2.0]
    assert result["category"].tolist() == [
        "str:x",
        "__DARKOFIT_MISSING_CATEGORY__",
    ]


def test_t7_uses_only_spent_development_rows():
    registry, rows = runner._rows()
    assert len(rows) == 8
    assert {
        int(row["task_id"]) for row in registry["development_tasks"]
    } == set(rows)
    assert not (
        set(rows)
        & {
            int(row["task_id"])
            for row in registry["confirmation_tasks"]
        }
    )


def test_depth_policy_is_fixed_by_samples_per_feature():
    assert depth_policy_arm(99, 1) == "depth_4"
    assert depth_policy_arm(100, 1) == "default"
    assert depth_policy_arm(2_499, 1) == "default"
    assert depth_policy_arm(2_500, 1) == "depth_8"


def test_catboost_warmup_target_is_nonconstant():
    pytest.importorskip("catboost")
    assert runner._warmup() > 0


def test_t7_artifacts_are_hash_bound_and_nonpromotional():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads(
        (root / "benchmarks" / "t7_catboost_attribution_raw.json").read_text()
    )
    raw_hash = raw.pop("raw_sha256")
    assert runner._json_sha256(raw) == raw_hash
    summary = json.loads(
        (
            root / "benchmarks" / "t7_catboost_attribution_summary.json"
        ).read_text()
    )
    summary_hash = summary.pop("summary_sha256")
    from benchmarks import analyze_t7_catboost_attribution as analyzer

    assert analyzer._json_sha256(summary) == summary_hash
    assert summary["frozen_research_candidates"] == ["depth_by_n_p"]
    assert summary["default_change_authorized"] is False
