import pandas as pd

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
