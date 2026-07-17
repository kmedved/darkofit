import pytest

from benchmarks import basketball_campaign_harness as campaign


def test_paired_ratio_summary_gates_ratios_not_noisy_arm_levels():
    summary = campaign.paired_ratio_summary(
        [10.0, 20.0, 40.0],
        [5.0, 10.0, 20.0],
        max_iqr_over_median=0.01,
        median_seconds_budget=25.0,
    )

    assert summary["paired_ratios"] == [2.0, 2.0, 2.0]
    assert summary["median_ratio"] == 2.0
    assert summary["iqr_over_median"] == 0.0
    assert summary["stable"] is True
    assert summary["seconds_budget_passed"] is True

    unstable = campaign.paired_ratio_summary(
        [5.0, 20.0, 60.0],
        [5.0, 10.0, 20.0],
        max_iqr_over_median=0.10,
    )
    assert unstable["paired_ratios"] == [1.0, 2.0, 3.0]
    assert unstable["stable"] is False


def test_paired_ratio_summary_is_fail_closed():
    with pytest.raises(RuntimeError, match="exactly 3"):
        campaign.paired_ratio_summary([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(RuntimeError, match="positive and finite"):
        campaign.paired_ratio_summary(
            [1.0, 2.0, 3.0], [1.0, 0.0, 3.0]
        )
    with pytest.raises(ValueError, match="max_iqr_over_median"):
        campaign.paired_ratio_summary(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            max_iqr_over_median=-0.1,
        )
    with pytest.raises(ValueError, match="median_seconds_budget"):
        campaign.paired_ratio_summary(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            median_seconds_budget=0.0,
        )
