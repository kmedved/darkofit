from pathlib import Path

from conftest import (
    CAMPAIGN_EXACT,
    _requested_partition,
    is_campaign_module,
    pytest_ignore_collect,
)


class _Config:
    def __init__(self, markexpr):
        self.markexpr = markexpr

    def getoption(self, name):
        assert name == "markexpr"
        return self.markexpr


def test_campaign_partition_is_explicit_complete_and_disjoint():
    tests = sorted(Path(__file__).parent.glob("test_*.py"))
    campaign = {path.name for path in tests if is_campaign_module(path)}
    library = {path.name for path in tests if not is_campaign_module(path)}
    assert campaign
    assert library
    assert campaign.isdisjoint(library)
    assert campaign | library == {path.name for path in tests}
    assert CAMPAIGN_EXACT <= campaign


def test_core_library_modules_stay_out_of_campaign_partition():
    for name in (
        "test_darkofit.py",
        "test_distributional.py",
        "test_input_validation.py",
        "test_ordinal_features.py",
        "test_payload_hardening.py",
        "test_tree_shap.py",
        "test_warmup.py",
    ):
        assert not is_campaign_module(name)

    campaign = _Config("campaign")
    library = _Config("not   campaign")
    unfiltered = _Config("")

    assert _requested_partition(campaign) == "campaign"
    assert _requested_partition(library) == "library"
    assert _requested_partition(_Config("campaign and slow")) is None
    assert pytest_ignore_collect(Path("test_darkofit.py"), campaign) is True
    assert (
        pytest_ignore_collect(
            Path("test_basketball_guardrails.py"), campaign
        )
        is False
    )
    assert pytest_ignore_collect(Path("test_darkofit.py"), library) is False
    assert (
        pytest_ignore_collect(
            Path("test_basketball_guardrails.py"), library
        )
        is True
    )
    assert pytest_ignore_collect(Path("test_darkofit.py"), unfiltered) is None
    assert pytest_ignore_collect(Path("tests"), campaign) is None
