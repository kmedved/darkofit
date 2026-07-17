from pathlib import Path

from conftest import CAMPAIGN_EXACT, is_campaign_module


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
