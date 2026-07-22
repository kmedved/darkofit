from pathlib import Path

from darkofit import DarkoClassifier, DarkoRegressor


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "benchmarks" / "ensemble_v3_public_contract.md"


def test_ensemble_v3_public_contract_is_frozen_and_complete():
    text = CONTRACT.read_text()
    required = {
        "ensemble-v3-public-contract-v1",
        'ensemble_mode="bootstrap"',
        'ensemble_mode="v3"',
        'ensemble_member_learning_rate="policy"',
        'ensemble_member_colsample="policy"',
        'n_ensembles=8',
        'ensemble_format_version=4',
        'recipe_version=1',
        'allow_pickle=False',
        'ordinal_features="auto"',
        'tree_mode="auto"',
        'auto_learning_rate_probe=True',
        "Distributional regression losses",
        "byte- and behavior-preserving non-regression",
    }
    missing = sorted(token for token in required if token not in text)
    assert not missing, f"public contract is missing: {missing}"


def test_authorized_public_ship_exposes_exact_v3_constructor_defaults():
    expected = {
        "ensemble_mode": "bootstrap",
        "ensemble_member_learning_rate": "policy",
        "ensemble_member_colsample": "policy",
    }
    for estimator in (DarkoRegressor(), DarkoClassifier()):
        params = estimator.get_params(deep=False)
        assert {name: params[name] for name in expected} == expected
