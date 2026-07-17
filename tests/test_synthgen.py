"""SynthGen generator contracts: determinism, goldens, meta, structure, suites.

Recipe goldens are the cross-platform NumPy-RNG-stream-drift tripwire. Dataset
goldens additionally cover floating transforms but are restricted to the
Darwin/arm64 platform on which the immutable df1 freeze was created. If either
fails after a NumPy upgrade on its declared platform (NEP 19 does not
guarantee distribution-method streams), bump synthgen.recipe.VERSION and
re-freeze the suite -- never re-pin goldens under the same version.

Modified by the DarkoFit project from ChimeraBoost 0.15.0 commit 851ab7f.
"""
import json
import os
import platform
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import synthgen
from synthgen import filters
from synthgen.suites import CANARIES, SUITES

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_synthgen.json")
RECIPE_GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "golden_synthgen_recipes.json"
)
FREEZE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "benchmarks", "synthgen_df1_freeze.json"
)


@pytest.fixture(autouse=True)
def _flat_memory():
    yield
    synthgen.build_dataset.cache_clear()


def test_key_roundtrip():
    key = synthgen.key_for(31)
    assert key == f"syn:{synthgen.VERSION}/031"
    version, did = synthgen.parse_key(key)
    assert (version, did) == (synthgen.VERSION, 31)
    with pytest.raises(ValueError):
        synthgen.parse_key("syn:v0/031")
    with pytest.raises(ValueError):
        synthgen.parse_key("gr:clf_num/pol")


def test_determinism_and_arg_independence():
    key = synthgen.key_for(3)
    h1 = synthgen.hash_dataset(key)
    synthgen.build_dataset.cache_clear()
    h2 = synthgen.hash_dataset(key)
    assert h1 == h2
    builder = synthgen.make_builder(key)
    X1, y1, c1, t1 = builder(1.0, np.random.default_rng(0))
    X2, y2, c2, t2 = builder(9.9, np.random.default_rng(12345))
    assert (t1, c1) == (t2, c2)
    assert np.array_equal(y1, y2)
    if X1.dtype == object:
        same = all((a == b) or (a != a and b != b)
                   for a, b in zip(X1.ravel(), X2.ravel()))
        assert same
    else:
        assert np.array_equal(X1, X2, equal_nan=True)


def test_generation_does_not_touch_global_numpy_rng():
    np.random.seed(1729)
    expected = np.random.random(5)
    np.random.seed(1729)
    synthgen.build_dataset.cache_clear()
    synthgen.build_dataset(synthgen.key_for(86))
    actual = np.random.random(5)
    np.testing.assert_array_equal(actual, expected)


def test_frozen_regression_target_is_feature_dependent():
    key = synthgen.key_for(86)
    X, y, cat, task, _ = synthgen.build_dataset(key)
    assert task == "regression"
    ok, detail = filters.learnable(X, y, cat, task)
    assert ok, detail


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="df1 dataset-byte goldens are frozen on Darwin arm64",
)
def test_reference_platform_dataset_goldens():
    goldens = json.load(open(GOLDEN_PATH, encoding="utf-8"))
    assert goldens, "golden_synthgen.json is empty"
    for key, expected in goldens.items():
        assert synthgen.hash_dataset(key) == expected, (
            f"{key} content drifted -- if this is a numpy upgrade, bump "
            "synthgen VERSION and re-freeze (see module docstring)")
        synthgen.build_dataset.cache_clear()


def test_cross_platform_recipe_goldens():
    goldens = json.load(open(RECIPE_GOLDEN_PATH, encoding="utf-8"))
    assert goldens, "golden_synthgen_recipes.json is empty"
    assert {
        key: synthgen.hash_recipe(key) for key in goldens
    } == goldens


def test_meta_and_dtype_contracts():
    for did in range(12):
        key = synthgen.key_for(did)
        X, y, cat, task, meta = synthgen.build_dataset(key)
        assert X.shape == (meta["n"], meta["d"])
        assert task == meta["task"] == synthgen.task_of(key)
        json.dumps(meta)  # JSON-safe
        if cat:
            assert X.dtype == object
            for j in cat:
                col = X[:100, j]
                assert all(isinstance(v, str) for v in col)
                assert not any(v.replace(".", "").lstrip("-").isdigit()
                               for v in col if v != "__nan__")
        else:
            assert X.dtype == np.float64
        if task == "regression":
            assert meta["noise_sigma"] and meta["noise_sigma"] > 0
            assert meta["bayes_brier"] is None
        else:
            assert y.dtype == np.int64
            counts = np.bincount(y)
            assert len(counts) == meta["n_classes"]
            assert 0.0 <= meta["bayes_brier"] <= 2.0
            if not meta["degenerate"]:
                assert counts.min() >= max(10, int(0.005 * meta["n"]))
        synthgen.build_dataset.cache_clear()


def test_saturated_floors_are_zero():
    seen = 0
    for did in range(40):
        rec = synthgen.sample_recipe(did)
        if not rec.saturated or rec.task == "regression":
            continue
        _, _, _, _, meta = synthgen.build_dataset(synthgen.key_for(did))
        assert meta["bayes_brier"] == 0.0
        synthgen.build_dataset.cache_clear()
        seen += 1
    assert seen >= 1


def test_classification_floor_matches_recomputation():
    # bayes_brier stored must equal sum-form Brier of the true p against y --
    # verified indirectly: floor <= observed Brier of the *ideal* constant
    # predictor and >= 0; exact recomputation happens inside emit. Here we
    # check the invariant bayes_brier < Brier(constant class prior).
    for did in range(20):
        key = synthgen.key_for(did)
        X, y, cat, task, meta = synthgen.build_dataset(key)
        if task == "regression" or meta["degenerate"]:
            synthgen.build_dataset.cache_clear()
            continue
        k = meta["n_classes"]
        prior = np.bincount(y, minlength=k) / len(y)
        onehot = np.zeros((len(y), k))
        onehot[np.arange(len(y)), y] = 1.0
        const_brier = float(((prior[None, :] - onehot) ** 2).sum(axis=1).mean())
        # expected Brier of the true p is <= any constant's; the REALIZED gap
        # can go slightly the other way on weak-signal data (O(1/sqrt(n)))
        assert meta["bayes_brier"] <= const_brier + 0.01
        synthgen.build_dataset.cache_clear()


def test_regression_noise_floor_below_target_std():
    for did in range(20):
        key = synthgen.key_for(did)
        _, y, _, task, meta = synthgen.build_dataset(key)
        if task == "regression":
            assert meta["noise_sigma"] < float(np.std(y))
        synthgen.build_dataset.cache_clear()


def test_irrelevant_columns_uncorrelated():
    # a dataset with irrelevant numeric columns: max |corr(x_j, y)| over the
    # weakest columns should be small on a decent-n dataset
    for did in range(40):
        key = synthgen.key_for(did)
        X, y, cat, task, meta = synthgen.build_dataset(key)
        if (task == "regression" and meta["irrelevant_fraction"] > 0.3
                and not cat and meta["n"] >= 3000):
            yz = (y - y.mean()) / y.std()
            cors = []
            for j in range(X.shape[1]):
                v = X[:, j].astype(float)
                m = np.isfinite(v)
                vz = v[m] - v[m].mean()
                s = vz.std()
                if s > 0:
                    cors.append(abs(float(np.mean(vz / s * yz[m]))))
            cors = np.sort(cors)
            n_irr = int(round(meta["irrelevant_fraction"] * meta["d"]))
            assert cors[: max(1, n_irr // 2)].max() < 0.15
            synthgen.build_dataset.cache_clear()
            return
        synthgen.build_dataset.cache_clear()
    pytest.skip("no qualifying dataset in probe range")


def test_entity_column_mechanism():
    from synthgen import emit
    rng = np.random.default_rng(7)
    codes, card, effect_rows, sigma_e = emit._entity_column(
        5000, np.log(10.0), 64, rng)
    assert codes.shape == (5000,) and codes.max() == card - 1
    assert 0.3 <= sigma_e <= 1.0
    counts = np.bincount(codes, minlength=card)
    # Zipf-ish frequencies: the heaviest level dwarfs the median level
    assert counts.max() > 5 * max(1.0, float(np.median(counts)))
    # singleton rare levels exist (the unseen-at-train stress)
    assert (counts == 1).sum() >= 2
    # per-row effect is a per-level lookup: constant within each level
    for lvl in range(min(card, 5)):
        v = effect_rows[codes == lvl]
        if len(v):
            assert np.allclose(v, v[0])


def test_entity_cats_present_in_meta():
    seen = 0
    for did in range(30):
        _, _, _, _, meta = synthgen.build_dataset(synthgen.key_for(did))
        if meta["n_cat_entity"] > 0:
            seen += 1
            assert meta["entity_strength"] > 0
            assert meta["n_cat_entity"] <= meta["n_cat"]
        synthgen.build_dataset.cache_clear()
        if seen >= 3:
            break
    assert seen >= 1


def test_canaries_contract():
    # canary status is earned at freeze time and frozen into suites.py
    assert CANARIES, "CANARIES empty -- freeze.py paste missing"
    assert set(CANARIES) <= set(SUITES["full"])
    for i in sorted(CANARIES)[:5]:
        assert synthgen.sample_recipe(i).saturated
    n_cat_bearing = 0
    for i in sorted(set(CANARIES) & set(SUITES["screen"])):
        _, _, _, _, meta = synthgen.build_dataset(synthgen.key_for(i))
        n_cat_bearing += meta["n_cat"] > 0
        synthgen.build_dataset.cache_clear()
    assert n_cat_bearing >= 3, "screen needs >=3 cat-bearing verified canaries"


def test_suite_nesting_and_registration():
    assert set(SUITES["smoke"]) <= set(SUITES["screen"]) <= set(SUITES["full"])
    keys = synthgen.frozen_keys("smoke")
    assert keys and all(k.startswith(f"syn:{synthgen.VERSION}/") for k in keys)
    assert len(synthgen.all_frozen_keys()) == len(set(synthgen.all_frozen_keys()))


def test_freeze_evidence_matches_committed_suite_and_goldens():
    with open(FREEZE_PATH, encoding="utf-8") as handle:
        evidence = json.load(handle)
    with open(GOLDEN_PATH, encoding="utf-8") as handle:
        goldens = json.load(handle)
    assert evidence["generator_version"] == synthgen.VERSION
    assert evidence["source"] == {
        "branch": "main",
        "clean": True,
        "commit": "fa1ab2cb790b83c20a39b5e175ad8d6f5c035ebf",
    }
    assert evidence["selection"]["suites"] == SUITES
    assert set(evidence["selection"]["canaries"]) == CANARIES
    assert evidence["selection"]["goldens"] == goldens
    records = {record["id"]: record for record in evidence["scan"]["records"]}
    for dataset_id in CANARIES:
        record = records[dataset_id]
        assert record["canary"] is True
        assert len(record["ceiling_values"]) == 3
        assert np.isfinite(record["ceiling_values"]).all()


def test_decision_slices_meet_frozen_minimum():
    slices = {
        "ordinary_regression": [],
        "noisy_nonlinear": [],
        "smooth_linear": [],
        "categorical_regression": [],
    }
    for dataset_id in SUITES["screen"]:
        _, _, _, task, meta = synthgen.build_dataset(
            synthgen.key_for(dataset_id)
        )
        if task != "regression" or meta["saturated"]:
            continue
        slices["ordinary_regression"].append(dataset_id)
        if (
            meta["noise_level"] >= 0.25
            and meta["interaction_depth"] >= 2
            and meta["func_dominant"] != "linear"
        ):
            slices["noisy_nonlinear"].append(dataset_id)
        if meta["func_dominant"] == "linear":
            slices["smooth_linear"].append(dataset_id)
        if meta["n_cat"] > 0:
            slices["categorical_regression"].append(dataset_id)
        synthgen.build_dataset.cache_clear()
    assert {name: len(ids) for name, ids in slices.items()} == {
        "ordinary_regression": 46,
        "noisy_nonlinear": 8,
        "smooth_linear": 11,
        "categorical_regression": 17,
    }


def test_frozen_key_task_registration():
    keys = synthgen.all_frozen_keys()
    assert keys
    assert all(
        synthgen.task_of(key) in ("regression", "binary", "multiclass")
        for key in keys
    )
    assert set(keys) == {
        synthgen.key_for(dataset_id)
        for dataset_id in SUITES["full"]
    }


def test_filters_behave():
    ok, why = filters.degeneracy_ok(np.zeros((100, 3)), np.zeros(100), "regression")
    assert not ok
    y = np.array([0] * 995 + [1] * 5)
    ok, _ = filters.degeneracy_ok(np.zeros((1000, 3)), y, "binary")
    assert not ok
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 5))
    y = X[:, 0] * 2.0 + rng.normal(size=1500) * 0.1
    ok, detail = filters.learnable(X, y, None, "regression")
    assert ok, detail
    ok, _ = filters.tractable({"n_cat": 3, "cat_fraction": 1.0, "max_cardinality": 80})
    assert not ok
    ok, _ = filters.tractable({"n_cat": 3, "cat_fraction": 1.0, "max_cardinality": 16})
    assert ok


def test_canary_verifier_preserves_each_seed_metric(monkeypatch):
    import darkofit

    class ConstantRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y, cat_features=None):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=np.float64)

    monkeypatch.setattr(darkofit, "DarkoRegressor", ConstantRegressor)
    X = np.arange(120, dtype=np.float64).reshape(40, 3)
    y = np.zeros(40, dtype=np.float64)
    ok, detail = filters.at_ceiling(
        X,
        y,
        None,
        "regression",
        {"noise_sigma": 1.0, "bayes_brier": None},
    )
    assert ok
    assert detail["ceiling_metric"] == "rmse_ratio"
    assert detail["ceiling_values"] == [0.0, 0.0, 0.0]
    assert detail["ceiling_mean"] == 0.0
    assert detail["ceiling_worst"] == 0.0
