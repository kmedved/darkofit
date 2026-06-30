import numpy as np
import pytest
from concurrent.futures import ProcessPoolExecutor
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.datasets import load_breast_cancer, load_diabetes

optuna = pytest.importorskip("optuna")

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
import chimeraboost.tuning.search as search_mod
from chimeraboost.tuning import ChimeraBoostSearchCV, ChimeraBoostStepwiseSearchCV
from chimeraboost.tuning.optuna_backend import make_storage
from chimeraboost.tuning.scoring import resolve_scorer
from chimeraboost.tuning.spaces import (
    LaneState,
    make_phase_spec,
    phase_names,
    SpaceContext,
    suggest_joint_compact,
    suggest_learning_rate,
    suggest_sampling_regularization,
    suggest_split_noise,
    suggest_structure,
)
from chimeraboost.tuning.validation import make_cv_splits, validate_cv_splits


class FakeTrial:
    def suggest_int(self, name, low, high, log=False):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


class HighRandomStrengthTrial(FakeTrial):
    def suggest_float(self, name, low, high, log=False):
        if name.endswith("_random_strength"):
            return high
        return low


def test_search_spaces_emit_legal_mode_specific_params():
    context = SpaceContext(
        estimator_params=ChimeraBoostClassifier().get_params(),
        has_categoricals=False,
        classifier=True,
    )

    cat = suggest_structure(FakeTrial(), context, LaneState("catboost"))
    assert cat["tree_mode"] == "catboost"
    assert cat["num_leaves"] is None
    assert cat["ordered_boosting"] == "auto"

    lgb = suggest_structure(FakeTrial(), context, LaneState("lightgbm"))
    assert lgb["tree_mode"] == "lightgbm"
    assert lgb["ordered_boosting"] == "auto"
    assert lgb["num_leaves"] >= 2


class LastChoiceTrial(FakeTrial):
    def suggest_categorical(self, name, choices):
        return choices[-1]


def test_search_spaces_preserve_requested_hybrid_lane():
    context = SpaceContext(
        estimator_params=ChimeraBoostRegressor().get_params(),
        has_categoricals=False,
        classifier=False,
        tree_modes=("catboost", "hybrid"),
    )

    joint = suggest_joint_compact(LastChoiceTrial(), context, LaneState("hybrid"))
    structure = suggest_structure(FakeTrial(), context, LaneState("hybrid"))

    assert joint["tree_mode"] == "hybrid"
    assert structure["tree_mode"] == "hybrid"


class GossTrial(FakeTrial):
    def suggest_categorical(self, name, choices):
        if name.endswith("_sampling"):
            return "goss"
        return choices[0]


def test_goss_space_forces_full_subsample():
    context = SpaceContext(
        estimator_params=ChimeraBoostClassifier().get_params(),
        has_categoricals=False,
        classifier=True,
    )
    params = suggest_sampling_regularization(
        GossTrial(), context, LaneState("lightgbm")
    )
    assert params["sampling"] == "goss"
    assert params["subsample"] == 1.0


def test_default_sampling_space_does_not_tune_random_strength():
    context = SpaceContext(
        estimator_params=ChimeraBoostClassifier().get_params(),
        has_categoricals=False,
        classifier=True,
    )
    params = suggest_sampling_regularization(
        HighRandomStrengthTrial(), context, LaneState("catboost")
    )
    split_noise = suggest_split_noise(
        HighRandomStrengthTrial(), context, LaneState("catboost")
    )
    spec = make_phase_spec("split_noise", "catboost", 2)

    assert params["random_strength"] == 0.0
    assert split_noise["random_strength"] == 2.0
    assert spec.tunable == ("random_strength",)


def test_phase_names_accepts_single_explicit_phase_string():
    assert phase_names("split_noise") == ("split_noise",)


def test_group_cv_splits_never_overlap_groups():
    X = np.arange(60).reshape(30, 2)
    y = np.tile([0, 1], 15)
    groups = np.repeat(np.arange(10), 3)
    splits = make_cv_splits(
        X, y, cv=5, groups=groups, classifier=True, random_state=0
    )
    for train_idx, valid_idx in splits:
        assert not set(groups[train_idx]).intersection(groups[valid_idx])


def test_custom_iterable_cv_splits_accept_lists_with_groups():
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    groups = np.repeat(np.arange(6), 2)
    cv = [([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11])]

    splits = make_cv_splits(
        X, y, cv=cv, groups=groups, classifier=True, random_state=0
    )

    train_idx, valid_idx = splits[0]
    assert train_idx.dtype == np.int64
    assert valid_idx.dtype == np.int64
    assert np.array_equal(train_idx, np.array(cv[0][0], dtype=np.int64))
    assert np.array_equal(valid_idx, np.array(cv[0][1], dtype=np.int64))


def test_cv_splits_require_positive_weight_mass_in_each_fold():
    y = np.tile([0, 1], 6)
    splits = [(np.arange(6), np.arange(6, 12))]
    weights = np.zeros(12, dtype=np.float64)
    weights[:6] = 1.0

    with pytest.raises(ValueError, match="positive sample_weight mass"):
        validate_cv_splits(
            splits, y, classifier=True, sample_weight=weights
        )


def test_cv_splits_reject_invalid_sample_weights():
    y = np.tile([0, 1], 6)
    splits = [(np.arange(6), np.arange(6, 12))]

    for bad, message in [
        (np.full(12, np.nan), "finite"),
        (np.r_[1.0, -1.0, np.ones(10)], "nonnegative"),
        (np.zeros(12), "positive total"),
    ]:
        with pytest.raises(ValueError, match=message):
            validate_cv_splits(
                splits, y, classifier=True, sample_weight=bad
            )


def test_cv_splits_require_positive_weight_mass_per_class():
    y = np.tile([0, 1], 6)
    splits = [(np.arange(6), np.arange(6, 12))]
    weights = np.ones(12, dtype=np.float64)
    weights[[7, 9, 11]] = 0.0

    with pytest.raises(ValueError, match="positive-mass class"):
        validate_cv_splits(
            splits, y, classifier=True, sample_weight=weights
        )


def test_validation_fraction_holdout_mode():
    X = np.arange(100).reshape(50, 2)
    y = np.tile([0, 1], 25)
    splits = make_cv_splits(
        X, y, cv=None, classifier=True, random_state=0,
        validation_fraction=0.2,
    )
    assert len(splits) == 1
    train_idx, valid_idx = splits[0]
    assert len(valid_idx) == 10
    assert len(train_idx) == 40
    assert set(np.unique(y[train_idx])) == {0, 1}


def test_chimeraboost_classifier_has_classifier_tag():
    assert is_classifier(ChimeraBoostClassifier())
    scorer = resolve_scorer(ChimeraBoostClassifier(), None, None)
    assert scorer.name == "neg_log_loss"
    assert ChimeraBoostSearchCV is ChimeraBoostStepwiseSearchCV


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    fit_calls = []
    score_weights = []

    def __init__(self, iterations=4, early_stopping=False,
                 early_stopping_rounds=None, eval_train_loss=True,
                 thread_count=None, refit=False, random_state=None,
                 tree_mode="catboost", verbose=False):
        self.iterations = iterations
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_train_loss = eval_train_loss
        self.thread_count = thread_count
        self.refit = refit
        self.random_state = random_state
        self.tree_mode = tree_mode
        self.verbose = verbose

    def fit(self, X, y, cat_features=None, eval_set=None,
            sample_weight=None, eval_sample_weight=None):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.best_n_estimators_ = self.iterations
        self.learning_rate_ = 0.1
        self.model_ = type("Model", (), {
            "auto_params_": {
                "threading": {
                    "thread_count_resolved": self.thread_count,
                }
            }
        })()
        self.majority_ = self.classes_[0]
        type(self).fit_calls.append({
            "sample_weight": None if sample_weight is None else sample_weight.copy(),
            "eval_sample_weight": (
                None if eval_sample_weight is None else eval_sample_weight.copy()
            ),
            "eval_set_present": eval_set is not None,
            "thread_count": self.thread_count,
            "early_stopping": self.early_stopping,
            "early_stopping_rounds": self.early_stopping_rounds,
        })
        return self

    def predict_proba(self, X):
        out = np.full((len(X), 2), 0.5)
        return out

    def predict(self, X):
        return np.repeat(self.majority_, len(X))


class FailingClassifier(RecordingClassifier):
    def fit(self, X, y, cat_features=None, eval_set=None,
            sample_weight=None, eval_sample_weight=None):
        raise RuntimeError("intentional fit failure")


def recording_scorer(estimator, X, y, sample_weight=None):
    RecordingClassifier.score_weights.append(
        None if sample_weight is None else sample_weight.copy()
    )
    return -float(np.average(np.ones(len(y)), weights=sample_weight))


def keyword_only_weight_scorer(estimator, X, y, *, sample_weight=None):
    return -float(np.average(np.ones(len(y)), weights=sample_weight))


def positional_weight_scorer(estimator, X, y, weight):
    return -float(np.average(np.ones(len(y)), weights=weight))


def exploding_kwargs_scorer(estimator, X, y, **kwargs):
    if "sample_weight" in kwargs:
        raise TypeError("sample_weight path reached")
    return 0.0


def test_keyword_only_sample_weight_scorer_is_supported():
    estimator = RecordingClassifier(iterations=3).fit(
        np.arange(12).reshape(6, 2),
        np.tile([0, 1], 3),
    )
    scorer = resolve_scorer(estimator, keyword_only_weight_scorer, None)
    score, loss = scorer(
        estimator,
        np.arange(8).reshape(4, 2),
        np.tile([0, 1], 2),
        sample_weight=np.arange(1, 5, dtype=float),
    )
    assert score == -1.0
    assert loss == 1.0


def test_positional_sample_weight_scorer_is_supported():
    estimator = RecordingClassifier(iterations=3).fit(
        np.arange(12).reshape(6, 2),
        np.tile([0, 1], 3),
    )
    scorer = resolve_scorer(estimator, positional_weight_scorer, None)
    score, loss = scorer(
        estimator,
        np.arange(8).reshape(4, 2),
        np.tile([0, 1], 2),
        sample_weight=np.arange(1, 5, dtype=float),
    )
    assert score == -1.0
    assert loss == 1.0


def test_custom_scorer_typeerror_is_not_silently_retried_without_weights():
    estimator = RecordingClassifier(iterations=3).fit(
        np.arange(12).reshape(6, 2),
        np.tile([0, 1], 3),
    )
    scorer = resolve_scorer(estimator, exploding_kwargs_scorer, None)
    with pytest.raises(TypeError, match="sample_weight path reached"):
        scorer(
            estimator,
            np.arange(8).reshape(4, 2),
            np.tile([0, 1], 2),
            sample_weight=np.arange(1, 5, dtype=float),
        )


def test_search_slices_weights_for_fit_eval_and_scoring():
    RecordingClassifier.fit_calls = []
    RecordingClassifier.score_weights = []
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    w = np.arange(1, 13, dtype=float)

    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=3,
        scoring=recording_scorer,
        refit=False,
        trial_thread_count=2,
        random_state=0,
        resume=False,
    )
    search.fit(X, y, sample_weight=w)

    fold_calls = RecordingClassifier.fit_calls
    assert len(fold_calls) == 3
    assert all(call["eval_set_present"] for call in fold_calls)
    assert all(call["thread_count"] == 2 for call in fold_calls)
    assert len(RecordingClassifier.score_weights) == 3
    for call, score_weight in zip(fold_calls, RecordingClassifier.score_weights):
        assert np.array_equal(call["eval_sample_weight"], score_weight)
        assert call["sample_weight"].sum() + score_weight.sum() == w.sum()


def test_search_validates_training_and_eval_payload_shapes():
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=3,
        scoring=recording_scorer,
        refit=False,
        resume=False,
    )

    with pytest.raises(ValueError, match=r"y must have shape \(12,\)"):
        search.fit(X, y[:10])
    with pytest.raises(ValueError, match=r"sample_weight must have shape \(12,\)"):
        search.fit(X, y, sample_weight=np.ones(13))
    with pytest.raises(
        ValueError, match="eval_sample_weight requires an explicit eval_set"
    ):
        search.fit(X, y, eval_sample_weight=np.ones(4))
    with pytest.raises(ValueError, match=r"eval_set\[1\] must have shape"):
        search.fit(X, y, eval_set=(X[:4], y[:5]))
    with pytest.raises(ValueError, match=r"eval_sample_weight must have shape"):
        search.fit(
            X, y, eval_set=(X[:4], y[:4]),
            eval_sample_weight=np.ones(5)
        )


def test_explicit_eval_set_uses_single_validation_payload():
    RecordingClassifier.fit_calls = []
    RecordingClassifier.score_weights = []
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    Xv = np.arange(12).reshape(6, 2)
    yv = np.tile([0, 1], 3)
    w = np.arange(1, 13, dtype=float)
    wv = np.arange(1, 7, dtype=float)

    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=5,
        scoring=recording_scorer,
        refit=False,
        random_state=0,
        resume=False,
    )
    search.fit(X, y, sample_weight=w, eval_set=(Xv, yv), eval_sample_weight=wv)

    assert search.n_splits_ == 1
    assert len(RecordingClassifier.fit_calls) == 1
    assert np.array_equal(RecordingClassifier.fit_calls[0]["sample_weight"], w)
    assert np.array_equal(RecordingClassifier.fit_calls[0]["eval_sample_weight"], wv)
    assert np.array_equal(RecordingClassifier.score_weights[0], wv)


def test_storage_defaults_to_journal_for_multiprocessing():
    cfg = make_storage(None, n_workers=2, study_name="cb-test", resume=True)
    assert cfg.storage_kind == "journal"
    assert cfg.storage_url.startswith("journal://")


def test_default_single_process_storage_is_fresh_in_memory():
    cfg = make_storage(None, n_workers=1, study_name="cb-test", resume=True)
    assert cfg.storage is None
    assert cfg.storage_kind == "in_memory"


def test_custom_storage_object_rejected_for_multiprocessing():
    storage = optuna.storages.InMemoryStorage()
    with pytest.raises(ValueError, match="custom storage object"):
        make_storage(storage, n_workers=2, study_name="cb-test", resume=True)


def test_resume_reopens_named_journal_study(tmp_path):
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    storage = f"journal://{tmp_path / 'resume.log'}"

    first = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=2,
        scoring=recording_scorer,
        refit=False,
        storage=storage,
        study_name="resume-smoke",
        random_state=0,
    ).fit(X, y)
    assert len(first.study_.trials) == 1

    second = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=2,
        scoring=recording_scorer,
        refit=False,
        storage=storage,
        study_name="resume-smoke",
        random_state=0,
    ).fit(X, y)
    assert len(second.study_.trials) == 2


def test_multiprocess_workers_share_journal_study(tmp_path):
    try:
        with ProcessPoolExecutor(max_workers=1):
            pass
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"process pools unavailable in this environment: {exc}")

    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 2},
        cv=2,
        scoring=recording_scorer,
        refit=False,
        n_workers=2,
        storage=f"journal://{tmp_path / 'study.log'}",
        study_name="mp-smoke",
        random_state=0,
        trial_thread_count=1,
        resume=False,
    )
    search.fit(X, y)
    assert len(search.study_.trials) == 2
    assert search.tuning_metadata_["storage_kind"] == "journal"
    assert search.best_trial_params_["thread_count"] == 1
    assert "thread_count" not in search.best_params_


def test_multiprocess_scheduler_uses_spawn_process_workers(monkeypatch, tmp_path):
    captured = {}

    class FakeExecutor:
        def __init__(self, max_workers, mp_context):
            captured["max_workers"] = max_workers
            captured["start_method"] = mp_context.get_start_method()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, iterable):
            captured["worker_fn"] = fn
            captured["worker_args"] = list(iterable)
            return []

    monkeypatch.setattr(search_mod, "ProcessPoolExecutor", FakeExecutor)

    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 2},
        cv=2,
        scoring=recording_scorer,
        refit=False,
        n_workers=2,
        storage=f"journal://{tmp_path / 'study.log'}",
        study_name="mp-contract",
        random_state=0,
        trial_thread_count=1,
        resume=False,
    )
    search.storage_config_ = make_storage(
        search.storage, n_workers=2, study_name=search.study_name, resume=True
    )
    search.study_name_ = search.study_name
    search.n_workers_ = 2
    payload = search_mod._ObjectivePayload(
        estimator=search.estimator,
        X=np.arange(12).reshape(6, 2),
        y=np.tile([0, 1], 3),
        sample_weight=None,
        cat_features=None,
        cv_splits=[(np.array([0, 1, 2, 3]), np.array([4, 5]))],
        eval_payload=None,
        scorer=resolve_scorer(search.estimator, recording_scorer, None),
        phase_spec=make_phase_spec("probe", "catboost", 2),
        lane_state=LaneState("catboost"),
        context=SpaceContext(search.estimator.get_params(), False, True),
        random_state=0,
        trial_thread_count=1,
        error_score=np.nan,
    )
    search._run_phase(payload, 2, optuna)

    assert captured["max_workers"] == 2
    assert captured["start_method"] == "spawn"
    assert captured["worker_fn"] is search_mod._optimize_worker
    assert [args[3] for args in captured["worker_args"]] == [True, True]
    assert [args[6] for args in captured["worker_args"]] == [1, 1]


def test_worker_calls_optuna_with_single_threaded_jobs(monkeypatch, tmp_path):
    captured = {}

    class FakeStudy:
        def optimize(self, objective, n_trials, timeout=None,
                     callbacks=None, n_jobs=1):
            captured["n_trials"] = n_trials
            captured["timeout"] = timeout
            captured["callbacks"] = callbacks
            captured["n_jobs"] = n_jobs
            captured["objective"] = objective

    def fake_create_study(**kwargs):
        return FakeStudy()

    monkeypatch.setattr(search_mod, "create_study", fake_create_study)

    payload = search_mod._ObjectivePayload(
        estimator=RecordingClassifier(iterations=3),
        X=np.arange(12).reshape(6, 2),
        y=np.tile([0, 1], 3),
        sample_weight=None,
        cat_features=None,
        cv_splits=[(np.array([0, 1, 2, 3]), np.array([4, 5]))],
        eval_payload=None,
        scorer=resolve_scorer(RecordingClassifier(), recording_scorer, None),
        phase_spec=make_phase_spec("probe", "catboost", 1),
        lane_state=LaneState("catboost"),
        context=SpaceContext(RecordingClassifier().get_params(), False, True),
        random_state=0,
        trial_thread_count=1,
        error_score=np.nan,
    )
    search_mod._optimize_worker((
        payload,
        f"journal://{tmp_path / 'worker.log'}",
        "worker-contract",
        True,
        None,
        None,
        3,
        None,
        None,
    ))
    assert captured["n_trials"] == 3
    assert captured["n_jobs"] == 1
    assert isinstance(captured["objective"], search_mod._TrialObjective)


class PruneAfterFirstFold(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        return bool(trial.intermediate_values)


def test_fold_boundary_pruning_stops_after_first_fold():
    RecordingClassifier.fit_calls = []
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=3,
        scoring=recording_scorer,
        refit=False,
        pruner=PruneAfterFirstFold(),
        random_state=0,
        resume=False,
    )
    with pytest.raises(ValueError, match="no completed tuning trials"):
        search.fit(X, y)
    assert len(RecordingClassifier.fit_calls) == 1


def test_pruning_is_not_swallowed_by_finite_error_score():
    RecordingClassifier.fit_calls = []
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=3,
        scoring=recording_scorer,
        refit=False,
        pruner=PruneAfterFirstFold(),
        random_state=0,
        resume=False,
        error_score=0.0,
    )
    with pytest.raises(ValueError, match="no completed tuning trials"):
        search.fit(X, y)
    assert search.study_.trials[0].state == optuna.trial.TrialState.PRUNED
    assert len(RecordingClassifier.fit_calls) == 1


class PruneTrialZero(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        return trial.number == 0 and bool(trial.intermediate_values)


def test_best_index_is_cv_results_row_index_after_pruned_trial():
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 2},
        cv=2,
        scoring=recording_scorer,
        refit=False,
        pruner=PruneTrialZero(),
        random_state=0,
        resume=False,
    )
    search.fit(X, y)
    assert search.cv_results_["trial_number"] == [0, 1]
    assert search.best_index_ == 1
    assert search.cv_results_["trial_number"][search.best_index_] == 1


def test_finite_error_score_trial_remains_visible():
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    search = ChimeraBoostStepwiseSearchCV(
        FailingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=2,
        scoring=recording_scorer,
        refit=False,
        random_state=0,
        resume=False,
        error_score=123.0,
    )
    search.fit(X, y)
    assert search.best_loss_ == 123.0
    assert search.cv_results_["status"] == ["ERROR_SCORE"]
    assert search.cv_results_["error"][0].startswith("RuntimeError")
    assert search.cv_results_["params"][0]["tree_mode"] == "catboost"


def test_trial_runtime_does_not_force_fixed_patience():
    RecordingClassifier.fit_calls = []
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=2,
        scoring=recording_scorer,
        refit=False,
        random_state=0,
        resume=False,
    )
    search.fit(X, y)
    assert {call["early_stopping_rounds"] for call in RecordingClassifier.fit_calls} == {None}


def test_learning_rate_phase_does_not_sample_patience():
    context = SpaceContext(
        estimator_params=ChimeraBoostClassifier().get_params(),
        has_categoricals=False,
        classifier=True,
    )
    params = suggest_learning_rate(FakeTrial(), context, LaneState("catboost"))
    assert "early_stopping_rounds" not in params


def test_lane_fixed_params_are_applied_without_suggestor_convention():
    params = search_mod._build_model_params(
        RecordingClassifier(),
        LaneState("catboost", fixed_params={"tree_mode": "catboost"}),
        {},
    )
    assert params["tree_mode"] == "catboost"


def test_lane_build_params_drop_leafwise_only_num_leaves_for_catboost():
    estimator = ChimeraBoostRegressor(tree_mode="lightgbm", num_leaves=31)

    catboost_params = search_mod._build_model_params(
        estimator,
        LaneState("catboost", fixed_params={"tree_mode": "catboost"}),
        {},
    )
    lightgbm_params = search_mod._build_model_params(
        estimator,
        LaneState("lightgbm", fixed_params={"tree_mode": "lightgbm"}),
        {},
    )

    assert catboost_params["num_leaves"] is None
    assert lightgbm_params["num_leaves"] == 31


def test_default_refit_preserves_rounds_and_auto_learning_rate():
    X, y = load_breast_cancer(return_X_y=True)
    search = ChimeraBoostStepwiseSearchCV(
        ChimeraBoostClassifier(iterations=16, learning_rate=None, random_state=0),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=2,
        refit=True,
        random_state=0,
        resume=False,
        trial_thread_count=1,
    )
    search.fit(X[:120], y[:120])
    assert search.refit_params_["early_stopping"] is False
    assert search.refit_params_["early_stopping_rounds"] is None
    assert search.refit_params_["iterations"] == 16
    assert "learning_rate" not in search.refit_params_
    assert search.predict(X[:3]).shape == (3,)


def test_median_fold_refit_is_opt_in():
    X, y = load_breast_cancer(return_X_y=True)
    search = ChimeraBoostStepwiseSearchCV(
        ChimeraBoostClassifier(iterations=16, random_state=0),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=2,
        refit=True,
        refit_rounds="median_best",
        refit_learning_rate="fold_median",
        random_state=0,
        resume=False,
        trial_thread_count=1,
    )
    search.fit(X[:120], y[:120])
    assert search.refit_params_["iterations"] == search.best_n_estimators_
    assert search.refit_params_["learning_rate"] == search.learning_rate_


def test_numeric_n_trials_is_global_across_lanes():
    X = np.arange(80).reshape(40, 2)
    y = np.tile([0, 1], 20)
    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        strategy="stepwise",
        tree_modes=("catboost", "lightgbm"),
        n_trials=7,
        cv=2,
        scoring=recording_scorer,
        refit=False,
        random_state=0,
        resume=False,
    )
    search.fit(X, y)
    assert len(search.study_.trials) == 7
    assert sum(block["n_trials"] for block in search.tuning_metadata_["budget_plan"]) == 7


def test_twenty_trial_auto_plan_is_optgbm_like_joint_budget():
    blocks = search_mod._make_budget_plan(
        n_trials=20,
        timeout=None,
        phases=phase_names("auto"),
        tree_modes=("catboost", "lightgbm"),
        strategy="joint",
    )
    assert blocks == [
        search_mod.TrialBlock("probe", "catboost", 1),
        search_mod.TrialBlock("probe", "lightgbm", 1),
        search_mod.TrialBlock("joint_compact", None, 18),
    ]


def test_thousand_trial_auto_plan_uses_large_budget_warmup():
    blocks = search_mod._make_budget_plan(
        n_trials=1000,
        timeout=None,
        phases=phase_names("auto"),
        tree_modes=("catboost", "lightgbm"),
        strategy="stepwise",
    )
    assert sum(block.n_trials for block in blocks) == 1000
    assert blocks[:3] == [
        search_mod.TrialBlock("probe", "catboost", 1),
        search_mod.TrialBlock("probe", "lightgbm", 1),
        search_mod.TrialBlock("joint_compact", None, 198),
    ]
    by_phase = {}
    for block in blocks:
        by_phase[block.phase] = by_phase.get(block.phase, 0) + block.n_trials
    assert by_phase["structure"] > by_phase["sampling_regularization"]
    assert by_phase["learning_rate"] > by_phase["binning_categorical"]
    assert any(
        block.phase == "structure" and block.lane is None
        for block in blocks
    )
    assert not any(
        block.phase == "structure" and block.lane is not None
        for block in blocks
    )


def test_thousand_trial_plan_honors_explicit_split_noise_phase():
    blocks = search_mod._make_budget_plan(
        n_trials=1000,
        timeout=None,
        phases=(
            "probe",
            "structure",
            "sampling_regularization",
            "learning_rate",
            "binning_categorical",
            "split_noise",
        ),
        tree_modes=("catboost", "lightgbm"),
        strategy="stepwise",
    )
    by_phase = {}
    for block in blocks:
        by_phase[block.phase] = by_phase.get(block.phase, 0) + block.n_trials
    assert sum(by_phase.values()) == 1000
    assert by_phase["joint_compact"] == 198
    assert by_phase["split_noise"] == 25


def test_adaptive_lane_allocation_favors_current_best_lane():
    lane_states = {
        "catboost": LaneState("catboost", best_loss=0.1),
        "lightgbm": LaneState("lightgbm", best_loss=0.2),
    }
    blocks = search_mod._adaptive_phase_to_lanes(
        "structure", 100, ("catboost", "lightgbm"), lane_states
    )
    assert blocks == [
        search_mod.TrialBlock("structure", "catboost", 70),
        search_mod.TrialBlock("structure", "lightgbm", 30),
    ]


def _tell_fake_complete_trial(search, payload, value=1.0):
    trial = search.study_.ask()
    params = {"tree_mode": payload.lane_state.tree_mode}
    trial.set_user_attr("phase", payload.phase_spec.name)
    trial.set_user_attr("tree_mode_lane", payload.lane_state.tree_mode)
    trial.set_user_attr("params_full", params)
    trial.set_user_attr("params_model", params)
    trial.set_user_attr("mean_score", -float(value))
    trial.set_user_attr("mean_loss", float(value))
    trial.set_user_attr("status", "OK")
    search.study_.tell(trial, float(value))


def test_fit_recomputes_timeout_for_each_adaptive_execution_block(monkeypatch):
    X = np.arange(80).reshape(40, 2)
    y = np.tile([0, 1], 20)
    calls = []
    remaining = iter([10.0, 0.0])

    def fake_remaining_timeout(timeout, started_at):
        return next(remaining, 0.0)

    def fake_run_phase(self, payload, n_trials, optuna, *,
                       timeout=None, callbacks=None):
        calls.append((payload.phase_spec.name, payload.lane_state.tree_mode, timeout))
        _tell_fake_complete_trial(self, payload, value=1.0)

    monkeypatch.setattr(search_mod, "_remaining_timeout", fake_remaining_timeout)
    monkeypatch.setattr(
        ChimeraBoostStepwiseSearchCV, "_run_phase", fake_run_phase
    )

    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("structure",),
        tree_modes=("catboost", "lightgbm"),
        n_trials={"structure": 2},
        cv=2,
        scoring=recording_scorer,
        timeout=30.0,
        refit=False,
        random_state=0,
        resume=False,
    )
    search.fit(X, y)

    assert calls == [("structure", "catboost", 10.0)]
    assert search.tuning_metadata_["budget_plan"] == [
        {"phase": "structure", "lane": "catboost", "n_trials": 1}
    ]


def test_shared_stop_flag_breaks_later_adaptive_execution_blocks(monkeypatch):
    X = np.arange(80).reshape(40, 2)
    y = np.tile([0, 1], 20)
    calls = []

    def fake_run_phase(self, payload, n_trials, optuna, *,
                       timeout=None, callbacks=None):
        calls.append((payload.phase_spec.name, payload.lane_state.tree_mode))
        _tell_fake_complete_trial(self, payload, value=1.0)
        self.study_.set_user_attr("_chimeraboost_stop", True)

    monkeypatch.setattr(
        ChimeraBoostStepwiseSearchCV, "_run_phase", fake_run_phase
    )

    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        phases=("structure",),
        tree_modes=("catboost", "lightgbm"),
        n_trials={"structure": 2},
        cv=2,
        scoring=recording_scorer,
        refit=False,
        random_state=0,
        resume=False,
    )
    search.fit(X, y)

    assert calls == [("structure", "catboost")]
    assert search.study_.user_attrs["_chimeraboost_stop"] is True


def test_auto_small_budget_uses_joint_strategy():
    X = np.arange(80).reshape(40, 2)
    y = np.tile([0, 1], 20)
    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        tree_modes=("catboost", "lightgbm"),
        n_trials=4,
        cv=2,
        scoring=recording_scorer,
        refit=False,
        random_state=0,
        resume=False,
    )
    search.fit(X, y)
    assert search.strategy_ == "joint"
    assert len(search.study_.trials) == 4
    assert "joint_compact" in search.cv_results_["phase"]


def test_timeout_only_auto_uses_joint_unbounded_block():
    blocks = search_mod._make_budget_plan(
        n_trials=None,
        timeout=30.0,
        phases=phase_names("auto"),
        tree_modes=("catboost", "lightgbm"),
        strategy="joint",
    )
    assert blocks[-1].phase == "joint_compact"
    assert blocks[-1].n_trials is None


def test_timeout_only_stepwise_requires_trial_budget():
    with pytest.raises(ValueError, match="timeout-only tuning requires"):
        search_mod._make_budget_plan(
            n_trials=None,
            timeout=30.0,
            phases=phase_names("auto"),
            tree_modes=("catboost", "lightgbm"),
            strategy="stepwise",
        )


def test_no_improvement_stopper_sets_shared_stop_flag():
    X = np.arange(40).reshape(20, 2)
    y = np.tile([0, 1], 10)
    search = ChimeraBoostStepwiseSearchCV(
        RecordingClassifier(iterations=3),
        strategy="joint",
        tree_modes=("catboost",),
        n_trials=5,
        cv=2,
        scoring=recording_scorer,
        refit=False,
        random_state=0,
        resume=False,
        early_stop_patience=1,
        early_stop_min_trials=1,
    )
    search.fit(X, y)
    assert len(search.study_.trials) < 5
    assert search.study_.user_attrs["_chimeraboost_stop"] is True


def test_classifier_and_regressor_smoke():
    X, y = load_breast_cancer(return_X_y=True)
    clf = ChimeraBoostStepwiseSearchCV(
        ChimeraBoostClassifier(iterations=12, random_state=0),
        phases=("probe", "structure"),
        tree_modes=("catboost", "lightgbm"),
        n_trials={"probe": 1, "structure": 1},
        cv=2,
        refit=True,
        random_state=0,
        resume=False,
        trial_thread_count=1,
    ).fit(X[:140], y[:140])
    assert clf.best_loss_ < np.inf
    assert clf.predict_proba(X[:5]).shape == (5, 2)

    Xr, yr = load_diabetes(return_X_y=True)
    reg = ChimeraBoostStepwiseSearchCV(
        ChimeraBoostRegressor(iterations=10, random_state=0),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=2,
        refit=True,
        random_state=0,
        resume=False,
        trial_thread_count=1,
    ).fit(Xr[:120], yr[:120])
    assert reg.predict(Xr[:4]).shape == (4,)
