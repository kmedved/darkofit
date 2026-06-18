"""Laned stepwise Optuna tuning around ChimeraBoost sklearn wrappers."""

from __future__ import annotations

import math
import os
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.utils.validation import check_is_fitted

from .._validation import n_features_from_array_like, normalize_cat_features
from .optuna_backend import create_study, import_optuna, load_study, make_storage
from .results import build_cv_results, phase_summary, weighted_mean
from .scoring import resolve_scorer, score_estimator
from .spaces import (
    LaneState,
    SpaceContext,
    make_phase_spec,
    phase_names,
)
from .validation import make_cv_splits, slice_fit_payload, validation_mass


class ChimeraBoostStepwiseSearchCV(BaseEstimator):
    """Stepwise Optuna search with separate CatBoost/LightGBM lanes."""

    def __init__(
        self,
        estimator,
        *,
        phases="auto",
        tree_modes=("catboost", "lightgbm"),
        cv=5,
        validation_fraction=0.2,
        scoring=None,
        greater_is_better=None,
        strategy="auto",
        n_trials=100,
        timeout=None,
        optuna_callbacks=None,
        study_stopper=None,
        early_stop_patience=None,
        early_stop_min_trials=20,
        early_stop_min_delta=0.0,
        refit=True,
        refit_rounds="preserve",
        refit_learning_rate="preserve",
        n_workers=1,
        trial_thread_count="auto",
        storage=None,
        study_name=None,
        resume=True,
        sampler=None,
        pruner=None,
        random_state=None,
        error_score=np.nan,
        verbose=False,
    ):
        self.estimator = estimator
        self.phases = phases
        self.tree_modes = tree_modes
        self.cv = cv
        self.validation_fraction = validation_fraction
        self.scoring = scoring
        self.greater_is_better = greater_is_better
        self.strategy = strategy
        self.n_trials = n_trials
        self.timeout = timeout
        self.optuna_callbacks = optuna_callbacks
        self.study_stopper = study_stopper
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_trials = early_stop_min_trials
        self.early_stop_min_delta = early_stop_min_delta
        self.refit = refit
        self.refit_rounds = refit_rounds
        self.refit_learning_rate = refit_learning_rate
        self.n_workers = n_workers
        self.trial_thread_count = trial_thread_count
        self.storage = storage
        self.study_name = study_name
        self.resume = resume
        self.sampler = sampler
        self.pruner = pruner
        self.random_state = random_state
        self.error_score = error_score
        self.verbose = verbose

    def fit(self, X, y, *, cat_features=None, groups=None, sample_weight=None,
            eval_set=None, eval_sample_weight=None):
        optuna = import_optuna()
        if self.n_trials is None and self.timeout is None:
            raise ValueError("at least one of n_trials or timeout must be set")
        y_arr = np.asarray(y)
        sample_weight_arr = (
            None if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        classifier = is_classifier(self.estimator)
        self.scorer_ = resolve_scorer(
            self.estimator, self.scoring, self.greater_is_better
        )
        n_features = n_features_from_array_like(X)
        cat_features_normalized = normalize_cat_features(cat_features, n_features)
        self.cat_features_ = (
            None if cat_features is None else list(cat_features_normalized)
        )
        self.n_workers_ = max(1, int(self.n_workers))
        self.trial_thread_count_ = _resolve_trial_thread_count(
            self.trial_thread_count, self.n_workers_
        )
        self.study_name_ = (
            self.study_name
            or f"chimeraboost-stepwise-search-{uuid.uuid4().hex}"
        )
        self.storage_config_ = make_storage(
            self.storage,
            n_workers=self.n_workers_,
            study_name=self.study_name_,
            resume=self.resume,
        )
        self.study_ = create_study(
            storage_config=self.storage_config_,
            study_name=self.study_name_,
            resume=self.resume,
            sampler=self.sampler,
            pruner=self.pruner,
        )

        eval_payload = None
        if eval_set is None:
            self.cv_splits_ = make_cv_splits(
                X, y_arr, cv=self.cv, groups=groups, classifier=classifier,
                random_state=self.random_state,
                validation_fraction=self.validation_fraction,
            )
        else:
            X_eval, y_eval = eval_set
            eval_payload = (
                X_eval,
                np.asarray(y_eval),
                None if eval_sample_weight is None else np.asarray(
                    eval_sample_weight, dtype=np.float64
                ),
            )
            self.cv_splits_ = [(np.arange(y_arr.shape[0]), None)]
        self.n_splits_ = len(self.cv_splits_)

        context = SpaceContext(
            estimator_params=self.estimator.get_params(),
            has_categoricals=bool(cat_features_normalized),
            classifier=classifier,
            tree_modes=tuple(self.tree_modes),
        )
        lane_states = {
            mode: LaneState(mode, fixed_params={"tree_mode": mode})
            for mode in self.tree_modes
        }
        self.strategy_ = _resolve_strategy(
            self.strategy, self.n_trials, self.phases
        )
        planned_blocks = _make_budget_plan(
            n_trials=self.n_trials,
            timeout=self.timeout,
            phases=phase_names(self.phases),
            tree_modes=tuple(self.tree_modes),
            strategy=self.strategy_,
        )
        self.budget_plan_requested_ = list(planned_blocks)
        self.budget_plan_ = []
        callbacks = _make_optuna_callbacks(
            self.optuna_callbacks,
            self.study_stopper,
            self.early_stop_patience,
            self.early_stop_min_trials,
            self.early_stop_min_delta,
        )
        started_at = time.monotonic()

        stop_requested = False
        for planned_block in planned_blocks:
            if _study_stop_requested(self.study_):
                break
            execution_blocks = _execution_blocks(
                planned_block, tuple(self.tree_modes), lane_states
            )
            for block in execution_blocks:
                if _study_stop_requested(self.study_):
                    stop_requested = True
                    break
                remaining_timeout = _remaining_timeout(self.timeout, started_at)
                if remaining_timeout is not None and remaining_timeout <= 0:
                    stop_requested = True
                    break
                self.budget_plan_.append(block)
                lane = block.lane
                state = (
                    lane_states[lane]
                    if lane is not None
                    else LaneState("joint", fixed_params={})
                )
                spec = make_phase_spec(block.phase, lane, block.n_trials)
                objective_payload = _ObjectivePayload(
                    estimator=self.estimator,
                    X=X,
                    y=y_arr,
                    sample_weight=sample_weight_arr,
                    cat_features=self.cat_features_,
                    cv_splits=self.cv_splits_,
                    eval_payload=eval_payload,
                    scorer=self.scorer_,
                    phase_spec=spec,
                    lane_state=state,
                    context=context,
                    random_state=self.random_state,
                    trial_thread_count=self.trial_thread_count_,
                    error_score=self.error_score,
                )
                self._run_phase(
                    objective_payload, spec.n_trials, optuna,
                    timeout=remaining_timeout, callbacks=callbacks,
                )
                if self.storage_config_.storage is not None:
                    self.study_ = load_study(
                        storage_config=self.storage_config_,
                        study_name=self.study_name_,
                    )
                if block.phase == "joint_compact":
                    for mode in self.tree_modes:
                        lane_states[mode] = _updated_lane_state(
                            self.study_, mode
                    )
                elif lane is not None:
                    lane_states[lane] = _updated_lane_state(self.study_, lane)
                if _study_stop_requested(self.study_):
                    stop_requested = True
                    break
            if stop_requested:
                break

        completed = [
            t for t in self.study_.trials
            if _is_usable_trial(t)
        ]
        if not completed:
            raise ValueError("no completed tuning trials")
        self.best_trial_ = min(completed, key=lambda t: t.value)
        self.cv_results_ = build_cv_results(self.study_.trials)
        trial_numbers = list(self.cv_results_["trial_number"])
        self.best_index_ = trial_numbers.index(self.best_trial_.number)
        self.best_params_ = dict(
            self.best_trial_.user_attrs.get("params_model")
            or self.best_trial_.user_attrs["params_full"]
        )
        self.best_trial_params_ = dict(self.best_trial_.user_attrs["params_full"])
        self.best_loss_ = float(self.best_trial_.value)
        self.best_score_ = float(self.best_trial_.user_attrs["mean_score"])
        self.phase_results_ = phase_summary(self.study_.trials)
        self.tuning_metadata_ = {
            "tree_modes": list(self.tree_modes),
            "phases": list(phase_names(self.phases)),
            "strategy": self.strategy_,
            "budget_plan_requested": [
                block.__dict__.copy() for block in self.budget_plan_requested_
            ],
            "budget_plan": [block.__dict__.copy() for block in self.budget_plan_],
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "n_workers": self.n_workers_,
            "trial_thread_count": self.trial_thread_count_,
            "parallel_backend": (
                "processes" if self.n_workers_ > 1 else "serial"
            ),
            "optuna_n_jobs": 1,
            "storage_kind": self.storage_config_.storage_kind,
            "storage_url": self.storage_config_.storage_url,
            "resume": bool(self.resume),
            "random_state": self.random_state,
            "refit_rounds": self.refit_rounds,
            "refit_learning_rate": self.refit_learning_rate,
            "objective": "weighted_cv_loss",
        }

        if self.refit:
            self._refit_best(X, y_arr, sample_weight_arr, cat_features)
        return self

    def predict(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "best_estimator_")
        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError("best_estimator_ does not support predict_proba")
        return self.best_estimator_.predict_proba(X)

    def staged_predict(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.staged_predict(X)

    def staged_predict_proba(self, X):
        check_is_fitted(self, "best_estimator_")
        if not hasattr(self.best_estimator_, "staged_predict_proba"):
            raise AttributeError("best_estimator_ does not support staged_predict_proba")
        return self.best_estimator_.staged_predict_proba(X)

    def _run_phase(self, objective_payload, n_trials, optuna, *,
                   timeout=None, callbacks=None):
        if self.n_workers_ <= 1:
            objective = _TrialObjective(objective_payload)
            self.study_.optimize(
                objective, n_trials=n_trials, timeout=timeout,
                callbacks=callbacks, n_jobs=1,
            )
            return

        if self.storage_config_.storage is None:
            raise ValueError("multiprocessing requires shared Optuna storage")
        quotas = _split_quota(n_trials, self.n_workers_)
        worker_payload = _WorkerPayload(
            objective_payload=objective_payload,
            storage_url=self.storage_config_.storage_url,
            study_name=self.study_name_,
            # The parent process already applied the user's resume policy when
            # creating the study. Workers must attach to that existing study.
            resume=True,
            sampler=self.sampler,
            pruner=self.pruner,
            n_trials_by_worker=quotas,
            timeout=timeout,
            callbacks=callbacks,
        )
        # Use process-level parallelism only. Optuna's n_jobs>1 would run
        # trials as threads inside one Python process, racing with Numba's
        # process-global thread pool that ChimeraBoost configures per fit.
        ctx = get_context("spawn")
        try:
            with ProcessPoolExecutor(
                max_workers=len(quotas), mp_context=ctx
            ) as executor:
                list(executor.map(_optimize_worker, worker_payload.iter_workers()))
        except (NotImplementedError, PermissionError) as exc:
            raise RuntimeError(
                "multiprocessing is unavailable in this Python environment; "
                "set n_workers=1 or run outside the restricted sandbox"
            ) from exc

    def _refit_best(self, X, y, sample_weight, cat_features):
        params = dict(self.best_params_)
        fold_iterations = self.best_trial_.user_attrs.get("fold_best_iterations") or []
        fold_lrs = self.best_trial_.user_attrs.get("fold_learning_rates") or []
        if self.refit_rounds == "preserve":
            pass
        elif self.refit_rounds == "median_best" and fold_iterations:
            params["iterations"] = max(1, int(math.ceil(np.median(fold_iterations))))
        else:
            raise ValueError(
                "refit_rounds must be 'preserve' or 'median_best'"
            )
        if self.refit_learning_rate == "preserve":
            pass
        elif self.refit_learning_rate == "explicit":
            pass
        elif self.refit_learning_rate == "fold_median":
            if fold_lrs:
                params["learning_rate"] = float(np.median(fold_lrs))
        else:
            raise ValueError(
                "refit_learning_rate must be 'preserve', 'explicit', "
                "or 'fold_median'"
            )
        params["early_stopping"] = False
        params["early_stopping_rounds"] = None
        params["refit"] = False
        params["thread_count"] = _resolve_trial_thread_count(
            self.trial_thread_count, 1
        )
        params = _filter_params(params, self.estimator)

        final = clone(self.estimator).set_params(**params)
        final.fit(X, y, cat_features=cat_features, sample_weight=sample_weight)
        self.best_estimator_ = final
        self.refit_params_ = params
        self.best_n_estimators_ = getattr(final, "best_n_estimators_", None)
        self.learning_rate_ = getattr(final, "learning_rate_", None)
        if hasattr(final, "classes_"):
            self.classes_ = final.classes_
            self.n_classes_ = final.n_classes_


@dataclass(frozen=True)
class TrialBlock:
    phase: str
    lane: str | None
    n_trials: int | None


@dataclass
class _ObjectivePayload:
    estimator: object
    X: object
    y: object
    sample_weight: object
    cat_features: object
    cv_splits: object
    eval_payload: object
    scorer: object
    phase_spec: object
    lane_state: object
    context: object
    random_state: object
    trial_thread_count: int
    error_score: float


class _TrialObjective:
    def __init__(self, payload):
        self.payload = payload

    def __call__(self, trial):
        optuna = import_optuna()
        start = time.perf_counter()
        payload = self.payload
        suggested = payload.phase_spec.suggest(
            trial, payload.context, payload.lane_state
        )
        params_model = _build_model_params(
            payload.estimator, payload.lane_state, suggested
        )
        params = _build_trial_params(
            payload.estimator, params_model, payload.trial_thread_count
        )
        tree_mode = params.get("tree_mode", payload.lane_state.tree_mode)
        _set_trial_identity_attrs(trial, payload, tree_mode, params,
                                  params_model, suggested)

        fold_scores = []
        fold_losses = []
        fold_masses = []
        fold_best_iterations = []
        fold_learning_rates = []
        fold_auto_params = []
        resolved_threads = []

        try:
            for fold_idx, (train_idx, valid_idx) in enumerate(payload.cv_splits):
                est = clone(payload.estimator).set_params(**params)
                if "random_state" in est.get_params():
                    est.set_params(random_state=_fold_seed(
                        payload.random_state, trial.number, fold_idx
                    ))

                if payload.eval_payload is None:
                    (X_train, y_train, X_valid, y_valid,
                     w_train, w_valid) = slice_fit_payload(
                        payload.X, payload.y, train_idx, valid_idx,
                        payload.sample_weight,
                    )
                    mass = validation_mass(valid_idx, payload.sample_weight)
                else:
                    X_valid, y_valid, w_valid = payload.eval_payload
                    X_train, y_train = payload.X, payload.y
                    w_train = payload.sample_weight
                    mass = float(len(y_valid) if w_valid is None else np.sum(w_valid))

                est.fit(
                    X_train,
                    y_train,
                    cat_features=payload.cat_features,
                    eval_set=(X_valid, y_valid),
                    sample_weight=w_train,
                    eval_sample_weight=w_valid,
                )
                score, loss = score_estimator(
                    est, payload.scorer, X_valid, y_valid, w_valid
                )
                fold_scores.append(score)
                fold_losses.append(loss)
                fold_masses.append(mass)
                fold_best_iterations.append(int(getattr(est, "best_n_estimators_", 0)))
                fold_learning_rates.append(float(getattr(est, "learning_rate_", np.nan)))
                auto_params = getattr(getattr(est, "model_", None), "auto_params_", {})
                fold_auto_params.append(auto_params)
                resolved = (
                    auto_params.get("threading", {})
                    .get("thread_count_resolved", payload.trial_thread_count)
                )
                resolved_threads.append(resolved)

                aggregate = weighted_mean(fold_losses, fold_masses)
                trial.report(aggregate, step=fold_idx)
                trial.set_user_attr("fold_losses", list(fold_losses))
                trial.set_user_attr("fold_scores", list(fold_scores))
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            if np.isnan(payload.error_score):
                raise
            error_loss = float(payload.error_score)
            trial.set_user_attr("status", "ERROR_SCORE")
            trial.set_user_attr("error", repr(exc))
            trial.set_user_attr("fold_scores", [])
            trial.set_user_attr("fold_losses", [])
            trial.set_user_attr("fold_weight_sums", [])
            trial.set_user_attr("mean_score", -error_loss)
            trial.set_user_attr("std_score", float("nan"))
            trial.set_user_attr("mean_loss", error_loss)
            trial.set_user_attr("std_loss", float("nan"))
            trial.set_user_attr("fit_time", float(time.perf_counter() - start))
            return float(payload.error_score)

        mean_loss = weighted_mean(fold_losses, fold_masses)
        mean_score = weighted_mean(fold_scores, fold_masses)
        trial.set_user_attr("status", "OK")
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_losses", fold_losses)
        trial.set_user_attr("fold_weight_sums", fold_masses)
        trial.set_user_attr("fold_best_iterations", fold_best_iterations)
        trial.set_user_attr("fold_learning_rates", fold_learning_rates)
        trial.set_user_attr("fold_auto_params", fold_auto_params)
        trial.set_user_attr("thread_count_requested", payload.trial_thread_count)
        trial.set_user_attr("thread_count_resolved", resolved_threads)
        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("std_score", float(np.std(fold_scores)))
        trial.set_user_attr("mean_loss", mean_loss)
        trial.set_user_attr("std_loss", float(np.std(fold_losses)))
        trial.set_user_attr("mean_best_iteration", float(np.mean(fold_best_iterations)))
        trial.set_user_attr("median_best_iteration", float(np.median(fold_best_iterations)))
        trial.set_user_attr("mean_learning_rate", float(np.mean(fold_learning_rates)))
        trial.set_user_attr("fit_time", float(time.perf_counter() - start))
        return mean_loss


@dataclass
class _WorkerPayload:
    objective_payload: object
    storage_url: str
    study_name: str
    resume: bool
    sampler: object
    pruner: object
    n_trials_by_worker: list[int]
    timeout: float | None = None
    callbacks: object = None

    def iter_workers(self):
        for n_trials in self.n_trials_by_worker:
            if n_trials is None or n_trials > 0:
                yield (
                    self.objective_payload,
                    self.storage_url,
                    self.study_name,
                    self.resume,
                    self.sampler,
                    self.pruner,
                    n_trials,
                    self.timeout,
                    self.callbacks,
                )


def _optimize_worker(args):
    (objective_payload, storage_url, study_name, resume, sampler, pruner,
     n_trials, timeout, callbacks) = args
    storage_config = make_storage(
        storage_url, n_workers=1, study_name=study_name, resume=resume
    )
    study = create_study(
        storage_config=storage_config,
        study_name=study_name,
        resume=resume,
        sampler=sampler,
        pruner=pruner,
    )
    # One trial thread per worker process: Chimeraboost itself receives a
    # bounded thread_count, so Optuna must not add thread-level parallelism.
    study.optimize(
        _TrialObjective(objective_payload),
        n_trials=n_trials,
        timeout=timeout,
        callbacks=callbacks,
        n_jobs=1,
    )


_RUNTIME_PARAM_KEYS = frozenset({
    "early_stopping",
    "early_stopping_rounds",
    "eval_train_loss",
    "thread_count",
    "verbose",
    "refit",
})


def _build_model_params(estimator, lane_state, suggested):
    params = estimator.get_params()
    params.update(lane_state.fixed_params)
    params.update(lane_state.best_params)
    params.update({
        key: value for key, value in suggested.items()
        if key not in _RUNTIME_PARAM_KEYS
    })
    return _filter_params(params, estimator)


def _build_trial_params(estimator, params_model, thread_count):
    params = dict(params_model)
    params.update(
        early_stopping=True,
        eval_train_loss=False,
        verbose=False,
        thread_count=int(thread_count),
        refit=False,
    )
    return _filter_params(params, estimator)


def _set_trial_identity_attrs(trial, payload, tree_mode, params, params_model,
                              suggested):
    trial.set_user_attr("phase", payload.phase_spec.name)
    trial.set_user_attr("tree_mode_lane", tree_mode)
    trial.set_user_attr("params_full", dict(params))
    trial.set_user_attr("params_model", dict(params_model))
    trial.set_user_attr("params_suggested", dict(suggested))
    trial.set_user_attr("sampled_keys", list(suggested.keys()))
    trial.set_user_attr("params_fixed", dict(payload.lane_state.best_params))


def _filter_params(params, estimator):
    known = estimator.get_params()
    out = {}
    for key, value in params.items():
        if key not in known:
            continue
        if value is None and key not in {"num_leaves", "early_stopping_rounds"}:
            continue
        out[key] = value
    return out


def _updated_lane_state(study, lane):
    candidates = [
        t for t in study.trials
        if _is_usable_trial(t)
        and t.user_attrs.get("tree_mode_lane") == lane
    ]
    if not candidates:
        return LaneState(lane, fixed_params={"tree_mode": lane})
    best = min(candidates, key=lambda t: t.value)
    return LaneState(
        tree_mode=lane,
        fixed_params={"tree_mode": lane},
        best_params=dict(
            best.user_attrs.get("params_model")
            or best.user_attrs["params_full"]
        ),
        best_loss=float(best.value),
        best_score=float(best.user_attrs.get("mean_score", -np.inf)),
        best_fold_iterations=list(best.user_attrs.get("fold_best_iterations", [])),
        best_fold_learning_rates=list(best.user_attrs.get("fold_learning_rates", [])),
    )


def _resolve_strategy(strategy, n_trials, phases):
    if strategy not in {"auto", "joint", "stepwise"}:
        raise ValueError("strategy must be 'auto', 'joint', or 'stepwise'")
    if strategy != "auto":
        return strategy
    if isinstance(n_trials, Mapping):
        return "stepwise"
    if phases != "auto":
        return "stepwise"
    if n_trials is None:
        return "joint"
    return "joint" if int(n_trials) <= 50 else "stepwise"


def _is_usable_trial(trial):
    state_name = getattr(trial.state, "name", str(trial.state).split(".")[-1])
    return (
        state_name == "COMPLETE"
        and trial.value is not None
        and trial.user_attrs.get("params_full") is not None
    )


def _study_stop_requested(study):
    return bool(
        getattr(study, "user_attrs", {}).get("_chimeraboost_stop")
        or getattr(study, "_stop_flag", False)
    )


def _make_budget_plan(n_trials, timeout, phases, tree_modes, strategy):
    tree_modes = tuple(tree_modes)
    if isinstance(n_trials, Mapping):
        return _mapping_budget_plan(n_trials, phases, tree_modes, strategy)
    if n_trials is None:
        if timeout is None:
            raise ValueError("timeout is required when n_trials=None")
        return _timeout_budget_plan(tree_modes, strategy)
    n_trials = max(0, int(n_trials))
    if n_trials <= 0:
        return []
    if strategy == "joint":
        return _joint_budget_plan(n_trials, tree_modes)
    return _stepwise_budget_plan(n_trials, phases, tree_modes)


def _mapping_budget_plan(n_trials, phases, tree_modes, strategy):
    if strategy == "joint":
        total = sum(max(0, int(v)) for v in n_trials.values())
        return _joint_budget_plan(total, tree_modes)
    blocks = []
    for phase in phases:
        phase_total = max(0, int(n_trials.get(phase, 0)))
        if phase_total > 0:
            blocks.append(TrialBlock(phase, None, phase_total))
    return blocks


def _timeout_budget_plan(tree_modes, strategy):
    if strategy == "joint":
        blocks = [TrialBlock("probe", lane, 1) for lane in tree_modes]
        blocks.append(TrialBlock("joint_compact", None, None))
        return blocks
    raise ValueError(
        "timeout-only tuning requires strategy='auto' or strategy='joint'; "
        "set n_trials for stepwise tuning"
    )


def _joint_budget_plan(n_trials, tree_modes):
    blocks = []
    remaining = int(n_trials)
    for lane in tree_modes:
        if remaining <= 0:
            break
        blocks.append(TrialBlock("probe", lane, 1))
        remaining -= 1
    if remaining > 0:
        blocks.append(TrialBlock("joint_compact", None, remaining))
    return blocks


def _stepwise_budget_plan(n_trials, phases, tree_modes):
    if int(n_trials) <= len(tree_modes):
        return [TrialBlock("probe", lane, 1) for lane in tree_modes[:n_trials]]
    weights = _phase_weights(int(n_trials))
    blocks = [TrialBlock("probe", lane, 1) for lane in tree_modes]
    remaining = int(n_trials) - len(blocks)
    warmup = _joint_warmup_trials(int(n_trials), remaining)
    if warmup > 0:
        blocks.append(TrialBlock("joint_compact", None, warmup))
        remaining -= warmup
    weighted_phases = [
        phase for phase in phases
        if phase not in {"probe", "joint_compact"}
    ]
    total_weight = sum(weights.get(phase, 1) for phase in weighted_phases)
    allocated = 0
    for idx, phase in enumerate(weighted_phases):
        if remaining <= allocated:
            break
        if idx == len(weighted_phases) - 1:
            phase_total = remaining - allocated
        else:
            phase_total = int(round(
                remaining * weights.get(phase, 1) / max(1, total_weight)
            ))
            phase_total = min(phase_total, remaining - allocated)
        allocated += phase_total
        blocks.append(TrialBlock(phase, None, phase_total))
    return [block for block in blocks if block.n_trials is None or block.n_trials > 0]


def _joint_warmup_trials(total_trials, remaining_after_probes):
    if total_trials < 200 or remaining_after_probes <= 0:
        return 0
    probe_trials = int(total_trials) - int(remaining_after_probes)
    target_warmup_bucket = max(probe_trials, round(int(total_trials) * 0.20))
    return min(
        int(remaining_after_probes),
        max(0, target_warmup_bucket - probe_trials),
    )


def _phase_weights(total_trials):
    if int(total_trials) >= 200:
        return {
            "probe": 0,
            "joint_compact": 0,
            "structure": 300,
            "sampling_regularization": 200,
            "learning_rate": 200,
            "binning_categorical": 75,
            "split_noise": 25,
        }
    return {
        "probe": 0,
        "joint_compact": 0,
        "structure": 4,
        "sampling_regularization": 3,
        "split_noise": 1,
        "learning_rate": 2,
        "binning_categorical": 1,
    }


def _allocate_phase_to_lanes(phase, n_trials, tree_modes):
    if int(n_trials) <= 0:
        return []
    quotas = _split_quota(n_trials, len(tree_modes))
    return [
        TrialBlock(phase, lane, quota)
        for lane, quota in zip(tree_modes, quotas)
        if quota > 0
    ]


def _execution_blocks(block, tree_modes, lane_states):
    if block.lane is not None or block.phase == "joint_compact":
        return [block]
    return _adaptive_phase_to_lanes(
        block.phase, block.n_trials, tree_modes, lane_states
    )


def _adaptive_phase_to_lanes(phase, n_trials, tree_modes, lane_states):
    n_trials = int(n_trials)
    if n_trials <= 0:
        return []
    tree_modes = tuple(tree_modes)
    if len(tree_modes) <= 1:
        return [TrialBlock(phase, tree_modes[0], n_trials)]
    scored = [
        (mode, lane_states[mode].best_loss)
        for mode in tree_modes
        if np.isfinite(lane_states[mode].best_loss)
    ]
    if not scored:
        return _allocate_phase_to_lanes(phase, n_trials, tree_modes)
    best_lane = min(scored, key=lambda item: item[1])[0]
    if n_trials < len(tree_modes):
        return [TrialBlock(phase, best_lane, n_trials)]
    best_quota = max(1, int(round(n_trials * 0.70)))
    best_quota = min(best_quota, n_trials - (len(tree_modes) - 1))
    other_modes = [mode for mode in tree_modes if mode != best_lane]
    other_quotas = _split_quota(n_trials - best_quota, len(other_modes))
    blocks = [TrialBlock(phase, best_lane, best_quota)]
    blocks.extend(
        TrialBlock(phase, mode, quota)
        for mode, quota in zip(other_modes, other_quotas)
        if quota > 0
    )
    return blocks


def _remaining_timeout(timeout, started_at):
    if timeout is None:
        return None
    return max(0.0, float(timeout) - (time.monotonic() - started_at))


def _make_optuna_callbacks(callbacks, study_stopper, patience, min_trials,
                           min_delta):
    out = []
    if callbacks is not None:
        if isinstance(callbacks, (list, tuple)):
            out.extend(callbacks)
        else:
            out.append(callbacks)
    if study_stopper is not None:
        out.append(study_stopper)
    if patience is not None:
        out.append(_NoImprovementStopper(
            patience=int(patience),
            min_trials=int(min_trials),
            min_delta=float(min_delta),
        ))
    return out or None


@dataclass
class _NoImprovementStopper:
    patience: int
    min_trials: int = 20
    min_delta: float = 0.0
    best_value: float | None = None
    best_trial_count: int = 0

    def __call__(self, study, trial):
        if study.user_attrs.get("_chimeraboost_stop"):
            study.stop()
            return
        completed = [
            t for t in study.trials
            if _is_usable_trial(t)
        ]
        n_completed = len(completed)
        if n_completed < self.min_trials:
            return
        current = min(float(t.value) for t in completed)
        if self.best_value is None or current < self.best_value - self.min_delta:
            self.best_value = current
            self.best_trial_count = n_completed
            return
        if n_completed - self.best_trial_count >= self.patience:
            study.set_user_attr("_chimeraboost_stop", True)
            study.stop()


def _resolve_trial_thread_count(setting, n_workers):
    if setting == "auto":
        return max(1, (os.cpu_count() or 1) // max(1, int(n_workers)))
    return max(1, int(setting))


def _split_quota(n_trials, n_workers):
    if n_trials is None:
        return [None] * max(1, int(n_workers))
    if int(n_trials) <= 0:
        return []
    n_workers = max(1, min(int(n_workers), int(n_trials)))
    base = int(n_trials) // n_workers
    extra = int(n_trials) % n_workers
    return [base + (1 if i < extra else 0) for i in range(n_workers)]


def _fold_seed(random_state, trial_number, fold_idx):
    if random_state is None:
        return None
    return int(random_state) + int(trial_number) * 1009 + int(fold_idx)


ChimeraBoostSearchCV = ChimeraBoostStepwiseSearchCV
