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

from ..auto_params import effective_sample_size
from ..booster import _normalize_tree_mode
from ..losses import VECTOR_LOSSES
from .._validation import (
    n_features_from_array_like,
    n_samples_from_array_like,
    normalize_random_state_seed,
    normalize_cat_features,
    validate_target_vector,
)
from .optuna_backend import create_study, import_optuna, load_study, make_storage
from .results import build_cv_results, phase_summary, weighted_mean
from .scoring import resolve_scorer, score_estimator
from .spaces import (
    KNOWN_PHASES,
    LaneState,
    SpaceContext,
    make_phase_spec,
    phase_names,
)
from .validation import make_cv_splits, slice_fit_payload, validation_mass
from ..sklearn_api import _normalize_dist_calibration


def _validate_search_sample_weight(sample_weight, n_samples, name="sample_weight"):
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.shape != (int(n_samples),):
        raise ValueError(f"{name} must have shape ({int(n_samples)},)")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(w < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    if float(np.sum(w)) <= 0.0:
        raise ValueError(f"{name} must have positive total weight")
    return w


def _pooled_trial_sigma_calibration(trial):
    fold_auto_params = trial.user_attrs.get("fold_auto_params") or []
    fold_masses = trial.user_attrs.get("fold_weight_sums") or []
    if not fold_auto_params or len(fold_auto_params) != len(fold_masses):
        return None

    method = None
    scale2_num = 0.0
    dispersion_log_scale_num = 0.0
    mean_calibration_num = 0.0
    mean_calibration_den = 0.0
    mean_calibration_seen = False
    mean_log_scale_num = 0.0
    mean_log_scale_seen = False
    affine_a_num = 0.0
    affine_b_num = 0.0
    mass_den = 0.0
    n_samples = 0
    positive_weight_n = 0
    effective_n = 0.0
    threshold = None
    any_small_fold = False
    fallback_reasons = set()

    for auto_params, mass in zip(fold_auto_params, fold_masses):
        if not isinstance(auto_params, Mapping):
            return None
        metadata = auto_params.get("dist_calibration")
        if not isinstance(metadata, Mapping):
            metadata = auto_params.get("sigma_calibration")
        if not isinstance(metadata, Mapping):
            return None
        fold_method = metadata.get("method")
        if fold_method not in {"scalar", "affine", "dispersion"}:
            return None
        if method is None:
            method = fold_method
        elif fold_method != method:
            return None
        scale = metadata.get("dist_scale", metadata.get("sigma_scale"))
        if scale is None:
            return None
        mass = float(mass)
        if not np.isfinite(mass) or mass <= 0.0:
            continue
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            return None

        scale2_num += mass * scale * scale
        if method == "dispersion":
            dispersion_log_scale_num += mass * math.log(scale)
        if (
            metadata.get("mean_calibration_numerator") is not None
            and metadata.get("mean_calibration_denominator") is not None
        ):
            mean_calibration_seen = True
            mean_calibration_num += float(metadata["mean_calibration_numerator"])
            mean_calibration_den += float(metadata["mean_calibration_denominator"])
        if metadata.get("mean_calibration_objective") == "negative_binomial_nll":
            mean_log_scale_seen = True
            mean_log_scale_num += mass * math.log(scale)
        if method == "affine":
            affine_a = metadata.get("sigma_affine_a")
            affine_b = metadata.get("sigma_affine_b")
            if affine_a is None or affine_b is None:
                return None
            affine_a = float(affine_a)
            affine_b = float(affine_b)
            if not np.isfinite(affine_a) or not np.isfinite(affine_b):
                return None
            affine_a_num += mass * affine_a
            affine_b_num += mass * affine_b
            fallback_reason = metadata.get("fallback_reason")
            if fallback_reason is not None:
                fallback_reasons.add(str(fallback_reason))
        mass_den += mass
        n_samples += int(metadata.get("validation_n_samples", 0) or 0)
        positive_weight_n += int(
            metadata.get("validation_positive_weight_n", 0) or 0
        )
        effective_n += float(metadata.get("validation_effective_n", 0.0) or 0.0)
        if threshold is None and metadata.get("small_fold_threshold") is not None:
            threshold = float(metadata["small_fold_threshold"])
        any_small_fold = any_small_fold or bool(
            metadata.get("small_fold_warning", False)
        )

    if mass_den <= 0.0:
        return None

    method = method or "scalar"
    stats = {
        "validation_n_samples": n_samples,
        "validation_positive_weight_n": positive_weight_n,
        "validation_effective_n": effective_n,
    }
    if threshold is not None:
        stats["small_fold_threshold"] = threshold
        stats["small_fold_warning"] = effective_n < threshold
    else:
        stats["small_fold_warning"] = any_small_fold
    pooled_scale = float(np.sqrt(max(scale2_num / mass_den, 1e-12)))
    pooling = "validation_mass_weighted_scale2"
    if method == "dispersion":
        pooled_scale = float(
            np.exp(np.clip(dispersion_log_scale_num / mass_den, -10.0, 10.0))
        )
        pooling = "validation_mass_weighted_log_dispersion_scale"
    if method == "scalar" and mean_log_scale_seen:
        pooled_scale = float(
            np.exp(np.clip(mean_log_scale_num / mass_den, -10.0, 10.0))
        )
        pooling = "validation_mass_weighted_log_mean_scale"
    if method == "scalar" and mean_calibration_seen:
        pooled_scale = float(
            max(mean_calibration_num / max(mean_calibration_den, 1e-12), 1e-12)
        )
        pooling = "exact_mean_sufficient_statistics"
    calibration = {
        "method": method,
        "dist_scale": pooled_scale,
        "sigma_scale": pooled_scale,
        "source": "search_cv_validation",
        "fold_stats": stats,
        "pooling": pooling,
    }
    if method == "scalar" and mean_calibration_seen:
        calibration["mean_calibration_numerator"] = float(mean_calibration_num)
        calibration["mean_calibration_denominator"] = float(mean_calibration_den)
        calibration["mean_calibration_objective"] = "poisson_closed_form"
    elif method == "scalar" and mean_log_scale_seen:
        calibration["mean_calibration_objective"] = "negative_binomial_nll"
    if method == "affine":
        calibration["sigma_affine_a"] = float(affine_a_num / mass_den)
        calibration["sigma_affine_b"] = float(affine_b_num / mass_den)
        calibration["dist_affine_a"] = calibration["sigma_affine_a"]
        calibration["dist_affine_b"] = calibration["sigma_affine_b"]
        calibration["sigma_scale"] = float(
            np.exp(np.clip(calibration["sigma_affine_a"], -700.0, 700.0))
        )
        calibration["dist_scale"] = calibration["sigma_scale"]
        calibration["pooling"] = "validation_mass_weighted_affine_coefficients"
        if fallback_reasons:
            calibration["fallback_reason"] = ",".join(sorted(fallback_reasons))
    return calibration


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

    @property
    def _estimator_type(self):
        estimator_type = getattr(self.estimator, "_estimator_type", None)
        if estimator_type is not None:
            return estimator_type
        return "classifier" if is_classifier(self.estimator) else None

    def fit(self, X, y, *, cat_features=None, groups=None, sample_weight=None,
            eval_set=None, eval_sample_weight=None):
        distributional = getattr(self.estimator, "loss", None) in VECTOR_LOSSES
        optuna = import_optuna()
        if self.n_trials is None and self.timeout is None:
            raise ValueError("at least one of n_trials or timeout must be set")
        n_samples = n_samples_from_array_like(X)
        y_arr = validate_target_vector(y, n_samples)
        sample_weight_arr = _validate_search_sample_weight(
            sample_weight, n_samples
        )
        self.random_state_ = normalize_random_state_seed(self.random_state)
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
        _reject_custom_sampler_multiprocessing(self.sampler, self.n_workers_)
        self.trial_thread_count_ = _resolve_trial_thread_count(
            self.trial_thread_count, self.n_workers_
        )
        if eval_sample_weight is not None and eval_set is None:
            raise ValueError(
                "eval_sample_weight requires an explicit eval_set; CV "
                "validation weights are derived from sample_weight"
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
            sampler_seed=self.random_state_,
        )

        eval_payload = None
        validation_fraction = self.validation_fraction
        if validation_fraction == "auto":
            n_eff = effective_sample_size(sample_weight_arr, n_samples)
            validation_fraction = float(
                np.clip(max(0.10, 200.0 / max(n_eff, 1.0)), 0.10, 0.25)
            )
        self.validation_fraction_ = validation_fraction

        if eval_set is None:
            self.cv_splits_ = make_cv_splits(
                X, y_arr, cv=self.cv, groups=groups, classifier=classifier,
                random_state=self.random_state_,
                validation_fraction=validation_fraction,
                sample_weight=sample_weight_arr,
            )
        else:
            X_eval, y_eval = eval_set
            eval_n_features = n_features_from_array_like(
                X_eval, name="eval_set[0]"
            )
            if eval_n_features != int(n_features):
                raise ValueError(
                    f"eval_set[0] has {eval_n_features} features, but X has "
                    f"{int(n_features)} features"
                )
            y_eval_arr = validate_target_vector(
                y_eval,
                n_samples_from_array_like(X_eval, name="eval_set[0]"),
                name="eval_set[1]",
            )
            eval_sample_weight_arr = _validate_search_sample_weight(
                eval_sample_weight, y_eval_arr.shape[0],
                name="eval_sample_weight"
            )
            eval_payload = (
                X_eval,
                y_eval_arr,
                eval_sample_weight_arr,
            )
            self.cv_splits_ = [(np.arange(y_arr.shape[0]), None)]
        self.n_splits_ = len(self.cv_splits_)

        requested_tree_modes = _coerce_search_tree_modes(self.tree_modes)
        self.tree_modes_requested_ = requested_tree_modes
        search_tree_modes = _resolve_search_tree_modes(
            requested_tree_modes, distributional=distributional
        )
        self.tree_modes_ = search_tree_modes
        context = SpaceContext(
            estimator_params=self.estimator.get_params(),
            has_categoricals=bool(cat_features_normalized),
            classifier=classifier,
            tree_modes=search_tree_modes,
        )
        lane_states = {
            mode: LaneState(mode, fixed_params={"tree_mode": mode})
            for mode in search_tree_modes
        }
        self.strategy_ = _resolve_strategy(
            self.strategy, self.n_trials, self.phases
        )
        self.phases_ = tuple(phase_names(self.phases))
        planned_blocks = _make_budget_plan(
            n_trials=self.n_trials,
            timeout=self.timeout,
            phases=self.phases_,
            tree_modes=search_tree_modes,
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
                planned_block, search_tree_modes, lane_states
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
                    random_state=self.random_state_,
                    trial_thread_count=self.trial_thread_count_,
                    error_score=self.error_score,
                )
                self._run_phase(
                    objective_payload, spec.n_trials, optuna,
                    timeout=remaining_timeout, callbacks=callbacks,
                )
                if self.verbose:
                    print(
                        "ChimeraBoostStepwiseSearchCV "
                        f"phase={spec.name} lane={state.tree_mode} "
                        f"trials={spec.n_trials}"
                    )
                if self.storage_config_.storage is not None:
                    self.study_ = load_study(
                        storage_config=self.storage_config_,
                        study_name=self.study_name_,
                    )
                if block.phase == "joint_compact":
                    for mode in search_tree_modes:
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
        self.cv_results_ = build_cv_results(self.study_.trials)
        self.phase_results_ = phase_summary(self.study_.trials)
        if not completed:
            raise ValueError("no completed tuning trials")
        self.best_trial_ = min(completed, key=lambda t: t.value)
        trial_numbers = list(self.cv_results_["trial_number"])
        self.best_index_ = trial_numbers.index(self.best_trial_.number)
        self.best_params_ = dict(
            self.best_trial_.user_attrs.get("params_model")
            or self.best_trial_.user_attrs["params_full"]
        )
        self.best_trial_params_ = dict(self.best_trial_.user_attrs["params_full"])
        self.best_loss_ = float(self.best_trial_.value)
        self.best_score_ = float(self.best_trial_.user_attrs["mean_score"])
        self.tuning_metadata_ = {
            "tree_modes": list(search_tree_modes),
            "tree_modes_requested": list(requested_tree_modes),
            "tree_modes_resolved": list(search_tree_modes),
            "phases": list(self.phases_),
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
            "random_state": self.random_state_,
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
        _reject_custom_sampler_multiprocessing(self.sampler, self.n_workers_)
        quotas = _split_quota(n_trials, self.n_workers_)
        if hasattr(self, "random_state_"):
            sampler_seed_base = self.random_state_
        else:
            sampler_seed_base = normalize_random_state_seed(self.random_state)
        sampler_seeds = _worker_sampler_seeds(sampler_seed_base, len(quotas))
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
            sampler_seeds=sampler_seeds,
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
            lr = self.best_trial_.user_attrs.get("mean_learning_rate")
            if lr is not None and np.isfinite(float(lr)):
                params["learning_rate"] = float(lr)
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
        estimator_params = self.estimator.get_params()
        dist_calibration = None
        if getattr(self.estimator, "loss", None) in VECTOR_LOSSES:
            dist_calibration = _normalize_dist_calibration(
                estimator_params.get("dist_calibration"),
                estimator_params.get("sigma_calibration"),
            )
            if dist_calibration is not None:
                params["dist_calibration"] = None
                params["sigma_calibration"] = None
        if "thread_count" in estimator_params:
            params["thread_count"] = estimator_params.get("thread_count")
        params = _filter_params(params, self.estimator)

        final = clone(self.estimator).set_params(**params)
        final.fit(X, y, cat_features=cat_features, sample_weight=sample_weight)
        if dist_calibration is not None:
            calibration = _pooled_trial_sigma_calibration(self.best_trial_)
            if calibration is None:
                raise ValueError(
                    "best trial does not contain distribution calibration metadata; "
                    "set refit=False or rerun the search with the current "
                    "ChimeraBoost version"
                )
            final.dist_calibration_ = calibration["method"]
            final.dist_scale_ = float(calibration["dist_scale"])
            final.dist_scale_source_ = calibration["source"]
            final.dist_calibration_fold_stats_ = calibration["fold_stats"]
            final.dist_calibration_pooling_ = calibration.get("pooling")
            if calibration.get("mean_calibration_numerator") is not None:
                final.dist_mean_calibration_numerator_ = float(
                    calibration["mean_calibration_numerator"]
                )
            if calibration.get("mean_calibration_denominator") is not None:
                final.dist_mean_calibration_denominator_ = float(
                    calibration["mean_calibration_denominator"]
                )
            if calibration.get("mean_calibration_objective") is not None:
                final.dist_mean_calibration_objective_ = str(
                    calibration["mean_calibration_objective"]
                )
            final.sigma_calibration_ = calibration["method"]
            final.sigma_scale_ = float(calibration["sigma_scale"])
            final.sigma_scale_source_ = calibration["source"]
            final.sigma_calibration_fold_stats_ = calibration["fold_stats"]
            final.sigma_calibration_pooling_ = calibration.get("pooling")
            if calibration["method"] == "affine":
                final.dist_affine_a_ = float(calibration["dist_affine_a"])
                final.dist_affine_b_ = float(calibration["dist_affine_b"])
                final.sigma_affine_a_ = float(calibration["sigma_affine_a"])
                final.sigma_affine_b_ = float(calibration["sigma_affine_b"])
                fallback_reason = calibration.get("fallback_reason")
                if fallback_reason is not None:
                    final.dist_calibration_fallback_reason_ = fallback_reason
                    final.sigma_calibration_fallback_reason_ = fallback_reason
            final._attach_dist_calibration_metadata()
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
    error_score: object


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
            if (
                isinstance(payload.error_score, str)
                and payload.error_score == "raise"
            ):
                raise
            error_loss = (
                float("inf")
                if _error_score_is_nan(payload.error_score)
                else float(payload.error_score)
            )
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
            return error_loss

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


def _error_score_is_nan(error_score):
    try:
        return bool(np.isnan(float(error_score)))
    except (TypeError, ValueError):
        return False


def _reject_custom_sampler_multiprocessing(sampler, n_workers):
    if sampler is not None and int(n_workers) > 1:
        raise ValueError(
            "n_workers > 1 with a custom Optuna sampler is not supported; "
            "arbitrary sampler instances cannot be cloned with deterministic "
            "per-worker seeds. Set n_workers=1 or leave sampler=None so "
            "ChimeraBoost can create seeded worker samplers."
        )


@dataclass
class _WorkerPayload:
    objective_payload: object
    storage_url: str
    study_name: str
    resume: bool
    sampler: object
    pruner: object
    n_trials_by_worker: list[int]
    sampler_seeds: list[int | None]
    timeout: float | None = None
    callbacks: object = None

    def iter_workers(self):
        for n_trials, sampler_seed in zip(
            self.n_trials_by_worker, self.sampler_seeds
        ):
            if n_trials is None or n_trials > 0:
                yield (
                    self.objective_payload,
                    self.storage_url,
                    self.study_name,
                    self.resume,
                    self.sampler,
                    self.pruner,
                    n_trials,
                    sampler_seed,
                    self.timeout,
                    self.callbacks,
                )


def _optimize_worker(args):
    (objective_payload, storage_url, study_name, resume, sampler, pruner,
     n_trials, sampler_seed, timeout, callbacks) = args
    storage_config = make_storage(
        storage_url, n_workers=1, study_name=study_name, resume=resume
    )
    study = create_study(
        storage_config=storage_config,
        study_name=study_name,
        resume=resume,
        sampler=sampler,
        pruner=pruner,
        sampler_seed=sampler_seed,
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
    tree_mode = str(params.get("tree_mode", "catboost")).lower().replace("-", "_")
    if tree_mode not in {"lightgbm", "leafwise", "leaf_wise",
                         "hybrid", "hybrid_leafwise", "shared_prefix",
                         "auto"}:
        params["num_leaves"] = None
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
    trial.set_user_attr("params_full", _metadata_safe_params(params))
    trial.set_user_attr("params_model", _metadata_safe_params(params_model))
    trial.set_user_attr("params_suggested", _metadata_safe_value(dict(suggested)))
    trial.set_user_attr("sampled_keys", list(suggested.keys()))
    trial.set_user_attr(
        "params_fixed", _metadata_safe_params(payload.lane_state.best_params)
    )


def _metadata_safe_params(params):
    out = {}
    for key, value in dict(params).items():
        if key == "random_state" and isinstance(
            value, (np.random.RandomState, np.random.Generator)
        ):
            continue
        out[key] = _metadata_safe_value(value)
    return out


def _metadata_safe_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_metadata_safe_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _metadata_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_safe_value(v) for v in value]
    return value


def _filter_params(params, estimator):
    known = estimator.get_params()
    out = {}
    for key, value in params.items():
        if key not in known:
            continue
        if (
            value is None
            and key not in {
                "num_leaves",
                "early_stopping_rounds",
                "dist_calibration",
                "sigma_calibration",
            }
        ):
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


def _coerce_search_tree_modes(tree_modes):
    if isinstance(tree_modes, str):
        return (tree_modes,)
    return tuple(tree_modes)


def _resolve_search_tree_modes(tree_modes, *, distributional):
    requested = _coerce_search_tree_modes(tree_modes)
    if not requested:
        raise ValueError("tree_modes must contain at least one tree mode")
    if not distributional:
        return requested

    resolved = []
    for mode in requested:
        normalized = _normalize_tree_mode(mode)
        if normalized == "lightgbm" and normalized not in resolved:
            resolved.append(normalized)
    if not resolved:
        raise ValueError(
            "distributional tuning supports only tree_mode='lightgbm'; "
            "include 'lightgbm' or a leafwise alias in tree_modes"
        )
    return tuple(resolved)


def _is_usable_trial(trial):
    state_name = getattr(trial.state, "name", str(trial.state).split(".")[-1])
    return (
        state_name == "COMPLETE"
        and trial.value is not None
        and trial.user_attrs.get("params_full") is not None
        and trial.user_attrs.get("status") != "ERROR_SCORE"
    )


def _study_stop_requested(study):
    return bool(
        getattr(study, "user_attrs", {}).get("_chimeraboost_stop")
        or getattr(study, "_stop_flag", False)
    )


def _make_budget_plan(n_trials, timeout, phases, tree_modes, strategy):
    tree_modes = tuple(tree_modes)
    phases = tuple(phases)
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
    _validate_budget_mapping_keys(n_trials, phases, strategy)
    if strategy == "joint":
        total = sum(max(0, int(v)) for v in n_trials.values())
        return _joint_budget_plan(total, tree_modes)
    blocks = []
    for phase in phases:
        phase_total = max(0, int(n_trials.get(phase, 0)))
        if phase_total > 0:
            blocks.append(TrialBlock(phase, None, phase_total))
    return blocks


def _validate_budget_mapping_keys(n_trials, phases, strategy):
    keys = tuple(n_trials.keys())
    unknown = [phase for phase in keys if phase not in KNOWN_PHASES]
    if unknown:
        _raise_unknown_phase(unknown[0])
    if strategy == "joint":
        return
    omitted = [
        phase for phase in keys
        if max(0, int(n_trials.get(phase, 0))) > 0 and phase not in phases
    ]
    if omitted:
        allowed = ", ".join(phases)
        raise ValueError(
            f"n_trials specifies phase {omitted[0]!r}, but phases does not "
            f"include it; configured phases: {allowed}"
        )


def _raise_unknown_phase(phase):
    valid = ", ".join(sorted(KNOWN_PHASES))
    raise ValueError(
        f"unknown tuning phase {phase!r}; expected one of: {valid}"
    )


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
        out.append(_SharedStopCallback(study_stopper))
    if patience is not None:
        out.append(_NoImprovementStopper(
            patience=int(patience),
            min_trials=int(min_trials),
            min_delta=float(min_delta),
        ))
    return out or None


@dataclass
class _SharedStopCallback:
    callback: object

    def __call__(self, study, trial):
        if study.user_attrs.get("_chimeraboost_stop"):
            study.stop()
            return
        self.callback(study, trial)
        if getattr(study, "_stop_flag", False):
            study.set_user_attr("_chimeraboost_stop", True)


@dataclass
class _NoImprovementStopper:
    patience: int
    min_trials: int = 20
    min_delta: float = 0.0

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
        best_value = float("inf")
        best_trial_count = 0
        for count, completed_trial in enumerate(completed, start=1):
            value = float(completed_trial.value)
            if value < best_value - self.min_delta:
                best_value = value
                best_trial_count = count
        if n_completed - best_trial_count >= self.patience:
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


def _worker_sampler_seeds(random_state, n_workers):
    n_workers = max(0, int(n_workers))
    if random_state is None:
        return [None] * n_workers
    base = int(random_state)
    modulus = 2 ** 32 - 1
    return [
        int((base + 1_000_003 * (idx + 1)) % modulus)
        for idx in range(n_workers)
    ]


def _fold_seed(random_state, trial_number, fold_idx):
    if random_state is None:
        return None
    return int(random_state) + int(trial_number) * 1009 + int(fold_idx)


ChimeraBoostSearchCV = ChimeraBoostStepwiseSearchCV
