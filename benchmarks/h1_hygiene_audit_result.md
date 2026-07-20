# H1 hygiene audit result

_Closed 2026-07-20 on clean DarkoFit source
`726e5d8e6131c580bce948db833a5007d0692dca`._

## Decision

Track H item H1 is complete. One confirmed thread-state bug and one
diagnostic gap were fixed; the remaining named risks were either already
absent or had correct behavior with incomplete documentation. The adjacent
test pass also exposed and fixed a legacy scikit-learn tag fallback.

This is the single post-hygiene DarkoFit source pin for Wave 1 M1/Q0
measurements. Later documentation-only commits do not change that package
source boundary.

## Dispositions

| Item | Disposition | Evidence |
| --- | --- | --- |
| Numba thread mask | Confirmed and fixed in `5537cff`. Fit, predict, nested predict-during-fit, and staged resumptions run at the fitted count and restore the caller thread's ambient mask. | `tests/test_thread_state_restoration.py` |
| Rebuildable predictor cache in serialization | Not present. `save_booster` serializes canonical tree/preprocessor state, not `_flat_cache_`; archives saved before and after cache construction are byte-identical, load without a cache, rebuild lazily, and predict bit-identically. | `test_h1_serialization_excludes_rebuildable_flat_predictor_cache` |
| Unseen classifier eval labels | Not present. Binary and multiclass wrappers reject labels outside the training class set before remapping or partial fitted state. The core multiclass path has the same guard. | `test_h1_classifier_eval_set_rejects_unseen_labels`; `test_eval_labels_must_be_training_classes` |
| Positional weights bound to `cat_features` | Confirmed diagnostic gap and fixed in `7ddb09d`. `fit(X, y, w)` already failed rather than training incorrectly, but now the error names `sample_weight=w`. | `test_h1_positional_sample_weight_misuse_names_keyword` |
| NumPy integer `cat_features` | Not present. Integer arrays normalize through `operator.index` and work across regressor, classifier, scalar core, and multiclass core paths. | `test_cat_features_accepts_numpy_arrays_across_public_layers` |
| `None` constructor semantics | Behavior correct; documentation corrected in `7ddb09d`. `None` is parameter-specific, not a universal `"auto"` alias. Named tests bind CatBoost classifier depth 6 and LightGBM depth `-1`; docs also record the small LightGBM/hybrid two-thread cap. | `test_h1_classifier_depth_none_uses_documented_mode_default`; `test_tree_mode_default_depth_resolution` |
| Legacy scikit-learn tags | Additional confirmed compatibility bug, fixed in `726e5d8`. On scikit-learn 1.5, MRO previously dropped DarkoFit's `allow_nan` tag from direct `__sklearn_tags__()` calls. The fallback now merges DarkoFit and `requires_y` tags explicitly. | `test_sklearn_messages_and_tags` |

The thread-mask scope is thread-local. This result does not describe a
process-global leak, and it does not change the existing warmup save/restore
path.

## Verification

Runtime: `/opt/anaconda3/bin/python3`, scikit-learn 1.5.1.

```text
python3 -m pytest -q -rs \
  tests/test_h1_hygiene_audit.py \
  tests/test_thread_state_restoration.py \
  tests/test_input_validation.py \
  tests/test_leafwise_packed_prediction.py

55 passed, 3 skipped in 1.71s
```

The skips are expected and explicit: two full estimator-compliance checks
require scikit-learn 1.6+, and one packed-prediction boundary check requires
18 available Numba threads.

```text
python3 -m pytest -q tests/test_darkofit.py -k \
  'eval_labels_must_be_training_classes or
   cat_features_accepts_numpy_arrays_across_public_layers or
   cat_features_validation_has_clear_errors or
   tree_mode_default_depth_resolution or
   loaded_refit_selection_persistence_flag_is_not_stale or flat_cache'

5 passed, 313 deselected in 0.95s
```
