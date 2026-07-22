# Ensemble-v3 public API and archive contract

_Contract frozen before release-candidate implementation on 2026-07-21._

## Authority and boundary

This contract is authorized by `NEXT_STEPS.md` §4.2.1 and §6 and starts from
DarkoFit source `445647a79e2f48d3169b0511d423896addcf3e3d`. It specifies the
future public surface so the authorized private release candidate can be built
against a stable target. It does **not** authorize public constructor changes,
exports, documentation, a default change, evidence access, or a release.

The existing `n_ensembles=1` path and the existing row/group bootstrap
ensemble are compatibility controls. Their constructor values, sampling,
predictions, fitted metadata version, safe-NPZ bytes, and errors must remain
unchanged until a separately authorized public ship. The future mode is
additive and explicit.

Contract identity: `ensemble-v3-public-contract-v1`.

## Future constructor surface

At public ship, both `DarkoRegressor` and `DarkoClassifier` add exactly these
parameters:

- `ensemble_mode="bootstrap"`, accepting only `"bootstrap"` and `"v3"`;
- `ensemble_member_learning_rate="policy"`; and
- `ensemble_member_colsample="policy"`.

The literal string `"policy"` is a JSON-safe, sklearn-clone-safe sentinel; it
is not a valid fitted member value. A raw object singleton is forbidden because
deep copying it can break sklearn's constructor-identity check. No call-stack
inspection or constructor-call provenance is permitted.

`ensemble_mode="bootstrap"` is the exact legacy behavior. In that mode both
member override parameters must remain `"policy"`; any other value is a loud
`ValueError`, because the legacy ensemble has no member policy.

`ensemble_mode="v3"` is the measured combined B1/B2 recipe and requires
`n_ensembles=8`. It uses without-replacement sampling at the fixed fraction
`0.8`, with `ensemble_bootstrap` continuing to select the sampling unit
(`"rows"` or `"groups"`), and the named `donor_balanced_v1` member policy:

- member `learning_rate=0.15`; and
- member `colsample=0.85`.

Eight is the only evaluated member count. Other counts fail loudly; the
implementation must not imply that eight was optimized or is generally best.
The fraction and policy are versioned recipe constants, not public tuning
knobs.

## Parameter precedence and sklearn semantics

The two dedicated member parameters are the only unambiguous public override
surface. Legacy top-level `learning_rate` and `colsample` retain their existing
meaning for singles and bootstrap ensembles and are recorded as the base
values; they do not silently override the v3 recipe.

For v3, resolve each policy field independently:

1. If the matching dedicated member parameter is not the exact sentinel
   `"policy"`, that value wins and its source is `"explicit_user"`. This
   includes an explicit `None` for learning rate and a value equal to the
   ordinary default or policy value.
2. Otherwise use the named policy value and record source `"member_policy"`.

`ensemble_member_learning_rate` accepts the sentinel, `None`, or the same
positive finite numeric values accepted by the ordinary constructor.
`ensemble_member_colsample` accepts the sentinel or a finite numeric value in
`(0, 1]`. Booleans are invalid for both.

Before fit, `get_params(deep=False)` and sklearn `clone` must preserve the
three constructor values exactly, including explicit `None` and numeric values.
`set_params` changes only future fits. A failed fit is transactional. After
fit, the outer constructor values remain unchanged; resolved member values live
in fitted metadata and member estimators. Loading restores the exact outer
constructor values, and cloning a loaded model produces an unfitted estimator
with the same constructor values.

The authorized private release candidate exposes the same semantics only
through a private helper. It must not add these parameters to either estimator
constructor or `_SKLEARN_ONLY`, and it must not be exported from
`darkofit.__init__`.

## Support matrix and pre-fit errors

The v3 candidate supports:

| Surface | Contract |
| --- | --- |
| Scalar regression | `RMSE`, `MAE`, and `Quantile` supported. |
| Classification | Binary and multiclass supported; aggregation is mean probability (soft vote). |
| Sample weights | Supported and aligned for training and OOB rows; both partitions require positive usable class mass. |
| Row sampling | Supported; passing `groups` is a loud error. |
| Group sampling | Supported; `groups` is required, consistently factorized, and train/OOB groups are disjoint. |
| Numeric inputs | Supported, including target-free shared preprocessing when eligible. |
| Categorical inputs | Supported through member-local preprocessing. |
| Explicit ordinal mapping | Supported through member-local preprocessing. |
| `ordinal_features="auto"` | Unsupported; loud error before sampling. |
| `preset` | Unsupported; loud error before sampling. |
| `tree_mode="auto"` | Unsupported; loud error before sampling. |
| Callbacks | Unsupported; loud error before sampling or member fitting. |
| `eval_set` / `eval_sample_weight` | Unsupported because each member's OOB complement is its validation set; loud error. |
| `refit=True` | Unsupported; loud error. |
| `auto_learning_rate_probe=True` | Unsupported; loud error before sampling. |
| Distributional regression losses | Unsupported because aggregate parameter semantics are undefined; loud error. |

Unsupported combinations are contract limits, not silent fallbacks. Validation
that does not need data must occur before sampling; no invalid combination may
partially fit a member or erase a previously fitted estimator.

## Sampling, fitted behavior, and invariants

The public recipe inherits the frozen B0 mechanics: deterministic seed-order
members, the exact rounded `0.8` row/group count, at most 128 deterministic
class/weight-safe attempts, exact complement OOB rows, group-disjoint OOB for
group sampling, OOB early stopping, sequential execution, mean regression and
SHAP aggregation, and soft-vote classification aggregation. There is no
bootstrap, fraction, row-sampling, or validation fallback.

The candidate must preserve the caller's ambient thread-local Numba mask across
fit, nested prediction-during-fit, prediction, staged prediction, save, load,
and failure. Each member still runs at its fitted `n_threads_` count.

Fitted metadata records the contract identity, recipe version, outer
constructor values, policy resolutions, sampling unit/fraction, deterministic
seeds, row/group counts, ordered sample/OOB SHA-256 digests, preprocessing
resolution, aggregation, and fitted member parameters. The release candidate
must be distinguishable from the historical private B1/B2 prototype while
remaining non-public.

## Future safe-NPZ schema

The future public archive is `archive_kind="darkofit_ensemble"` with
`ensemble_format_version=4`. Versions 1 (legacy public bootstrap) and 3
(historical private B1/B2 provenance) remain readable exactly as before;
version 4 is never inferred from either.

The version-4 header contains:

- exact outer wrapper constructor parameters, including the three new public
  values;
- public metadata with `ensemble_mode="v3"`,
  `recipe_contract="ensemble-v3-public-contract-v1"`, and
  `recipe_version=1`; and
- the existing class, feature, aggregation, preprocessing, seed, sampling,
  policy-resolution, and fitted-member provenance.

Payloads contain one non-pickle safe-NPZ member archive per member, exact
little-endian int64 sampled/OOB index arrays per member, and the canonical
little-endian factorized group-code vector only for group sampling. Loading
uses `allow_pickle=False`, rejects nested ensembles, unknown or extra arrays,
wrong dtype/shape/count/order, contradictory digests or metadata, unsupported
versions, and any outer/member/booster constructor mismatch. Corruption checks
are schema-derived; archive-size ratios are telemetry, not validity gates.

Load, predict, and byte-identical re-save are required. No public v3 archive
may be written until the public-ship row is separately authorized. During the
current authorized stage, the private candidate may use a private marker and
schema, but its round-trip validation must exercise every field needed by this
future version-4 mapping.

## Release-candidate exit checks

Before the private implementation checkpoint can close, tests must prove:

1. the future value normalizers and precedence resolver, including sentinel,
   explicit `None`, explicit defaults, invalid booleans, and clone-safe values;
2. deterministic row/group sampling and exact OOB/group disjointness;
3. regression, binary, multiclass, weighted, categorical, and explicit ordinal
   fits across both sampling units where applicable;
4. loud, transactional rejection of every unsupported matrix row;
5. prediction, probability, staged prediction, SHAP, and thread-state behavior;
6. exact safe-NPZ round-trip/re-save plus schema-derived corruption rejection,
   including uneven groups;
7. member failure propagation with no partial fitted state; and
8. byte- and behavior-preserving non-regression for singles and the existing
   public bootstrap ensemble.

Public parameters, exports, product documentation, characterization claims,
M2/M4 evidence, and release work remain forbidden until their §6 owner gates.
