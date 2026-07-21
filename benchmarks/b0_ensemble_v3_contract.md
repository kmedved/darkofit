# Track B0 private ensemble-v3 design contract

_Design frozen before B1/B2 implementation or prototype outcome access on
2026-07-20._

## Purpose and authority

This contract defines the funded private, sequential B1/B2 attribution from
Track B. It starts from DarkoFit source
`89556419a0cb77a89801c381db4c09b400963a64`, after the Wave 1 hygiene and
release-note checkpoints. It does not reopen the failed M3a group-bootstrap
route, change the existing meaning of `n_ensembles`, or authorize a public
option or default.

The prototype asks two causal questions:

1. Does replacing full-size bootstrap sampling with a fixed-fraction,
   without-replacement sample improve the existing sequential ensemble?
2. Does a named member policy improve the existing sequential ensemble when
   sampling is held fixed?

The combined arm tests whether the mechanisms compose. B3 parallelism is not
funded and must not enter the implementation or evidence. All members run in
seed order in one process with the full declared per-member thread budget.

## Private surface and preserved behavior

The implementation is a private prototype entry point. It must not add a
constructor parameter, alter `get_params()`, change a default, or change any
fit reached through the public estimator API. Existing `n_ensembles=1` and
existing row/group bootstrap fits must remain behavior-identical.

The private entry point accepts an unfitted DarkoFit estimator plus the normal
fit inputs and these mechanism controls:

- `sampling`: `"bootstrap"` or `"without_replacement"`;
- `sampling_unit`: `"rows"` or `"groups"`;
- `sample_fraction`: fixed to `0.8` for the funded prototype when sampling is
  without replacement, and absent for bootstrap;
- `member_policy`: `"none"` or `"donor_balanced_v1"`; and
- `explicit_user_params`: the policy-controlled constructor fields explicitly
  supplied by the hypothetical user.

`explicit_user_params` is prototype provenance, not a proposed public API.
Any future public surface must have an unambiguous way to distinguish an
omitted value from an explicitly supplied value before it can claim the same
precedence rule.

## B1 sampling contract

The current bootstrap plan remains the control. B1 adds sampling without
replacement at the fixed fraction `0.8`.

For row sampling, let
`m = min(n_rows - 1, max(1, floor(0.8 * n_rows + 0.5)))`. Draw exactly `m`
distinct row indices uniformly without replacement. The OOB indices are the
sorted complement. A `groups` fit is invalid for row sampling; the prototype
must not silently split entities.

For group sampling, factor the one-dimensional group vector using the existing
consistently-comparable-scalar rule. With `g` unique groups, let
`m = min(g - 1, max(1, floor(0.8 * g + 0.5)))`. Draw exactly `m` distinct
group codes uniformly without replacement, include every row belonging to a
drawn group, and use every remaining group as OOB. Training and OOB groups
must be disjoint by construction. The realized row fraction is allowed to
differ from `0.8` and must be recorded.

Each member uses its existing deterministic member seed for its sampling RNG.
Up to 128 deterministic attempts are allowed to obtain a usable plan. A plan
is usable only when:

- training and OOB are both nonempty;
- both sides have positive total `sample_weight`, when weights are supplied;
- classification training and OOB each contain every class present in the
  full fit input; and
- group sampling has at least one training and one OOB group.

Exhaustion is a `RuntimeError`; there is no fallback to bootstrap, row
sampling, a contaminated validation split, or a different fraction.

Every member record must contain the requested and realized fractions, sample
method and unit, attempt count, row/group counts, group-disjoint flag, and
SHA-256 fingerprints of the ordered training and OOB indices. Sampling is
always OOB-selected: the complement is the member's explicit early-stopping
evaluation set, including its aligned evaluation weights.

## B2 member-policy contract

The only funded named policy is `donor_balanced_v1`, derived from the sibling
library's validated bagged-member rule:

- `learning_rate = 0.15`; and
- `colsample = 0.85`.

For each policy-controlled field, a value explicitly named in
`explicit_user_params` wins exactly, including an explicit `None` or a value
equal to the normal default. Otherwise the named policy value is applied.
The base estimator is not mutated. Every member record and the outer fitted
metadata must store, for both fields, the base value, resolved value, and one
of the exact sources `"explicit_user"`, `"member_policy"`, or `"base"`.

The existing OOB selection requirements remain mechanical invariants rather
than policy choices: private members have `n_ensembles=1`, their deterministic
member seed, `early_stopping=True`, `use_best_model=True`, and `refit=False`.
No other constructor field may be changed by the policy.

## Fitted-state, persistence, and failure obligations

Successful private fits must use the normal aggregate prediction, probability,
SHAP, and expected-value paths. They must expose a versioned
`ensemble_metadata_` that distinguishes bootstrap from without-replacement
sampling and `none` from `donor_balanced_v1`. Metadata must be sufficient to
audit every resolution and sampling plan without retaining training rows.

The normal pickle-free NPZ format must round-trip every private arm with exact
predictions (and exact probabilities for classifiers), identical fitted
metadata, member count/order, class schema, feature schema, OOB provenance,
and resolved policy values. Loading must fail closed on forged or
contradictory sampling, fraction, policy, resolution, count, digest, fitted
thread, or member-parameter metadata. A loaded private prototype is for
prediction and re-save only; it does not make the private fit surface public.

Any exception during sampling or any member fit must restore the estimator's
entire pre-call fitted state. A fresh estimator must remain unfitted; an
already-fitted estimator must retain its prior predictions and metadata.
Partial members and partial final-looking benchmark artifacts are forbidden.

## Invariants required before M3b freeze

Tests must establish all of the following without using M3b outcomes:

1. exact deterministic indices and member order at a fixed seed;
2. no duplicate training rows/groups in without-replacement plans;
3. exact row complement or group-disjoint OOB membership;
4. correct class and positive-weight retries and fail-closed exhaustion;
5. aligned pandas, Polars/PyArrow where installed, NumPy, weight, target, and
   group row slicing;
6. policy-only changes exactly the two declared fields;
7. explicit-user precedence, including explicit `None` and explicit normal
   defaults;
8. public single and existing bootstrap fits remain unchanged;
9. regression mean, classification soft vote, and SHAP aggregation remain
   correct;
10. exact safe serialization and corruption rejection; and
11. nested prediction during member fitting preserves the caller's ambient
    thread mask.

M5 remains an invariant/drift suite, not a quality scoreboard. No sports,
general-slice, TabArena, or lockbox outcome may be inspected to develop these
invariants.

## Memory-efficiency design objective

Memory efficiency is a first-class design objective because DarkoFit's
standing advantage is compact sequential ensembles, while the sibling's
historical eight-member arm cost about `5.90x` single-model bytes and `6.16x`
single-model peak RSS. The private eight-member design targets:

- median safe-NPZ archive bytes no greater than `4.0x` its matched single;
- aggregate worker peak RSS no greater than `2.0x` its matched single; and
- one fitted numeric preprocessing/binning definition per outer ensemble in
  memory and, if needed to meet the archive target, one serialized definition
  referenced by members rather than eight duplicated definitions.

These are design targets, not B0 acceptance gates. M3b must prospectively bind
the actual archive/RSS gates, measurement method, and handling of categorical
member-local preprocessing before any prototype outcome is opened.

## M3b boundary and forbidden work

After implementation and invariant tests, a new M3b protocol, runner,
analyzer, and create-only machine contract must be committed and frozen before
quality execution. It must compare matched control, sampling-only,
member-policy-only, and combined arms; complete a paired weighted-holdout
quality pass before timing; and bind the strict `paired-evidence-v1`
environment or a stricter successor. M3a files remain byte-preserved.

The funded work may produce a private recommendation only. It does not
authorize B3, a public recipe/preset, a constructor change, a default change,
fresh confirmation, TabArena, or CTR23 lockbox access.
