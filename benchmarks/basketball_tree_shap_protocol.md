# Basketball exact TreeSHAP confirmation protocol

## Decision and source boundary

This campaign decides whether DarkoFit may ship exact interventional TreeSHAP
for its scalar oblivious-tree path. It is a product capability, not a modeling
default: fitting, predictions, learning-rate policy, tree selection, and every
existing estimator parameter must remain unchanged.

The implementation may adapt ChimeraBoost's Apache-2.0 exact-enumeration
algorithm introduced in commit
`ff6f248d09f92d608ed8cc366463b61f1af04acc`. DarkoFit must retain a prominent
source notice and extend `NOTICE` with the upstream author, license, feature,
and commit. The formal run binds:

- DarkoFit base commit `efb24036a8938cd967ab84d4629c9c470f33a601`;
- synced ChimeraBoost 0.15.0 commit
  `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`; and
- this protocol, the runner, both clean source states, and the exact DarkoFit
  package manifest after implementation.

No CTR23 coordinate or lockbox data is used. Basketball is the primary and
complete development gate for this feature because it is fast and matches the
intended sports-data workload: exploratory profiling used creator fold 0;
formal confirmation uses fold 1 for matched speed and attribution comparison,
plus the corrected genuinely cold-player view as the sports-noise correctness
guardrail. Broader tabular panels are neither required nor consulted for this
decision.

## Supported API

`DarkoRegressor.shap_values(X, X_background=None)` returns an array of shape
`(n_samples, n_input_features)` and sets `expected_value_`. Rows must satisfy

```text
phi.sum(axis=1) + expected_value_ == predict(X)
```

within floating-point tolerance. Supported regressors are scalar,
non-`linear_residual` fits whose retained trees are all oblivious; both
constant and local-linear leaves are required. MAE and Quantile are supported
because their public prediction is the scalar raw score.

`DarkoClassifier.shap_values` supports binary classifiers and explains raw
log-odds, setting `expected_value_` in margin space. Multiclass classifiers,
distributional regressors, active global linear residuals, and any retained
leaf-wise/level-wise tree must fail explicitly with `NotImplementedError`.

At fit, scalar boosters retain at most 200 deterministically sampled binned
training rows as the default empirical background. A caller-supplied
background overrides it. `max_background` and `random_state` are core-level
controls for deterministic subsampling; the sklearn wrappers expose the
simple full-background API above. Empty backgrounds and nonpositive caps are
errors. Legacy archives without a stored background remain explainable only
when a custom background is supplied.

Attributions are returned in original input-feature space. Internal raw-code,
target-statistic, or other columns mapping to one source feature form one
coalition player. Linear-leaf intercepts and slopes are explained exactly.
Because exact enumeration is exponential, a tree using more than 16 distinct
original features fails explicitly before allocating coalition storage.

## Correctness and compatibility gates

Before the basketball run, focused tests must prove:

- exact agreement with an independent brute-force coalition oracle on a tiny
  forest, not merely self-consistency with the ported kernel;
- Shapley efficiency for numeric constant leaves, local-linear leaves,
  categorical features, MAE/Quantile regression, and binary margins;
- the custom-background baseline equals the mean corresponding raw prediction;
- deterministic background sampling/subsampling and exact one-thread versus
  multithread output;
- original-feature mapping when multiple fitted columns represent one input;
- explicit unsupported-mode and invalid-argument failures;
- `.npz` round trips preserve background, attributions, and baseline exactly,
  while malformed background payloads fail closed under `allow_pickle=False`;
- fitted predictions, prediction goldens, callbacks, tuning, exact refit, and
  all five distributional heads remain unchanged; and
- the complete test suite passes.

No SHAP state may alter the fitted tree archive, prediction hash, feature
importance, or training RNG. The stored background payload is the only allowed
model-size increase and must contain at most `200 * fitted_internal_features`
bin values.

## Formal basketball lane

Both libraries fit creator fold 1 with the matched 1,000-tree constant-leaf
configuration from `basketball_chimera_v015_protocol.md` at 18 threads. They
must retain exactly 1,000 depth-six trees and produce array-exact predictions.

The formal SHAP request explains the first eight fold rows against the first
32 training rows. A second correctness request explains the first eight
genuinely cold-player rows against the same background. Both libraries are
warmed with a one-row/two-background-row call before timing. The fold request
then runs in 11 reciprocal blocks with five calls per block; medians and
IQR/median are computed per call.

The feature advances only if:

- DarkoFit and ChimeraBoost attributions are array-close at `rtol=0`,
  `atol=1e-12`, and their expected values differ by at most `1e-12`;
- DarkoFit fold and cold-player efficiency error is at most `1e-9`;
- repeated DarkoFit calls are array-exact;
- DarkoFit median SHAP time is at most 1.50 times ChimeraBoost;
- both timing IQR fractions are at most 0.30;
- the default stored-background call also satisfies efficiency at
  `max_background=32`; and
- source, license, serialization, storage, and complete-test gates pass.

Exploratory ChimeraBoost fold-0 timing was 0.0301 seconds median for this
8-row/32-background request, with efficiency error `1.64e-13`. That observation
sets the comparator-relative gate; it is not formal confirmation evidence.

Failure rejects the feature rather than relaxing tolerances or changing the
background on fold 1. A pass promotes only the supported SHAP API. It does not
authorize multiclass/vector TreeSHAP, any default change, or CTR23 use.
