# SHAP explanations

`model.shap_values(X)` returns **exact** SHAP feature attributions — a faithful,
additive decomposition of each prediction into per-feature contributions.

```python
reg = ChimeraBoostRegressor(random_state=0).fit(X_train, y_train)
phi = reg.shap_values(X_test)        # (n_samples, n_features)
base = reg.expected_value_           # baseline, set by the call above
```

## Why "exact" matters

Most SHAP tooling **approximates** — it samples feature coalitions because a tree of
arbitrary shape makes the exact computation expensive. ChimeraBoost computes it
**exactly**, with no sampling, by exploiting the oblivious tree structure: every level
of a tree splits on the *same* feature, so a depth-`D` tree touches **at most `D`
distinct features**. The Shapley coalition game therefore has at most `D` players, and
all `≤2**D` coalitions are enumerated directly inside a numba kernel (64 evaluations
per tree at the default depth 6). What is intractable in general is cheap here.

This is the [interventional](https://proceedings.mlr.press/v119/janizek20a.html)
formulation of TreeSHAP, integrated over a background distribution.

## The efficiency guarantee

The defining property — and the reason these numbers can be trusted as *the model's
own accounting* of each prediction — is **Shapley efficiency**: the contributions plus
the baseline reconstruct the prediction exactly (to floating-point tolerance).

```text
phi.sum(axis=1) + expected_value_  ==  prediction
```

```python
i = 0
recon = phi[i].sum() + base
assert abs(recon - reg.predict(X_test)[i]) < 1e-6   # holds to ~1e-14
```

Nothing is double-counted, nothing is lost. Contrast this with **gain importance**
(`feature_importances_`), which measures what the trees *split on* — a structural,
winner-take-all signal that ignores the per-leaf linear models and does not decompose
any individual prediction.

## What the numbers mean

`phi[i, j]` is feature `j`'s signed, additive contribution to the raw score of row `i`,
measured against `expected_value_` (the mean raw score over the background):

- **Regressor** → contributions to the **predicted target**.
- **Binary classifier** → contributions to the **pre-temperature log-odds** of the
  positive class. (Probabilities are a nonlinear squash of the margin, so SHAP, like
  the wider SHAP ecosystem, explains the additive margin.)

The **linear-leaf slope terms are included exactly** — a leaf that predicts
`intercept + slope·(x − center)` folds its slope contribution straight into the
attribution, so SHAP explains the actual fitted model, not just its split skeleton.

## Global importance

Average the absolute contributions across rows for a consistent, prediction-faithful
global ranking (a better-justified alternative to gain importance):

```python
import numpy as np
global_importance = np.abs(phi).mean(axis=0)
for j in np.argsort(global_importance)[::-1][:10]:
    print(f"feature {j}: {global_importance[j]:.4f}")
```

## Local explanations

Explain a single prediction by reading its row of `phi`:

```python
i = 0
print(f"baseline (expected value): {base:.3f}")
for j in np.argsort(np.abs(phi[i]))[::-1][:5]:
    sign = "increases" if phi[i, j] > 0 else "decreases"
    print(f"  feature {j} {sign} the output by {abs(phi[i, j]):.3f}")
print(f"  => prediction: {phi[i].sum() + base:.3f}")
```

## The background distribution

SHAP attributions are defined *relative to a reference*: "how does this feature move
the prediction away from a typical input?" That reference is the **background**, which
by default is a sample of the training data captured at fit time. Override it to
explain against a specific cohort:

```python
phi = clf.shap_values(X_test, X_background=X_reference)
```

`expected_value_` equals the model's mean prediction over whatever background is used.
Cost scales linearly with the background size; the default sample keeps it fast
(~3 ms/row at depth 6 with 200 background rows).

## Bagged models

When `n_ensembles > 1`, the attributions are averaged across members. For regression
this is exact (the bag prediction is the members' mean, and Shapley values are linear).
For classification it is an additive surrogate for the soft-voted probability.

## Limitations

!!! warning "Binary and regression only"
    Multiclass SHAP is not supported yet — `shap_values` raises
    `NotImplementedError` for a 3+ class model.

- Explanations are in **raw score / log-odds** space, not probability space.
- The values are attributions for *this* model; they are not causal effects.

## Compared to `feature_importances_`

| | `feature_importances_` | `shap_values` |
|---|---|---|
| What it measures | total split **gain** | exact contribution to each prediction |
| Granularity | global only | per-prediction **and** global |
| Includes linear leaves | no | **yes** |
| Faithful to the output | no (structural) | **yes** (sums to the prediction) |
| Cost | free (tracked at fit) | ~ms per row |

Use gain importance for a quick, free global glance; use SHAP when you need a faithful
or per-prediction explanation.
