# TreeSHAP

DarkoFit provides exact interventional TreeSHAP for supported scalar
oblivious-tree regressors and binary classifiers.

```python
import numpy as np

values = model.shap_values(X_test, X_background=X_reference)
raw = model.model_.predict_raw(X_test)
assert np.allclose(
    values.sum(axis=1) + model.expected_value_,
    raw,
)
```

Contributions are mapped back to original input features. Binary classifier
values explain raw log-odds. A deterministic fitted background is preserved
by safe model serialization; callers may also supply an explicit background.

TreeSHAP currently rejects multiclass, distributional, global
linear-residual, and non-oblivious models instead of returning a partial or
misleading explanation. The supported implementation is limited to at most
16 coalition players after feature grouping.

The frozen basketball comparison matched ChimeraBoost 0.15 attributions; see
the [TreeSHAP result](https://github.com/kmedved/darkofit/blob/main/benchmarks/basketball_tree_shap_result.md).
