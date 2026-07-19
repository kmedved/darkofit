# Getting started

## Install

DarkoFit requires Python 3.9 or newer.

DarkoFit is distributed from tagged GitHub releases rather than PyPI:

```bash
python -m pip install "darkofit @ git+https://github.com/kmedved/darkofit.git@v0.10.0"
```

The release wheel can also be installed directly:

```bash
python -m pip install "https://github.com/kmedved/darkofit/releases/download/v0.10.0/darkofit-0.10.0-py3-none-any.whl"
```

For a source checkout:

```bash
python -m pip install -e ".[dev,tuning]"
```

## Classification

```python
from darkofit import DarkoClassifier

model = DarkoClassifier(
    early_stopping=True,
    random_state=4,
)
model.fit(
    X_train,
    y_train,
    cat_features=["team", "position"],
    sample_weight=weights,
)
probability = model.predict_proba(X_test)
```

## Regression

```python
from darkofit import DarkoRegressor

model = DarkoRegressor(random_state=4)
model.fit(X_train, y_train, cat_features=["team"])
prediction = model.predict(X_test)
```

Named inputs enforce the fitted column names and order at prediction. For
unnamed arrays, categorical features must be integer indices.

## Save and load

DarkoFit archives are NumPy `.npz` files and load with pickle disabled.

```python
model.save_model("model.npz")
restored = type(model).load_model("model.npz")
```

## Warm fresh workers

Call `darkofit.warmup()` before the first latency-sensitive fit. Import-time
warmup is controlled by `DARKOFIT_WARMUP`: `1`, `true`, `on`, and `yes` block;
`background`, `thread`, and `bg` start the guarded background path; `0`,
`false`, `off`, `no`, and an empty value disable it.

## Next

Read [Parameters](parameters.md) for model controls and
[Core concepts](concepts.md) for validation and categorical behavior.
