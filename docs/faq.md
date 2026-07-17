# FAQ

## Is DarkoFit a drop-in CatBoost or LightGBM replacement?

No. The estimators follow scikit-learn conventions and borrow modeling ideas,
but model files and tree representations are DarkoFit-specific.

## Why did a nonnumeric column fail?

Declare categorical columns with `cat_features` or encode them first. Named
tables may use column names; arrays use integer indices.

## How are missing and infinite values handled?

Use NaN for missing values. Infinity is rejected by default. Trusted pipelines
can bypass feature infinity checks with
`sklearn.config_context(assume_finite=True)`; unchecked infinity then uses the
missing-value bin.

## Can prediction accept an empty batch?

Yes, if the batch still has the fitted feature count and named-column schema.
The output keeps its normal trailing dimensions with zero rows.

## Why is my first fit slower?

Numba may compile kernels on first use. Call `darkofit.warmup()` during worker
startup and configure a persistent `NUMBA_CACHE_DIR`.

## Which tree mode should I choose?

Start with the default `catboost` mode. Use `lightgbm` when leaf-wise trees or
distributional heads are required. Treat `hybrid`, `auto`, local linear
leaves, and explicit ordinal mappings as opt-in mechanisms that require
problem-specific validation.

## Are benchmark-winning experimental knobs defaults?

No. DarkoFit requires preregistered quality, stability, runtime, memory, and
confirmation gates before promoting a policy. Several promising mechanisms
remain explicit because they failed at least one gate.
