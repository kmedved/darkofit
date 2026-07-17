# Invalid fresh selector confirmation attempt

The first formal launch from DarkoFit commit `7e34282` completed the three
DarkoFit waves and the ChimeraBoost wave, then failed closed in the first
CatBoost batch before writing a result artifact:

```text
TypeError: int() argument must be a string, a bytes-like object or a real
number, not 'NoneType'
```

CatBoost product defaults do not use an evaluation set, so
`get_best_iteration()` legitimately returns `None`. The runner incorrectly
required that report-only metadata value to be an integer. CatBoost fitting,
prediction, scoring, the five frozen arms, the 60 coordinates, and every
decision threshold are unchanged.

No result artifact was written and no RMSE or selection outcome was emitted
or inspected. In-memory worker results were discarded. The restart is not
adaptive: it must rerun the identical complete campaign from a clean committed
source and preserve `None` as JSON `null`. This attempt cannot support a
quality or performance claim. The CTR23 lockbox remains sealed.
