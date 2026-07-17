# Invalid group-linear-selector attempt

The first formal invocation from clean `2b7b3dc` completed all six workers but
failed closed before writing a result artifact:

```text
RuntimeError: selector behavior changed across repeats
```

The fingerprint included raw NPZ archive byte length. The archive serializes
`verbose_timing` phase durations, whose decimal string lengths can differ
between otherwise identical fits. The binding canonical model-state hash
already replaces only that observational timing field with `null`; raw
container length is therefore not model behavior and was removed from the
fingerprint payload.

No block was dropped, no result file existed, no quality or timing statistic
was inspected, and no decision threshold changed. The protocol is unchanged.
The corrected runner must be committed and the complete reciprocal campaign
rerun from clean source.
