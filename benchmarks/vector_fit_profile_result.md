# Classification and distributional fit-path profile result

_Run 2026-07-17 from clean DarkoFit `447190e`. The create-only
[`vector_fit_profile.json`](vector_fit_profile.json) artifact has SHA-256
`e1ae6facac2a9c465fe0bbde6b99c7c649b3652e9600777935238ca92f8b876f`
and binds the frozen
[`vector_fit_profile_protocol.md`](vector_fit_profile_protocol.md)._

## Result

Tree construction, not loss/gradient evaluation, dominates every unmeasured
path.

| Path | Median ms / round | Tree-build share | Grad/Hess share | Component trees |
| --- | ---: | ---: | ---: | ---: |
| Scalar RMSE CatBoost control | 3.72 | 69.6% | 0.8% | 40 |
| Binary CatBoost | 4.73 | 73.9% | 3.5% | 40 |
| Multiclass CatBoost per-class | 15.05 | 90.7% | 1.6% | 160 |
| Multiclass LightGBM shared-vector | 22.76 | 92.4% | 2.3% | 40 |
| Gaussian LightGBM | 20.39 | 92.6% | 1.7% | 40 |
| Student-t LightGBM | 20.45 | 92.5% | 1.8% | 40 |

Fresh-worker fit IQR / median ranged from 0.015 to 0.047, and predictions were
identical across all three workers for every path.

## Decision

Profile the tree builder inside Gaussian LightGBM before changing E1. The
automatic opportunity selector chose that path because tree construction
accounted for 92.56% of attributed time.

The profile rules out loss-kernel optimization as a first move: even Student-t
gradient/Hessian work was under 2% of attributed time. It also shows that the
four-class shared-vector LightGBM path is slower per round than four per-class
CatBoost trees on this workload; that is a diagnostic lead, not a strategy
default decision.

For the planned hessian-carrying fused CatBoost expansion, binary classification
remains the closest low-risk implementation lane: it is 1.27x the scalar
control per round and 73.9% tree-bound. Any E1 implementation still needs a
direct tree-kernel profile plus bit-identity gates.

No external comparison, model-policy change, CTR23 coordinate, or lockbox task
was used.
