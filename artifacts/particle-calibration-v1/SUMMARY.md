# Provider-Free Particle Calibration V1

This artifact calibrates the evidence-shortlist SMC estimator under the frozen 
`structured-singleton-execution-ranking-oracle-v1` proposal. It contains no live or cached provider calls.

## Exact references

| Task | Programs | Exact mass | Mean loss |
|---|---:|---:|---:|
| foldr-bounded-square | 36000 | 0.49444174 | 1.8836792 |
| calibration-lower-shift | 36000 | 0.89882874 | 0.44682263 |
| calibration-upper-negate | 36000 | 0.92089535 | 0.38055544 |
| calibration-equality-scale | 36000 | 0.11012515 | 3.4902229 |

## Pooled calibration

| N | Exact-mass bias | Exact-mass RMSE | Mean-loss bias | Mean-loss RMSE | Mean relative ESS |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.131589 | 0.324685 | 0.459449 | 4.21255 | 0.639046 |
| 64 | 0.107314 | 0.267758 | 0.335186 | 3.3211 | 0.593144 |
| 128 | 0.11931 | 0.265361 | -0.0864667 | 2.01438 | 0.596491 |
| 256 | 0.113018 | 0.239527 | -0.284705 | 1.16578 | 0.564292 |
| 512 | 0.0879051 | 0.255915 | -0.227311 | 1.05474 | 0.505415 |

## Frozen N=256 primary gate

Overall pass: **False**; pooled exact-mass RMSE 0.239527 (<0.10 required), signed bias 0.113018 ([-0.03,+0.03] required).

Per-task N=256 estimates are recorded under `analysis.json` at `primary_gate.per_task_descriptive`; no per-task pass was frozen.

## Scientific boundary

The oracle is deterministic and mechanically evidence-aware. These results measure 
finite-sample calibration for that proposal and these enumerable tasks. They do not 
measure an LLM, provider variability, arbitrary-task generalization, wall-clock 
speed, or an asymptotic convergence rate.

N counts terminal particles and 1+4N sampled program slots per repetition. The exact 
reference enumerates every grammar program, and the oracle exhaustively ranks each 
queried one-hole neighborhood using public execution loss. Those costs are reported 
separately and are not included in N, so this is not a compute-efficiency benchmark.
