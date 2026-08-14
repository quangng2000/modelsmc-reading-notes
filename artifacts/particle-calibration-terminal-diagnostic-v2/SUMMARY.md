# Provider-Free Particle Calibration Terminal Diagnostic V2

This is a posthoc terminal-only mechanism diagnostic. The frozen V1 gate remains failed and unchanged.

The global top-64 arm is an exhaustive developmental positive control, not a search algorithm and not evidence of search efficiency.

## Exact terminal references

| Task | Exact syntaxes | Exact mass | Mean loss |
|---|---:|---:|---:|
| foldr-bounded-square | 2 | 0.49444174 | 1.8836792 |
| calibration-lower-shift | 18 | 0.89882874 | 0.44682263 |
| calibration-upper-negate | 18 | 0.92089535 | 0.38055544 |
| calibration-equality-scale | 126 | 0.11012515 | 3.4902229 |

## Pooled Monte Carlo exact-mass error

| Arm | N | Bias | RMSE | Mean relative ESS |
|---|---:|---:|---:|---:|
| sticky-epsilon-005 | 256 | 0.120419 | 0.280944 | 0.668054 |
| sticky-epsilon-005 | 512 | 0.124946 | 0.279219 | 0.647397 |
| sticky-epsilon-025 | 256 | 0.0909341 | 0.286585 | 0.477204 |
| sticky-epsilon-025 | 512 | 0.0320031 | 0.296817 | 0.380993 |
| sticky-epsilon-050 | 256 | 0.0584419 | 0.285879 | 0.305726 |
| sticky-epsilon-050 | 512 | -0.00311772 | 0.276844 | 0.237345 |
| grammar-only-epsilon-100 | 256 | -0.492485 | 0.640075 | 0.0826629 |
| grammar-only-epsilon-100 | 512 | -0.421394 | 0.591938 | 0.0535451 |
| nonsticky-mapper-epsilon-005 | 256 | -0.101677 | 0.282737 | 0.240939 |
| nonsticky-mapper-epsilon-005 | 512 | -0.114787 | 0.296105 | 0.223913 |
| global-top64-epsilon-050-positive-control | 256 | -0.0103019 | 0.0975706 | 0.075705 |
| global-top64-epsilon-050-positive-control | 512 | -0.0016066 | 0.0626348 | 0.0751722 |

## Diagnostic conclusion

Exact importance identities pass: **True**.

Positive-control population ESS exceeds sticky epsilon=0.05 for every task: **True**.

Positive-control pooled N=512 exact-mass RMSE is below sticky epsilon=0.05: **True**.

Proposal-target overlap diagnosis supported: **True**.

This conclusion is diagnostic only. It does not turn V1 into a passing calibration study.
