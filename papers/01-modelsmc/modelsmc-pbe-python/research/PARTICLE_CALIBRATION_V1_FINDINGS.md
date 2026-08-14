# Provider-Free Particle Calibration V1: Post-Run Audit

> This post-run findings note is outside the immutable experiment inventory. The
> sealed calibration artifact remains unchanged at `artifacts/particle-calibration-v1`.

## Outcome

The frozen N=256 primary gate **failed**. Across all four tasks and 32 frozen
repetitions per task:

- exact-mass RMSE: **0.2395265** (required `< 0.10`);
- signed exact-mass bias: **+0.1130184** (required `[-0.03, +0.03]`).

Both gate components failed. This is a negative calibration result and is not
evidence that the current estimator is accurate at N=256.

## Frozen design and integrity

- Four developmental, non-blind filter-map tasks, each with 36,000 complete
  program syntaxes.
- Particle counts 32, 64, 128, 256, and 512; 32 frozen seeds per cell.
- Complete design: 4 tasks x 5 particle counts x 32 repetitions = 640 runs.
- Budget: one fixed initial complete program plus `4N` proposal draws, or
  `1+4N` logical complete-program slots per run; 508,544 slots in total.
- Zero provider calls authorized and zero observed.
- Proposal: the frozen deterministic public-evidence/public-loss ranking oracle
  mixed with a 0.05 recursive-grammar restart.

All 649 inventory records rehashed successfully. The inventory records digest
is `a9096e0a4ec2f042b5f7e53f53bc535c4f25ec159b98e51e557389698ae5fd3b`;
the analysis SHA-256 is
`34cad4e333820853993fc80a61bb1345d5ed39e23cb1f30bf301a89066e257bd`.
An independent post-run computation reproduced the 20 complete task/N cells,
seed order, `1+4N` budgets, zero-call total, and pooled bias/RMSE values.

## N=256 task heterogeneity

| Task | Exact-mass bias | Exact-mass RMSE | Mean-loss bias | Mean-loss RMSE | Mean relative ESS |
|---|---:|---:|---:|---:|---:|
| bounded square | +0.34013 | 0.41565 | -1.01169 | 1.82692 | 0.44360 |
| lower shift | +0.09576 | 0.09600 | -0.36107 | 0.37258 | 0.91861 |
| upper negate | +0.06628 | 0.07533 | -0.16871 | 0.59948 | 0.87762 |
| equality scale | -0.05010 | 0.20453 | +0.40266 | 1.26506 | 0.01734 |

The easier interval tasks do not hide the bounded-square and equality failures.
Per-task results were descriptive rather than separately gated, but they are
necessary to interpret the pooled failure.

## Scaling and the N=512 uptick

Pooled exact-mass RMSE was 0.32468, 0.26776, 0.26536, 0.23953, and 0.25592 for
N=32 through N=512. The descriptive log-log slope was only -0.08475. Mean-loss
RMSE improved more clearly (slope -0.55060), while mean terminal ESS grew with
slope 0.92512. ESS growth therefore did not imply calibration of exact mass.

The increase in pooled RMSE from N=256 to N=512 is **not statistically resolved
by 32 repetitions per task**. The observed mean-squared-error difference was
+0.00812. An unpaired task-pooled nonparametric bootstrap over the 128 results
at each N gave a 95% interval of approximately [-0.0177, +0.0349], spanning
zero. Thus the specific non-monotonic step can be sampling variability and must
not be claimed as proof that increasing N makes the estimator structurally
worse.

The uptick is largely driven by two upper-negate N=512 repetitions whose
exact-mass estimates were 0.0817 and 0.0888 versus a 0.9209 reference. Their
terminal ESS values were 1.19 and 1.21 and maximum weights were 0.918 and 0.908.
These outliers demonstrate real importance-weight degeneracy risk under the
proposal, but 32 repetitions do not identify the population N=512 error as
higher than N=256 error.

The broader conclusion is firmer: bias and RMSE remain materially large across
the frozen grid, especially for bounded square, and there is no empirical
support here for the reference `N^-1/2` exact-mass RMSE scaling.

## Computational caveat

This is not a search-efficiency benchmark. Exact-reference construction first
evaluated all 36,000 complete programs per task and populated a task-local
public-loss cache. During 640 runs the oracle made 3,095,027 logical candidate
score lookups across 5,533 previously unranked one-hole neighborhoods; all
scorer requests were cache hits. The oracle does not read target ASTs, target
weights, or reference aggregates, but it exhaustively ranks each queried local
neighborhood by public execution loss.

## Defensible claim

Under the frozen mechanical oracle and this four-task finite suite, the current
evidence-shortlist SMC estimator did not meet the preregistered N=256
exact-mass calibration gate. Exact-program discovery can be high while
posterior-mass estimation remains inaccurate. No LLM-quality, provider,
wall-clock, arbitrary-task, or asymptotic claim follows.

Any v2 proposal or schedule must be frozen separately. The failed v1 artifact
must remain unchanged and reported alongside later results.
