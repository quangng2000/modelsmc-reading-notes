# Particle Calibration V1 Mathematical Audit

Status: **unsealed post-result methodology audit**  
Date: 2026-08-13

This note was written after the provider-free particle-calibration V1 results were
known. It is not a preregistration and must not be used to revise, replace, or
reinterpret the frozen V1 gate. The V1 gate remains failed:

- pooled exact-mass RMSE at `N=256`: `0.23952653096132373` (required `< 0.10`);
- pooled exact-mass signed bias at `N=256`: `0.11301839407999795` (required to
  lie in `[-0.03, 0.03]`);
- pooled exact-mass RMSE at `N=512`: `0.2559150661200694`;
- pooled exact-mass signed bias at `N=512`: `0.08790507058595476`.

The frozen V1 inputs are unchanged. In particular, the audited harness SHA-256 is
`b93939fdb208ec652a92a1be5f35d85aff3851fa852eef3ede2fa9e84f9cead4`, the
protocol SHA-256 is
`dd7d5ba3d5e2e8b3427b9ef0db5bc7443ae2fcfb40a642497da31590fe34d0cb`, and the
shared evidence-shortlist implementation SHA-256 is
`3737ab0e9a71b7347611ed852ac1bb43465dfd413c4a29f00a4c59606c9b6b8b`.

## Diagnosis

The observed failure is best explained by extreme finite-population
self-normalized importance-sampling variance caused by proposal/target mismatch.
The audit did not find a terminal-target mismatch, proposal-normalization error,
stale parent-weight error, or missing Feynman--Kac ratio in the recurrence that V1
actually declares.

The key failure occurs after an exact program becomes a parent. V1 replaces its
four proposal slots with four copies of that same syntax. At the terminal round,
the resulting proposal is

\[
q(y\mid x_\star)=0.95\,\delta_{x_\star}(y)+0.05\,g(y),
\]

where `g` is the normalized recursive-grammar prior. This is a useful search
policy: it preserves a discovered zero-loss program and avoids another provider
call. It is a poor posterior proposal: it spends 95% of draws on one exact syntax
while asking a 5% grammar restart to represent every other exact syntax and all
nonexact terminal mass. The importance correction is mathematically valid, but
the required correction is carried by events that are much too rare at
`N=256` or `N=512`.

## The recurrence V1 implements

Let `x_0` be the fixed initial program. For stage `t`, V1 defines an unnormalized
single-program density `gamma_t(x_t)` and an adaptive proposal
`q_t(x_t | x_{t-1})`. The declared path-space target is

\[
\Gamma_t(x_{1:t})=\prod_{s=1}^t \gamma_s(x_s),
\]

and the proposal law is

\[
Q_t(x_{1:t})=q_1(x_1\mid x_0)
             \prod_{s=2}^t q_s(x_s\mid x_{s-1}).
\]

Therefore the unnormalized path weight is

\[
W_t(x_{1:t})=
\frac{\Gamma_t(x_{1:t})}{Q_t(x_{1:t})}
=W_{t-1}(x_{1:t-1})
 \frac{\gamma_t(x_t)}{q_t(x_t\mid x_{t-1})}.
\]

That is the factor used by V1. A ratio
`gamma_t(x_t) / gamma_{t-1}(x_t)` would be appropriate for a different model: a
same-state tempering bridge with an identity or invariant mutation kernel. It is
not the incremental factor for V1's explicitly declared product-path target.

The terminal marginal is the intended stage-four target. Marginalizing the prior
path states gives

\[
\int \Gamma_4(x_{1:4})\,dx_{1:3}
=\gamma_4(x_4)\prod_{s=1}^3 Z_s,
\qquad
Z_s=\int\gamma_s(x)\,dx,
\]

so normalizing over `x_4` yields `gamma_4(x_4) / Z_4`. The extra product of early
normalizers is constant in `x_4` and cancels.

### Parent weights and resampling

At round one, the fixed parent produces `N` children. V1 divides its unit mass by
`N`, hence the `-log(N)` term. This common factor cancels during normalization but
is correct accounting.

At later rounds each of the `N` resampled parents produces one child. Systematic
resampling draws parent indices according to the normalized child weights and
then assigns every resampled parent weight `1/N`. Resetting weights after
resampling is correct. Carrying the pre-resampling normalized weight as well would
double-count it.

Conditionally on any realized parent population, with one child from every
parent,

\[
\mathbb{E}\left[\frac1N\sum_{i=1}^N
 \frac{\gamma_t(Y_i)h(Y_i)}{q_i(Y_i)}
 \middle| X_{t-1}^{1:N}\right]
=\sum_y\gamma_t(y)h(y).
\]

Thus imperfect earlier stages adapt the proposals but do not change the target of
the unnormalized terminal numerator. The final ratio of estimated numerator and
normalizer is self-normalized and is biased at finite `N`, as expected, but is
consistent under the full-support proposal.

## Proposal and estimator checks

The recursive grammar maps normalize separately over predicate and mapper
catalogs, and their product normalizes over all 36,000 complete programs. The
application samples the declared mixture and evaluates the same mixture mass,
including the multiplicity of repeated slots. The epsilon grammar component is
strictly positive on every complete program.

The reference and Monte Carlo endpoint also match. Both use
`ScoredProgram.exact_program`, meaning zero loss on all public examples, and both
aggregate all matching grammar syntaxes. Neither estimates the probability of a
single privileged target AST. The exact program multiplicities in the four tasks
are respectively 2, 18, 18, and 126.

These checks rule out the three most plausible accounting alternatives:

1. no missing parent weight remains after resampling;
2. no `gamma_t/gamma_{t-1}` term belongs to the declared product-path law;
3. no syntax-versus-semantics mismatch separates the reference from the
   estimator.

## Evidence in the sealed V1 artifact

The per-task terminal distributions and estimates have the pattern expected from
rare importance corrections:

- `foldr-bounded-square` has exact target mass `0.4944417370`, but its median
  estimate is `0.9350167872` at `N=256` and `0.9212297796` at `N=512`;
- `calibration-lower-shift` has exact mass `0.8988287374`, while the corresponding
  medians are `0.9969842135` and `0.9954832909`;
- `calibration-upper-negate` has exact mass `0.9208953488`, while the medians are
  `0.9932972893` and `0.9924280045`;
- `calibration-equality-scale` has exact mass `0.1101251499`, while the medians
  are only `0.0018195265` and `0.0061455086`.

Individual terminal populations sometimes contain a draw with 80--99.6% of all
normalized weight. These rare draws can move an estimate from near zero to near
one. At `N=512`, two `calibration-upper-negate` repetitions estimated `0.0816924`
and `0.0888012` against the `0.9208953` reference; their maximum normalized
weights were `0.918` and `0.908`. The two squared errors sum to about `1.397`,
which is larger than the entire pooled squared-error increase from `N=256` to
`N=512` (about `1.039`). The apparent RMSE reversal is therefore explained by a
few heavy-tail realizations, not evidence of worsening asymptotic behavior.

Treating the 128 pooled task/repetition errors as descriptive observations, the
approximate Monte Carlo standard errors of pooled RMSE are `0.0177` at `N=256`
and `0.0204` at `N=512`; the two RMSE estimates are not distinguishable at that
resolution. More importantly, independence across pooled tasks is not assumed
for a confirmatory claim; these numbers are only a diagnostic scale check.

## Exact finite-state overlap calculation

For a fixed exact parent, enumerate all terminal programs and define

\[
\pi(y)=\gamma_4(y)/Z_4,\qquad w(y)=\pi(y)/q(y).
\]

The population importance-ESS fraction is

\[
\rho_{\mathrm{ESS}}=
\frac{1}{\sum_y \pi(y)^2/q(y)},
\]

and the exact-mass self-normalized asymptotic variance is

\[
\sigma^2=
\sum_y \frac{\pi(y)^2}{q(y)}
\left(\mathbf{1}_{\mathrm{exact}}(y)-\mu\right)^2,
\qquad
\mu=\sum_y\pi(y)\mathbf{1}_{\mathrm{exact}}(y).
\]

For representative exact parents under the frozen `epsilon=0.05` sticky
proposal, the exact ESS fractions are:

- bounded square: `4.0056e-5`;
- lower shift: `5.3895e-5`;
- upper negate: `5.1474e-5`;
- equality scale: `7.3804e-3`.

The corresponding asymptotic particle counts needed for exact-mass standard
deviation near `0.03` are about 7.04 million, 328 thousand, 219 thousand, and 75
thousand. Changing which exact syntax is the fixed parent does not repair the
problem: over every exact parent the first three ESS fractions remain between
approximately `3.7e-5` and `5.4e-5`, and equality scale remains near
`0.0071--0.0082`.

At `N=256`, the probability of even one exact grammar-restart draw other than the
sticky parent is only about `0.00074`, `0.00582`, `0.00582`, and `0.0314` for the
four tasks. That is the rare-event mechanism visible in the artifact.

## Why increasing epsilon is insufficient

An epsilon ladder of `0.05`, `0.25`, `0.50`, and `1.0` increases global coverage,
but the grammar prior is itself diffuse relative to the high-likelihood terminal
region. At `epsilon=1`, the exact population ESS fractions for the four
representative-parent calculations improve to only about `0.000434`, `0.000774`,
`0.000739`, and `0.1412`. Pure grammar sampling often misses the exact/high-target
region entirely at these particle counts.

Removing the sticky special case while retaining a four-program local mapper
slate is also not a general fix. It helps some tasks and worsens bounded square,
because a one-hole local slice can still omit important global modes. The failure
is not exact absorption by itself; it is insufficient overlap between the full
terminal target and a proposal concentrated on a tiny state-dependent slate.

## Terminal-only V2 diagnostic

A new, separately frozen, provider-free terminal diagnostic is appropriate. It
does not rerun or overwrite V1 and does not rehabilitate the failed V1 gate. For
each task it mechanically chooses a fixed canonical exact parent from the public
finite grammar, enumerates the exact 36,000-state terminal law, and evaluates six
proposal arms:

1. sticky exact parent with `epsilon=0.05`;
2. sticky exact parent with `epsilon=0.25`;
3. sticky exact parent with `epsilon=0.50`;
4. grammar-only with `epsilon=1.00`;
5. nonsticky four-slot mapper repair with `epsilon=0.05`;
6. a parent-independent global top-64 mixture with `epsilon=0.50`.

Arms use `N in {256, 512}` and 128 frozen repetitions per task/arm/particle count,
for 6,144 repetitions and 2,359,296 terminal proposal draws. Exact census checks
record proposal normalization, importance identities, chi-square divergence,
population ESS, and endpoint asymptotic variance before Monte Carlo draws.

The sixth arm is explicitly a **posthoc developmental positive control**. Its
top-64 slate is constructed by exhaustively ranking all complete programs using
public-example loss, cost, and canonical syntax. It is intended to show whether
better proposal/target overlap removes the calibration pathology. It is not an
implementable low-cost search method and supports no claim of search efficiency,
provider quality, or generalization.

The decisive diagnostic pattern is:

- exact finite-state proposal and importance identities hold for every arm;
- the sticky arm has the predicted poor overlap and reproduces unstable
  self-normalized estimates;
- the global positive control has much better exact overlap and lower observed
  error.

That pattern distinguishes a mathematically coherent but impractical finite-`N`
proposal from a target, proposal-probability, weight, resampling, or endpoint bug.

### Observed V2 diagnostic outcome

The terminal diagnostic was subsequently frozen under protocol SHA-256
`4c09087728794b5a4652cb020b72b161089ac96cb7efb85fac68e8419450c945` and
executed provider-free on CPU. The sealed artifact contains all 6,144 planned
repetitions and 2,359,296 terminal draws. Its inventory-record digest is
`c4b18f976a06ae8a7eab69c152039ed384f90f95a1809e4d37597a6513a49969`.

The results match the diagnostic pattern:

- every exact proposal/importance identity passed; the maximum absolute census
  error was `2.220446049250313e-16` against the frozen `1e-12` tolerance;
- sticky `epsilon=0.05` pooled exact-mass RMSE was `0.2809441` at `N=256` and
  `0.2792190` at `N=512`;
- raising sticky epsilon did not repair finite-`N` error: the `epsilon=0.25` and
  `epsilon=0.50` RMSE values remained approximately `0.277--0.297` across the two
  particle counts;
- grammar-only sampling was worse, with RMSE `0.6400749` and `0.5919375`;
- removing the sticky parent but retaining the local top-four mapper slate also
  failed to help, with RMSE `0.2827374` and `0.2961051`;
- the exhaustive global positive control reduced RMSE to `0.0975706` at `N=256`
  and `0.0626348` at `N=512`, with biases `-0.0103019` and `-0.0016066`;
- its exact population importance-ESS fraction exceeded the sticky proposal by
  factors of about 876, 763, 763, and 18.7 on bounded square, lower shift, upper
  negate, and equality scale respectively.

This is direct evidence for the proposal/target-overlap diagnosis. The exact
finite-state identities contradict a terminal target, proposal-probability, or
importance-weight mismatch, while neither a larger grammar restart nor removal
of exact absorption alone fixes the problem. A proposal that globally covers the
high-target region does fix it at the tested particle counts.

The positive-control result remains posthoc and developmental. In particular,
its `N=256` RMSE being below `0.10` is **not** a new calibration pass: the arm was
constructed after V1 failed, exhaustively ranks every program, and was frozen as
a mechanism control rather than a confirmatory gate. The V1 failure is unchanged.

## Candidate for a future search-compatible method

A future full-SMC method should separate discovery retention from posterior
representation. One candidate is deterministic-mixture or balance-heuristic
weighting over all realized parent proposals,

\[
\bar q_t(y)=\frac1N\sum_{i=1}^N q_t(y\mid X_{t-1}^i),
\]

combined with deliberately diversified exact-parent slates. Because every slate
and recursive-grammar mass is application-controlled, `bar q_t` is exactly
evaluable. This may reduce the penalty from duplicated state-dependent proposals,
but it requires its own frozen calibration study. It is not tested or claimed by
V1 or by the terminal-only positive control.
