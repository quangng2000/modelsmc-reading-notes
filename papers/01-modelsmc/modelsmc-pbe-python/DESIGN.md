# ModelSMC-PBE design

## Research question

Can the population, feedback, resampling, and refinement mechanism in
[ModelSMC](https://arxiv.org/abs/2602.18266) improve typed
programming-by-example (PBE), and which probabilistic conclusions survive when
the program proposer is a black-box LLM?

This project uses the authors' [upstream
implementation](https://github.com/mackelab/ModelSMC) as a behavioral reference,
not a package dependency or vendored library. Upstream solves scientific
simulator discovery with a broad learned-likelihood stack. Here the state is a
typed program AST and observations are fixed input-output examples, so a
focused implementation makes the changed target explicit.

## Three modes, three claims

### `paper-search`: practical resample and revise

This mode preserves the practical ModelSMC lifecycle:

1. compute ESS from current normalized allocation scores;
2. systematically resample when relative ESS is below the threshold;
3. clone an ancestor with probability `alpha`, otherwise request a revised AST;
4. validate and score every returned AST through the semantic core;
5. replace, rather than accumulate, weights from the fixed-dataset objective;
6. generate counterexample feedback and repeat.

For PBE data $D=\{(x_j,y_j)\}_{j=1}^M$, the allocation objective is

$$
\log G(e)
=
-\lambda_L\sum_{j=1}^{M}d(e(x_j),y_j)
-\lambda_C\,\mathrm{cost}(e).
$$

The LLM or catalog induces a history- and feedback-conditioned proposal kernel.
For an LLM, its probability
$q(e'\mid e,\text{prompt},\text{feedback})$ is unavailable, and the shell does
not apply an importance correction. Normalized particle weights are therefore
search-allocation scores, **not** samples from a calibrated posterior. Artifacts
label this mode `heuristic_search_uncorrected_proposal_kernel`.

### `grammar-smc`: finite calibrated control

For one complete finite skeleton support $\mathcal E$, define

$$
p_0(e)
\propto
\exp[-\lambda_C\mathrm{cost}(e)],
\qquad
\pi_\beta(e)
\propto
p_0(e)\exp[-\beta\lambda_L\mathrm{loss}(e)].
$$

For $0=\beta_0<\cdots<\beta_T=\beta_{\max}$, the incremental potential is

$$
G_t(e)
=
\exp[-(\beta_t-\beta_{t-1})\lambda_L\mathrm{loss}(e)].
$$

Particles begin from known normalized $p_0$. Rejuvenation proposes an
independent state from $p_0$, so prior and proposal factors cancel in the
Metropolis-Hastings ratio:

$$
\log a(e,e')
=
\min\!\left(0,-\beta_t\lambda_L
[\mathrm{loss}(e')-\mathrm{loss}(e)]\right).
$$

The enumerator constructs the complete declared skeleton support or fails at
the safety ceiling. Exact enumeration supplies reference state probabilities,
mean loss, exact-program mass, log-normalizer ratio, and total-variation error.

These weights are calibrated to the declared finite-skeleton Gibbs target. The
claim does not extend to an unbounded program language or turn the soft loss
into a real-world likelihood.

### `importance-smc`: typed holes and an evaluable Qwen-energy proposal

This mode fixes the proposal-density gap rather than assigning posterior
meaning to free-form LLM output. Before sampling, it performs this fixed
symbolic pipeline:

1. infer the PBE signature and inspect structural list relationships;
2. generate type-correct expression, `map`, and `foldr` skeletons;
3. soundly refute inconsistent skeletons;
4. derive typed input-output specifications for their holes;
5. completely enumerate each hole through the declared structural-cost bound;
6. assemble and semantically validate every complete construction;
7. reject duplicate construction traces rather than hiding an AST alias sum.

Deduction-derived hole examples guide Qwen's scores; they do not hard-filter
otherwise valid candidates. The target support $\mathcal E_D$ therefore uses
only sound skeleton refutations plus the explicit type, grammar, constant,
cost, depth, node, and enumeration bounds. It is fixed before Qwen is queried
or any particle is sampled.

For each hole candidate $u$ and common prompt prefix $P$, vLLM supplies a
teacher-forced energy for the complete prompt tokenization

$$
s_\theta(P,u)=\sum_{r=2}^{|\operatorname{tok}(P\Vert u)|}
\log p_\theta(t_r\mid t_{<r}).
$$

The application—not vLLM's output sampler—normalizes and samples

$$
q_j(u\mid P,\mathcal C_j)
=(1-\varepsilon)\operatorname{softmax}_{u\in\mathcal C_j}
\left(s_\theta(P,u)/\tau\right)
+\varepsilon/|\mathcal C_j|,
$$

with $\tau>0$ and $\varepsilon>0$. A complete program has one family choice
and an ordered sequence of hole choices, so

$$
\log q_{\mathrm{Qwen}}(e\mid a,D,F)
=-\log H+\sum_j\log q_j(u_j\mid a,D,F,h,u_{<j}).
$$

Only choices with at least one scorer-approved completion are exposed at each
step. Every complete construction has one trace, so this product is the
program's probability rather than merely one contribution to it.

Cloning is part of the same transition law:

$$
Q_\alpha(e'\mid e)
=\alpha\mathbf1[e'=e]+(1-\alpha)q_{\mathrm{Qwen}}(e'\mid e,D,F).
$$

When $e'=e$, `logaddexp` combines both routes. When a sampled Qwen proposal is
different, its log probability includes $\log(1-\alpha)$. The implementation
requires $\alpha<1$, and provider failure aborts instead of adding unknown
fallback mass.

The finite unnormalized program target is

$$
\widetilde\pi_\beta(e)
=\mathbf1[e\in\mathcal E_D]
\exp[-\beta\lambda_L\mathrm{loss}(e)-\lambda_C\mathrm{cost}(e)].
$$

Repeated `pi/q` updates are given an explicit Feynman--Kac meaning. The path
target is

$$
\gamma_t(e_{1:t})=\prod_{s=1}^{t}\widetilde\pi_{\beta_s}(e_s),
$$

the transition is $M_t=Q_\alpha$, and the incremental potential is

$$
G_t(e_{t-1},e_t)
=\frac{\widetilde\pi_{\beta_t}(e_t)}
       {Q_\alpha(e_t\mid e_{t-1},D,F)}.
$$

Thus $M_tG_t=\widetilde\pi_{\beta_t}$, and the current-program marginal is
exactly the normalized finite target $\pi_{\beta_t}$. If resampling is skipped,
the previous normalized path weight is multiplied by $G_t$. After systematic
resampling, the base weight resets to $1/N$. Examples are fixed observations;
only program particles are resampled.

The path normalizer is $\prod_t Z_{\beta_t}$. Artifacts compare its SMC estimate
with $\sum_t\log Z_{\beta_t}$ from exact enumeration and separately compare the
terminal marginal using total variation, exact-program mass, mean loss, and
mean cost. They do not mislabel the accumulated value as the single final
posterior normalizer.

This proposal is a finite categorical whose energies come from Qwen, not
vLLM's free-form output distribution. Canonical serialization removes JSON
aliases; no generated EOS, rationale, invalid output, native top-p, or output
grammar mask enters its probability. Scoring the full prompt avoids a false
assumption that separately tokenized `P` is a token-ID prefix of `P || u`;
Qwen tokenizers can merge tokens across that boundary. Incompatible provider
responses still abort the run.

## Initial state and proposal context

The practical algorithm starts all $N$ particles from the same deterministic
$m_0$, derived from the declared signature:

- identity when input and output types match;
- the first allowed integer constant for `Int` output;
- `false` for `Bool` output;
- the corresponding typed empty list for list output.

The semantic scorer evaluates this AST once before it is copied. Each LLM
request includes the current AST, its failures, and at most four recent ancestry
snapshots. Particle artifacts retain full lineage identifiers, but prompts do
not grow without bound.

## Provider and rejection policy

Concurrent provider requests resolve independently. One timeout, HTTP failure,
or malformed response does not cancel the batch. That slot retains its selected
ancestor and records `proposal.failed`. A scorer-rejected AST follows the same
fallback and records `proposal.rejected`.

The result separates proposal calls, responses, provider errors, scorer
rejections, and accepted proposals. It reports `degraded: true` when every
provider call fails before producing an AST. `exact` and `degraded` answer
different questions.

That recoverable fallback belongs only to `paper-search`. In
`importance-smc`, every candidate is enumerated, canonical, typed, and known
before provider I/O. vLLM only scores it. A timeout, malformed prompt-logprob
path, token-boundary mismatch, or other provider failure aborts the calibrated
run because retaining the ancestor without its exact rejection probability
would change $Q_\alpha$.

## Configuration boundary

Pydantic parses either the compact flat specification or nested experiment form
into one immutable `ExperimentConfig`. CLI overrides are applied there.
`ProgramScorer` receives that object directly, so the signature, examples,
constants, loss scales, and AST bounds cannot diverge across processes or
language runtimes. Python owns particle count, iterations, resampling, and
proposal scheduling.

## Semantic and tested boundaries

All candidate decisions route through the pure-Python semantic core:

| Responsibility | Status |
| --- | --- |
| AST normalization and bounds | tested Python core |
| scope-aware type inference | tested Python core |
| program evaluation | tested Python core |
| structural cost | tested Python core |
| per-example loss and exact acceptance | tested Python core |
| ESS, resampling, MH, and normalizer estimates | tested Python SMC |
| provider I/O, prompting, fallback, and logging | tested Python shell |

The semantic core is not formally verified. Its tests establish strong
implementation evidence but do not constitute a proof of progress,
preservation, termination, or evaluator soundness. Calibration of
`grammar-smc` follows from its explicit finite target and proposal accounting;
it is separate from formal software verification.

## Modules and dependencies

Dependencies point inward toward domain types and pure calculations:

```text
shell/ ──────────┐
                 v
search/paper/ + search/grammar_control/ + search/importance/
      |                    |                    |
      v                    v                    v
 proposals/              smc/                 core/
      |                    |                    |
      └────────────────────┴────────────────────┘
                           v
       induction/ + deduction/ + enumeration/ + domain/ + runtime/

observability/ is injected at orchestration boundaries.
```

- `shell/` owns CLI parsing and construction, not search behavior.
- `search/paper/` separates initialization, objective, particles, propagation,
  provider batches, prompts, counters, and orchestration.
- `search/grammar_control/` separates support construction, target math,
  Markov transitions, result records, and annealing.
- `search/importance/` separates fixed support, Qwen-energy prompts, exact
  proposal accounting, Feynman--Kac updates, and reference metrics.
- `induction/`, `deduction/`, and `enumeration/` implement the Paper-2-inspired
  typed skeleton, refutation, hole-example, and increasing-cost front end.
- `core/` owns deterministic program semantics and scoring, not search.
- `grammar/` declares finite supports; it does not score programs.
- `proposals/` owns provider transport and AST extraction, not semantics.
- `smc/` contains tensor numerics independent of CLI and providers.

Small `__init__.py` files define public import surfaces; logic stays in
responsibility-named modules. Typed protocols, immutable Pydantic models and
dataclasses, explicit dependency injection, and one-way dependencies make the
architecture recognizable without a heavy application framework.

## Devices and reproducibility

`auto` resolves CUDA first, then Apple MPS, then CPU. Explicitly requesting an
unavailable accelerator is an error. Torch accelerates population-scale
potential calculations and metrics; Ollama or vLLM separately owns model
inference placement. AST execution is deterministic Python semantics.

Seeds, stable normalization, ESS, systematic resampling, and population
decisions use CPU `float64`. Repeating a run on the same device and software
stack is designed to be reproducible. CPU, CUDA, and MPS are not promised to be
bit-identical because accelerator kernels and precision can round differently.

## Artifacts and test claims

Each run writes a manifest, append-only JSONL event stream, result summary, and
final particle JSONL. Prompts and raw responses are omitted by default; hashes,
latency, failure class, proposal counts, ESS, resampling decisions, and scoring
diagnostics remain observable.

Tests cover configuration normalization, semantic type/evaluation/cost/loss
behavior, AST bounds, device resolution, population numerics, grammar
completeness, seeded lineage, independent provider failures, grammar-SMC
agreement with enumeration, importance proposal and clone-mixture accounting,
symbolic induction and deduction, exact hole enumeration, and artifact schemas.
Live provider calls are not part of the suite.
