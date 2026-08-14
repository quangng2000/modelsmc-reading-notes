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

### `importance-smc`: typed holes and an evaluable deduction/Qwen proposal

This mode fixes the proposal-density gap rather than assigning posterior
meaning to free-form LLM output. Before sampling, it performs this fixed
symbolic pipeline:

1. infer the PBE signature and inspect structural list relationships;
2. generate type-correct expression, `map`, `foldr`, conditioned
   `foldr-filter-map`, and signed piecewise filter/map skeletons;
3. soundly refute inconsistent skeletons;
4. derive typed input-output specifications for their holes;
5. completely enumerate each generic hole through its structural-cost bound,
   or every canonical choice in a declared factorized skeleton;
6. assemble and semantically validate every complete construction;
7. assign exact cross-family aliases to a declared canonical owner, count the
   discarded traces, and reject every unexpected duplicate.

Deduction-derived hole examples enter both the prompt and a separate, exactly
normalized proposal guide; they do not hard-filter otherwise valid candidates.
The target support $\mathcal E_D$ therefore uses
only sound skeleton refutations, a deterministic bounded-catalog selection
policy, plus the explicit type, grammar, constant, cost, depth, node, and
enumeration bounds. It is fixed before Qwen is queried or any particle is
sampled.

In multi-family `auto` mode, structural inspection does not commit to one
family. It retains every generated type-correct hypothesis, and deduction
removes hypotheses contradicted by the examples. The two specialized
filter/map hypotheses share one abstract shape but declare different bounded
mapper catalogs. Auto retains the simple arithmetic catalog when one candidate
satisfies every derived mapped-value example; otherwise it retains the signed
piecewise catalog. This is an explicit finite-support policy, not a refutation
of the omitted abstract higher-order family. The deduction/Qwen mixture then
scores the surviving family before its holes. Explicitly naming a family
remains a reproducible single-family ablation.

For the hardest included task, deduction refutes `map` because list length is
not preserved. The surviving support contains expression, generic `foldr`, and
the specialized skeleton

$$
\operatorname{foldr}\Bigl(
  \lambda item,acc.\ \mathbf{if}\ p(item)\ \mathbf{then}\ v(item)::acc\ \mathbf{else}\ acc,
  [],xs
\Bigr).
$$

Suffix deduction derives examples for $p:\mathrm{Int}\to\mathrm{Bool}$ and
$v:\mathrm{Int}\to\mathrm{Int}$. With eight integer constants, their exact
catalogs contain 600 and 60 choices respectively, giving 36,000 unique complete
programs. Together with the bounded expression and generic-fold families, the
default full support contains 36,198 programs. The two ordered hole choices
preserve sequential proposal factorization while avoiding an intractable
generic reducer catalog.

The signed-window task uses the refinement

$$
\operatorname{foldr}\Bigl(
  \lambda item,acc.\ \mathbf{if}\ p(item)\ \mathbf{then}\
  g(item)::acc\ \mathbf{else}\ acc,
  [],xs
\Bigr),
$$

where $g$ is one canonical zero-sign conditional with two distinct branches.
Each branch is identity, negation, squaring, or a declared constant. With eight
constants, $g$ has $2\cdot11\cdot10=220$ choices; together with 600 outer
predicates this yields 132,000 specialized programs. Keeping $g$ as one typed
hole makes direct mapped-value deductions sound: the engine never invents an
unknown branch label merely to manufacture independent subproblems.

For each hole candidate $u$ and common prompt prefix $P$, vLLM supplies a
teacher-forced energy for the complete prompt tokenization

$$
s_\theta(P,u)=\sum_{r=2}^{|\operatorname{tok}(P\Vert u)|}
\log p_\theta(t_r\mid t_{<r}).
$$

The configurable model energy is either this total (the backward-compatible
default) or the arithmetic mean over the scored full-prompt positions. The
latter is a length-normalized energy, not a candidate-suffix probability: no
separate token boundary for textual `P` is assumed. Run results retain a
replayable, deduplicated score ledger with the full prefix plus digest, every
canonical candidate token path, total and configured energy, local
component/mixture probabilities, selections, score semantics, model identity,
and optional model/tokenizer revisions. Provider credentials and endpoints are
not copied into that ledger.

Write the configured energy as
$\widetilde{s}_\theta(P,u)\in\{s_\theta(P,u),s_\theta(P,u)/n(P,u)\}$, where
$n(P,u)$ counts scored full-prompt positions.

The application—not vLLM's output sampler—first normalizes the model component

$$
q_{\mathrm{LLM},j}(u\mid P,\mathcal C_j)
=\operatorname{softmax}_{u\in\mathcal C_j}
\left(\widetilde{s}_\theta(P,u)/\tau\right),
$$

with $\tau>0$. For a retained program $e$, let $D(e)$ count the deduplicated
derived hole examples violated by its fillings. The stage-$t$ deduction guide is

$$
r_t(e)\propto p_0(e)\exp\!\left[-\kappa_{\max}
\frac{\beta_t}{\beta_{\max}}D(e)\right].
$$

Here $p_0$ is the same equal-family, within-family Occam prior used by the
target. Exact sums of $r_t$ over each proposal subtree define $r_{t,A}(c)$ for
each family or sequential hole choice $c$ at node $A$. The sampled categorical is

$$
q_A(c)=(1-\varepsilon)\left[(1-\lambda)q_{\mathrm{LLM},A}(c)
+\lambda r_{t,A}(c)\right]+\frac{\varepsilon}{|A|}.
$$

Thus deduction cannot be overwhelmed by cumulative JSON token log-probability,
while every finite state remains reachable. A complete program has one family
choice and an ordered sequence of hole choices:

$$
q_{\mathrm{construct}}(e\mid a,D,F)
=q_H(h\mid a,D,F)\prod_j q_j(u_j\mid a,D,F,h,u_{<j}).
$$

When one family survives, $q_H=1$ and no provider family-scoring call is made.
Only choices with at least one scorer-approved completion are exposed at each
step. Identical scoring requests across resampled paths are evaluated once and
fanned back out, reducing paid model work without changing `q`.
After deduplication, a run-wide candidate-prompt budget is reserved before each
scoring wave. A wave that would cross the configured ceiling fails before any
provider request; completed artifacts persist both used and allowed prompts.

Every accepted complete program has one owned trace. Exact overlaps between
the specialized family and generic `foldr` belong to the specialized family;
discarded aliases are counted. Non-overlapping generic folds remain eligible,
and every other duplicate is an invariant failure rather than silently dropped
probability mass.

Cloning is part of the same transition law:

$$
Q_\alpha(e'\mid e)
=\alpha\mathbf1[e'=e]+(1-\alpha)q_{\mathrm{construct}}(e'\mid e,D,F).
$$

When $e'=e$, `logaddexp` combines both routes. When a sampled construction is
different, its log probability includes $\log(1-\alpha)$. The implementation
requires $\alpha<1$, and provider failure aborts instead of adding unknown
fallback mass.

Let $\mathcal E_h$ be the retained states owned by surviving family $h$. The
base prior uses a uniform family prior and a normalized Occam prior within each
family:

$$
p_0(e)
=\frac{1}{H}
 \frac{\exp[-\lambda_C\mathrm{cost}(e)]}
      {\sum_{u\in\mathcal E_{h(e)}}\exp[-\lambda_C\mathrm{cost}(u)]}.
$$

Thus every family has prior mass $1/H$ even when catalog sizes differ. The
finite unnormalized program target is

$$
\widetilde\pi_\beta(e)
=\mathbf1[e\in\mathcal E_D]p_0(e)
 \exp[-\beta\lambda_L\mathrm{loss}(e)].
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

#### Lazy joint-semantic proposal

The `joint-semantic` strategy is a model-backed proposal over complete
construction traces. It never uses observed execution loss in its prompt.
Instead, for each unique canonical slate program it computes a
label-prior-symmetrized compatibility score from two swapped binary-label
comparisons. Each comparison is accepted only when its A/B teacher-forced paths
have equal lengths, identical prefix token IDs, and one final, distinct label
token. The two raw prefix-logprob vectors remain separately hash-committed and
their maximum absolute difference is retained as a provider-numerics diagnostic;
prefix-logprob equality is not part of the token-boundary contract.
The derived evidence uses the explicit nested schema identifier
`joint-semantic-proposal-ledger-v2`.

For a deterministic slate $A$, the normalized semantic component and defensive
proposal are

$$
r_A(e)\propto \mathbf1[e\in A]p_0(e)\exp[\eta a_{\rm LLM}(e;D)],
\qquad
q(e)=\epsilon p_0(e)+(1-\epsilon)r_A(e).
$$

The factorized implementation combines exact prior suffix partitions with
semantic slate prefix masses. Each next-choice probability is child prefix
mass divided by parent prefix mass, so the sequential product equals the direct
leaf mixture. The prior floor preserves absolute continuity outside a bounded
slate. `alpha=0` is required by this first implementation.

Complete slate ASTs are assembled for model prompts but are not executed.
Only sampled traces enter `ProgramScorer`; their realized loss then enters
`log p0 - beta * lossScale * loss - log q`. Thus the importance correction is
exact conditional on the recorded slate and scores even when the LLM surrogate
is poor. Four raw label paths are charged for each unique canonical slate
program. The result uses a distinct semantic ledger and never labels these
scores as deduction-guide or full-prompt fluency energies.

The compact ledger independently replays the swapped-label score, semantic
normalization, defensive mixture, and every selected density. Its stored prior
factors are inputs to that compact replay; the runtime separately checks slate
priors against the factorized support, and an artifact-only prior audit must
rebuild the support from the bound experiment configuration. The ledger stores
hash commitments rather than duplicate raw token vectors, so an independent
token-boundary or prefix-drift audit also needs the retained content-addressed
cache entries.

#### Materialized joint-target oracle

The optional `joint-target` strategy is a separate exhaustive control, not an
LLM energy mode. It reuses the fixed `FiniteImportanceTarget` logits

$$
\log\widetilde\pi_\beta(e)
=\log p_0(e)-\beta\lambda_L\mathrm{loss}(e)
$$

after support construction has assembled and executed every complete state.
For each family or hole prefix, it groups compatible terminal states and takes
exact log-sum-exp subtree masses. Conditional log probabilities are differences
of parent and child subtree masses, so they telescope to
$\log\widetilde\pi_\beta(e)-\log Z_\beta$. The runtime checks this identity for
every selected terminal state.

The strategy requires materialized support and $\alpha=0$. It records no model
identity, performs no candidate-score request, and leaves the LLM score ledger
empty. Under these constraints every log incremental importance ratio equals
$\log Z_\beta$, normalized particle weights remain uniform, and the estimated
path normalizer matches enumeration up to floating-point tolerance. The
up-front time and memory scale with the entire bounded support, so this mode is
an oracle for proposal/target correctness rather than the scalable search
algorithm.

The model component uses finite-candidate energies from Qwen, not vLLM's
free-form output distribution; it is mixed with the exact deduction guide.
Canonical serialization removes JSON
aliases; no generated EOS, rationale, invalid output, native top-p, or output
grammar mask enters its probability. The server is configured for processed
prompt log probabilities; the client requests zero output tokens and applies
its own positive local categorical temperature. Scoring the full prompt avoids a false
assumption that separately tokenized `P` is a token-ID prefix of `P || u`;
Qwen tokenizers can merge tokens across that boundary. Incompatible provider
responses still abort the run.

### Immutable finite-score cache

Remote finite scoring may be replayed from a persistent, content-addressed
cache. Its key includes the complete score-producing and reconstruction
contract: provider/model alias and repository, immutable model and tokenizer
revisions, declared vLLM runtime/logprob configuration, score semantics,
tokenization flag, configured energy reduction, exact prompt and candidate
bytes, and the typed expression-validation context. The value contains the raw
token IDs and log probabilities plus provider metadata; AST expressions are
re-parsed under the bound validation context on every read.

Cache files are installed atomically without replacing an existing key.
Checksum, schema, key, metadata, candidate order, types, and logprob invariants
are all validated before use; any corruption aborts instead of falling back to
the provider. `replay-only` also aborts on a miss. Cache disposition is attached
to each returned score batch and copied into the score ledger. Run metrics keep
algorithm-requested candidates, cache hits/misses, provider-scored token
positions, actual HTTP request telemetry, and provider await time separate.
The scientific `max_scored_candidates` budget is charged before cache lookup,
so warm evidence cannot buy a larger search.

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
- `search/importance/` separates fixed support, reusable subtree/trie math,
  modular `guided/` and `joint/` proposals, Feynman--Kac updates, and reference
  metrics; touched Python modules stay below 300 lines.
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
