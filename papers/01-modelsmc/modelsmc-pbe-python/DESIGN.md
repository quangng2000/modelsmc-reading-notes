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

## Two modes, two claims

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
search/paper/ + search/grammar_control/
      |          |       |
      v          v       v
 proposals/   smc/     core/
      |          |       |
      └──────────┴───────┘
                 v
          domain/ + runtime/

observability/ is injected at orchestration boundaries.
```

- `shell/` owns CLI parsing and construction, not search behavior.
- `search/paper/` separates initialization, objective, particles, propagation,
  provider batches, prompts, counters, and orchestration.
- `search/grammar_control/` separates support construction, target math,
  Markov transitions, result records, and annealing.
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
agreement with enumeration, and artifact schemas. Live provider calls are not
part of the suite.
