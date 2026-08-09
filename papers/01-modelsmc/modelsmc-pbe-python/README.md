# ModelSMC for programming by example

This standalone Python package studies whether ModelSMC's population,
feedback, resampling, and revision loop helps synthesize a typed program from
input-output examples. It contains the bounded program language, semantic
scorer, search engines, LLM adapters, GPU-aware numerics, and structured
logging in one installable project.

The package has three deliberately different modes:

| Mode | Proposal mechanism | What the weights mean |
| --- | --- | --- |
| `paper-search` | finite catalog or black-box LLM | Heuristic allocation scores from an uncorrected proposal kernel; **not** posterior probabilities |
| `grammar-smc` | known finite skeleton prior | An SMC approximation to the declared finite-skeleton Gibbs target, checked against exact enumeration |
| `importance-smc` | finite typed holes scored by Qwen through vLLM | Importance-corrected SMC for the fixed, bounded, deduction-refuted support, checked against exact enumeration |

The latter two modes are calibrated to their declared computational targets.
They are not thereby Bayesian posteriors over every possible program or over a
real-world system: their supports are bounded, and the soft PBE loss is not
claimed to be a scientific observation model.

See [DESIGN.md](DESIGN.md) for the algorithms and exact assurance boundary.

## Relationship to ModelSMC

The design follows [A Probabilistic Framework for LLM-Based Model
Discovery](https://arxiv.org/abs/2602.18266) and uses the authors'
[ModelSMC repository](https://github.com/mackelab/ModelSMC) as an implementation
reference. It does not import or vendor that project. Upstream targets
scientific simulator discovery and uses a substantially larger stack including
Hydra, DSPy, Ray, SBI, neural likelihood/posterior estimation, and TabPFN. This
project keeps the population lifecycle but replaces that likelihood pipeline
with deterministic PBE scoring.

See [NOTICE.md](NOTICE.md) for provenance and licensing notes.

## Install

From the repository root:

```bash
cd papers/01-modelsmc/modelsmc-pbe-python
uv sync --dev
```

Python 3.12 is pinned by `.python-version`.

## Run

All commands below start in `papers/01-modelsmc/modelsmc-pbe-python`.

### Deterministic catalog search

This exercises the practical resample/revise lifecycle without a model server:

```bash
uv run modelsmc-pbe synthesize \
  examples/map-increment.json \
  --mode paper-search \
  --proposal catalog \
  --skeleton map-arithmetic \
  --particles 8 --iterations 6 \
  --device cpu --trace
```

### Uncorrected Qwen through Ollama (baseline only)

Start Ollama in another terminal, ensure the model is installed, then run:

```bash
ollama serve
```

```bash
uv run modelsmc-pbe synthesize \
  examples/foldr-bounded-square.json \
  --mode paper-search \
  --proposal ollama \
  --model qwen3-coder:30b-a3b-q8_0 \
  --particles 2 --iterations 8 --alpha 0 \
  --ess-threshold 1 --temperature 0.7 \
  --max-tokens 4096 --timeout-seconds 300 \
  --device auto --trace
```

Ollama controls the LLM's device placement. `--device` controls Torch work in
this process; it does not move an Ollama model. Ollama is intentionally not
accepted by `importance-smc`; this command is only the uncorrected search
baseline.

### Uncorrected free-form generation through vLLM

Start vLLM with a model and stable served name, for example:

```bash
vllm serve YOUR_HUGGING_FACE_MODEL --served-model-name qwen-coder --port 8000
```

Then point the OpenAI-compatible proposal client at it:

```bash
uv run modelsmc-pbe synthesize \
  examples/map-increment.json \
  --mode paper-search \
  --proposal vllm \
  --base-url http://localhost:8000/v1 \
  --model qwen-coder \
  --particles 8 --iterations 8 \
  --max-concurrency 8 --device auto --trace
```

This is still `paper-search`: it asks vLLM for a free-form complete AST and does
not know the probability of that proposal.

### Importance-corrected Qwen proposal

`importance-smc` combines Paper 2's typed generalization and deduction with an
explicit finite proposal law:

```text
examples
  -> infer the signature and structural relationships
  -> generate viable typed skeletons
  -> refute impossible skeletons
  -> derive typed examples for each hole
  -> enumerate a bounded canonical candidate set for each hole
  -> ask Qwen to score only those candidates
  -> sample locally from a known categorical q
  -> assemble, type-check, execute, and score complete programs
  -> importance-weight and resample program particles
```

The sound skeleton refutations restrict the fixed support. Derived hole
examples are supplied to Qwen as deduction guidance but do not hard-delete
imperfect hole candidates; keeping them preserves a meaningful soft-loss
target instead of reducing expression and map families to exact solutions only.

Run this primary experiment against vLLM on the CUDA machine; no Ollama bridge
is involved. Start vLLM with Qwen. `processed_logprobs` is also the required
setting for any later experiment that samples output tokens directly. This
mode uses teacher-forced prompt log probabilities, for which vLLM's raw and
processed values coincide:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --served-model-name qwen-coder \
  --port 8000 \
  --logprobs-mode processed_logprobs
```

Then run:

```bash
uv run modelsmc-pbe synthesize \
  examples/map-increment.json \
  --mode importance-smc \
  --proposal vllm \
  --base-url http://localhost:8000/v1 \
  --model qwen-coder \
  --particles 128 --iterations 6 \
  --alpha 0.25 --temperature 0.7 --proposal-epsilon 0.05 \
  --hole-max-cost 3 --device auto --trace
```

For a provider-free probability-accounting control, replace `--proposal vllm`
with `--proposal catalog`. Every finite hole candidate then receives equal
energy.

The Qwen path does **not** sample free-form JSON. For prompt `P` and canonical
candidate `u`, it teacher-forces the complete concatenated prompt and obtains
an energy from that prompt's tokenization

$$
s_\theta(P,u)=\sum_{r=2}^{|\operatorname{tok}(P\Vert u)|}
\log p_\theta(t_r\mid t_{<r}).
$$

The shell constructs and samples its own categorical distribution

$$
q_j(u)=(1-\varepsilon)\operatorname{softmax}
\left(\frac{s_\theta(P,u)}{\tau}\right)
+\frac{\varepsilon}{|\mathcal C_j|},
\qquad \tau>0,\ \varepsilon>0.
$$

Family probability and every conditional hole probability are included in
`log q`. The uniform component keeps every target program reachable and bounds
otherwise extreme importance corrections. The clone transition is also
accounted exactly:

$$
Q_\alpha(e'\mid e)
=\alpha\mathbf 1[e'=e]+(1-\alpha)q_{\mathrm{Qwen}}(e'\mid e,D,F).
$$

If both cloning and Qwen can return the ancestor, both masses enter the
denominator. Provider errors abort this strict mode; there is no hidden
ancestor fallback. `alpha=1` is rejected because it destroys full support.

Canonical ASTs and a one-to-one construction trace remove the otherwise
intractable sum over whitespace, key order, prose, and alternative
serializations. Because vLLM supplies scores rather than performing the draw,
native top-p, output temperature, grammar masking, and EOS accounting do not
enter this `q`; `--temperature` is the positive local categorical temperature.
Scoring the complete prompt also avoids assuming that tokenizing `P` separately
produces a token prefix of `P || u`, an assumption that fails for Qwen at some
text boundaries.

At stage `t`, the incremental potential is

$$
G_t(e_{t-1},e_t)
=\frac{\widetilde\pi_{\beta_t}(e_t)}
       {Q_\alpha(e_t\mid e_{t-1},D,F)},
\quad
\widetilde\pi_\beta(e)
=\exp[-\beta\lambda_L\operatorname{loss}(e)-\lambda_C\operatorname{cost}(e)].
$$

The declared Feynman--Kac path target is the product of these stage targets,
so its terminal marginal is the desired finite `pi_beta`. Accordingly, the
reported accumulated log normalizer is for the **product path target**
`product_t Z_beta_t`, not just the final posterior's `Z_1`.

### Calibrated finite-grammar control

Map increment has a small arithmetic-map support:

```bash
uv run modelsmc-pbe synthesize \
  examples/map-increment.json \
  --mode grammar-smc \
  --skeleton map-arithmetic \
  --particles 2048 --iterations 12 \
  --beta-max 1 --moves-per-stage 1 \
  --device auto --trace
```

The bounded-square task needs the conditioned fold skeleton:

```bash
uv run modelsmc-pbe synthesize \
  examples/foldr-bounded-square.json \
  --mode grammar-smc \
  --skeleton foldr-filter-map \
  --particles 4096 --iterations 16 \
  --beta-max 1 --moves-per-stage 1 \
  --grammar-limit 250000 --device auto --trace
```

`--grammar-limit` is a safety ceiling. Enumeration either returns the complete
skeleton support or fails; it never takes an order-dependent prefix and calls
that a probability space.

A skeleton is a named, bounded hypothesis family, not a Python language
restriction or a general-purpose synthesis grammar. `expression-arithmetic`
completes one scalar arithmetic body, `map-arithmetic` completes an arithmetic
mapper, and `foldr-filter-map` completes the particular fold/filter/map shape
needed by the harder control. This conditioning makes exact enumeration and
probability accounting possible.

`importance-smc` has analogous `--hole-state-limit` and `--support-limit`
ceilings. Its default `--hole-max-cost 3` keeps live Qwen scoring tractable and
contains the `map (item + 1)` and fold-sum examples. Increasing the bound
changes the declared finite target and can grow the catalog combinatorially.

## What happens at startup

`paper-search` begins from one fixed model, `m0`, derived from the signature:

| Signature relation | Initial AST |
| --- | --- |
| input type equals output type | identity (`Input`) |
| output is `Int` | first declared integer constant |
| output is `Bool` | `false` |
| output is `List<Int>` or `List<Bool>` | corresponding typed empty list |

The semantic core scores `m0` once, and the search copies it into all `N`
particle slots. For revisions, each prompt receives the current AST,
counterexamples, and at most four recent ancestry snapshots. Full lineage IDs
remain in artifacts while bounded prompt ancestry prevents unbounded context
growth.

Provider failures do not become successful proposals. A failed call or a
scorer-rejected AST retains its selected ancestor, records a typed warning, and
increments separate response, provider-error, scorer-rejection, and accepted-
proposal counters. `result.json` sets `degraded: true` when every provider call
failed before returning an AST.

That fallback policy applies only to heuristic `paper-search`.
`importance-smc` never generates or parses a candidate response: it scores an
already validated finite catalog, samples locally, and aborts if any score
request fails. This is necessary because an unmeasured fallback would add
unknown probability mass to the ancestor.

## Configuration and semantics

Pydantic parses either the compact flat specification or the nested experiment
form into one immutable `ExperimentConfig`. CLI overrides are applied to that
same normalized object. `ProgramScorer` receives it directly, so examples,
signature, constants, loss scales, and AST bounds have one in-process source of
truth.

The semantic core strictly normalizes ASTs, enforces allowed constants and
structural bounds, performs scope-aware type inference, evaluates accepted
programs, computes structural cost and capped per-example loss, and decides
exact acceptance. Rejected candidates are typed values, so one bad model
proposal does not abort a population batch.

The bounded DSL supports four value types—`Int`, `Bool`, `List<Int>`, and
`List<Bool>`—and three complete program forms:

| Program form | Available expressions |
| --- | --- |
| scalar expression | input, integer/Boolean literals, arithmetic, comparisons, Boolean operations, conditionals, typed empty lists, and typed prepend |
| `map` | the same expressions with `Item` in scope; element type may change between `Int` and `Bool` |
| right fold | the same expressions with `Item` and `Accumulator` in scope; the initial value fixes the result type |

Map and fold bodies cannot capture the outer input list. This keeps their
semantics first-order and makes scope checking explicit. The named grammar-SMC
skeletons are finite subsets of this full transport language; model-backed
`paper-search` proposals may return any AST in the bounded DSL.

Integer values in specifications may be JSON integers or canonical decimal
strings such as `"-2"`; a leading `+` is rejected. The integer-constant catalog
must be nonempty. A signature can be inferred from nonempty scalar or list
examples, while all-empty list examples require an explicit signature. SMC
numeric settings are strict and finite rather than silently coercing strings or
accepting `NaN`/infinity.

## Artifacts

Every run that passes startup validation creates
`runs/<UTC timestamp>-<name>-<run id>/`:

| File | Contents |
| --- | --- |
| `manifest.json` | normalized configuration, git state, package versions, seed, requested/resolved device, and probabilistic claim |
| `events.jsonl` | append-only run, ESS, resampling, proposal, scoring, fallback, timing, and failure events |
| `result.json` | champion plus aggregate and degradation diagnostics |
| `final_particles.jsonl` | final ASTs, weights, scores, sources, and ancestry |

Raw prompts and free-form model rationale are omitted by default; their hashes
and operational diagnostics remain. Accepted program ASTs are retained because
they are the search states being studied. `--trace` renders the structured
event stream without changing stored records.

## Package structure

The code is organized by responsibility:

```text
src/modelsmc_pbe/
  shell/          CLI options, command assembly, provider and skeleton selection
  search/paper/   initial model, particles, objective, propagation, prompts, engine
  search/grammar_control/
                  support, target math, transitions, engine, result records
  search/importance/
                  fixed support, prompts, exact q, FK engine, reference metrics
  induction/      typed hypotheses and structural relationships
  deduction/      sound refutations and typed hole-example inference
  enumeration/    complete increasing-cost typed expression catalogs
  core/           type checking, evaluation, cost, loss, rendering, batch scorer
  proposals/      catalog and OpenAI-compatible provider adapters
  grammar/        named, complete finite skeleton enumerators
  smc/            pure population numerics
  domain/         validated PBE and AST transport types
  runtime/        device resolution and seeded generators
  observability/  structured run artifacts
```

The small `__init__.py` files mark importable packages and expose intentional
public names; they contain no algorithm. Recognizable boundaries come from
narrow modules, typed protocols, immutable records, dependency injection, and
one-way imports rather than from a large application framework.

## Assurance and reproducibility

The pure-Python semantic core is strictly validated and covered by unit,
property, integration, and golden-target tests. It is **not formally verified**.
Do not describe this package as a verified synthesizer or claim the previous
formal guarantees transfer automatically to this reimplementation.

The probabilistic shell, floating-point potentials, provider clients, and
logging are also test-backed rather than formally verified. `grammar-smc` is
calibrated because its finite target and proposal law are explicit, not because
the implementation has a machine-checked proof. `importance-smc` has the same
finite-target limitation; its extra claim is that its locally sampled proposal
law is explicit and appears in the importance denominator.

`auto` resolves Torch devices in the order CUDA, Apple MPS, then CPU. Explicitly
requesting an unavailable accelerator fails instead of silently using CPU.
Normalization, ESS, resampling, and population decisions use seeded CPU
`float64`; accelerator kernels can still differ across platforms.

## Development

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```
