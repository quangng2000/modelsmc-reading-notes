# Verified SMC Programming by Example

This CLI combines the two papers in the repository:

- Paper 1 contributes a population of program particles, potential-based weighting, ESS, and resampling.
- Paper 2 contributes typed input-output specifications, program-family hypotheses, deduction, and a small functional language.

An offline catalog, local LLM, or cloud LLM can propose complete program ASTs for the exploratory search. A separate exact finite-grammar backend removes the LLM entirely. In both paths, a LemmaScript/Dafny-verified core type-checks and evaluates programs; the unverified shell owns scoring, floating-point inference, and orchestration.

The language now supports recursive `List<Int>` and `List<Bool>` values, `map`, and `foldr`. It is a **defunctionalized higher-order combinator DSL**, not a full lambda calculus: the displayed mapper and reducer lambdas are scoped surface notation, not general `Lambda`/`Apply` AST nodes.

## Quick start

Requirements:

- Node.js 20 or newer
- npm
- Dafny 4.11 on `PATH` for formal verification

```bash
cd papers/01-modelsmc/program-synthesis-prototype
npm install
```

Run the original scalar task:

```bash
npm run synthesize -- examples/affine-int.json --trace
```

It discovers:

```text
λx: Int. (1 + (2 * x))
```

Run the recursive-list examples:

```bash
npm run synthesize -- examples/map-increment.json --trace
npm run synthesize -- examples/foldr-sum.json --trace
npm run synthesize -- examples/foldr-filter-positive.json --trace
```

Their exact results are equivalent to:

```text
λxs: List<Int>. map (λitem: Int. item + 1) xs

λxs: List<Int>.
  foldr (λitem: Int. λacc: Int. item + acc) 0 xs

λxs: List<Int>.
  foldr
    (λitem: Int. λacc: List<Int>.
      if 0 < item then item :: acc else acc)
    []
    xs
```

The final line of each run reports whether the discovered program exactly satisfies every example.

## Specifications

Scalar tasks remain backward compatible: decimal strings represent exact integers, while JSON Booleans represent Boolean values.

```json
{
  "examples": [
    { "input": "-1", "output": "-1" },
    { "input": "0", "output": "1" },
    { "input": "2", "output": "5" }
  ],
  "integerConstants": ["-1", "0", "1", "2"]
}
```

List tasks require an explicit signature because an empty JSON array does not reveal its element type:

```json
{
  "signature": {
    "input": "List<Int>",
    "output": "List<Int>"
  },
  "examples": [
    { "input": [], "output": [] },
    { "input": ["1"], "output": ["2"] },
    { "input": ["-2", "0", "3"], "output": ["-1", "1", "4"] }
  ]
}
```

Supported signatures use `Int`, `Bool`, `List<Int>`, and `List<Bool>`. Lists are flat and homogeneous; nested lists are not in this milestone. `map` can change element type, such as `List<Int> -> List<Bool>`.

## The language

| Category | Forms |
| --- | --- |
| Program families | expression, `map`, `foldr` |
| Scoped variables | outer `Input`; mapper/fold `Item`; fold `Accumulator` |
| Values | integer and Boolean literals; typed empty integer/Boolean lists |
| Recursive constructors | typed integer or Boolean prepend (`head :: tail`) |
| Integer operations | add, subtract, multiply |
| Predicates | integer less-than and equality |
| Boolean operations | not, and |
| Control flow | typed if-then-else |

The verified runtime represents lists recursively:

```text
IntList  = IntNil  | IntCons(bigint, IntList)
BoolList = BoolNil | BoolCons(boolean, BoolList)
```

`map` and `foldr` are top-level program forms. Their semantics are structural recursion on a finite list:

```text
map f []       = []
map f (x::xs)  = f(x) :: map f xs

foldr f z []       = z
foldr f z (x::xs)  = f(x, foldr f z xs)
```

The mapper can use `Item`. The fold reducer can use `Item` and `Accumulator`; `acc` is the already-folded result of the tail. The core rejects outer `Input` inside a mapper, fold initial expression, or reducer. This closure restriction is important: it makes aligned map examples and suffix-based fold deductions valid for the admitted combinator language.

There is no unrestricted recursion, `Fix`, general application, closure value, or first-class function. Consequently, list processing terminates structurally without evaluator fuel. Nested combinator pipelines are also deferred.

## Generalization, deduction, and SMC

Before sampling, the trace identifies possible program families.

For a map task it can infer a skeleton and element-level subexamples:

```text
[trace] family map: viable skeleton ... ?mapper
[trace] deduction for map ?mapper: 1 -> 2, 2 -> 3, -2 -> -1
```

The map family is refuted when list lengths differ or the same input element would need two different outputs.

For `foldr`, an empty-list example determines the initial accumulator. If the specification includes an observed suffix and its output, deduction obtains a reducer example. For example:

```text
[]        -> 0
[3]       -> 3
[2, 3]    -> 5
[1, 2, 3] -> 6
```

induces:

```text
?initial = 0
?reducer(3, 0) = 3
?reducer(2, 3) = 5
?reducer(1, 5) = 6
```

Missing suffix observations produce a `partial` deduction message, not an unsound refutation. These hypothesis and deduction rules are implemented in the unverified search shell; every complete candidate is still checked against every top-level example by the verified evaluator.

One SMC iteration then:

1. computes effective sample size from current normalized weights;
2. resamples ancestors when relative ESS is below the threshold;
3. clones an ancestor or asks the proposer for a complete program;
4. decodes, type-checks, and evaluates the proposal;
5. converts fresh potential values into the next normalized population.

For candidate program $m$,

$$
\log G(m)
=
-\lambda\sum_j d\!\left(m(x_j),y_j\right)
-\beta\,\mathrm{cost}(m).
$$

Integer loss is capped absolute distance. Boolean loss is zero-or-one disagreement. List loss is a bounded sequence edit distance: insertion and deletion cost one, while substitution uses the scalar loss but never costs more than delete-plus-insert. This prevents one extra element from making every following element look misaligned. The potential $G(m)=\exp(\log G(m))$ is a nonnegative relative score, not a normalized probability.

The shell deliberately does not multiply old weights into this fixed-dataset score again at every refinement step. It also lacks an importance correction for the unknown LLM proposal probability. The population is therefore an approximate search distribution, not an exact Bayesian posterior.

## Exact finite-grammar SMC control

The `grammar-smc` backend is a scientific control for that limitation. It makes **no LLM or network call**. Instead, it exhaustively enumerates every well-typed AST through a small cost bound and defines a normalized Occam prior on that finite universe:

$$
p_0(e)
=
\frac{\exp[-\lambda\,\mathrm{cost}(e)]}
{\sum_{e'\in\mathcal E_C}\exp[-\lambda\,\mathrm{cost}(e')]}.
$$

Here $\mathcal E_C$ is the set of unique typed ASTs with structural cost at most $C$, and `costScale` supplies $\lambda$. A fixed inverse-temperature ladder $0=\beta_0<\cdots<\beta_T=\beta_{\max}$ then defines

$$
\pi_{\beta_t}(e)
\propto
p_0(e)\exp[-\beta_t\,s\,L(e)],
\qquad
G_t(e)=\exp[-(\beta_t-\beta_{t-1})sL(e)],
$$

where `lossScale` supplies $s$ and $L(e)$ is the verified program's total example loss. Early stages retain imperfect programs; later stages concentrate mass on low-loss programs.

Particles are sampled exactly from $p_0$, reweighted by the incremental potential, and systematically resampled when relative ESS crosses the configured threshold. Each stage also applies an independent Metropolis–Hastings move whose proposal is $p_0$. Because that proposal is known and equals the prior, the prior/proposal terms cancel and the log acceptance ratio is simply

$$
-\beta_t s\,[L(e')-L(e)].
$$

The bounded universe lets the CLI calculate the final target by enumeration as ground truth. It reports particle and exact-enumeration values for exact-program mass, mean loss, $\log(Z_{\beta}/Z_0)$, and total-variation distance. This makes particles, tempering, ESS, resampling, and the normalizing-constant estimate testable independently of LLM capability.

Run the control on the map example:

```bash
npm run synthesize -- examples/map-increment.json \
  --proposal grammar-smc \
  --particles 1024 --iterations 8 --ess-threshold 0.8 \
  --grammar-max-cost 5 --beta-max 1 --moves-per-stage 1 \
  --trace
```

`--iterations` is the number of temperature stages in this mode. `--alpha` is not used because this mode has no clone-or-LLM-propose decision. `--grammar-max-cost` is intentionally small because exhaustive typed enumeration grows combinatorially; `--grammar-limit` fails clearly before an accidental explosion. The exact grammar control is calibrated **inside its declared bounded DSL**. It is not evidence that the separate black-box LLM proposal distribution is calibrated.

## Experiment: when iterative SMC can help

[`examples/foldr-bounded-square.json`](examples/foldr-bounded-square.json) is intentionally harder than the small demonstrations. Its examples specify a list transformation that must discover three ideas together:

1. traverse with `foldr` so retained values keep their order;
2. keep only values strictly between `-2` and `3`;
3. square each retained value before prepending it.

This creates useful intermediate candidates. “Square every item,” “filter only the lower boundary,” and the complete two-boundary filter receive progressively better example scores. After every proposal, the engine records the parent-to-child loss, exact-match count, cost, and rationale. The next Ollama prompt receives those diagnostics, the inferred map/fold subexamples, its iteration and particle slot, and sibling programs to avoid.

The deterministic test suite compares two searches with the same four-proposal budget:

```text
one-shot:  4 particles × 1 iteration = 4 independent proposals
iterative: 2 particles × 2 iterations = 4 ancestry-aware proposals
```

The controlled proposer cannot finish the task in the one-shot configuration. In the iterative configuration its winning lineage strictly improves from the seed, to a partial transformation, to the exact program. This proves that the implemented feedback path can create value; it does not establish that every LLM run will do so.

Run the same equal-budget comparison against Ollama:

```bash
# Four independent calls
npm run synthesize -- examples/foldr-bounded-square.json \
  --proposal ollama --model gpt-oss:20b \
  --particles 4 --iterations 1 --alpha 0 \
  --temperature 0.7 --max-tokens 2048 --trace

# Four calls arranged as two rounds of refinement
npm run synthesize -- examples/foldr-bounded-square.json \
  --proposal ollama --model gpt-oss:20b \
  --particles 2 --iterations 2 --alpha 0 --ess-threshold 1 \
  --temperature 0.7 --max-tokens 2048 --trace
```

Evidence for SMC value is an exact solution first appearing after iteration 1, along a lineage whose losses improve, when the equal-call one-shot run does not solve the task. A reliable empirical claim requires repeating both configurations across seeds and reporting success rates—not selecting one favorable run.

That multi-seed program has been run — three experiments, **270 runs**, reported with statistics and adversarially verified figures in [`experiments/RESULTS.md`](experiments/RESULTS.md). The arc: **(E1)** at a 4-call budget, feedback value is gated by proposer capability — unnecessary at the frontier (Claude Sonnet 5 solves one-shot), visible near the threshold (Claude Haiku 4.5), absent below it (Qwen3-Coder 30B mode-collapses); **(E2)** two harder tasks show a task's "difficulty" is model-relative and that call-level failures (token truncation) can masquerade as search-strategy effects; **(E3)** at a 16-call budget, on the one cell swept to depth (Haiku × bounded-square), pure sampling gives 0/10 while deep refinement (2×8, 4×4) gives 9/10 at equal budget (Fisher p = 1.2e-4). Capability gates whether search can succeed; in that cell, refinement depth — not population width — decides whether it does (generalization to other cells is left open). The report grounds the loop in ModelSMC's Feynman–Kac potential and proves (machine-checked, via the Dafny core's `EvaluateProgramSound`) that the verified evaluator is what makes that potential a well-defined total function. Reproduce with `npx tsx experiments/run-matrix.ts` (see RESULTS.md for per-experiment flags).

## Reading the trace

`--trace` reports:

- the signature and every example;
- viable and refuted expression/map/fold families;
- inferred mapper, initial, and reducer subexamples;
- initial particle predictions and losses;
- ESS, resampling decisions, and ancestor indices;
- clone-versus-propose draws;
- decoded proposal types and rejection reasons;
- parent-to-child loss, exact-match, and cost changes;
- per-example predictions and potential scores;
- population diversity, first-exact proposal call, best-so-far improvements, and champion lineage.

Write the same events as JSONL:

```bash
npm run synthesize -- examples/foldr-sum.json \
  --trace \
  --log-file runs/foldr-sum.jsonl
```

Run `npm run synthesize -- --help` for all overrides.

## Choose a proposal backend

`--proposal` selects either the LLM-guided resample-and-revise search or the
finite-grammar SMC control. Every candidate still passes through the same
verified type checker, evaluator, structural cost, and exact acceptance check.

| Backend | Where proposals run | CLI value | Credentials |
| --- | --- | --- | --- |
| Catalog | Offline bounded enumerator | `catalog` | None |
| Ollama | Local OpenAI-compatible endpoint | `ollama` | None by default |
| Anthropic | Claude through the cloud Messages API | `anthropic` | `ANTHROPIC_API_KEY` |
| Exact grammar SMC | Offline exhaustive finite grammar; no LLM | `grammar-smc` | None |

The cloud backend can incur provider API charges; the local and catalog
backends do not make an external API request.

## Optional local open-weight LLM

The LLM proposer uses Ollama's OpenAI-compatible local endpoint:

```bash
ollama pull gpt-oss:20b
ollama serve
```

Then:

```bash
npm run synthesize -- examples/map-increment.json \
  --proposal ollama \
  --model gpt-oss:20b \
  --ollama-url http://localhost:11434/v1 \
  --max-tokens 2048 \
  --trace
```

The frozen model proposes strict JSON `Program` ASTs, never executable TypeScript or Python source. The boundary decoder enforces exact keys, tags, allowed constants, depth, node count, and acyclicity. The verified checker then rejects ill-typed or incorrectly scoped programs. A failed proposal falls back to its ancestor. The response budget defaults to 2,048 tokens and can be changed with `--max-tokens`.

The request omits the provider-specific reasoning option by default so non-thinking models work without rejecting the request. For a model that supports Ollama's OpenAI-compatible reasoning option, set it explicitly with `--reasoning-effort low`, `medium`, or `high`.

## Optional Claude cloud LLM

To propose with a hosted Claude model instead of a local one, use the `anthropic`
backend. It calls the native Messages API (`/v1/messages`) and forces a tool call
so the model must return the same strict typed-AST envelope the boundary decoder
already verifies.

The API key is read from the `ANTHROPIC_API_KEY` environment variable; it is never
passed on the command line or written to the repository. Export it in your shell:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Then:

```bash
npm run synthesize -- examples/map-increment.json \
  --proposal anthropic \
  --model claude-sonnet-5 \
  --max-tokens 2048 \
  --trace
```

`--model` defaults to `claude-sonnet-5` for this backend; override it with any
Claude model id. `--anthropic-url` overrides the API base URL (default
`https://api.anthropic.com`). `--reasoning-effort` applies only to the `ollama`
backend and is ignored here. The decode, verification, and ancestor-fallback path
is identical to the local backend—only the transport differs.

## Verification boundary

[`src/core/language.verify.ts`](src/core/language.verify.ts) is executable TypeScript and verified through the Dafny backend. Generated Dafny is in [`language.verify.dfy.gen`](src/core/language.verify.dfy.gen); additions-only proofs are in [`language.verify.dfy`](src/core/language.verify.dfy).

The proof establishes:

- scoped expression progress and preservation;
- whole-program progress and preservation for expression, map, and foldr programs;
- homogeneous typed recursive-list construction;
- fold accumulator-type preservation for integer and Boolean input lists;
- positive structural cost and direct-child dominance;
- exact characterization of example matching and program acceptance;
- nonempty homogeneous signatures for accepted example sets.

Current verification result:

```text
Dafny program verifier finished with 416 verified, 0 errors
```

The proof does **not** cover JSON parsing, LLM or HTTP behavior, grammar/catalog completeness beyond the implemented bound, hypothesis/deduction completeness, floating-point loss and normalization, ESS, randomness, resampling, logging, rendering, or global minimum cost. The exact grammar mode has a fully specified finite target and an enumerated reference distribution, but its numerical SMC implementation remains tested rather than Dafny-verified. The LLM modes remain approximate resample-and-revise searches, not calibrated posterior samplers.

## Test and inspect

```bash
npm run typecheck
npm test
npm run verify
```

The deterministic suite currently contains 45 passing tests, including scalar regression, type-changing map, right-associative fold, recursive filter construction, scope rejection, large `bigint` list values, deduction traces, exact catalog synthesis, champion retention, the equal-budget iterative-feedback scenario, CLI backend selection, mocked Ollama and Anthropic schema/decoder round trips, bounded typed enumeration, exact target normalization, deterministic grammar SMC, and agreement with enumeration.

Project layout:

```text
src/core/              executable verified semantics and Dafny proof
src/shell/ast/         strict AST decoding, JSON conversion, rendering
src/shell/catalog/     bounded offline candidate families
src/shell/cli/         argument parsing, proposer selection, command runner
src/shell/config/      JSON validation, typed values, overrides
src/shell/deduction/   family inference plus map/fold deduction
src/shell/engine/      SMC lifecycle, propagation, ESS, resampling, trace
src/shell/grammar-smc/ finite typed enumeration, exact target, tempered SMC control
src/shell/ollama/      OpenAI-compatible local HTTP proposer
src/shell/anthropic/   native Messages API cloud proposer
src/shell/proposal/    shared context, result, prompt, schema, decode, error contract
src/shell/scoring/     sequence loss and potential evaluation
src/shell/cli.ts       executable entry point only
examples/              scalar and recursive-list PBE specifications
tests/                 focused core, config, catalog, engine, Ollama, and Anthropic suites
LemmaScript-files.txt
DESIGN.md
PROOF_FINDINGS.md
```

Each feature directory exposes a narrow `index.ts`; tests may import a package-internal numerical primitive when that primitive is the test subject. Internal modules can otherwise evolve without making callers depend on their implementation layout. The verified core remains one proof unit because splitting it without verified cross-module contracts would enlarge the trusted boundary rather than improve it.
