# Verified SMC Programming by Example

This CLI combines the two papers in the repository:

- Paper 1 contributes a population of program particles, potential-based weighting, ESS, and resampling.
- Paper 2 contributes typed input-output specifications, program-family hypotheses, deduction, and a small functional language.

An offline catalog or frozen local LLM proposes complete program ASTs. A LemmaScript/Dafny-verified core type-checks and evaluates them. The unverified SMC shell scores each candidate against the examples, favors smaller programs, and keeps a best-so-far exact discovery even if later resampling removes it from the live population.

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
-\beta\,\operatorname{cost}(m).
$$

Integer loss is capped absolute distance. Boolean loss is zero-or-one disagreement. List loss is a bounded sequence edit distance: insertion and deletion cost one, while substitution uses the scalar loss but never costs more than delete-plus-insert. This prevents one extra element from making every following element look misaligned. The potential $G(m)=\exp(\log G(m))$ is a nonnegative relative score, not a normalized probability.

The shell deliberately does not multiply old weights into this fixed-dataset score again at every refinement step. It also lacks an importance correction for the unknown LLM proposal probability. The population is therefore an approximate search distribution, not an exact Bayesian posterior.

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

The proof does **not** cover JSON parsing, LLM or HTTP behavior, catalog completeness, hypothesis/deduction completeness, floating-point loss and normalization, ESS, randomness, resampling, logging, rendering, global minimum cost, or posterior correctness. This is a verified semantic core inside an approximate synthesis experiment—not an end-to-end verified synthesizer.

## Test and inspect

```bash
npm run typecheck
npm test
npm run verify
```

The deterministic suite currently contains 35 passing tests, including scalar regression, type-changing map, right-associative fold, recursive filter construction, scope rejection, large `bigint` list values, deduction traces, exact catalog synthesis, champion retention, the equal-budget iterative-feedback scenario, CLI backend selection, and mocked Ollama and Anthropic schema/decoder round trips.

Project layout:

```text
src/core/              executable verified semantics and Dafny proof
src/shell/ast/         strict AST decoding, JSON conversion, rendering
src/shell/catalog/     bounded offline candidate families
src/shell/cli/         argument parsing, proposer selection, command runner
src/shell/config/      JSON validation, typed values, overrides
src/shell/deduction/   family inference plus map/fold deduction
src/shell/engine/      SMC lifecycle, propagation, ESS, resampling, trace
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
