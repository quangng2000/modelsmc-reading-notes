# A small λ² synthesizer in TypeScript

This directory turns the core of the λ² paper into an executable TypeScript tool. Three typed skeleton families — `map`, `filter`, and left `fold` — compete inside one cost-ordered search, each with its own sound deduction rule:

```text
list examples
  -> read or infer a typed signature
  -> select type-compatible map/filter/fold skeletons
  -> propose typed skeletons (map/filter, or foldl)
  -> deduction refutes impossible skeletons and derives hole examples
  -> cost-ordered enumeration fills the remaining holes
  -> check deduced constraints, then the original examples
  -> first surviving program is minimum-cost across all families
```

For example, `[1, 2] -> [3, 4]` and `[3] -> [5]` synthesize

```text
(xs: list<int>) => map((x: int) => (x + 2), xs)          (cost 5)
```

while `[1, -2, 3, 0] -> [1, 3]` refutes `map` (lengths differ) and synthesizes

```text
(xs: list<int>) => filter((x: int) => (0 < x), xs)       (cost 5)
```

and the scalar examples `[] -> 0`, `[1] -> 1`, `[1, 2] -> 3`, `[1, 2, 3] -> 6` synthesize

```text
(xs: list<int>) => foldl((acc: int) => (x: int) => (acc + x), 0, xs)   (cost 6)
```

## Run it

```bash
cd papers/02-data-structure-synthesis/prototype
npm ci
npm test
npm run synth -- --trace examples/paper-map-example.json
```

The `cd` command assumes you are starting at this repository's root.

## Examples in, program out

The synthesizer receives **behavior examples**, not the target source code. Each example says, “for this input list, produce this output.” It searches the supported grammar and returns the lowest-cost program that satisfies every supplied example.

### Paper example: inferring a `map` hole

The paper's introductory **Example inference** discussion uses the top-level example `[1, 2] -> [3, 4]` and the hypothesis `λx. map f* x`. Its deduction rule turns that one list example into two examples for the unknown function: `1 -> 3` and `2 -> 4`.

That exact top-level example is available as [`examples/paper-map-example.json`](examples/paper-map-example.json):

```json
{
  "examples": [
    {
      "input": [1, 2],
      "output": [3, 4]
    }
  ],
  "maxCost": 4
}
```

Run it with full tracing:

```bash
npm run synth -- --trace examples/paper-map-example.json
```

The console shows the complete reasoning path:

```text
[trace] loaded 1 example
[trace] signature: list<int> -> list<int>
[trace] example 1: [1, 2] -> [3, 4]
[trace] search started; possible families: map, filter
[trace] family map: viable skeleton (xs: list<int>) => map(?f: int -> int, xs) (cost 2)
[trace] deduction subproblem type: ?f: int -> int
[trace] deduction for ?f: 1 -> 3, 2 -> 4
[trace] family filter: refuted — filter cannot introduce 3; the output needs more copies than the input provides
[trace] candidate 1: map, cost 3, (xs: list<int>) => map((x: int) => x, xs) -> rejected by deduction — inferred sub-example ?f(1): expected 3, got 1
...
[trace] candidate 11: map, cost 5, (xs: list<int>) => map((x: int) => (x + 2), xs) -> accepted
[trace] search finished: synthesized after 11 tested candidates
```

The CLI prints candidates 1 through 11; the `...` above only shortens the README excerpt. Trace mode includes every loaded example, candidate family, deduction result, viable skeleton, completed candidate in cost order, rejection reason, accepted program, and final count. Its output therefore grows quickly when `maxCost` is increased.

The generated program is:

```text
(xs: list<int>) => map((x: int) => (x + 2), xs)
```

### Paper example with multiple examples

Section 4.3's second example combines two top-level observations. The fixture
[`examples/paper-multiple-map-example.json`](examples/paper-multiple-map-example.json)
contains them exactly:

```json
{
  "examples": [
    { "input": [1, 2], "output": [2, 3] },
    { "input": [2, 4], "output": [3, 5] }
  ],
  "maxCost": 4
}
```

Run it with:

```bash
npm run synth -- --trace examples/paper-multiple-map-example.json
```

Deduction merges information from both observations. The repeated input value
`2` consistently maps to `3`, leaving three distinct constraints for the hole:

```text
[trace] loaded 2 examples
[trace] signature: list<int> -> list<int>
[trace] example 1: [1, 2] -> [2, 3]
[trace] example 2: [2, 4] -> [3, 5]
[trace] search started; possible families: map, filter
[trace] family map: viable skeleton (xs: list<int>) => map(?f: int -> int, xs) (cost 2)
[trace] deduction subproblem type: ?f: int -> int
[trace] deduction for ?f: 1 -> 2, 2 -> 3, 4 -> 5
[trace] family filter: refuted — filter cannot introduce 3; the output needs more copies than the input provides
[trace] candidate 1: map, cost 3, (xs: list<int>) => map((x: int) => x, xs) -> rejected by deduction — inferred sub-example ?f(1): expected 2, got 1
...
[trace] candidate 10: map, cost 5, (xs: list<int>) => map((x: int) => (x + 1), xs) -> accepted
[trace] search finished: synthesized after 10 tested candidates
```

The actual console includes candidates 1 through 10. It synthesizes:

```text
(xs: list<int>) => map((x: int) => (x + 1), xs)
```

### Additional filter example

The included [`examples/keep-positive.json`](examples/keep-positive.json) request is:

```json
{
  "examples": [
    {
      "input": [1, -2, 3, 0],
      "output": [1, 3]
    }
  ],
  "maxCost": 4
}
```

Run it with:

```bash
npm run synth -- examples/keep-positive.json
```

From that input-output behavior, the tool generates:

```text
(xs: list<int>) => filter((x: int) => (0 < x), xs)
```

The CLI's complete report includes the selected family, program cost, number of tested candidates, and deduction results:

```text
family:  filter
program: (xs: list<int>) => filter((x: int) => (0 < x), xs)
cost:    5
tested:  13 candidates
deduction:
  map: refuted — map preserves list length
  filter: viable
```

### Complex specification with multiple examples

[`examples/multiple-sum-fold.json`](examples/multiple-sum-fold.json) supplies five examples, including an empty list, a prefix chain, and an additional non-prefix case:

```json
{
  "examples": [
    { "input": [], "output": 0 },
    { "input": [1], "output": 1 },
    { "input": [1, 2], "output": 3 },
    { "input": [1, 2, 3], "output": 6 },
    { "input": [-2, 5, 1], "output": 4 }
  ],
  "maxCost": 4
}
```

Run all five examples through the traced search:

```bash
npm run synth -- --trace examples/multiple-sum-fold.json
```

The scalar outputs select the `fold` family. Deduction fixes `?init` from the empty example and derives reducer steps from the prefix-related examples. The final non-prefix example is still checked during full-program validation:

```text
[trace] loaded 5 examples
[trace] signature: list<int> -> int
[trace] example 1: [] -> 0
[trace] example 2: [1] -> 1
[trace] example 3: [1, 2] -> 3
[trace] example 4: [1, 2, 3] -> 6
[trace] example 5: [-2, 5, 1] -> 4
[trace] search started; possible families: fold
[trace] family fold: viable skeleton (xs: list<int>) => foldl(?f: int -> int -> int, ?init: int, xs) (cost 2)
[trace] deduction subproblem types: ?f: int -> int -> int, ?init: int
[trace] deduction for ?init: 0
[trace] deduction for reducer ?f: (0, 1) -> 1, (1, 2) -> 3, (3, 3) -> 6
[trace] candidate 1: fold, cost 5, (xs: list<int>) => foldl((acc: int) => (x: int) => acc, 0, xs) -> rejected by deduction — inferred reducer sub-example ?f(0, 1): expected 1, got 0
...
[trace] candidate 8: fold, cost 6, (xs: list<int>) => foldl((acc: int) => (x: int) => (acc + x), 0, xs) -> accepted
[trace] search finished: synthesized after 8 tested candidates
```

Again, the CLI prints every candidate; the README elides candidates 2 through 7 only to stay readable. The generated program is:

```text
(xs: list<int>) => foldl((acc: int) => (x: int) => (acc + x), 0, xs)
```

### Explicit string and Boolean specifications

Non-integer examples declare their signature explicitly. For example,
[`examples/string-is-yes-map.json`](examples/string-is-yes-map.json) asks for
a `list<string> -> list<bool>` program:

```json
{
  "inputType": "list<string>",
  "outputType": "list<bool>",
  "examples": [
    {
      "input": ["yes", "no"],
      "output": [true, false]
    },
    {
      "input": ["maybe", "yes"],
      "output": [false, true]
    }
  ],
  "stringConstants": ["yes"],
  "maxCost": 2
}
```

Run it with:

```bash
npm run synth -- --trace examples/string-is-yes-map.json
```

Type-directed family selection considers only `map` because `filter` cannot
change the element type from `string` to `bool`. Deduction produces
`"yes" -> true`, `"no" -> false`, and `"maybe" -> false`, then enumeration
finds:

```text
(xs: list<string>) => map((x: string) => (x == "yes"), xs)
```

Three more typed fixtures exercise the rest of the primitive grammar:

```bash
# string length: list<string> -> list<int>
npm run synth -- --trace examples/string-lengths-map.json

# Boolean disjunction: list<bool> -> bool
npm run synth -- --trace examples/boolean-any-fold.json

# string concatenation: list<string> -> string
npm run synth -- --trace examples/concatenate-strings-fold.json
```

They synthesize `length(x)`, `acc || x`, and `acc ++ x`, respectively.

## CLI input format

The CLI reads a JSON request from a file or inline:

```bash
npm run build
node dist/src/cli.js examples/keep-positive.json
node dist/src/cli.js --trace examples/keep-positive.json
node dist/src/cli.js -e '{"examples": [{"input": [1, -2, 3, 0], "output": [1, 3]}]}'
```

Legacy integer requests may omit types. They are interpreted as
`list<int> -> list<int>` or `list<int> -> int`:

```json
{
  "examples": [ { "input": [1, -2, 3, 0], "output": [1, 3] } ],
  "maxCost": 4,
  "constants": [-1, 0, 1, 2]
}
```

Typed requests add both `inputType` and `outputType`:

```json
{
  "inputType": "list<string>",
  "outputType": "list<bool>",
  "examples": [
    { "input": ["yes", "no"], "output": [true, false] }
  ],
  "maxCost": 4,
  "constants": [-1, 0, 1, 2],
  "stringConstants": ["", "yes"]
}
```

Both type fields must be present together. Supported signatures are:

| Signature | Candidate families |
| --- | --- |
| `list<A> -> list<B>` | `map` |
| `list<A> -> list<A>` | `map`, `filter` |
| `list<A> -> B` | `fold` |

Here `A` and `B` are `int`, `bool`, or `string`. Nested lists and scalar inputs
are rejected because this prototype does not yet enumerate nested-list or
direct scalar programs. `constants` supplies integer literals;
`stringConstants` supplies string literals. Boolean literals always include
`true` and `false`.

Exit codes: `0` synthesized, `1` refuted or not found within the cost bound, `2` invalid input. The package also exposes the binary as `lambda2-synth` via its `bin` entry.

## What is implemented

| Component | File | Role |
| --- | --- | --- |
| Hypothesis AST | [`src/ast.ts`](src/ast.ts) | Object-language terms plus an explicit typed-hole node for open synthesis hypotheses |
| Cost calculation | [`src/cost.ts`](src/cost.ts) | Computes structural cost; an unresolved hole contributes a lower bound of zero |
| Type checking | [`src/typecheck.ts`](src/typecheck.ts) | Infers object-language types and rejects malformed programs such as `int + list<int>` or `int && bool` |
| Evaluation | [`src/evaluation/`](src/evaluation/) | Evaluates integers, Booleans, strings, lists, closures, `map`, `filter`, and curried left `foldl`; rejects open holes |
| Deduction | [`src/deduction/`](src/deduction/) | One rule per combinator: element examples for `map`, predicate examples for `filter`, init and step examples for `fold`, or refutation |
| Best-first frontier | [`src/frontier.ts`](src/frontier.ts) | Pops lower-cost items first and preserves insertion order for equal costs |
| Enumeration | [`src/enumeration/`](src/enumeration/) | Builds well-typed `int`, `bool`, or `string` expressions over typed variables, in exact-cost buckets |
| Integrated search | [`src/synthesis/`](src/synthesis/) | Races viable skeleton families in one frontier ordered by total program cost |
| CLI | [`src/cli.ts`](src/cli.ts) | JSON in, synthesized program out, with per-family deduction reports |

The type checker and evaluator operate on the modeled object language. Holes, deduction, enumeration, and frontier management are synthesis-level machinery; TypeScript's own type checker does not stand in for either layer.

## Toy object language

```text
type       ::= int | bool | string | list<type> | type -> type
expression ::= integer | true | false | string
             | variable
             | [expression, ...] : list<type>
             | expression (+ | - | * | %) expression
             | expression (< | <= | ==) expression
             | expression (&& | ||) expression
             | expression ++ expression
             | length(expression)
             | !(expression)
             | (variable: type) => expression
             | map(expression, expression)
             | filter(expression, expression)
             | foldl(expression, expression, expression)
hypothesis ::= expression | expression containing ?hole: type
```

`foldl(f, init, xs)` is a left fold whose reducer is curried — in general,
`f: B -> A -> B` is applied as `f(acc)(x)` — so the language needs only unary
lambdas. A closed expression contains no holes and can be evaluated; the
`hole` node represents a typed unknown in an open synthesis hypothesis.

The scalar search enumerates expressions over typed hole variables (`x` for
map and filter bodies; `acc` and `x` for fold reducers), the default integer
vocabulary `{-1, 0, 1, 2}`, the string vocabulary supplied by the request
(defaulting to `""` for string-aware searches), and a default per-hole cost
bound of 4. A finite vocabulary and cost bound keep the search finite; growth
remains exponential, so larger bounds should be chosen deliberately.

## Cost model

| Construct | Cost |
| --- | ---: |
| Variable | 0 |
| Hole lower bound | 0 |
| Integer, Boolean, or string constant | 1 |
| `+`, `-` | 1 plus child costs |
| `*`, `%` | 2 plus child costs |
| `<`, `<=`, `==`, `&&`, `\|\|`, `!` | 1 plus child costs |
| String `++` or `length` | 1 plus child costs |
| List, lambda, `map`, `filter` | 1 plus child costs |
| `foldl` | 1 plus all three child costs |

Completion overheads follow from the skeletons: a `map` or `filter` program costs 3 plus its hole body (skeleton 2, lambda 1); a `fold` program costs 5 plus its reducer body (skeleton 2, two curried lambdas 2, init literal 1). Hence `map (x + 2)` costs 5, `filter (0 < x)` costs 5, and `foldl (acc + x) init 0` costs 6.

## Deduction rules

Deduction rules live in [`src/deduction/`](src/deduction/), one file per combinator. Every rule is sound: it only infers examples that any correct completion must satisfy, and it only refutes hypotheses that no completion can satisfy.

### `map` ([`src/deduction/map.ts`](src/deduction/map.ts))

- **Example inference:** `[1, 2] -> [3, 4]` implies `1 -> 3` and `2 -> 4` for the missing element function.
- **Refutation:** `[1, 1] -> [2, 3]` cannot be implemented by deterministic `map f`, because the same input `1` would need two outputs. Different input and output lengths are also impossible because `map` preserves length.

### `filter` ([`src/deduction/filter.ts`](src/deduction/filter.ts))

- **Example inference:** `[1, 2, 3, 4] -> [2, 4]` implies `1 -> false`, `2 -> true`, `3 -> false`, and `4 -> true` for the missing predicate. Because the predicate is a pure function of the element value, it must keep every copy of a value or none, so per-value occurrence counts decide each element exactly.
- **Refutation:** `[1] -> [2]` introduces an element, `[1, 2, 1] -> [1]` keeps only one of two equal elements, `[1, 2] -> [2, 1]` reorders survivors, and keeping `1` in one example while dropping it in another is contradictory. No pure predicate can produce any of these.

### `fold` ([`src/deduction/fold.ts`](src/deduction/fold.ts))

- **Init inference:** `[] -> 0` fixes the initial accumulator to `0`, because a left fold of the empty list returns its init unchanged.
- **Step peeling:** whenever one example's input extends another's by exactly one trailing element — `[1, 2] -> 3` and `[1, 2, 3] -> 6` — the reducer must satisfy the step `f(3, 3) = 6`. A known init acts as a virtual `[] -> init` example, so `[x] -> b` also yields `f(init, x) = b`.
- **Refutation:** identical inputs with different outputs, or two derived steps sending the same `(accumulator, element)` pair to different outputs.

Fold deduction is deliberately weak — intermediate accumulators are unobservable, so examples that are not prefix-related yield no steps at all. The engine then leans on enumeration plus full-example validation, which mirrors why fold-heavy tasks dominate the slow tail of the paper's evaluation. When no `[] -> init` example exists, the init hole is enumerated from the constant vocabulary alongside the reducer body; a deduced init is used directly even when it is absent from that vocabulary.

Each rule deduplicates consistent examples across all top-level examples. The overall search can still fail because no family fits or the cost bound is too low; those are search-space limitations, not failures of these deduction rules.

## Best-first behavior across families

The engine iterates total program cost `T` upward. At each `T`, every viable family whose completion overhead fits contributes all completions whose hole bodies cost exactly `T - overhead`, and the shared frontier then pops candidates in nondecreasing total cost. Each popped candidate is first checked against its family's deduced constraints (cheap scalar or predicate checks); only survivors are evaluated against the original examples. The first program passing both is therefore **minimum-cost across every viable family**, not just within one skeleton.

Ties are broken deterministically and are documented behavior: equal-cost candidates pop in insertion order, which is family order (`map`, `filter`, `fold`) then enumeration order; within each exact-cost bucket, candidates are stably ordered by AST node count, so structurally simpler expressions such as `(0 < x)` precede equal-cost noise such as `(x < (x + x))`. Identity examples demonstrate cross-family minimality: `map (x => x)` at cost 3 wins over `filter (x => true)` at cost 4.

The returned trace aggregates consecutive pops as `(stage, cost, count)`. Tests pin complete default buckets, nondecreasing popped costs, cross-family cost comparisons, frontier persistence, and stable tie handling.

## Evaluation semantics

Evaluation represents primitives with JavaScript safe integers, Booleans, and
strings. A numeric candidate is discarded if an intermediate value leaves the
safe-integer range. `%` follows JavaScript remainder semantics (sign of the
dividend) and raises a discarding error on a zero divisor. `&&` and `||`
evaluate **both** operands—no short-circuiting—so an error anywhere inside a
candidate discards it deterministically rather than depending on operand
order. Equality is available for two operands of the same primitive type;
`<` and `<=` remain integer-only.

## Scope

This is not the full λ² implementation. It supports one list of primitive
values as input, but not nested lists, algebraic data types, trees, pattern
matching, general recursion, `foldr`/`foldt`, nested skeleton composition
(such as `map` inside `map`), or the paper's complete deduction system. The
minimum-cost statement applies to the finite toy grammar, supplied constants,
the type-compatible instances of the three skeleton families, and the chosen
cost bound.

The modules are kept small, pure, and deterministic to make later LemmaScript contracts practical. This directory contains no LemmaScript annotations and makes no formal-verification claim.
