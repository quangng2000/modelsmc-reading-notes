# Minimal λ² TypeScript slice

This directory turns one small vertical slice of the λ² paper into executable TypeScript. It includes every stage needed to explain one synthesis result:

```text
list examples
  -> typed map skeleton
  -> deduction for the function hole
  -> cost-ordered scalar enumeration
  -> fill the hole
  -> type-check and evaluate the completed program
```

For example, these top-level examples

```text
[1, 2] -> [3, 4]
[3]    -> [5]
```

produce the open hypothesis

```text
(xs: list<int>) => map(?f: int -> int, xs)
```

Deduction turns the list examples into `1 -> 3`, `2 -> 4`, and `3 -> 5` for `f`. The scalar enumerator then finds `x + 2`, giving

```text
(xs: list<int>) => map((x: int) => x + 2, xs)
```

## Run it

```bash
npm ci
npm test
npm run demo
```

Run these commands from this directory.

## What is implemented

| Component | File | Role |
| --- | --- | --- |
| Hypothesis AST | [`src/ast.ts`](src/ast.ts) | Object-language terms plus an explicit typed-hole node for open synthesis hypotheses |
| Cost calculation | [`src/cost.ts`](src/cost.ts) | Computes structural cost; an unresolved hole contributes a lower bound of zero |
| Type checking | [`src/typecheck.ts`](src/typecheck.ts) | Infers object-language types and rejects malformed programs such as `int + list<int>` |
| Evaluation | [`src/evaluate.ts`](src/evaluate.ts) | Evaluates safe integers, lists, closures, and `map`; rejects open holes |
| Deduction | [`src/deduction.ts`](src/deduction.ts) | Infers element examples for a `map` hole or refutes an impossible `map` hypothesis |
| Best-first frontier | [`src/frontier.ts`](src/frontier.ts) | Pops lower-cost items first and preserves insertion order for equal costs |
| Enumeration | [`src/enumerator.ts`](src/enumerator.ts) | Builds well-typed unary integer expressions in exact-cost buckets |
| Integrated search | [`src/synthesizer.ts`](src/synthesizer.ts) | Refines the map skeleton, fills its hole, and validates the result against the original examples |

The type checker and evaluator operate on the modeled object language. Holes, deduction, enumeration, and frontier management are synthesis-level machinery; TypeScript's own type checker does not stand in for either layer.

## Toy object language

```text
type       ::= int | list<type> | type -> type
expression ::= integer
             | variable
             | [expression, ...] : list<type>
             | expression + expression
             | expression - expression
             | expression * expression
             | (variable: type) => expression
             | map(expression, expression)
hypothesis ::= expression | expression containing ?hole: type
```

A closed object-language expression contains no holes and can be evaluated. The separate `hole` AST node is an implementation representation of a typed unknown in an open synthesis hypothesis.

The scalar search uses one variable `x`, the default finite constant vocabulary `{-1, 0, 1, 2}`, and a default maximum cost of 4. Callers may supply a different finite list or bound. A finite vocabulary and maximum cost are necessary to keep the search finite; growth remains exponential, so larger bounds should be chosen deliberately.

## Cost model

| Construct | Cost |
| --- | ---: |
| Variable | 0 |
| Hole lower bound | 0 |
| Integer constant | 1 |
| Addition or subtraction | 1 plus child costs |
| Multiplication | 2 plus child costs |
| List, lambda, or `map` | 1 plus child costs |

For the scalar expression,

```text
cost(x + 2) = cost(+) + cost(x) + cost(2) = 1 + 0 + 1 = 2.
```

The open map skeleton has cost 2. Filling its hole with `(x: int) => x + 2` produces a completed program of cost 5.

## Deduction rules

This prototype knows one combinator, `map`, and uses two sound rules:

- **Example inference:** `[1, 2] -> [3, 4]` implies `1 -> 3` and `2 -> 4` for the missing element function.
- **Refutation:** `[1, 1] -> [2, 3]` cannot be implemented by deterministic `map f`, because the same input `1` would need two outputs. Different input and output lengths are also impossible because `map` preserves length.

The rule deduplicates consistent element examples across all top-level examples and exactly reduces this fixed `map` hypothesis to its finite element constraints. The overall search can still fail because it does not consider a different hypothesis family or because the scalar cost bound is too low; those are search-space limitations, not failures of this deduction rule.

## Best-first behavior

Unary expressions are generated in buckets satisfying

```text
operator cost + left cost + right cost = total cost.
```

The buckets are visited from cost 0 upward. For each bucket, bounded batches of same-cost completions enter a stable minimum-cost frontier and compete before the search advances to the next cost. Batching does not change best-first semantics because every item in a batch and every later item in that bucket has the same cost. Each popped completion is first checked against the inferred element examples; only survivors are evaluated against the original lists. Thus the first result is minimum-cost within this fixed map hypothesis, scalar grammar, and cost bound.

The returned trace aggregates consecutive pops as `(stage, cost, count)` instead of retaining one object per rejected program. Tests check complete default buckets through cost 2, nondecreasing popped costs, competing equal-cost programs, frontier persistence, and stable tie handling.

## Numeric semantics

Evaluation uses JavaScript `number` values but accepts only safe integers. A candidate is discarded if any intermediate operation leaves the safe-integer range, preventing floating-point rounding from creating a false match.

## Scope

This is not the full λ² implementation. It does not yet include algebraic data types, pattern matching, recursion, polymorphism, `filter`, folds, multiple competing skeleton families, or the paper's complete deduction system. The minimum-cost statement applies only to the finite toy grammar, supplied constants, fixed map hypothesis, and chosen cost bound.

The modules are kept small, pure, and deterministic to make later LemmaScript contracts practical. This PR contains no LemmaScript annotations and makes no formal-verification claim.
