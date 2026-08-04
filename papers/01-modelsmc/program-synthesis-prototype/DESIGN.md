# Verified SMC Program Synthesis — Design

## 1. Product and boundary

The product is a command-line programming-by-example experiment. A deterministic catalog or frozen local LLM proposes typed program ASTs. An SMC shell scores them against input-output examples, resamples the population, and retains the best program encountered.

The verified core owns all semantic decisions:

- runtime types and recursive list representations;
- scoped type inference;
- total evaluation with explicit failure;
- structural cost;
- tagged equality and exact example matching;
- final exact-program acceptance.

The shell owns JSON parsing, proposal generation, Paper 2-style hypothesis/deduction messages, soft loss, floating-point potentials, ESS, randomness, resampling, logging, and rendering. Those pieces are tested but not proved.

The claim is therefore **a verified language core inside an approximate synthesis experiment**, not an end-to-end verified synthesizer or exact Bayesian posterior sampler.

## 2. Why dedicated combinators instead of full lambda calculus

The first implementation used a first-order `Int`/`Bool` expression body. This milestone adds recursive lists through two dedicated top-level program forms:

```text
MapProgram(mapper)
FoldRightProgram(initial, reducer)
```

The renderer displays lambda-shaped mapper and reducer arguments, but the AST has no general `Lambda`, `Apply`, closures, arrow types, substitution, or `Fix`.

This defunctionalized design is intentional:

- `map` and `foldr` recurse structurally on finite lists and need no fuel;
- scope is limited to explicit `Item` and `Accumulator` variables;
- the proof decomposes into expression recursion and list recursion;
- search avoids alpha/beta-equivalent programs;
- first-order examples never need to compare function values.

A full simply typed lambda calculus is a separate milestone. An ordinary closure evaluator cannot prove termination merely by structural recursion at application: the closure body is not syntactically a child of the current `Apply` node. Fuel would weaken the current progress theorem, while a fuel-free evaluator would require a normalization proof.

## 3. Verified data model

The exact source is [`src/core/language.verify.ts`](src/core/language.verify.ts). Its central shapes are:

```typescript
type StaticType =
  | "IntType"
  | "BoolType"
  | "IntListType"
  | "BoolListType";

type IntList =
  | { kind: "IntNil" }
  | { kind: "IntCons"; head: bigint; tail: IntList };

type BoolList =
  | { kind: "BoolNil" }
  | { kind: "BoolCons"; head: boolean; tail: BoolList };

type RuntimeValue =
  | { kind: "IntValue"; intValue: bigint }
  | { kind: "BoolValue"; boolValue: boolean }
  | { kind: "IntListValue"; intListValue: IntList }
  | { kind: "BoolListValue"; boolListValue: BoolList };
```

Specialized recursive list datatypes make empty-list element types explicit and make heterogeneous runtime lists unrepresentable.

Expression bodies include:

```text
Input | Item | Accumulator
IntLiteral | BoolLiteral
EmptyIntList | EmptyBoolList
PrependInt(head, tail) | PrependBool(head, tail)
Add | Subtract | Multiply | LessThan | EqualInt | Not | And
IfThenElse
```

Complete programs are:

```typescript
type Program =
  | { kind: "ExpressionProgram"; body: Expr }
  | { kind: "MapProgram"; mapper: Expr }
  | { kind: "FoldRightProgram"; initial: Expr; reducer: Expr };
```

`ExpressionProgram` has the outer `Input` binding. A mapper binds only `Item`. A fold reducer binds `Item` and `Accumulator`; its initial expression is closed. The verified `expressionUsesInput` guard rejects an outer `Input` anywhere in map/fold scoped expressions.

## 4. Static semantics

Let `A` be the input-list element type and `B` the accumulator or output type.

| Form | Premises | Result |
| --- | --- | --- |
| expression body | body type-checks with `Input : inputType` and scoped variables unbound | body type |
| `map mapper` | input is `List<A>`; mapper is closed except `Item : A`; mapper returns scalar `B` | `List<B>` |
| `foldr reducer initial` | input is `List<A>`; closed initial has type `B`; reducer with `Item : A, Accumulator : B` returns `B` | `B` |
| integer prepend | head is `Int`; tail is `List<Int>` | `List<Int>` |
| Boolean prepend | head is `Bool`; tail is `List<Bool>` | `List<Bool>` |

`map` deliberately returns only a flat scalar list. `foldr` may return a scalar or flat list, which permits sum, predicates, filters, duplication, and other constructor-based transformations.

The closure restriction is also what makes the shell's local deductions valid. If a mapper or reducer could inspect the entire outer list, the same item or suffix could behave differently in different top-level examples.

## 5. Dynamic semantics and termination

Expression evaluation is structural on `Expr`. Map and fold helpers recurse on `IntList` or `BoolList`:

```text
map f []       = []
map f (x::xs)  = f[x/Item] :: map f xs

foldr f z []       = z
foldr f z (x::xs)  = f[x/Item, foldr f z xs/Accumulator]
```

The recursive call receives the strict list tail, so Dafny accepts a structural decreases argument. There is no user-defined recursion and no possibility of a cyclic verified list value.

Invalid dynamic combinations return `EvalError`. The soundness proof shows that a successfully inferred program, run on a matching input value, never takes that error path and returns a value of the inferred type.

## 6. Cost model

Expression terminals and variables cost one. Unary, binary, conditional, and prepend nodes cost one plus their contained bodies. Program wrappers add a family cost:

```text
ExpressionProgram(body)       = bodyCost(body)
MapProgram(mapper)             = 2 + bodyCost(mapper)
FoldRightProgram(init, reduce) = 3 + bodyCost(init) + bodyCost(reduce)
```

All costs use `bigint`, matching Dafny's mathematical integers. The shell converts to JavaScript `number` only after checking the configured safe-integer maximum.

The model prefers smaller programs but does not prove global minimality or catalog completeness.

## 7. Architecture

```text
┌──────────────────────────── unverified shell ────────────────────────────┐
│ explicit-signature JSON parser                                           │
│ bounded strict Program decoder                                           │
│ offline catalog or Ollama proposer                                       │
│ family hypotheses and map/fold deduction traces                          │
│ list-aware soft loss, floating potential, ESS, resampling, champion       │
│                                  │ direct imports                         │
└──────────────────────────────────┼────────────────────────────────────────┘
                                   ▼
┌──────────────────────── verified pure core ──────────────────────────────┐
│ recursive IntList / BoolList and tagged RuntimeValue                     │
│ Expr scopes and Program family                                           │
│ inferExpression · inferType · evaluateExpression · evaluate              │
│ expressionBodyCost · expressionCost                                      │
│ sameValue · matchesExample · examplesHaveSignature · acceptProgram       │
│ Dafny progress/preservation, cost, signature, and exactness lemmas        │
└───────────────────────────────────────────────────────────────────────────┘
```

The shell imports the verified functions directly. It does not duplicate the type checker, evaluator, equality relation, structural cost, or exact acceptance decision.

The shell is organized as feature packages with narrow public indexes:

| Package | One reason to change |
| --- | --- |
| `ast/` | change untrusted AST decoding, JSON conversion, or rendering |
| `catalog/` | add or reorder bounded offline candidate families |
| `cli/` | change command arguments or command-line orchestration |
| `config/` | change specification decoding, validation, or overrides |
| `deduction/` | add sound family refutations or inferred subexamples |
| `engine/` | change SMC population lifecycle, propagation, or lineage |
| `ollama/` | change prompt diagnostics, response schema, or HTTP transport |
| `proposal/` | change the common proposal request/result contract |
| `scoring/` | change soft sequence loss or potential evaluation |

ESS normalization, seeded randomness, resampling, and trace emission live inside `engine/` because they serve the SMC lifecycle. `cli.ts` is the only shell-root file and contains only the executable entry-point guard. The verified language remains a single formal unit so its semantic calls are proved together rather than replaced by unproved cross-module assumptions.

## 8. Proof catalog

The additions-only Dafny proof file establishes:

1. `EvaluateExpressionSound`: scoped expression progress and preservation.
2. `EvaluateMapIntSound` and `EvaluateMapBoolSound`: mapper evaluation succeeds and produces the list tag selected by the mapper's scalar type.
3. `EvaluateFoldRightIntSound` and `EvaluateFoldRightBoolSound`: each recursive fold result preserves the accumulator type.
4. `EvaluateProgramSound`: every inferred complete program evaluates successfully to its inferred type.
5. `ExpressionBodyCostExceedsDirectChildren` and `ProgramCostExceedsContainedBodies`.
6. `MatchesExampleExact`, `MatchesAllExamplesExact`, and `AcceptProgramExact`.
7. `ExamplesHaveSignatureExact` and `AcceptedProgramSound`.

Current result:

```text
Dafny program verifier finished with 416 verified, 0 errors
```

No proof uses `assume`, `havoc`, `extern`, `skip`, or `autohavoc`.

## 9. Deduction and search

The unverified deduction module produces guidance rather than executable holes:

- **Map:** equal lengths yield aligned element examples; length mismatches or contradictory duplicate inputs refute the family.
- **Foldr:** an empty input fixes the initial value; an observed tail example yields a reducer subexample. Missing suffixes are reported as partial information.

Only complete `Program` values become particles. There is no `Hole` variant in the executable grammar.

SMC scores every candidate against all original examples. A best-so-far archive prevents a later stochastic iteration from forgetting an exact program, while the live population and resampling equations remain unchanged.

Propagation is explicitly refinement-aware. A proposal receives its ancestor AST and score, its iteration and population slot, diagnostics for failing examples, family deductions, and already proposed siblings. The engine records parent/child loss, exact matches, cost, and ancestry. It reports the first exact iteration/call and reconstructs the winning lineage from an archive of immutable particles.

The hard bounded-square example and deterministic equal-budget test distinguish independent sampling from iterative refinement. Four one-round proposals do not solve the controlled task; two particles refined for two rounds do, using the same four calls. This validates the feedback mechanism, not posterior correctness or universal LLM effectiveness.

## 10. Verification and test workflow

After any verified TypeScript change:

```bash
npx lsc regen src/core/language.verify.ts
npx lsc check
```

For the whole project:

```bash
npm run typecheck
npm test
npm run verify
```

The current deterministic test matrix covers scalar regression, empty lists, type-changing map, integer sum, order-sensitive right subtraction, filter construction, invalid scopes and prepends, exact list equality, large `bigint` elements, config/decoder bounds, deduction events, deterministic map/fold SMC discovery, refinement lineage, and equal-budget one-shot versus iterative search.

## 11. Deferred work

- general `Lambda`/`Apply`, arrow types, closures, and beta-normal enumeration;
- nested lists, trees, and nested map/fold pipelines;
- strings and richer primitive libraries;
- verified JSON decoding and rendering round trips;
- verified map/fold deduction rules;
- proof of search completeness or minimum-cost synthesis;
- proposal-density importance corrections for an exact SMC target.
