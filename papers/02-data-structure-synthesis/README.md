# Synthesizing Data Structure Transformations from Input-Output Examples

[← Back to the two-paper index](../../README.md)

John K. Feser, Swarat Chaudhuri, and Isil Dillig. PLDI 2015. [Publisher page and DOI](https://doi.org/10.1145/2737924.2737977).

> **Core idea:** λ², the “Lambda Learner,” synthesizes typed functional programs over lists, trees, and nested data structures by combining type-aware program skeletons, semantic deduction, and best-first enumerative search.

## Research question

Can a synthesizer recover useful recursive data-structure transformations from only a few input-output examples, without requiring the user to provide a program template or complete logical specification?

The desired result is not merely any program that memorizes the examples. The system searches for the **minimum-cost program** that satisfies them, using a cost model that prefers structurally simple programs.

## Formal synthesis problem

The input is a finite set of examples

$$
E_{\mathrm{in}}=\{a_i\mapsto b_i\}_{i=1}^{n}.
$$

The synthesizer searches for a closed program $e^\star$ such that evaluating it on every example input produces the corresponding output:

$$
(e^\star\,a_i)\Downarrow b_i
\qquad
\text{for every }i.
$$

Among all consistent programs, it returns one with minimum cost:

$$
C(e^\star)=
\min\left\{
C(e)\mid
\text{for every }i,(e\,a_i)\Downarrow b_i
\right\}.
$$

The cost function is part of the specification. For example, assigning a larger cost to pattern matching than to folds biases the synthesizer toward fold-based programs.

## Target language

The synthesized programs live in a typed functional language with:

- Lambda abstraction and application.
- Algebraic data types, records, and variants.
- ML-style pattern matching.
- Recursion.
- Polymorphic lists and trees.
- Higher-order combinators such as `map`, `filter`, `foldl`, `foldr`, and `foldt`.
- User-supplied primitive operators, constants, types, and external predicates.

This explicit language is both a strength and a boundary: λ² can search it systematically, but it cannot synthesize a mechanism that the language and available primitives cannot express.

## Hypotheses and holes

A **hypothesis** is a partial program that may contain free variables called **holes**. For example,

```text
λx. map f* x
```

states that the target probably maps an unknown function `f*` over its input. Completing the hypothesis means finding a concrete program for every hole.

Each synthesis subtask has the form

```text
(current hypothesis, hole to fill, examples for that hole)
```

The examples for an inner hole may be derived automatically from the user's top-level examples.

## The three-part algorithm

### 1. Type-aware inductive generalization

The system infers the required type from the examples and generates only hypotheses compatible with that type. A `map` skeleton, for example, is considered only when the inferred input and output types make such a program possible.

Types prune large parts of the search space before expensive candidate enumeration begins.

### 2. Deduction

The system uses known semantics of combinators in two ways.

**Refutation:** reject a hypothesis that no completion could make consistent. If an example maps `[1, 1]` to `[2, 3]`, a pure `map f` hypothesis is impossible because one function cannot map the same input element `1` to two different outputs.

**Example inference:** derive smaller synthesis problems for holes. From

```text
[1, 2] -> [3, 4]
```

and the hypothesis `λx. map f* x`, the system derives examples `1 -> 3` and `2 -> 4` for `f*`.

Deduction is sound but incomplete. It can safely eliminate some impossible hypotheses and generate useful subexamples, but failure to deduce an answer does not prove that no answer exists.

### 3. Best-first enumerative search

The system keeps synthesis subtasks in a priority queue ordered by program cost. It repeatedly expands the least-cost hypothesis:

1. Infer the hole's type.
2. Generate type-compatible open and closed hypotheses.
3. Use deduction to reject conflicts or infer examples for new holes.
4. Add viable subtasks back to the priority queue.
5. Return when the least-cost closed hypothesis satisfies all original examples.

Because the search expands candidates in cost order, the first returned solution is minimum-cost under the chosen cost model.

## Mental model

```text
input-output examples
        -> infer types
        -> propose program skeletons
        -> reject impossible skeletons
        -> derive examples for holes
        -> enumerate least-cost completions
        -> validate against the original examples
```

The important idea is not enumeration alone. It is the recursive conversion of one synthesis problem into smaller synthesis problems using the semantics of typed combinators.

## Motivating examples

| Program | Transformation | Reported synthesis time |
| --- | --- | ---: |
| `dropmins` | Remove each inner list's minimum value | 114.65 s |
| `selectnodes` | Select tree nodes satisfying an external predicate | 15.97 s |
| `cprod` | Compute the Cartesian product of a list of lists | 83.83 s |

For `cprod`, λ² rediscovered the Barron-Strachey Cartesian-product program, often described as an early functional-programming pearl.

## Evaluation

The implementation is approximately 4,000 lines of OCaml. The evaluation contains 41 synthesis tasks over lists, trees, and nested structures, with a 10-minute and 8 GB resource limit.

- Median synthesis time: **0.43 seconds**.
- **88%** of tasks finish within one minute.
- Median number of expert-written examples: **4**.
- More than 75% of tasks need at most 5 expert examples.
- Without type-aware hypothesis generation, more than **60%** of tasks time out.
- Removing deduction makes synthesis about **6 times slower on average**.
- Naively generated random examples require a median of **8 examples** and **0.93 seconds** for 90% success.

The difficult cases show that the search remains combinatorial: `droplast` takes 316.39 seconds and `tconcat` takes 551.84 seconds in the reported evaluation.

## What the guarantees do and do not mean

The paper proves that a returned program is minimum-cost among programs in the search language that satisfy the supplied examples, under the search and cost assumptions.

That does **not** guarantee that the program matches the user's unstated intention on unseen inputs:

- Finite examples are ambiguous.
- “Simplest” depends on the chosen cost model.
- Weak examples may permit an unintended but cheaper program.
- Deduction rules require hand-encoded semantics for known combinators.
- Unknown external operations often force a fallback to enumeration.

The random-example experiment illustrates this issue. A membership function needs both positive and negative cases, but naive random generation often produces almost only negative examples.

## Relationship to Paper 1

| Dimension | λ² | ModelSMC |
| --- | --- | --- |
| Candidate representation | Typed functional expression | Executable scientific simulator |
| Proposal mechanism | Typed skeletons and enumeration | LLM program revisions |
| Evidence | Exact consistency with examples | Likelihood-based fit to observed data |
| Search control | Cost-ordered priority queue | Particle weights and resampling |
| Structural knowledge | Types, combinators, and deduction rules | Prompt, base simulator, priors, and LLM knowledge |
| Main output | One minimum-cost consistent program | Weighted population of plausible programs |
| Main guarantee | Optimality inside the explicit search language | Idealized particle consistency under strong assumptions |

Both systems perform inference-time program synthesis, but their uncertainty is handled differently. λ² resolves ambiguity using a cost bias and returns one program. ModelSMC represents ambiguity explicitly through a weighted population of models.

## Limitations and open questions

- How sensitive is the synthesized program to the cost model?
- How should users choose examples that distinguish intended behavior from cheaper alternatives?
- Can deduction rules be learned or synthesized rather than implemented manually?
- How does the approach scale from textbook transformations to production programs?
- Could probabilistic or LLM-based proposals accelerate this structured search without losing its guarantees?

## Section map

- **Section 1:** Motivation and contributions.
- **Section 2:** `dropmins`, `selectnodes`, and `cprod` examples.
- **Section 3:** Language, cost model, examples, and hypotheses.
- **Section 4:** Synthesis architecture, hypothesis generation, deduction, and optimality.
- **Section 5:** Evaluation and ablations.
- **Section 6:** Related program-synthesis work.
- **Section 7:** Conclusions and future directions.
