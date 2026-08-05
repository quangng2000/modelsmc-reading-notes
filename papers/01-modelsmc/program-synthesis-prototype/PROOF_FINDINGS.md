# LemmaScript Proof Findings

Audit date: 2026-08-04

## Executive Summary

The current map/foldr/list milestone passes the required LemmaScript gate. From the project root, npx lsc check regenerated the Dafny model, enforced the additions-only proof boundary, exited successfully, and reported:

> Dafny program verifier finished with 416 verified, 0 errors

The verified model establishes progress and preservation for scoped expressions, type-preserving map and fold-right evaluation, sound evaluation of every successfully inferred complete program, positive structural costs with strict direct-child dominance, homogeneous example signatures, and exact tagged-value matching and acceptance. These guarantees apply to finite values in the Dafny model and to the TypeScript logic translated into that model.

Overall, the design's narrow claim is supportable: this is a verified language core inside an approximate synthesis experiment. It is not an end-to-end verified synthesizer. JSON decoding, proposal generation, deduction, soft loss, floating-point potentials, ESS, randomness, resampling, logging, rendering, and the runtime boundary remain outside the proof. Intent review is also incomplete: no current claimcheck guarantees artifact exists because the attempted claimcheck could not run without an Anthropic API credential.

## What Is Safely Guaranteed

### Scoped expression and complete-program soundness

EvaluateExpressionSound requires:

- the input value to match inputType;
- the Item and Accumulator values to match their binding types; and
- inferExpression to return TypeOk.

It ensures that evaluateExpression returns EvalOk and that the output tag matches the inferred static type. The proof covers every Expr constructor, including arithmetic, Boolean operations, conditionals, integer and Boolean prepend, and the three scoped variable forms.

EvaluateProgramSound lifts that result to ExpressionProgram, MapProgram, and FoldRightProgram. Given a runtime input matching inputType and a successful inferType result, evaluate succeeds and returns a value matching the inferred program type.

The inferType definition enforces the stated scopes: ExpressionProgram binds only Input; MapProgram rejects Input, binds Item, leaves Accumulator unbound, requires a list input, and permits only scalar mapper results; FoldRightProgram rejects Input in both bodies, checks a closed initial expression, binds Item and an Accumulator of the initial type in the reducer, and requires the reducer to return that same type. This matches the prose rules on independent inspection. The theorem proves soundness, not completeness of inference against a separately formalized declarative type system.

### Map preservation

EvaluateMapIntSound and EvaluateMapBoolSound require a correctly tagged list input, a mapper inferred at the corresponding Item type, and a scalar mapper type of IntType or BoolType. They ensure evaluation succeeds for every finite modeled list and produces IntListValue exactly when the mapper type is IntType, or BoolListValue exactly when it is BoolType.

The list evaluator definitions recurse on the strict tail and prepend each mapped head, matching the design's order-preserving map equation. The lemmas establish successful evaluation and output tags; they do not prove facts about synthesis discovery, catalog completeness, or shell deductions.

### Fold-right preservation

EvaluateFoldRightIntSound and EvaluateFoldRightBoolSound require the initial runtime value to match the accumulator type and the reducer to infer that same accumulator type with the appropriate Item binding. They ensure every finite modeled fold succeeds and preserves the accumulator/output type.

The evaluator first folds the tail and then evaluates the reducer with the current head and folded tail, matching the stated foldr equation. The proof covers scalar or flat-list accumulators admitted by the static type model. It does not establish a general lambda calculus, user-defined recursion, or higher-order behavior.

### Structural cost

The source-level contracts on expressionBodyCost and expressionCost each ensure a result of at least one. The generated expressionBodyCost_ensures and expressionCost_ensures lemmas verify those postconditions.

ExpressionBodyCostExceedsDirectChildren proves strict dominance over:

- the operand of Not;
- both children of every binary and prepend form; and
- the condition and both branches of IfThenElse.

ProgramCostExceedsContainedBodies proves that the MapProgram wrapper costs more than its mapper and that the FoldRightProgram wrapper costs more than each contained body. ExpressionProgram deliberately has the same cost as its body. The transparent definitions encode the exact wrapper constants in DESIGN.md. These proofs do not show global minimum cost, search optimality, or catalog completeness.

### Exact equality, signatures, matching, and acceptance

sameValue is tagged equality: scalar tags and contents must match, and integer or Boolean lists are compared recursively with the same element tag and order.

MatchesExampleExact proves that matchesExample is true exactly when evaluation succeeds and sameValue relates the result to the expected output. MatchesAllExamplesExact extends this to every example. ExamplesHaveSignatureExact characterizes examplesHaveSignature exactly as a nonempty sequence with all input tags equal to the first input tag and all output tags equal to the first output tag.

AcceptProgramExact characterizes acceptance exactly as:

- a nonempty homogeneous example sequence;
- successful program inference at the shared input type;
- equality of the inferred and expected output tags; and
- exact matching of every example.

AcceptedProgramSound then ensures successful evaluation and sameValue equality for every accepted example. Exactness is relative to the modeled evaluator, tagged equality, and supplied examples; it does not imply that the examples uniquely determine the intended program or that the stochastic shell will find one.

### Termination in the model

Dafny accepted termination for the recursive expression and list functions. Map and fold helpers carry explicit decreases clauses on the list, and the expression recursions are structural over algebraic datatypes. This establishes totality for finite Dafny values. It does not establish JavaScript stack bounds, resource bounds, or termination on malformed or cyclic objects introduced outside the typed construction boundary.

## Trust Surface Inventory

- **Verified boundary:** LemmaScript-files.txt lists exactly src/core/language.verify.ts, with its corresponding .dfy.gen and .dfy files present and inspected.
- **Verification coverage:** there is no selective //@ verify marker. All 41 of 41 TypeScript function declarations are present as 41 generated Dafny functions.
- **Backend restriction:** the source begins with //@ backend dafny. The audit used the Dafny backend, so the file was included. It would be silently skipped under a different backend; no Lean verification claim is made.
- **//@ assume:** none.
- **//@ havoc or keyed havoc:** none.
- **//@ extern:** none.
- **Cross-file auto-externs:** none. The verified source has no imports or cross-file calls, and neither Dafny artifact contains an axiom declaration.
- **//@ skip:** none.
- **//@ autohavoc:** none.
- **Other abstraction directives:** no safe-slice, declare-type, or unmodeled-expression abstraction is present.
- **Dafny proof bypasses:** no axiom, assume, admit, verify-false attribute, or external declaration appears in the generated or maintained proof file. EvaluateExpressionSound uses timeLimit and isolate_assertions attributes only; these tune verification and do not weaken obligations.
- **Generated/additions boundary:** the maintained .dfy differs from .dfy.gen only by additions: 528 inserted lines and no generated deletion or replacement. One addition attaches the proved positivity postcondition directly to expressionBodyCost for recursive composition; the remaining additions include the proof catalog's 15 handwritten lemmas. Successful lsc check enforced the additions-only gate.
- **Ordinary trusted computing base:** the result relies on LemmaScript 0.5.16's TypeScript-to-Dafny translation, Dafny 4.11.0 and its solver, and correspondence between runtime TypeScript values and the finite modeled datatypes. Unverified boundary code must prevent malformed or cyclic runtime objects if the model-level guarantees are to apply.

The file-level pure, contract, requires, ensures, and decreases annotations do not bypass proof obligations. Preconditions were separately checked for satisfiability in the design-claim review below.

## Intent Coverage

No verified TypeScript function has a formal //@ ensures annotation without an adjacent natural-language //@ contract. The four source-level pairs are:

| Function | Natural-language intent | Formal postcondition |
| --- | --- | --- |
| expressionBodyCost | Structural expression-body cost is always at least one | result >= 1n |
| expressionCost | Complete-program structural cost is positive | result >= 1n |
| examplesHaveSignature | Acceptance by the signature checker implies a nonempty list | result true implies examples.length > 0 |
| acceptProgram | Acceptance implies a nonempty example list | result true implies examples.length > 0 |

That syntactic coverage is narrow: only 4 of 41 TypeScript functions have source-level postconditions and intent text. The substantive progress/preservation, map/fold, child-cost, signature biconditional, and exact-acceptance claims are handwritten Dafny lemmas, not source annotations consumed by claimcheck.

No *.guarantees.json or *.guarantees.md artifact is present. The attempted npx lsc claimcheck run failed solely because no Anthropic API credential was available, and stale artifacts were removed. Consequently there are no current confirmed, disputed, or gap verdicts to report. Absence of an artifact is an open intent-review finding, not a clean claimcheck result. Per the proof-review workflow, this audit did not rerun claimcheck.

## Design-Doc Claims: Status Table

| Design claim | Status | Evidence and qualification |
| --- | --- | --- |
| §1: a verified language core exists inside an approximate, unverified synthesis experiment | **Partially supported** | The core's model and proof obligations verify. The proof does not establish shell integration, runtime validation, or that every production semantic path goes through the core. The design's exclusion of end-to-end verification is accurate. |
| §§3-4: specialized tagged values, recursive homogeneous lists, and the stated Expression/Map/FoldRight scoping and typing rules | **Supported in the Dafny model** | Closed datatypes and the inspected inferExpression, expressionUsesInput, and inferType definitions match the prose rules. Runtime object validity remains a boundary assumption. |
| §5 and §8.1: scoped expression progress and preservation | **Supported** | EvaluateExpressionSound proves EvalOk and output-tag preservation from satisfiable matching and TypeOk preconditions. |
| §5 and §8.2: integer/Boolean map evaluation succeeds and selects the mapper's result-list tag | **Supported** | EvaluateMapIntSound and EvaluateMapBoolSound. |
| §5 and §8.3: integer/Boolean fold-right evaluation succeeds and preserves the accumulator type | **Supported** | EvaluateFoldRightIntSound and EvaluateFoldRightBoolSound. |
| §8.4: every successfully inferred complete program evaluates successfully to its inferred type | **Supported** | EvaluateProgramSound. |
| §6 and §8.5: costs are positive and composite/wrapper costs dominate their direct contained bodies | **Supported** | Source contracts, generated ensures lemmas, ExpressionBodyCostExceedsDirectChildren, and ProgramCostExceedsContainedBodies. No optimality theorem follows. |
| §8.6: exact one-example, all-example, and final acceptance semantics | **Supported** | MatchesExampleExact, MatchesAllExamplesExact, AcceptProgramExact, and the transparent tagged equality definitions. |
| §8.7: homogeneous signatures and accepted-program soundness | **Supported** | ExamplesHaveSignatureExact and AcceptedProgramSound. |
| §8 current result: 416 verified, 0 errors, with none of the listed trust-bypass annotations | **Supported** | Reproduced by npx lsc check; independent trust-surface scan is clean. |
| §9: map/fold deductions and SMC search behavior | **Not yet supported by proof** | The design explicitly assigns these to the unverified shell. Separate formal models would be needed for deduction validity, proposal/search properties, weights, ESS, and resampling. |
| §11: full lambda calculus, nested data/pipelines, verified JSON/rendering, verified deductions, search completeness/minimum cost, and exact SMC target | **Not yet supported** | These are correctly listed as deferred and have no corresponding verified function or theorem. |

The theorem preconditions are not vacuous. Examples include:

- IntLiteral(0) with matching unbound bindings for EvaluateExpressionSound;
- Item as an integer or Boolean mapper with a correspondingly tagged list for both map lemmas;
- IntLiteral(0) as the initial expression and Accumulator as the reducer for fold soundness;
- ExpressionProgram(Input) with a matching input for EvaluateProgramSound; and
- ExpressionProgram(Input) with the singleton example IntValue(0) to IntValue(0) for AcceptedProgramSound.

The postconditions are nontrivial: an always-error evaluator, a map returning the wrong list tag, a fold changing accumulator tags, a constant-one compound cost, or unconditional acceptance would violate at least one theorem. The exactness lemmas are nevertheless close to definitional expansions; their value depends on inferType, evaluate, sameValue, and acceptProgram representing the intended semantics. This audit independently compared those definitions to the product rules, but there is no second formal declarative semantics proving correspondence, so that co-vacuity risk remains in the trusted specification boundary.

## Main Gaps

1. **Intent checking is not yet complete.** Run claimcheck with a configured supported credential against the current annotations and retain fresh guarantees artifacts. To cover the headline claims rather than only four elementary source postconditions, add claimcheck-vettable intent/spec artifacts for the substantive progress, preservation, map/fold, and exact-acceptance theorems.
2. **The formal specification is not yet independent of the implementation functions.** A stronger assurance case would define declarative typing and evaluation relations separately, then prove inferType and evaluate sound and complete with respect to them. That would reduce co-vacuity in the current definition-unfolding exactness results.
3. **The runtime and shell boundary is not yet verified.** End-to-end wording would require a proved or mechanically connected decoder/validator that constructs finite well-formed model values, plus evidence that shell decisions route through the verified type checker, evaluator, equality, cost, and acceptance functions.
4. **Synthesis and probabilistic properties are not yet proved.** Deduction validity, proposal completeness, best-program discovery, minimum cost, proposal-density correction, floating-point normalization, ESS, and resampling correctness need separate specifications and proofs.
5. **Deferred language and export features are not yet proved.** General Lambda/Apply, closures, normalization, nested data/pipelines, strings, and parser/renderer semantic preservation require new model extensions and theorem families.

## Safe External Wording

Safe claim:

> This prototype contains a Dafny-verified pure core for a restricted first-order language with scalar values, homogeneous integer and Boolean lists, expression programs, map, and fold-right. For every finite modeled input, successful program inference implies evaluation succeeds and returns the inferred tagged type. The map and fold-right helpers preserve their stated output or accumulator types. Structural costs are positive and strictly dominate direct children as specified, and final acceptance is exactly characterized over nonempty homogeneous examples using tagged structural equality. The current LemmaScript batch check reports 416 verified obligations and 0 errors. Proposal generation, decoding, deductions, soft scoring, floating-point SMC operations, randomness, resampling, logging, rendering, and runtime boundary validation remain unverified.

Do not claim yet:

- an end-to-end verified synthesizer or an exact Bayesian/SMC implementation;
- that claimcheck currently confirms specification intent;
- verified validity, finiteness, termination, or resource safety for arbitrary JavaScript or JSON objects;
- a verified general lambda calculus, higher-order synthesis, nested-list language, or normalization result;
- verified deduction rules, proposal/search completeness, guaranteed discovery, global minimum cost, or catalog completeness;
- verified parser/renderer round trips or semantic preservation; or
- that 416 is a count of product theorems rather than Dafny verifier obligations.

## Files Changed

- PROOF_FINDINGS.md only.

npx lsc check regenerated/touched language.verify.dfy.gen as part of verification, but no TypeScript source, maintained Dafny proof, design document, or other code content was changed by this audit.
