import { deriveSynthesisTrace } from "../deduction/index.js";
import type { ProposalContext } from "../proposal/index.js";
import {
  jsonStringify,
  programToJsonValue,
  renderProgram,
  renderType,
  renderValue,
} from "../ast/render.js";
import { mismatchDiagnostic } from "./diagnostics.js";

function refinementIntent(context: ProposalContext): string {
  if (context.ancestorScore.exactProgram) {
    return "The ancestor is exact. Preserve every output and remove redundant structure to lower its cost; if no smaller exact form is apparent, return the ancestor unchanged.";
  }
  if (context.ancestorFeedback?.startsWith("proposal rejected:")) {
    return `The previous proposal was rejected (${context.ancestorFeedback.slice("proposal rejected: ".length)}). Repair that validation failure first. If it exceeded cost, replace enumerated literal cases with compact relational predicates and shared arithmetic.`;
  }
  if (context.ancestorFeedback?.startsWith("proposal failed:")) {
    return `The previous proposal call failed (${context.ancestorFeedback.slice("proposal failed: ".length)}). Return a complete, bounded JSON AST and keep the semantic repair small.`;
  }
  const intents = [
    "Make the smallest semantic repair aimed at the highest-loss failures while preserving examples that already match.",
    "Use the inferred family and hole examples to repair the mapper, fold initial value, or reducer rather than restarting blindly.",
    "Try a structurally different viable family or operator combination that explains the failures; do not make a cosmetic rewrite.",
    "Focus on boundary conditions, predicates, and arithmetic in the current candidate, preserving its useful subexpressions.",
  ] as const;
  return intents[context.requestIndex % intents.length]!;
}

export function promptFor(context: ProposalContext): string {
  const examples = context.examples.map((example, index) => ({
    index: index + 1,
    input: renderValue(example.input),
    output: renderValue(example.output),
  }));
  const failures = context.ancestorScore.evaluations
    .map((evaluation, index) => ({
      index: index + 1,
      input: renderValue(evaluation.input),
      expected: renderValue(evaluation.expected),
      ancestorPrediction: renderValue(evaluation.predicted),
      loss: evaluation.loss,
      exact: evaluation.exact,
      diagnostic: mismatchDiagnostic(evaluation.predicted, evaluation.expected),
    }))
    .filter((evaluation) => !evaluation.exact)
    .sort((left, right) => right.loss - left.loss || left.index - right.index);
  const deductions = deriveSynthesisTrace(
    context.inputType,
    context.outputType,
    context.examples,
  ).map((event) => event.message);
  const avoidPrograms = (context.avoidPrograms ?? []).slice(-16).map((program) =>
    programToJsonValue(program)
  );
  const location = context.iteration === undefined || context.slot === undefined
    ? "SMC iteration/slot unavailable"
    : `SMC iteration ${context.iteration}, slot ${context.slot}`;
  return [
    `Synthesize a ${renderType(context.inputType)} -> ${renderType(context.outputType)} program.`,
    `Proposal request ${context.requestIndex}; ${location}.`,
    "Return one complete JSON Program AST, never source code.",
    `Allowed integer constants: ${context.integerConstants.map((value) => value.toString()).join(", ")}.`,
    "The root must be ExpressionProgram{body}, MapProgram{mapper}, or FoldRightProgram{initial,reducer}.",
    "Allowed expression nodes: Input, Item, Accumulator, IntLiteral, BoolLiteral, EmptyIntList, EmptyBoolList, PrependInt, PrependBool, Add, Subtract, Multiply, LessThan, EqualInt, Not, And, IfThenElse.",
    "Item is bound only in a MapProgram mapper or FoldRightProgram reducer. Accumulator is bound only in a FoldRightProgram reducer. MapProgram and FoldRightProgram scoped expressions must not reference the outer Input.",
    "MapProgram requires a list input and preserves list length. FoldRightProgram traverses a list from right to left; its reducer must return the same type as its initial value.",
    `Maximum cost: ${context.maxCost}; maximum depth: ${context.maxDepth}; maximum nodes: ${context.maxNodes}.`,
    `Synthesis deductions: ${jsonStringify(deductions)}`,
    `Current ancestor: ${renderProgram(context.ancestor, context.inputType)}`,
    `Current ancestor JSON: ${jsonStringify(programToJsonValue(context.ancestor))}`,
    `Current loss: ${context.ancestorScore.totalLoss}; exact matches: ${context.ancestorScore.exactMatches}/${context.examples.length}; cost: ${context.ancestorScore.cost}.`,
    `Previous transition feedback: ${context.ancestorFeedback ?? "none (initial proposal)"}`,
    `Specification examples: ${jsonStringify(examples)}`,
    `Failing examples sorted by descending loss: ${jsonStringify(failures)}`,
    avoidPrograms.length === 0
      ? "No sibling proposals need to be avoided yet."
      : `Already proposed sibling ASTs (do not duplicate them): ${jsonStringify(avoidPrograms)}`,
    `Refinement intent: ${refinementIntent(context)}`,
    context.ancestorScore.exactProgram
      ? "An unchanged ancestor is allowed only when you cannot identify a smaller exact equivalent."
      : "Return a substantive revision: do not duplicate the ancestor or any listed sibling AST.",
    "Prefer a small well-typed expression that improves the candidate-specific feedback.",
  ].join("\n");
}
