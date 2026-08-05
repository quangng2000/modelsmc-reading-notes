import {
  acceptProgram,
  evaluate,
  expressionCost,
  inferType,
  matchesExample,
  valueMatchesType,
  type Program,
} from "../../core/language.verify.js";
import { CoreInvariantError } from "./errors.js";
import { softLoss } from "./loss.js";
import type {
  ExampleEvaluation,
  ProgramScore,
  ScoringOptions,
} from "./types.js";

export { CoreInvariantError } from "./errors.js";
export type {
  ExampleEvaluation,
  ProgramScore,
  RejectedScore,
  ScoringOptions,
  ValidScore,
} from "./types.js";

export function scoreProgram(program: Program, options: ScoringOptions): ProgramScore {
  const inferred = inferType(program, options.inputType);
  if (inferred.kind === "TypeError") {
    return { kind: "Rejected", reason: "the verified type checker rejected the AST" };
  }
  if (inferred.inferred !== options.outputType) {
    return {
      kind: "Rejected",
      reason: `output type mismatch: inferred ${inferred.inferred}, expected ${options.outputType}`,
      inferredType: inferred.inferred,
    };
  }

  const exactCost = expressionCost(program);
  if (exactCost > BigInt(options.maxCost)) {
    return {
      kind: "Rejected",
      reason: `expression cost ${exactCost.toString()} exceeds maximum ${options.maxCost}`,
      inferredType: inferred.inferred,
    };
  }
  // This conversion is exact because maxCost is validated as a safe integer and
  // the comparison above rejects every larger structural cost.
  const cost = Number(exactCost);

  const evaluations: ExampleEvaluation[] = [];
  let totalLoss = 0;
  let exactMatches = 0;
  for (const example of options.examples) {
    if (!valueMatchesType(example.input, options.inputType)) {
      throw new CoreInvariantError("an example input crossed the shell/core boundary with the wrong type");
    }
    const evaluated = evaluate(program, example.input);
    if (evaluated.kind === "EvalError") {
      throw new CoreInvariantError(
        "a program accepted by the verified type checker failed during evaluation",
      );
    }
    if (!valueMatchesType(evaluated.output, options.outputType)) {
      throw new CoreInvariantError(
        "a program accepted by the verified type checker returned the wrong output type",
      );
    }
    const exact = matchesExample(program, example);
    const loss = softLoss(evaluated.output, example.output, options.lossCap);
    totalLoss += loss;
    if (exact) exactMatches += 1;
    evaluations.push({
      input: example.input,
      expected: example.output,
      predicted: evaluated.output,
      exact,
      loss,
    });
  }

  return {
    kind: "Scored",
    inferredType: inferred.inferred,
    evaluations,
    totalLoss,
    exactMatches,
    cost,
    logTarget: -options.lossScale * totalLoss - options.costScale * cost,
    exactProgram: acceptProgram(program, [...options.examples]),
  };
}
