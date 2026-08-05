import {
  evaluate,
  expressionCost,
  inferType,
  type Program,
  type RuntimeValue,
  type StaticType,
} from "../../core/language.verify.js";
import { CoreInvariantError } from "./errors.js";
import { logExampleLikelihood, type NoiseModel } from "./emission.js";

/**
 * Calibrated scoring: log prior (up to the shared normalizer) and true
 * log likelihood of a candidate program. Proposition 1 (EvaluateProgramSound)
 * is what makes the likelihood total on the type-checker's accepted set:
 * evaluation cannot crash and always yields a value of the expected type.
 */
export interface CalibratedScoreOptions {
  readonly inputType: StaticType;
  readonly outputType: StaticType;
  readonly examples: readonly { readonly input: RuntimeValue; readonly output: RuntimeValue }[];
  readonly beta: number;
  readonly noise: NoiseModel;
}

export interface CalibratedScore {
  readonly cost: number;
  /** -beta * cost; add -logZ (from the prior tables) for the normalized log prior. */
  readonly logPriorUnnormalized: number;
  readonly logLikelihood: number;
  /** logPriorUnnormalized + logLikelihood: the unnormalized log posterior. */
  readonly logPosteriorUnnormalized: number;
}

export function scoreCalibrated(
  program: Program,
  options: CalibratedScoreOptions,
): CalibratedScore | { readonly rejected: string } {
  const inferred = inferType(program, options.inputType);
  if (inferred.kind === "TypeError") return { rejected: "type checker rejected the program" };
  if (inferred.inferred !== options.outputType) {
    return { rejected: `output type mismatch: ${inferred.inferred}` };
  }
  const cost = Number(expressionCost(program));

  let logLikelihood = 0;
  for (const example of options.examples) {
    const evaluated = evaluate(program, example.input);
    if (evaluated.kind === "EvalError") {
      throw new CoreInvariantError(
        "a type-checked program failed to evaluate (contradicts EvaluateProgramSound)",
      );
    }
    logLikelihood += logExampleLikelihood(evaluated.output, example.output, options.noise);
  }
  const logPriorUnnormalized = -options.beta * cost;
  return {
    cost,
    logPriorUnnormalized,
    logLikelihood,
    logPosteriorUnnormalized: logPriorUnnormalized + logLikelihood,
  };
}
