import {
  DEFAULT_TARGET_TYPE,
  DEFAULT_VARIABLES,
} from "./constants.js";
import { enumerateBestFirst } from "./enumerate.js";
import { evaluateScalarSafely } from "./evaluation.js";
import type {
  Example,
  SearchOptions,
  SynthesisResult,
} from "./types.js";
import { validateIntegers } from "./validation.js";

// synthesize is the scalar helper: it matches int outputs over the single
// variable "x", so it stays int-only by design. Bool-typed synthesis goes
// through the family engine instead.
export function synthesize(
  examples: readonly Example[],
  options: SearchOptions = {},
): SynthesisResult | undefined {
  if (examples.length === 0) {
    throw new Error("At least one input-output example is required.");
  }

  if ((options.targetType ?? DEFAULT_TARGET_TYPE) !== "int") {
    throw new Error(
      "synthesize matches int outputs only; it does not support targetType \"bool\".",
    );
  }

  const variables = options.variables ?? DEFAULT_VARIABLES;
  if (variables.length !== 1 || variables[0] !== "x") {
    throw new Error(
      "synthesize evaluates candidates over the single scalar variable \"x\".",
    );
  }

  for (const example of examples) {
    validateIntegers([example.input, example.output], "example values");
  }

  let candidatesTested = 0;

  for (const candidate of enumerateBestFirst(options)) {
    candidatesTested += 1;

    const satisfiesEveryExample = examples.every(
      ({ input, output }) =>
        evaluateScalarSafely(candidate.expression, input) === output,
    );

    if (satisfiesEveryExample) {
      return {
        expression: candidate.expression,
        cost: candidate.cost,
        candidatesTested,
      };
    }
  }

  return undefined;
}
