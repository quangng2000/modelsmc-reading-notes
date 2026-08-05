import {
  INT,
  primitiveValueEquals,
  renderPrimitiveValue,
  type PrimitiveType,
  type PrimitiveValue,
} from "../ast.js";
import {
  validatePrimitiveList,
  type DeductionResult,
  type ScalarExample,
} from "./types.js";

export interface MapExample<
  Input extends PrimitiveValue = number,
  Output extends PrimitiveValue = number,
> {
  readonly input: readonly Input[];
  readonly output: readonly Output[];
}

export type MapDeductionResult<
  Input extends PrimitiveValue = number,
  Output extends PrimitiveValue = number,
> = DeductionResult<ScalarExample<Input, Output>>;

export function deduceMapExamples<
  Input extends PrimitiveValue = number,
  Output extends PrimitiveValue = number,
>(
  examples: readonly MapExample<Input, Output>[],
  inputType: PrimitiveType = INT,
  outputType: PrimitiveType = INT,
): MapDeductionResult<Input, Output> {
  const outputByInput = new Map<Input, Output>();
  const inferred: ScalarExample<Input, Output>[] = [];

  for (const example of examples) {
    validatePrimitiveList(example.input, inputType, "map input");
    validatePrimitiveList(example.output, outputType, "map output");

    if (example.input.length !== example.output.length) {
      return {
        kind: "refuted",
        reason: "map preserves list length",
      };
    }

    for (let index = 0; index < example.input.length; index += 1) {
      const input = example.input[index];
      const output = example.output[index];

      if (input === undefined || output === undefined) {
        throw new Error("A length-checked list element was unexpectedly missing.");
      }

      const hasPriorOutput = outputByInput.has(input);
      const priorOutput = outputByInput.get(input);
      if (
        hasPriorOutput &&
        priorOutput !== undefined &&
        !primitiveValueEquals(priorOutput, output)
      ) {
        return {
          kind: "refuted",
          reason: `map cannot send ${renderPrimitiveValue(input)} to both ${renderPrimitiveValue(priorOutput)} and ${renderPrimitiveValue(output)}`,
        };
      }

      if (!hasPriorOutput) {
        outputByInput.set(input, output);
        inferred.push({ input, output });
      }
    }
  }

  return { kind: "inferred", examples: inferred };
}
