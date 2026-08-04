import type { Example } from "./enumerator.js";

export interface MapExample {
  readonly input: readonly number[];
  readonly output: readonly number[];
}

export type MapDeductionResult =
  | {
      readonly kind: "inferred";
      readonly examples: readonly Example[];
    }
  | {
      readonly kind: "refuted";
      readonly reason: string;
    };

export function deduceMapExamples(
  examples: readonly MapExample[],
): MapDeductionResult {
  const outputByInput = new Map<number, number>();
  const inferred: Example[] = [];

  for (const example of examples) {
    validateList(example.input, "map input");
    validateList(example.output, "map output");

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

      const priorOutput = outputByInput.get(input);
      if (priorOutput !== undefined && priorOutput !== output) {
        return {
          kind: "refuted",
          reason: `map cannot send ${input} to both ${priorOutput} and ${output}`,
        };
      }

      if (priorOutput === undefined) {
        outputByInput.set(input, output);
        inferred.push({ input, output });
      }
    }
  }

  return { kind: "inferred", examples: inferred };
}

function validateList(values: readonly number[], label: string): void {
  if (!values.every(Number.isSafeInteger)) {
    throw new Error(`${label} must contain only safe integers.`);
  }
}
