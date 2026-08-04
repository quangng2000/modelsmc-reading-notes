import {
  functionOf,
  primitiveValueEquals,
  typeEquals,
  type Expression,
  type PrimitiveValue,
} from "../ast.js";
import { evaluateExpression, type Value } from "../evaluation/index.js";
import { inferType } from "../typecheck.js";
import type { IOExample, SynthesisSignature } from "./types.js";

// Full-program validation is type-directed by the normalized signature.
// Evaluation failures discard a candidate rather than crashing the search.
export function programMatches(
  program: Expression,
  example: IOExample,
  signature: SynthesisSignature,
): boolean {
  if (program.kind !== "lambda") {
    return false;
  }

  const programType = inferType(program);
  if (
    programType === undefined ||
    !typeEquals(
      programType,
      functionOf(signature.inputType, signature.outputType),
    )
  ) {
    return false;
  }

  try {
    const actual = evaluateExpression(program.body, [
      {
        name: program.parameter,
        type: signature.inputType,
        value: example.input,
      },
    ]);

    if (signature.outputType.kind === "list") {
      return (
        !isPrimitiveOutput(example.output) &&
        Array.isArray(actual) &&
        primitiveListsEqual(actual, example.output)
      );
    }

    return (
      isPrimitiveOutput(example.output) &&
      isPrimitiveValue(actual) &&
      primitiveValueEquals(actual, example.output)
    );
  } catch {
    return false;
  }
}

function primitiveListsEqual(
  left: readonly Value[],
  right: readonly PrimitiveValue[],
): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => {
      const expected = right[index];
      return (
        expected !== undefined &&
        isPrimitiveValue(value) &&
        primitiveValueEquals(value, expected)
      );
    })
  );
}

function isPrimitiveValue(value: Value): value is PrimitiveValue {
  return (
    typeof value === "number" ||
    typeof value === "boolean" ||
    typeof value === "string"
  );
}

function isPrimitiveOutput(
  output: IOExample["output"],
): output is PrimitiveValue {
  return (
    typeof output === "number" ||
    typeof output === "boolean" ||
    typeof output === "string"
  );
}
