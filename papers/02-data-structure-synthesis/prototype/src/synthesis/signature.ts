import {
  INT,
  isPrimitiveType,
  primitiveTypeOf,
  renderType,
  type ObjectType,
  type PrimitiveType,
  type PrimitiveValue,
} from "../ast.js";
import type {
  IOExample,
  SynthesisOptions,
  SynthesisSignature,
} from "./types.js";

export function resolveSignature(
  examples: readonly IOExample[],
  options: SynthesisOptions,
): SynthesisSignature {
  const declaredInput = options.inputType;
  const declaredOutput = options.outputType;

  if ((declaredInput === undefined) !== (declaredOutput === undefined)) {
    throw new Error("inputType and outputType must be provided together");
  }

  const signature =
    declaredInput === undefined || declaredOutput === undefined
      ? legacySignature(examples)
      : declaredSignature(declaredInput, declaredOutput);

  validateExamples(examples, signature);
  return signature;
}

function legacySignature(examples: readonly IOExample[]): SynthesisSignature {
  const listOutputs = examples.filter((example) =>
    Array.isArray(example.output),
  ).length;
  if (listOutputs !== 0 && listOutputs !== examples.length) {
    throw new Error("examples mix list and scalar outputs");
  }

  const inputType = { kind: "list" as const, element: INT };
  const outputType =
    listOutputs === examples.length
      ? ({ kind: "list" as const, element: INT })
      : INT;
  return { inputType, outputType };
}

function declaredSignature(
  inputType: ObjectType,
  outputType: ObjectType,
): SynthesisSignature {
  if (inputType.kind !== "list" || !isPrimitiveType(inputType.element)) {
    throw new Error(
      `inputType must be list<int>, list<bool>, or list<string>; received ${renderType(inputType)}`,
    );
  }
  const normalizedInput = {
    kind: "list" as const,
    element: inputType.element,
  };
  if (isPrimitiveType(outputType)) {
    return { inputType: normalizedInput, outputType };
  }
  if (outputType.kind === "list" && isPrimitiveType(outputType.element)) {
    return {
      inputType: normalizedInput,
      outputType: { kind: "list", element: outputType.element },
    };
  }
  throw new Error(
    `outputType must be a primitive or list of primitives; received ${renderType(outputType)}`,
  );
}

function validateExamples(
  examples: readonly IOExample[],
  signature: SynthesisSignature,
): void {
  for (let exampleIndex = 0; exampleIndex < examples.length; exampleIndex += 1) {
    const example = examples[exampleIndex];
    if (example === undefined) {
      throw new Error("An example was unexpectedly missing.");
    }

    if (!Array.isArray(example.input)) {
      throw new Error(
        `example ${exampleIndex + 1} input must match ${renderType(signature.inputType)}`,
      );
    }
    validatePrimitiveList(
      example.input,
      signature.inputType.element,
      `example ${exampleIndex + 1} input`,
    );

    if (signature.outputType.kind === "list") {
      if (!Array.isArray(example.output)) {
        throw new Error(
          `example ${exampleIndex + 1} output must match ${renderType(signature.outputType)}`,
        );
      }
      validatePrimitiveList(
        example.output,
        signature.outputType.element,
        `example ${exampleIndex + 1} output`,
      );
      continue;
    }

    if (!isPrimitiveOutput(example.output)) {
      throw new Error(
        `example ${exampleIndex + 1} output must match ${renderType(signature.outputType)}`,
      );
    }
    validatePrimitive(
      example.output,
      signature.outputType,
      `example ${exampleIndex + 1} output`,
    );
  }
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

function validatePrimitiveList(
  values: readonly PrimitiveValue[],
  type: PrimitiveType,
  label: string,
): void {
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === undefined) {
      throw new Error(`${label}[${index}] was unexpectedly missing`);
    }
    validatePrimitive(value, type, `${label}[${index}]`);
  }
}

function validatePrimitive(
  value: PrimitiveValue,
  type: PrimitiveType,
  label: string,
): void {
  if (
    primitiveTypeOf(value).kind !== type.kind ||
    (type.kind === "int" && !Number.isSafeInteger(value))
  ) {
    throw new Error(`${label} must match ${renderType(type)}`);
  }
}
