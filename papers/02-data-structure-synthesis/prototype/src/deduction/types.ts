import {
  primitiveTypeOf,
  type PrimitiveType,
  type PrimitiveValue,
} from "../ast.js";

export type DeductionResult<E> =
  | {
      readonly kind: "inferred";
      readonly examples: readonly E[];
    }
  | {
      readonly kind: "refuted";
      readonly reason: string;
    };

export interface ScalarExample<
  Input extends PrimitiveValue = number,
  Output extends PrimitiveValue = number,
> {
  readonly input: Input;
  readonly output: Output;
}

export function validatePrimitiveList(
  values: readonly PrimitiveValue[],
  expectedType: PrimitiveType,
  label: string,
): void {
  for (const value of values) {
    validatePrimitiveValue(value, expectedType, label);
  }
}

export function validatePrimitiveValue(
  value: PrimitiveValue,
  expectedType: PrimitiveType,
  label: string,
): void {
  if (
    primitiveTypeOf(value).kind !== expectedType.kind ||
    (expectedType.kind === "int" && !Number.isSafeInteger(value))
  ) {
    throw new Error(`${label} contains a value of the wrong type.`);
  }
}
