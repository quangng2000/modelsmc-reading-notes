import {
  BOOL,
  INT,
  STRING,
  type Expression,
  type ObjectType,
} from "../ast.js";
import {
  evaluateExpression,
  expectInt,
  type Value,
} from "../evaluation/index.js";
import type {
  EvaluationBinding,
  PrimitiveTypeName,
  TypedEvaluationBinding,
} from "./types.js";

export function evaluateScalar(
  expression: Expression,
  input: number,
): number {
  return expectInt(
    evaluateExpression(expression, [{ name: "x", type: INT, value: input }]),
  );
}

export function evaluateWith(
  expression: Expression,
  bindings: readonly EvaluationBinding[],
): Value {
  return evaluateWithTyped(
    expression,
    bindings.map((binding) =>
      "type" in binding
        ? binding
        : { name: binding.name, type: "int", value: binding.value },
    ),
  );
}

export function evaluateWithTyped(
  expression: Expression,
  bindings: readonly TypedEvaluationBinding[],
): Value {
  return evaluateExpression(
    expression,
    bindings.map(({ name, type, value }) => ({
      name,
      type: typeof type === "string" ? primitiveType(type) : type,
      value,
    })),
  );
}

export function evaluateScalarSafely(
  expression: Expression,
  input: number,
): number | undefined {
  try {
    return evaluateScalar(expression, input);
  } catch (error) {
    if (error instanceof RangeError) {
      return undefined;
    }
    throw error;
  }
}

function primitiveType(type: PrimitiveTypeName): ObjectType {
  switch (type) {
    case "int":
      return INT;
    case "bool":
      return BOOL;
    case "string":
      return STRING;
  }
}
