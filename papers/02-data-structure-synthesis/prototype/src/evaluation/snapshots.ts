import type { Expression, ObjectType } from "../ast.js";
import { isClosure } from "./closures.js";
import type { EvaluationEnvironment, Value } from "./types.js";

export function snapshotEnvironment(
  environment: EvaluationEnvironment,
): EvaluationEnvironment {
  return Object.freeze(
    environment.map((binding) =>
      Object.freeze({
        name: binding.name,
        type: snapshotType(binding.type),
        value: snapshotValue(binding.value),
      }),
    ),
  );
}

export function snapshotValue(value: Value): Value {
  if (
    typeof value === "number" ||
    typeof value === "boolean" ||
    typeof value === "string"
  ) {
    return value;
  }
  if (Array.isArray(value)) {
    return Object.freeze(value.map(snapshotValue));
  }
  if (isClosure(value)) {
    return value;
  }
  throw new TypeError("Environment contains a value not created by the evaluator.");
}

export function snapshotType(type: ObjectType): ObjectType {
  switch (type.kind) {
    case "int":
      return Object.freeze({ kind: "int" });
    case "bool":
      return Object.freeze({ kind: "bool" });
    case "string":
      return Object.freeze({ kind: "string" });
    case "list":
      return Object.freeze({
        kind: "list",
        element: snapshotType(type.element),
      });
    case "function":
      return Object.freeze({
        kind: "function",
        parameter: snapshotType(type.parameter),
        result: snapshotType(type.result),
      });
  }
}

export function snapshotExpression(expression: Expression): Expression {
  switch (expression.kind) {
    case "int":
      return Object.freeze({ kind: "int", value: expression.value });
    case "bool":
      return Object.freeze({ kind: "bool", value: expression.value });
    case "string":
      return Object.freeze({ kind: "string", value: expression.value });
    case "list":
      return Object.freeze({
        kind: "list",
        elementType: snapshotType(expression.elementType),
        elements: Object.freeze(expression.elements.map(snapshotExpression)),
      });
    case "variable":
      return Object.freeze({ kind: "variable", name: expression.name });
    case "binary":
      return Object.freeze({
        kind: "binary",
        operator: expression.operator,
        left: snapshotExpression(expression.left),
        right: snapshotExpression(expression.right),
      });
    case "concat":
      return Object.freeze({
        kind: "concat",
        left: snapshotExpression(expression.left),
        right: snapshotExpression(expression.right),
      });
    case "length":
      return Object.freeze({
        kind: "length",
        operand: snapshotExpression(expression.operand),
      });
    case "comparison":
      return Object.freeze({
        kind: "comparison",
        operator: expression.operator,
        left: snapshotExpression(expression.left),
        right: snapshotExpression(expression.right),
      });
    case "logic":
      return Object.freeze({
        kind: "logic",
        operator: expression.operator,
        left: snapshotExpression(expression.left),
        right: snapshotExpression(expression.right),
      });
    case "not":
      return Object.freeze({
        kind: "not",
        operand: snapshotExpression(expression.operand),
      });
    case "lambda":
      return Object.freeze({
        kind: "lambda",
        parameter: expression.parameter,
        parameterType: snapshotType(expression.parameterType),
        body: snapshotExpression(expression.body),
      });
    case "map":
      return Object.freeze({
        kind: "map",
        mapper: snapshotExpression(expression.mapper),
        list: snapshotExpression(expression.list),
      });
    case "filter":
      return Object.freeze({
        kind: "filter",
        predicate: snapshotExpression(expression.predicate),
        list: snapshotExpression(expression.list),
      });
    case "fold":
      return Object.freeze({
        kind: "fold",
        reducer: snapshotExpression(expression.reducer),
        initial: snapshotExpression(expression.initial),
        list: snapshotExpression(expression.list),
      });
    case "hole":
      return Object.freeze({
        kind: "hole",
        name: expression.name,
        expectedType: snapshotType(expression.expectedType),
      });
  }
}
