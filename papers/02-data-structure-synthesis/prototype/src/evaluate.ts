import {
  typeEquals,
  type Expression,
  type ObjectType,
} from "./ast.js";
import { inferType, type TypeBinding } from "./typecheck.js";

const EVALUATOR_CLOSURES = new WeakSet<object>();

interface ClosureValue {
  readonly kind: "closure";
  readonly parameter: string;
  readonly parameterType: ObjectType;
  readonly resultType: ObjectType;
  readonly body: Expression;
  readonly environment: EvaluationEnvironment;
}

export type Value = number | readonly Value[] | ClosureValue;

export interface ValueBinding extends TypeBinding {
  readonly value: Value;
}

export type EvaluationEnvironment = readonly ValueBinding[];

export function evaluateExpression(
  expression: Expression,
  environment: EvaluationEnvironment = [],
): Value {
  if (containsHole(expression)) {
    throw new Error("Cannot evaluate an expression with unresolved holes.");
  }

  const stableEnvironment = snapshotEnvironment(environment);
  validateEnvironment(stableEnvironment);
  const typeEnvironment: readonly TypeBinding[] = stableEnvironment;
  if (inferType(expression, typeEnvironment) === undefined) {
    throw new TypeError("Cannot evaluate an ill-typed expression.");
  }
  return evaluateWellTyped(expression, stableEnvironment);
}

export function expectInt(value: Value): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new TypeError("Expected a safe integer value.");
  }
  return value;
}

export function expectIntList(value: Value): readonly number[] {
  if (!Array.isArray(value)) {
    throw new TypeError("Expected a list of safe integers.");
  }
  return value.map((element) => expectInt(element));
}

function evaluateWellTyped(
  expression: Expression,
  environment: EvaluationEnvironment,
): Value {
  switch (expression.kind) {
    case "int":
      return expectInt(expression.value);
    case "list": {
      const values = expression.elements.map((element) =>
        evaluateWellTyped(element, environment),
      );
      if (
        !values.every((value) => valueMatchesType(value, expression.elementType))
      ) {
        throw new TypeError("A list element does not match its declared type.");
      }
      return Object.freeze(values);
    }
    case "variable": {
      const binding = findBinding(environment, expression.name);
      if (binding === undefined) {
        throw new ReferenceError(`Unbound variable: ${expression.name}`);
      }
      return binding.value;
    }
    case "binary": {
      const left = expectInt(evaluateWellTyped(expression.left, environment));
      const right = expectInt(evaluateWellTyped(expression.right, environment));
      let result: number;
      switch (expression.operator) {
        case "+":
          result = left + right;
          break;
        case "-":
          result = left - right;
          break;
        case "*":
          result = left * right;
          break;
      }
      if (!Number.isSafeInteger(result)) {
        throw new RangeError(
          "Arithmetic result is outside JavaScript's safe-integer range.",
        );
      }
      return result;
    }
    case "lambda": {
      const lambdaType = inferType(expression, environment);
      if (lambdaType === undefined || lambdaType.kind !== "function") {
        throw new TypeError("Cannot evaluate an ill-typed lambda.");
      }
      const closure: ClosureValue = {
        kind: "closure",
        parameter: expression.parameter,
        parameterType: snapshotType(expression.parameterType),
        resultType: snapshotType(lambdaType.result),
        body: snapshotExpression(expression.body),
        environment: snapshotEnvironment(environment),
      };
      EVALUATOR_CLOSURES.add(closure);
      return Object.freeze(closure);
    }
    case "map": {
      const mapper = evaluateWellTyped(expression.mapper, environment);
      const list = evaluateWellTyped(expression.list, environment);
      if (!isClosure(mapper)) {
        throw new TypeError("map expects a function as its first argument.");
      }
      if (!Array.isArray(list)) {
        throw new TypeError("map expects a list as its second argument.");
      }
      return Object.freeze(
        list.map((element) => applyClosure(mapper, element)),
      );
    }
    case "hole":
      throw new Error(`Cannot evaluate unresolved hole ?${expression.name}.`);
  }
}

function applyClosure(closure: ClosureValue, argument: Value): Value {
  if (!valueMatchesType(argument, closure.parameterType)) {
    throw new TypeError("Function argument does not match its parameter type.");
  }
  const result = evaluateWellTyped(closure.body, [
    ...closure.environment,
    Object.freeze({
      name: closure.parameter,
      type: closure.parameterType,
      value: snapshotValue(argument),
    }),
  ]);
  if (!valueMatchesType(result, closure.resultType)) {
    throw new TypeError("Function result does not match its inferred type.");
  }
  return result;
}

function findBinding(
  environment: EvaluationEnvironment,
  name: string,
): ValueBinding | undefined {
  for (let index = environment.length - 1; index >= 0; index -= 1) {
    const binding = environment[index];
    if (binding !== undefined && binding.name === name) {
      return binding;
    }
  }
  return undefined;
}

function isClosure(value: unknown): value is ClosureValue {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }

  const candidate = value as Partial<ClosureValue>;
  return candidate.kind === "closure" && EVALUATOR_CLOSURES.has(value);
}

function valueMatchesType(value: Value, type: ObjectType): boolean {
  switch (type.kind) {
    case "int":
      return typeof value === "number" && Number.isSafeInteger(value);
    case "list":
      return (
        Array.isArray(value) &&
        value.every((element) => valueMatchesType(element, type.element))
      );
    case "function":
      return (
        isClosure(value) &&
        typeEquals(value.parameterType, type.parameter) &&
        typeEquals(value.resultType, type.result)
      );
  }
}

function validateEnvironment(environment: EvaluationEnvironment): void {
  for (const binding of environment) {
    if (!valueMatchesType(binding.value, binding.type)) {
      throw new TypeError(
        `Environment value for ${binding.name} does not match its declared type.`,
      );
    }
  }
}

function snapshotEnvironment(
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

function snapshotValue(value: Value): Value {
  if (typeof value === "number") {
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

function containsHole(expression: Expression): boolean {
  switch (expression.kind) {
    case "int":
    case "variable":
      return false;
    case "hole":
      return true;
    case "list":
      return expression.elements.some(containsHole);
    case "binary":
      return containsHole(expression.left) || containsHole(expression.right);
    case "lambda":
      return containsHole(expression.body);
    case "map":
      return containsHole(expression.mapper) || containsHole(expression.list);
  }
}

function snapshotType(type: ObjectType): ObjectType {
  switch (type.kind) {
    case "int":
      return Object.freeze({ kind: "int" });
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

function snapshotExpression(expression: Expression): Expression {
  switch (expression.kind) {
    case "int":
      return Object.freeze({ kind: "int", value: expression.value });
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
    case "hole":
      return Object.freeze({
        kind: "hole",
        name: expression.name,
        expectedType: snapshotType(expression.expectedType),
      });
  }
}
