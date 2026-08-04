import { primitiveValueEquals, type Expression } from "../ast.js";
import { inferType } from "../typecheck.js";
import { isClosure, registerClosure } from "./closures.js";
import { findBinding } from "./environment.js";
import {
  snapshotEnvironment,
  snapshotExpression,
  snapshotType,
  snapshotValue,
} from "./snapshots.js";
import type {
  ClosureValue,
  EvaluationEnvironment,
  Value,
} from "./types.js";
import {
  expectBool,
  expectInt,
  expectPrimitive,
  expectString,
  valueMatchesType,
} from "./values.js";

export function evaluateWellTyped(
  expression: Expression,
  environment: EvaluationEnvironment,
): Value {
  switch (expression.kind) {
    case "int":
      return expectInt(expression.value);
    case "bool":
      return expression.value;
    case "string":
      return expression.value;
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
        case "%":
          // JS remainder semantics: the result's sign follows the dividend
          // (left operand), e.g. -7 % 3 === -1 and 7 % -3 === 1. A zero
          // divisor raises RangeError, which the search layer treats as a
          // deterministic candidate rejection.
          if (right === 0) {
            throw new RangeError("Modulo by zero.");
          }
          result = left % right;
          break;
      }
      if (!Number.isSafeInteger(result)) {
        throw new RangeError(
          "Arithmetic result is outside JavaScript's safe-integer range.",
        );
      }
      return result;
    }
    case "concat": {
      const left = expectString(
        evaluateWellTyped(expression.left, environment),
      );
      const right = expectString(
        evaluateWellTyped(expression.right, environment),
      );
      return left + right;
    }
    case "length":
      return expectString(
        evaluateWellTyped(expression.operand, environment),
      ).length;
    case "comparison": {
      if (expression.operator === "==") {
        const left = expectPrimitive(
          evaluateWellTyped(expression.left, environment),
        );
        const right = expectPrimitive(
          evaluateWellTyped(expression.right, environment),
        );
        return primitiveValueEquals(left, right);
      }

      const left = expectInt(evaluateWellTyped(expression.left, environment));
      const right = expectInt(evaluateWellTyped(expression.right, environment));
      switch (expression.operator) {
        case "<":
          return left < right;
        case "<=":
          return left <= right;
      }
    }
    case "logic": {
      // STRICT evaluation: BOTH operands are evaluated with no short-circuit,
      // so a RangeError anywhere inside a candidate discards it
      // deterministically regardless of the other operand's value.
      const left = expectBool(evaluateWellTyped(expression.left, environment));
      const right = expectBool(
        evaluateWellTyped(expression.right, environment),
      );
      let result: boolean;
      switch (expression.operator) {
        case "&&":
          result = left && right;
          break;
        case "||":
          result = left || right;
          break;
      }
      return result;
    }
    case "not":
      return !expectBool(evaluateWellTyped(expression.operand, environment));
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
      registerClosure(closure);
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
    case "filter": {
      const predicate = evaluateWellTyped(expression.predicate, environment);
      const list = evaluateWellTyped(expression.list, environment);
      if (!isClosure(predicate)) {
        throw new TypeError("filter expects a function as its first argument.");
      }
      if (!Array.isArray(list)) {
        throw new TypeError("filter expects a list as its second argument.");
      }
      return Object.freeze(
        list.filter((element) => expectBool(applyClosure(predicate, element))),
      );
    }
    case "fold": {
      const reducer = evaluateWellTyped(expression.reducer, environment);
      const initial = evaluateWellTyped(expression.initial, environment);
      const list = evaluateWellTyped(expression.list, environment);
      if (!isClosure(reducer)) {
        throw new TypeError("fold expects a function as its first argument.");
      }
      if (!Array.isArray(list)) {
        throw new TypeError("fold expects a list as its third argument.");
      }
      // LEFT fold with a CURRIED reducer: applying the reducer to the
      // accumulator yields the inner lambda's closure, which is then applied
      // to the element. applyClosure validates both argument and result types
      // at each step.
      let accumulator: Value = initial;
      for (const element of list) {
        const partial = applyClosure(reducer, accumulator);
        if (!isClosure(partial)) {
          throw new TypeError(
            "fold expects its curried reducer to return a function.",
          );
        }
        accumulator = applyClosure(partial, element);
      }
      return accumulator;
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
