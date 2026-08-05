import {
  BOOL,
  INT,
  STRING,
  functionOf,
  isPrimitiveType,
  listOf,
  typeEquals,
  type Expression,
  type ObjectType,
} from "./ast.js";

export interface TypeBinding {
  readonly name: string;
  readonly type: ObjectType;
}

export type TypeEnvironment = readonly TypeBinding[];

export function inferType(
  expression: Expression,
  environment: TypeEnvironment = [],
): ObjectType | undefined {
  switch (expression.kind) {
    case "int":
      return Number.isSafeInteger(expression.value) ? INT : undefined;
    case "bool":
      return BOOL;
    case "string":
      return STRING;
    case "list": {
      for (const element of expression.elements) {
        const elementType = inferType(element, environment);
        if (
          elementType === undefined ||
          !typeEquals(elementType, expression.elementType)
        ) {
          return undefined;
        }
      }
      return listOf(expression.elementType);
    }
    case "variable":
      return findBinding(environment, expression.name)?.type;
    case "binary": {
      const leftType = inferType(expression.left, environment);
      const rightType = inferType(expression.right, environment);
      return leftType !== undefined &&
        rightType !== undefined &&
        typeEquals(leftType, INT) &&
        typeEquals(rightType, INT)
        ? INT
        : undefined;
    }
    case "concat": {
      const leftType = inferType(expression.left, environment);
      const rightType = inferType(expression.right, environment);
      return leftType !== undefined &&
        rightType !== undefined &&
        typeEquals(leftType, STRING) &&
        typeEquals(rightType, STRING)
        ? STRING
        : undefined;
    }
    case "length": {
      const operandType = inferType(expression.operand, environment);
      return operandType !== undefined && typeEquals(operandType, STRING)
        ? INT
        : undefined;
    }
    case "comparison": {
      const leftType = inferType(expression.left, environment);
      const rightType = inferType(expression.right, environment);
      if (leftType === undefined || rightType === undefined) {
        return undefined;
      }
      if (expression.operator === "==") {
        return isPrimitiveType(leftType) && typeEquals(leftType, rightType)
          ? BOOL
          : undefined;
      }
      return typeEquals(leftType, INT) && typeEquals(rightType, INT)
        ? BOOL
        : undefined;
    }
    case "logic": {
      const leftType = inferType(expression.left, environment);
      const rightType = inferType(expression.right, environment);
      return leftType !== undefined &&
        rightType !== undefined &&
        typeEquals(leftType, BOOL) &&
        typeEquals(rightType, BOOL)
        ? BOOL
        : undefined;
    }
    case "not": {
      const operandType = inferType(expression.operand, environment);
      return operandType !== undefined && typeEquals(operandType, BOOL)
        ? BOOL
        : undefined;
    }
    case "lambda": {
      const bodyType = inferType(expression.body, [
        ...environment,
        { name: expression.parameter, type: expression.parameterType },
      ]);
      return bodyType === undefined
        ? undefined
        : functionOf(expression.parameterType, bodyType);
    }
    case "map": {
      const mapperType = inferType(expression.mapper, environment);
      const listType = inferType(expression.list, environment);
      if (
        mapperType === undefined ||
        listType === undefined ||
        mapperType.kind !== "function" ||
        listType.kind !== "list" ||
        !typeEquals(mapperType.parameter, listType.element)
      ) {
        return undefined;
      }
      return listOf(mapperType.result);
    }
    case "filter": {
      // predicate: a -> bool, list: list<a>  =>  list<a>
      const predicateType = inferType(expression.predicate, environment);
      const listType = inferType(expression.list, environment);
      if (
        predicateType === undefined ||
        listType === undefined ||
        predicateType.kind !== "function" ||
        listType.kind !== "list" ||
        !typeEquals(predicateType.result, BOOL) ||
        !typeEquals(predicateType.parameter, listType.element)
      ) {
        return undefined;
      }
      return listOf(listType.element);
    }
    case "fold": {
      // reducer: b -> (a -> b) (curried), initial: b, list: list<a>  =>  b
      const reducerType = inferType(expression.reducer, environment);
      const initialType = inferType(expression.initial, environment);
      const listType = inferType(expression.list, environment);
      if (
        reducerType === undefined ||
        initialType === undefined ||
        listType === undefined ||
        reducerType.kind !== "function" ||
        reducerType.result.kind !== "function" ||
        listType.kind !== "list" ||
        !typeEquals(reducerType.parameter, initialType) ||
        !typeEquals(reducerType.result.parameter, listType.element) ||
        !typeEquals(reducerType.result.result, initialType)
      ) {
        return undefined;
      }
      return initialType;
    }
    case "hole":
      return expression.expectedType;
  }
}

function findBinding(
  environment: TypeEnvironment,
  name: string,
): TypeBinding | undefined {
  for (let index = environment.length - 1; index >= 0; index -= 1) {
    const binding = environment[index];
    if (binding !== undefined && binding.name === name) {
      return binding;
    }
  }
  return undefined;
}
