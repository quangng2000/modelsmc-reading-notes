import {
  INT,
  functionOf,
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
