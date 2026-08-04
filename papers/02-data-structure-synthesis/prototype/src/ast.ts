export type BinaryOperator = "+" | "-" | "*";

export type ObjectType =
  | { readonly kind: "int" }
  | { readonly kind: "list"; readonly element: ObjectType }
  | {
      readonly kind: "function";
      readonly parameter: ObjectType;
      readonly result: ObjectType;
    };

export type Expression =
  | { readonly kind: "int"; readonly value: number }
  | {
      readonly kind: "list";
      readonly elementType: ObjectType;
      readonly elements: readonly Expression[];
    }
  | { readonly kind: "variable"; readonly name: string }
  | {
      readonly kind: "binary";
      readonly operator: BinaryOperator;
      readonly left: Expression;
      readonly right: Expression;
    }
  | {
      readonly kind: "lambda";
      readonly parameter: string;
      readonly parameterType: ObjectType;
      readonly body: Expression;
    }
  | {
      readonly kind: "map";
      readonly mapper: Expression;
      readonly list: Expression;
    }
  | {
      readonly kind: "hole";
      readonly name: string;
      readonly expectedType: ObjectType;
    };

export const INT: ObjectType = Object.freeze({ kind: "int" });

export function listOf(element: ObjectType): ObjectType {
  return Object.freeze({ kind: "list", element });
}

export function functionOf(
  parameter: ObjectType,
  result: ObjectType,
): ObjectType {
  return Object.freeze({ kind: "function", parameter, result });
}

export function typeEquals(left: ObjectType, right: ObjectType): boolean {
  switch (left.kind) {
    case "int":
      return right.kind === "int";
    case "list":
      return (
        right.kind === "list" && typeEquals(left.element, right.element)
      );
    case "function":
      return (
        right.kind === "function" &&
        typeEquals(left.parameter, right.parameter) &&
        typeEquals(left.result, right.result)
      );
  }
}

export function renderType(type: ObjectType): string {
  switch (type.kind) {
    case "int":
      return "int";
    case "list":
      return `list<${renderType(type.element)}>`;
    case "function": {
      const parameter =
        type.parameter.kind === "function"
          ? `(${renderType(type.parameter)})`
          : renderType(type.parameter);
      return `${parameter} -> ${renderType(type.result)}`;
    }
  }
}

export function renderExpression(expression: Expression): string {
  switch (expression.kind) {
    case "int":
      return String(expression.value);
    case "list":
      return `[${expression.elements.map(renderExpression).join(", ")}] : ${renderType(listOf(expression.elementType))}`;
    case "variable":
      return expression.name;
    case "binary":
      return `(${renderExpression(expression.left)} ${expression.operator} ${renderExpression(expression.right)})`;
    case "lambda":
      return `(${expression.parameter}: ${renderType(expression.parameterType)}) => ${renderExpression(expression.body)}`;
    case "map":
      return `map(${renderExpression(expression.mapper)}, ${renderExpression(expression.list)})`;
    case "hole":
      return `?${expression.name}: ${renderType(expression.expectedType)}`;
  }
}
