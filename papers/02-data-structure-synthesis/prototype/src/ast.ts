export type ArithmeticOperator = "+" | "-" | "*" | "%";
export type ComparisonOperator = "<" | "<=" | "==";
export type LogicOperator = "&&" | "||";

/** @deprecated Use {@link ArithmeticOperator}; kept so older imports compile. */
export type BinaryOperator = ArithmeticOperator;

export type PrimitiveType =
  | { readonly kind: "int" }
  | { readonly kind: "bool" }
  | { readonly kind: "string" };

export type PrimitiveValue = number | boolean | string;

export type ObjectType =
  | PrimitiveType
  | { readonly kind: "list"; readonly element: ObjectType }
  | {
      readonly kind: "function";
      readonly parameter: ObjectType;
      readonly result: ObjectType;
    };

export type PrimitiveLiteral =
  | { readonly kind: "int"; readonly value: number }
  | { readonly kind: "bool"; readonly value: boolean }
  | { readonly kind: "string"; readonly value: string };

export type Expression =
  | PrimitiveLiteral
  | {
      readonly kind: "list";
      readonly elementType: ObjectType;
      readonly elements: readonly Expression[];
    }
  | { readonly kind: "variable"; readonly name: string }
  | {
      readonly kind: "binary";
      readonly operator: ArithmeticOperator;
      readonly left: Expression;
      readonly right: Expression;
    }
  | {
      readonly kind: "concat";
      readonly left: Expression;
      readonly right: Expression;
    }
  | { readonly kind: "length"; readonly operand: Expression }
  | {
      readonly kind: "comparison";
      readonly operator: ComparisonOperator;
      readonly left: Expression;
      readonly right: Expression;
    }
  | {
      readonly kind: "logic";
      readonly operator: LogicOperator;
      readonly left: Expression;
      readonly right: Expression;
    }
  | { readonly kind: "not"; readonly operand: Expression }
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
      readonly kind: "filter";
      readonly predicate: Expression;
      readonly list: Expression;
    }
  | {
      // A LEFT fold whose reducer is CURRIED: generally b -> a -> b,
      // int -> (int -> int) in practice.
      readonly kind: "fold";
      readonly reducer: Expression;
      readonly initial: Expression;
      readonly list: Expression;
    }
  | {
      readonly kind: "hole";
      readonly name: string;
      readonly expectedType: ObjectType;
    };

export const INT: PrimitiveType = Object.freeze({ kind: "int" });
export const BOOL: PrimitiveType = Object.freeze({ kind: "bool" });
export const STRING: PrimitiveType = Object.freeze({ kind: "string" });

export function isPrimitiveType(type: ObjectType): type is PrimitiveType {
  return type.kind === "int" || type.kind === "bool" || type.kind === "string";
}

export function primitiveTypeOf(value: PrimitiveValue): PrimitiveType {
  switch (typeof value) {
    case "number":
      if (!Number.isSafeInteger(value)) {
        throw new TypeError("Integer primitive values must be safe integers.");
      }
      return INT;
    case "boolean":
      return BOOL;
    case "string":
      return STRING;
  }
}

export function primitiveLiteral(value: PrimitiveValue): PrimitiveLiteral {
  primitiveTypeOf(value);
  switch (typeof value) {
    case "number":
      return { kind: "int", value };
    case "boolean":
      return { kind: "bool", value };
    case "string":
      return { kind: "string", value };
  }
}

export function primitiveValueEquals(
  left: PrimitiveValue,
  right: PrimitiveValue,
): boolean {
  primitiveTypeOf(left);
  primitiveTypeOf(right);
  return typeof left === typeof right && left === right;
}

export function renderPrimitiveValue(value: PrimitiveValue): string {
  primitiveTypeOf(value);
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

export function listOf<Element extends ObjectType>(
  element: Element,
): Readonly<{ kind: "list"; element: Element }> {
  return Object.freeze({ kind: "list", element });
}

export function functionOf<
  Parameter extends ObjectType,
  Result extends ObjectType,
>(
  parameter: Parameter,
  result: Result,
): Readonly<{
  kind: "function";
  parameter: Parameter;
  result: Result;
}> {
  return Object.freeze({ kind: "function", parameter, result });
}

export function typeEquals(left: ObjectType, right: ObjectType): boolean {
  switch (left.kind) {
    case "int":
      return right.kind === "int";
    case "bool":
      return right.kind === "bool";
    case "string":
      return right.kind === "string";
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
    case "bool":
      return "bool";
    case "string":
      return "string";
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
    case "bool":
      return expression.value ? "true" : "false";
    case "string":
      return renderPrimitiveValue(expression.value);
    case "list":
      return `[${expression.elements.map(renderExpression).join(", ")}] : ${renderType(listOf(expression.elementType))}`;
    case "variable":
      return expression.name;
    case "binary":
    case "comparison":
    case "logic":
      return `(${renderExpression(expression.left)} ${expression.operator} ${renderExpression(expression.right)})`;
    case "concat":
      return `(${renderExpression(expression.left)} ++ ${renderExpression(expression.right)})`;
    case "length":
      return `length(${renderExpression(expression.operand)})`;
    case "not":
      return `!(${renderExpression(expression.operand)})`;
    case "lambda":
      return `(${expression.parameter}: ${renderType(expression.parameterType)}) => ${renderExpression(expression.body)}`;
    case "map":
      return `map(${renderExpression(expression.mapper)}, ${renderExpression(expression.list)})`;
    case "filter":
      return `filter(${renderExpression(expression.predicate)}, ${renderExpression(expression.list)})`;
    case "fold":
      return `foldl(${renderExpression(expression.reducer)}, ${renderExpression(expression.initial)}, ${renderExpression(expression.list)})`;
    case "hole":
      return `?${expression.name}: ${renderType(expression.expectedType)}`;
  }
}
