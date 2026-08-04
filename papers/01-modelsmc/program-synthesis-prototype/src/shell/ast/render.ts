import {
  inferType,
  type BoolList,
  type Expr,
  type IntList,
  type Program,
  type RuntimeValue,
  type StaticType,
} from "../../core/language.verify.js";

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | readonly JsonValue[]
  | { readonly [key: string]: JsonValue };

export function renderType(type: StaticType): string {
  if (type === "IntType") return "Int";
  if (type === "BoolType") return "Bool";
  if (type === "IntListType") return "List<Int>";
  return "List<Bool>";
}

function renderIntList(list: IntList): string {
  const items: string[] = [];
  let remaining = list;
  while (remaining.kind === "IntCons") {
    items.push(remaining.head.toString());
    remaining = remaining.tail;
  }
  return `[${items.join(", ")}]`;
}

function renderBoolList(list: BoolList): string {
  const items: string[] = [];
  let remaining = list;
  while (remaining.kind === "BoolCons") {
    items.push(String(remaining.head));
    remaining = remaining.tail;
  }
  return `[${items.join(", ")}]`;
}

export function renderValue(value: RuntimeValue): string {
  if (value.kind === "IntValue") return value.intValue.toString();
  if (value.kind === "BoolValue") return String(value.boolValue);
  if (value.kind === "IntListValue") return renderIntList(value.intListValue);
  return renderBoolList(value.boolListValue);
}

function renderExpression(expr: Expr, inputName: string): string {
  if (expr.kind === "Input") return inputName;
  if (expr.kind === "Item") return "item";
  if (expr.kind === "Accumulator") return "acc";
  if (expr.kind === "IntLiteral") return expr.intValue.toString();
  if (expr.kind === "BoolLiteral") return String(expr.boolValue);
  if (expr.kind === "EmptyIntList") return "([]: List<Int>)";
  if (expr.kind === "EmptyBoolList") return "([]: List<Bool>)";
  if (expr.kind === "PrependInt" || expr.kind === "PrependBool") {
    return `(${renderExpression(expr.head, inputName)} :: ${renderExpression(expr.tail, inputName)})`;
  }
  if (expr.kind === "Not") return `(!${renderExpression(expr.operand, inputName)})`;
  if (expr.kind === "IfThenElse") {
    return `(if ${renderExpression(expr.condition, inputName)} then ${renderExpression(expr.thenExpr, inputName)} else ${renderExpression(expr.elseExpr, inputName)})`;
  }
  const operator = {
    Add: "+",
    Subtract: "-",
    Multiply: "*",
    LessThan: "<",
    EqualInt: "==",
    And: "&&",
  }[expr.kind];
  return `(${renderExpression(expr.left, inputName)} ${operator} ${renderExpression(expr.right, inputName)})`;
}

export function renderExpr(program: Program): string {
  if (program.kind === "ExpressionProgram") return renderExpression(program.body, "x");
  if (program.kind === "MapProgram") {
    return `map (λitem. ${renderExpression(program.mapper, "xs")}) xs`;
  }
  return `foldr (λitem. λacc. ${renderExpression(program.reducer, "xs")}) ${renderExpression(program.initial, "xs")} xs`;
}

function renderItemType(inputType: StaticType): string {
  if (inputType === "IntListType") return "Int";
  if (inputType === "BoolListType") return "Bool";
  return "?";
}

/** Render the single implicit input as an explicit lambda abstraction. */
export function renderProgram(program: Program, inputType: StaticType): string {
  const inputName = inputType === "IntListType" || inputType === "BoolListType" ? "xs" : "x";
  if (program.kind === "MapProgram") {
    const itemType = renderItemType(inputType);
    return `λ${inputName}: ${renderType(inputType)}. map (λitem: ${itemType}. ${renderExpression(program.mapper, inputName)}) ${inputName}`;
  }
  if (program.kind === "FoldRightProgram") {
    const itemType = renderItemType(inputType);
    const inferred = inferType(program, inputType);
    const accumulatorType = inferred.kind === "TypeOk" ? renderType(inferred.inferred) : "?";
    return `λ${inputName}: ${renderType(inputType)}. foldr (λitem: ${itemType}. λacc: ${accumulatorType}. ${renderExpression(program.reducer, inputName)}) ${renderExpression(program.initial, inputName)} ${inputName}`;
  }
  return `λ${inputName}: ${renderType(inputType)}. ${renderExpression(program.body, inputName)}`;
}

function expressionToJsonValue(expr: Expr): JsonValue {
  if (expr.kind === "Input") return { kind: "Input" };
  if (expr.kind === "Item") return { kind: "Item" };
  if (expr.kind === "Accumulator") return { kind: "Accumulator" };
  if (expr.kind === "IntLiteral") {
    return { kind: "IntLiteral", intValue: expr.intValue.toString() };
  }
  if (expr.kind === "BoolLiteral") {
    return { kind: "BoolLiteral", boolValue: expr.boolValue };
  }
  if (expr.kind === "EmptyIntList") return { kind: "EmptyIntList" };
  if (expr.kind === "EmptyBoolList") return { kind: "EmptyBoolList" };
  if (expr.kind === "PrependInt" || expr.kind === "PrependBool") {
    return {
      kind: expr.kind,
      head: expressionToJsonValue(expr.head),
      tail: expressionToJsonValue(expr.tail),
    };
  }
  if (expr.kind === "Not") {
    return { kind: "Not", operand: expressionToJsonValue(expr.operand) };
  }
  if (expr.kind === "IfThenElse") {
    return {
      kind: "IfThenElse",
      condition: expressionToJsonValue(expr.condition),
      thenExpr: expressionToJsonValue(expr.thenExpr),
      elseExpr: expressionToJsonValue(expr.elseExpr),
    };
  }
  return {
    kind: expr.kind,
    left: expressionToJsonValue(expr.left),
    right: expressionToJsonValue(expr.right),
  };
}

export function exprToJsonValue(program: Program): JsonValue {
  if (program.kind === "ExpressionProgram") {
    return { kind: "ExpressionProgram", body: expressionToJsonValue(program.body) };
  }
  if (program.kind === "MapProgram") {
    return { kind: "MapProgram", mapper: expressionToJsonValue(program.mapper) };
  }
  return {
    kind: "FoldRightProgram",
    initial: expressionToJsonValue(program.initial),
    reducer: expressionToJsonValue(program.reducer),
  };
}

export function programToJsonValue(program: Program): JsonValue {
  return exprToJsonValue(program);
}

export function jsonStringify(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) =>
    typeof item === "bigint" ? item.toString() : item,
  );
}
