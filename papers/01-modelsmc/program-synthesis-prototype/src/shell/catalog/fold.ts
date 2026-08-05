import type { Expr, StaticType } from "../../core/language.verify.js";
import {
  addIntegerPredicateCandidates,
  binary,
  boolLiteral,
  intLiteral,
  type ElementType,
} from "./expressions.js";

export function foldInitials(outputType: StaticType, constants: readonly bigint[]): Expr[] {
  if (outputType === "IntType") return constants.map(intLiteral);
  if (outputType === "BoolType") return [boolLiteral(false), boolLiteral(true)];
  if (outputType === "IntListType") return [{ kind: "EmptyIntList" }];
  return [{ kind: "EmptyBoolList" }];
}

function intFoldReducers(inputElementType: ElementType, constants: readonly bigint[]): Expr[] {
  const item: Expr = { kind: "Item" };
  const accumulator: Expr = { kind: "Accumulator" };
  const reducers: Expr[] = [accumulator];
  if (inputElementType === "IntType") {
    reducers.push(
      item,
      binary("Add", item, accumulator),
      binary("Add", accumulator, item),
      binary("Subtract", item, accumulator),
      binary("Subtract", accumulator, item),
      binary("Multiply", item, accumulator),
    );
  }
  for (const constant of constants) reducers.push(intLiteral(constant));
  if (inputElementType === "BoolType") {
    for (const whenTrue of constants) {
      for (const whenFalse of constants) {
        reducers.push({
          kind: "IfThenElse",
          condition: item,
          thenExpr: intLiteral(whenTrue),
          elseExpr: accumulator,
        });
        reducers.push({
          kind: "IfThenElse",
          condition: item,
          thenExpr: intLiteral(whenTrue),
          elseExpr: intLiteral(whenFalse),
        });
      }
    }
  }
  return reducers;
}

function boolFoldReducers(inputElementType: ElementType, constants: readonly bigint[]): Expr[] {
  const item: Expr = { kind: "Item" };
  const accumulator: Expr = { kind: "Accumulator" };
  const reducers: Expr[] = [accumulator, boolLiteral(false), boolLiteral(true)];
  const predicates: Expr[] = [];
  if (inputElementType === "BoolType") {
    predicates.push(item, { kind: "Not", operand: item });
  } else {
    addIntegerPredicateCandidates(predicates, constants, item);
  }
  for (const predicate of predicates) {
    reducers.push(
      predicate,
      binary("And", accumulator, predicate),
      {
        kind: "IfThenElse",
        condition: predicate,
        thenExpr: boolLiteral(true),
        elseExpr: accumulator,
      },
    );
  }
  return reducers;
}

function listFoldReducers(
  inputElementType: ElementType,
  outputType: "IntListType" | "BoolListType",
  constants: readonly bigint[],
): Expr[] {
  const item: Expr = { kind: "Item" };
  const accumulator: Expr = { kind: "Accumulator" };
  const reducers: Expr[] = [accumulator];
  if (inputElementType === "IntType" && outputType === "IntListType") {
    const prepend: Expr = { kind: "PrependInt", head: item, tail: accumulator };
    reducers.push(prepend);
    for (const pivot of constants) {
      const literal = intLiteral(pivot);
      for (const condition of [
        binary("LessThan", literal, item),
        binary("LessThan", item, literal),
        binary("EqualInt", item, literal),
      ] as const) {
        reducers.push({
          kind: "IfThenElse",
          condition,
          thenExpr: prepend,
          elseExpr: accumulator,
        });
      }
    }
  } else if (inputElementType === "BoolType" && outputType === "BoolListType") {
    const prepend: Expr = { kind: "PrependBool", head: item, tail: accumulator };
    reducers.push(
      prepend,
      { kind: "IfThenElse", condition: item, thenExpr: prepend, elseExpr: accumulator },
      {
        kind: "IfThenElse",
        condition: { kind: "Not", operand: item },
        thenExpr: prepend,
        elseExpr: accumulator,
      },
    );
  } else if (inputElementType === "IntType" && outputType === "BoolListType") {
    for (const pivot of constants) {
      const head = binary("LessThan", item, intLiteral(pivot));
      reducers.push({ kind: "PrependBool", head, tail: accumulator });
    }
  } else {
    for (const whenTrue of constants) {
      const head: Expr = {
        kind: "IfThenElse",
        condition: item,
        thenExpr: intLiteral(whenTrue),
        elseExpr: intLiteral(constants[0]!),
      };
      reducers.push({ kind: "PrependInt", head, tail: accumulator });
    }
  }
  return reducers;
}

export function foldReducers(
  inputElementType: ElementType,
  outputType: StaticType,
  constants: readonly bigint[],
): Expr[] {
  if (outputType === "IntType") return intFoldReducers(inputElementType, constants);
  if (outputType === "BoolType") return boolFoldReducers(inputElementType, constants);
  return listFoldReducers(inputElementType, outputType, constants);
}
