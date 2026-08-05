import type { Expr, Program, StaticType } from "../../core/language.verify.js";
import { listElementType, type ExprScope } from "./count.js";

/**
 * Exhaustive enumeration of the accepted program space up to a cost cap.
 *
 * Mirrors the grammar productions counted in count.ts and yields concrete
 * ASTs; the exact-posterior harness asserts per-cost agreement between this
 * enumeration and the counting DP, and the test suite additionally checks
 * every yielded program against the verified core's inferType.
 */

function* expressionsOf(
  type: StaticType,
  cost: number,
  scope: ExprScope,
  constants: readonly bigint[],
): Generator<Expr> {
  if (cost < 1) return;
  if (cost === 1) {
    if (scope.input === type) yield { kind: "Input" };
    if (scope.item === type) yield { kind: "Item" };
    if (scope.accumulator === type) yield { kind: "Accumulator" };
    if (type === "IntType") {
      for (const value of constants) yield { kind: "IntLiteral", intValue: value };
    } else if (type === "BoolType") {
      yield { kind: "BoolLiteral", boolValue: false };
      yield { kind: "BoolLiteral", boolValue: true };
    } else if (type === "IntListType") {
      yield { kind: "EmptyIntList" };
    } else {
      yield { kind: "EmptyBoolList" };
    }
    return;
  }

  const inner = cost - 1;
  if (type === "BoolType") {
    for (const operand of expressionsOf("BoolType", inner, scope, constants)) {
      yield { kind: "Not", operand };
    }
  }
  const binaryProductions: readonly {
    readonly kinds: readonly ("Add" | "Subtract" | "Multiply" | "LessThan" | "EqualInt" | "And" | "PrependInt" | "PrependBool")[];
    readonly left: StaticType;
    readonly right: StaticType;
    readonly result: StaticType;
  }[] = [
    { kinds: ["Add", "Subtract", "Multiply"], left: "IntType", right: "IntType", result: "IntType" },
    { kinds: ["LessThan", "EqualInt"], left: "IntType", right: "IntType", result: "BoolType" },
    { kinds: ["And"], left: "BoolType", right: "BoolType", result: "BoolType" },
    { kinds: ["PrependInt"], left: "IntType", right: "IntListType", result: "IntListType" },
    { kinds: ["PrependBool"], left: "BoolType", right: "BoolListType", result: "BoolListType" },
  ];
  for (const production of binaryProductions) {
    if (production.result !== type) continue;
    for (let leftCost = 1; leftCost <= inner - 1; leftCost += 1) {
      for (const left of expressionsOf(production.left, leftCost, scope, constants)) {
        for (const right of expressionsOf(production.right, inner - leftCost, scope, constants)) {
          for (const kind of production.kinds) {
            if (kind === "PrependInt" || kind === "PrependBool") {
              yield { kind, head: left, tail: right };
            } else {
              yield { kind, left, right };
            }
          }
        }
      }
    }
  }
  for (let conditionCost = 1; conditionCost <= inner - 2; conditionCost += 1) {
    for (const condition of expressionsOf("BoolType", conditionCost, scope, constants)) {
      for (let thenCost = 1; thenCost <= inner - conditionCost - 1; thenCost += 1) {
        for (const thenExpr of expressionsOf(type, thenCost, scope, constants)) {
          for (const elseExpr of expressionsOf(type, inner - conditionCost - thenCost, scope, constants)) {
            yield { kind: "IfThenElse", condition, thenExpr, elseExpr };
          }
        }
      }
    }
  }
}

export interface EnumeratedProgram {
  readonly program: Program;
  readonly cost: number;
}

export function* enumeratePrograms(
  inputType: StaticType,
  outputType: StaticType,
  costCap: number,
  constants: readonly bigint[],
): Generator<EnumeratedProgram> {
  for (let cost = 1; cost <= costCap; cost += 1) {
    for (const body of expressionsOf(outputType, cost, { input: inputType }, constants)) {
      yield { program: { kind: "ExpressionProgram", body }, cost };
    }
  }

  const elementType = listElementType(inputType);
  const outputElementType = listElementType(outputType);
  if (elementType !== undefined && outputElementType !== undefined) {
    for (let cost = 3; cost <= costCap; cost += 1) {
      for (const mapper of expressionsOf(outputElementType, cost - 2, { item: elementType }, constants)) {
        yield { program: { kind: "MapProgram", mapper }, cost };
      }
    }
  }

  if (elementType !== undefined) {
    for (let cost = 5; cost <= costCap; cost += 1) {
      const budget = cost - 3;
      for (let initialCost = 1; initialCost <= budget - 1; initialCost += 1) {
        for (const initial of expressionsOf(outputType, initialCost, {}, constants)) {
          for (const reducer of expressionsOf(
            outputType,
            budget - initialCost,
            { item: elementType, accumulator: outputType },
            constants,
          )) {
            yield { program: { kind: "FoldRightProgram", initial, reducer }, cost };
          }
        }
      }
    }
  }
}
