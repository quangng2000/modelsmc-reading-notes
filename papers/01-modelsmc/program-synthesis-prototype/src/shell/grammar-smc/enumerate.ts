import {
  expressionCost,
  inferType,
  type Expr,
  type Program,
  type StaticType,
} from "../../core/language.verify.js";
import { jsonStringify } from "../ast/index.js";
import type { ExperimentConfig } from "../config/index.js";

interface Scope {
  readonly allowInput: boolean;
  readonly itemType?: StaticType;
  readonly accumulatorType?: StaticType;
}

interface EnumerationContext {
  readonly config: ExperimentConfig;
  readonly generationLimit: number;
  generated: number;
  readonly cache: Map<string, readonly Expr[]>;
}

function scalarForList(type: StaticType): StaticType | undefined {
  if (type === "IntListType") return "IntType";
  if (type === "BoolListType") return "BoolType";
  return undefined;
}

function cacheKey(type: StaticType, cost: number, scope: Scope): string {
  return [
    type,
    cost,
    scope.allowInput ? "input" : "closed",
    scope.itemType ?? "no-item",
    scope.accumulatorType ?? "no-acc",
  ].join("|");
}

function partitions(total: number, parts: number): number[][] {
  if (parts === 1) return total >= 1 ? [[total]] : [];
  const result: number[][] = [];
  for (let first = 1; first <= total - (parts - 1); first += 1) {
    for (const rest of partitions(total - first, parts - 1)) result.push([first, ...rest]);
  }
  return result;
}

function noteGenerated(context: EnumerationContext, amount: number): void {
  context.generated += amount;
  if (context.generated > context.generationLimit) {
    throw new Error(
      `grammar enumeration exceeded --grammar-limit ${context.generationLimit}; lower --grammar-max-cost or raise the limit`,
    );
  }
}

function unique(expressions: readonly Expr[]): Expr[] {
  const seen = new Set<string>();
  const result: Expr[] = [];
  for (const expression of expressions) {
    const key = jsonStringify(expression);
    if (!seen.has(key)) {
      seen.add(key);
      result.push(expression);
    }
  }
  return result;
}

function enumerateExpressions(
  context: EnumerationContext,
  type: StaticType,
  cost: number,
  scope: Scope,
): readonly Expr[] {
  if (cost < 1) return [];
  const key = cacheKey(type, cost, scope);
  const cached = context.cache.get(key);
  if (cached !== undefined) return cached;

  const expressions: Expr[] = [];
  if (cost === 1) {
    if (scope.allowInput && context.config.inputType === type) expressions.push({ kind: "Input" });
    if (scope.itemType === type) expressions.push({ kind: "Item" });
    if (scope.accumulatorType === type) expressions.push({ kind: "Accumulator" });
    if (type === "IntType") {
      const constants = new Set(context.config.integerConstants.map((value) => value.toString()));
      for (const value of constants) expressions.push({ kind: "IntLiteral", intValue: BigInt(value) });
    } else if (type === "BoolType") {
      expressions.push({ kind: "BoolLiteral", boolValue: false });
      expressions.push({ kind: "BoolLiteral", boolValue: true });
    } else if (type === "IntListType") {
      expressions.push({ kind: "EmptyIntList" });
    } else {
      expressions.push({ kind: "EmptyBoolList" });
    }
  }

  if (cost >= 2 && type === "BoolType") {
    for (const [operandCost] of partitions(cost - 1, 1)) {
      for (const operand of enumerateExpressions(context, "BoolType", operandCost!, scope)) {
        expressions.push({ kind: "Not", operand });
      }
    }
  }

  if (cost >= 3) {
    for (const [leftCost, rightCost] of partitions(cost - 1, 2)) {
      if (type === "IntType") {
        const lefts = enumerateExpressions(context, "IntType", leftCost!, scope);
        const rights = enumerateExpressions(context, "IntType", rightCost!, scope);
        for (const left of lefts) {
          for (const right of rights) {
            expressions.push({ kind: "Add", left, right });
            expressions.push({ kind: "Subtract", left, right });
            expressions.push({ kind: "Multiply", left, right });
          }
        }
      } else if (type === "BoolType") {
        const integersLeft = enumerateExpressions(context, "IntType", leftCost!, scope);
        const integersRight = enumerateExpressions(context, "IntType", rightCost!, scope);
        for (const left of integersLeft) {
          for (const right of integersRight) {
            expressions.push({ kind: "LessThan", left, right });
            expressions.push({ kind: "EqualInt", left, right });
          }
        }
        const booleansLeft = enumerateExpressions(context, "BoolType", leftCost!, scope);
        const booleansRight = enumerateExpressions(context, "BoolType", rightCost!, scope);
        for (const left of booleansLeft) {
          for (const right of booleansRight) expressions.push({ kind: "And", left, right });
        }
      } else {
        const scalar = scalarForList(type)!;
        const heads = enumerateExpressions(context, scalar, leftCost!, scope);
        const tails = enumerateExpressions(context, type, rightCost!, scope);
        for (const head of heads) {
          for (const tail of tails) {
            expressions.push(
              type === "IntListType"
                ? { kind: "PrependInt", head, tail }
                : { kind: "PrependBool", head, tail },
            );
          }
        }
      }
    }
  }

  if (cost >= 4) {
    for (const [conditionCost, thenCost, elseCost] of partitions(cost - 1, 3)) {
      const conditions = enumerateExpressions(context, "BoolType", conditionCost!, scope);
      const thenExpressions = enumerateExpressions(context, type, thenCost!, scope);
      const elseExpressions = enumerateExpressions(context, type, elseCost!, scope);
      for (const condition of conditions) {
        for (const thenExpr of thenExpressions) {
          for (const elseExpr of elseExpressions) {
            expressions.push({ kind: "IfThenElse", condition, thenExpr, elseExpr });
          }
        }
      }
    }
  }

  const result = unique(expressions);
  noteGenerated(context, result.length);
  context.cache.set(key, result);
  return result;
}

function addProgram(
  context: EnumerationContext,
  programs: Map<string, Program>,
  program: Program,
): void {
  const key = jsonStringify(program);
  if (programs.has(key)) return;
  noteGenerated(context, 1);
  programs.set(key, program);
}

export function enumeratePrograms(
  config: ExperimentConfig,
  maxCost: number,
  generationLimit: number,
): Program[] {
  if (!Number.isSafeInteger(maxCost) || maxCost < 1) throw new Error("grammar max cost must be positive");
  if (!Number.isSafeInteger(generationLimit) || generationLimit < 1) {
    throw new Error("grammar generation limit must be positive");
  }
  const context: EnumerationContext = {
    config,
    generationLimit,
    generated: 0,
    cache: new Map(),
  };
  const programs = new Map<string, Program>();

  for (let cost = 1; cost <= maxCost; cost += 1) {
    for (const body of enumerateExpressions(context, config.outputType, cost, { allowInput: true })) {
      addProgram(context, programs, { kind: "ExpressionProgram", body });
    }
  }

  const inputItemType = scalarForList(config.inputType);
  const outputItemType = scalarForList(config.outputType);
  if (inputItemType !== undefined && outputItemType !== undefined) {
    for (let mapperCost = 1; mapperCost <= maxCost - 2; mapperCost += 1) {
      for (const mapper of enumerateExpressions(context, outputItemType, mapperCost, {
        allowInput: false,
        itemType: inputItemType,
      })) {
        addProgram(context, programs, { kind: "MapProgram", mapper });
      }
    }
  }

  if (inputItemType !== undefined) {
    for (let totalCost = 5; totalCost <= maxCost; totalCost += 1) {
      for (const [initialCost, reducerCost] of partitions(totalCost - 3, 2)) {
        const initials = enumerateExpressions(context, config.outputType, initialCost!, { allowInput: false });
        const reducers = enumerateExpressions(context, config.outputType, reducerCost!, {
          allowInput: false,
          itemType: inputItemType,
          accumulatorType: config.outputType,
        });
        for (const initial of initials) {
          for (const reducer of reducers) {
            addProgram(context, programs, { kind: "FoldRightProgram", initial, reducer });
          }
        }
      }
    }
  }

  const result = [...programs.values()].filter((program) => {
    const inferred = inferType(program, config.inputType);
    return (
      inferred.kind === "TypeOk" &&
      inferred.inferred === config.outputType &&
      expressionCost(program) <= BigInt(maxCost)
    );
  });
  result.sort((left, right) => {
    const costDifference = Number(expressionCost(left) - expressionCost(right));
    return costDifference === 0
      ? jsonStringify(left).localeCompare(jsonStringify(right))
      : costDifference;
  });
  if (result.length === 0) throw new Error("the bounded grammar contains no well-typed programs");
  return result;
}
