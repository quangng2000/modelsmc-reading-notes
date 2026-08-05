import {
  BOOL,
  INT,
  STRING,
  renderExpression,
  typeEquals,
  type Expression,
  type ObjectType,
  type PrimitiveType,
} from "../ast.js";
import { expressionCost, OPERATOR_COSTS } from "../cost.js";
import { inferType, type TypeEnvironment } from "../typecheck.js";
import {
  ARITHMETIC_OPERATORS,
  BOOL_LITERAL_COST,
  COMPARISON_COST,
  COMPARISON_OPERATORS,
  CONCAT_COST,
  CONSTANT_COST,
  DEFAULT_CONSTANTS,
  DEFAULT_MAX_COST,
  DEFAULT_STRING_CONSTANTS,
  DEFAULT_TARGET_TYPE,
  DEFAULT_VARIABLES,
  LENGTH_COST,
  LOGIC_COST,
  LOGIC_OPERATORS,
  NOT_COST,
} from "./constants.js";
import { sortBucketBySize } from "./ordering.js";
import type {
  BestFirstCandidate,
  CostBucket,
  PrimitiveTypeName,
  SearchOptions,
} from "./types.js";
import {
  normalizeVariables,
  validateSearchOptions,
} from "./validation.js";

export function* enumerateExpressionsByCost(
  options: SearchOptions = {},
): Generator<CostBucket> {
  const maxCost = options.maxCost ?? DEFAULT_MAX_COST;
  const constants = options.constants ?? DEFAULT_CONSTANTS;
  const variableOptions = options.variables ?? DEFAULT_VARIABLES;
  const targetType = options.targetType ?? DEFAULT_TARGET_TYPE;

  validateSearchOptions(options);

  const variables = normalizeVariables(variableOptions);
  const environment: TypeEnvironment = variables.map(({ name, type }) => ({
    name,
    type: primitiveType(type),
  }));
  const hasStringVariable = variables.some(({ type }) => type === "string");
  const stringConstants =
    options.stringConstants ??
    (targetType === "string" || hasStringVariable
      ? DEFAULT_STRING_CONSTANTS
      : []);
  // In this scalar grammar, bool expressions are only useful when bool is the
  // requested result type. They can still be produced without a bool source
  // variable: int comparisons and string equalities both populate the bool
  // table. Enable bool equality for every bool search so those generated
  // expressions can be composed as operands as well.
  const enableBoolEquality = targetType === "bool";
  const enableStringEquality =
    targetType === "string" ||
    hasStringVariable ||
    options.stringConstants !== undefined;

  // Three tables per cost level let productions draw only correctly typed
  // operands. Legacy string-name variables populate the int table exactly as
  // before; typed variables are routed to their declared primitive table.
  const intByCost: Expression[][] = [];
  const boolByCost: Expression[][] = [];
  const stringByCost: Expression[][] = [];

  for (let cost = 0; cost <= maxCost; cost += 1) {
    const intBucket: Expression[] = [];
    const boolBucket: Expression[] = [];
    const stringBucket: Expression[] = [];
    const seenInt = new Set<string>();
    const seenBool = new Set<string>();
    const seenString = new Set<string>();

    const add = (
      candidate: Expression,
      expected: ObjectType,
      bucket: Expression[],
      seen: Set<string>,
    ): void => {
      const candidateType = inferType(candidate, environment);
      if (
        candidateType === undefined ||
        !typeEquals(candidateType, expected)
      ) {
        return;
      }

      if (expressionCost(candidate) !== cost) {
        throw new Error("A candidate was generated in the wrong cost bucket.");
      }

      const key = renderExpression(candidate);
      if (!seen.has(key)) {
        seen.add(key);
        bucket.push(candidate);
      }
    };

    const addInt = (candidate: Expression): void =>
      add(candidate, INT, intBucket, seenInt);
    const addBool = (candidate: Expression): void =>
      add(candidate, BOOL, boolBucket, seenBool);
    const addString = (candidate: Expression): void =>
      add(candidate, STRING, stringBucket, seenString);

    if (cost === 0) {
      for (const { name, type } of variables) {
        const variable: Expression = { kind: "variable", name };
        switch (type) {
          case "int":
            addInt(variable);
            break;
          case "bool":
            addBool(variable);
            break;
          case "string":
            addString(variable);
            break;
        }
      }
    }

    if (cost === CONSTANT_COST) {
      for (const value of constants) {
        addInt({ kind: "int", value });
      }
      for (const value of stringConstants) {
        addString({ kind: "string", value });
      }
    }

    for (const operator of ARITHMETIC_OPERATORS) {
      const operandBudget = cost - OPERATOR_COSTS[operator];
      if (operandBudget < 0) {
        continue;
      }

      for (let leftCost = 0; leftCost <= operandBudget; leftCost += 1) {
        const rightCost = operandBudget - leftCost;
        const leftExpressions = intByCost[leftCost] ?? [];
        const rightExpressions = intByCost[rightCost] ?? [];

        for (const left of leftExpressions) {
          for (const right of rightExpressions) {
            addInt({ kind: "binary", operator, left, right });
          }
        }
      }
    }

    const lengthOperandBudget = cost - LENGTH_COST;
    if (lengthOperandBudget >= 0) {
      for (const operand of stringByCost[lengthOperandBudget] ?? []) {
        addInt({ kind: "length", operand });
      }
    }

    const concatOperandBudget = cost - CONCAT_COST;
    if (concatOperandBudget >= 0) {
      for (
        let leftCost = 0;
        leftCost <= concatOperandBudget;
        leftCost += 1
      ) {
        const rightCost = concatOperandBudget - leftCost;
        const leftExpressions = stringByCost[leftCost] ?? [];
        const rightExpressions = stringByCost[rightCost] ?? [];

        for (const left of leftExpressions) {
          for (const right of rightExpressions) {
            addString({ kind: "concat", left, right });
          }
        }
      }
    }

    if (cost === BOOL_LITERAL_COST) {
      addBool({ kind: "bool", value: true });
      addBool({ kind: "bool", value: false });
    }

    for (const operator of COMPARISON_OPERATORS) {
      const operandBudget = cost - COMPARISON_COST;
      if (operandBudget < 0) {
        continue;
      }

      for (let leftCost = 0; leftCost <= operandBudget; leftCost += 1) {
        const rightCost = operandBudget - leftCost;
        const leftExpressions = intByCost[leftCost] ?? [];
        const rightExpressions = intByCost[rightCost] ?? [];

        for (const left of leftExpressions) {
          for (const right of rightExpressions) {
            addBool({ kind: "comparison", operator, left, right });
          }
        }
      }
    }

    const addEqualities = (byCost: readonly Expression[][]): void => {
      const operandBudget = cost - COMPARISON_COST;
      if (operandBudget < 0) {
        return;
      }

      for (let leftCost = 0; leftCost <= operandBudget; leftCost += 1) {
        const rightCost = operandBudget - leftCost;
        const leftExpressions = byCost[leftCost] ?? [];
        const rightExpressions = byCost[rightCost] ?? [];

        for (const left of leftExpressions) {
          for (const right of rightExpressions) {
            addBool({ kind: "comparison", operator: "==", left, right });
          }
        }
      }
    };

    // The old grammar already includes int equality. Boolean equality is
    // relevant only for a bool target, while string equality is enabled when
    // strings participate in the search. Keeping those gates avoids changing
    // legacy int-target buckets and candidate counts.
    if (enableBoolEquality) {
      addEqualities(boolByCost);
    }
    if (enableStringEquality) {
      addEqualities(stringByCost);
    }

    for (const operator of LOGIC_OPERATORS) {
      const operandBudget = cost - LOGIC_COST;
      if (operandBudget < 0) {
        continue;
      }

      for (let leftCost = 0; leftCost <= operandBudget; leftCost += 1) {
        const rightCost = operandBudget - leftCost;
        const leftExpressions = boolByCost[leftCost] ?? [];
        const rightExpressions = boolByCost[rightCost] ?? [];

        for (const left of leftExpressions) {
          for (const right of rightExpressions) {
            addBool({ kind: "logic", operator, left, right });
          }
        }
      }
    }

    const operandBudget = cost - NOT_COST;
    if (operandBudget >= 0) {
      for (const operand of boolByCost[operandBudget] ?? []) {
        addBool({ kind: "not", operand });
      }
    }

    // Ties inside a cost bucket resolve toward structurally smaller
    // candidates: each bucket is stable-sorted by AST node count, so
    // equal-cost programs like (0 < x) and (x < (x + x)) surface the simpler
    // form first. Search layers that test candidates in insertion order
    // inherit this preference.
    sortBucketBySize(intBucket);
    sortBucketBySize(boolBucket);
    sortBucketBySize(stringBucket);

    intByCost[cost] = intBucket;
    boolByCost[cost] = boolBucket;
    stringByCost[cost] = stringBucket;
    yield {
      cost,
      expressions:
        targetType === "int"
          ? intBucket
          : targetType === "bool"
            ? boolBucket
            : stringBucket,
    };
  }
}

export function* enumerateBestFirst(
  options: SearchOptions = {},
): Generator<BestFirstCandidate> {
  for (const bucket of enumerateExpressionsByCost(options)) {
    for (const expression of bucket.expressions) {
      yield { expression, cost: bucket.cost };
    }
  }
}

function primitiveType(type: PrimitiveTypeName): PrimitiveType {
  switch (type) {
    case "int":
      return INT;
    case "bool":
      return BOOL;
    case "string":
      return STRING;
  }
}
