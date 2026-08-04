import {
  INT,
  renderExpression,
  typeEquals,
  type BinaryOperator,
  type Expression,
} from "./ast.js";
import { expressionCost, OPERATOR_COSTS } from "./cost.js";
import { evaluateExpression, expectInt } from "./evaluate.js";
import { inferType, type TypeEnvironment } from "./typecheck.js";

export interface Example {
  readonly input: number;
  readonly output: number;
}

export interface CostBucket {
  readonly cost: number;
  readonly expressions: readonly Expression[];
}

export interface BestFirstCandidate {
  readonly expression: Expression;
  readonly cost: number;
}

export interface SearchOptions {
  readonly maxCost?: number;
  readonly constants?: readonly number[];
}

export interface SynthesisResult {
  readonly expression: Expression;
  readonly cost: number;
  readonly candidatesTested: number;
}

export const VARIABLE: Expression = { kind: "variable", name: "x" };
export const DEFAULT_CONSTANTS: readonly number[] = [-1, 0, 1, 2];

const OPERATORS: readonly BinaryOperator[] = ["+", "-", "*"];
const SCALAR_TYPES: TypeEnvironment = [{ name: "x", type: INT }];
const CONSTANT_COST = 1;
const DEFAULT_MAX_COST = 4;

export function evaluateScalar(
  expression: Expression,
  input: number,
): number {
  return expectInt(
    evaluateExpression(expression, [{ name: "x", type: INT, value: input }]),
  );
}

export function* enumerateExpressionsByCost(
  options: SearchOptions = {},
): Generator<CostBucket> {
  const maxCost = options.maxCost ?? DEFAULT_MAX_COST;
  const constants = options.constants ?? DEFAULT_CONSTANTS;

  validateSearchOptions(options);

  const expressions: Expression[][] = [];

  for (let cost = 0; cost <= maxCost; cost += 1) {
    const bucket: Expression[] = [];
    const seen = new Set<string>();

    const add = (candidate: Expression): void => {
      const candidateType = inferType(candidate, SCALAR_TYPES);
      if (
        candidateType === undefined ||
        !typeEquals(candidateType, INT)
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

    if (cost === 0) {
      add(VARIABLE);
    }

    if (cost === CONSTANT_COST) {
      for (const value of constants) {
        add({ kind: "int", value });
      }
    }

    for (const operator of OPERATORS) {
      const operandBudget = cost - OPERATOR_COSTS[operator];
      if (operandBudget < 0) {
        continue;
      }

      for (let leftCost = 0; leftCost <= operandBudget; leftCost += 1) {
        const rightCost = operandBudget - leftCost;
        const leftExpressions = expressions[leftCost] ?? [];
        const rightExpressions = expressions[rightCost] ?? [];

        for (const left of leftExpressions) {
          for (const right of rightExpressions) {
            add({ kind: "binary", operator, left, right });
          }
        }
      }
    }

    expressions[cost] = bucket;
    yield { cost, expressions: bucket };
  }
}

export function validateSearchOptions(options: SearchOptions = {}): void {
  validateMaxCost(options.maxCost ?? DEFAULT_MAX_COST);
  validateIntegers(options.constants ?? DEFAULT_CONSTANTS, "constants");
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

export function synthesize(
  examples: readonly Example[],
  options: SearchOptions = {},
): SynthesisResult | undefined {
  if (examples.length === 0) {
    throw new Error("At least one input-output example is required.");
  }

  for (const example of examples) {
    validateIntegers([example.input, example.output], "example values");
  }

  let candidatesTested = 0;

  for (const candidate of enumerateBestFirst(options)) {
    candidatesTested += 1;

    const satisfiesEveryExample = examples.every(
      ({ input, output }) =>
        evaluateScalarSafely(candidate.expression, input) === output,
    );

    if (satisfiesEveryExample) {
      return {
        expression: candidate.expression,
        cost: candidate.cost,
        candidatesTested,
      };
    }
  }

  return undefined;
}

function evaluateScalarSafely(
  expression: Expression,
  input: number,
): number | undefined {
  try {
    return evaluateScalar(expression, input);
  } catch (error) {
    if (error instanceof RangeError) {
      return undefined;
    }
    throw error;
  }
}

function validateMaxCost(maxCost: number): void {
  if (!Number.isSafeInteger(maxCost) || maxCost < 0) {
    throw new Error("maxCost must be a nonnegative safe integer.");
  }
}

function validateIntegers(values: readonly number[], label: string): void {
  if (!values.every(Number.isSafeInteger)) {
    throw new Error(`${label} must contain only safe integers.`);
  }
}
