import {
  INT,
  functionOf,
  listOf,
  typeEquals,
  type Expression,
} from "./ast.js";
import { expressionCost } from "./cost.js";
import {
  deduceMapExamples,
  type MapExample,
} from "./deduction.js";
import { evaluateExpression, expectIntList } from "./evaluate.js";
import {
  enumerateExpressionsByCost,
  evaluateScalar,
  type Example,
  type SearchOptions,
  validateSearchOptions,
} from "./enumerator.js";
import {
  emptyFrontier,
  popMinFrontier,
  pushFrontier,
  type Frontier,
} from "./frontier.js";
import { inferType, type TypeEnvironment } from "./typecheck.js";

export interface SearchTraceEntry {
  readonly stage: "skeleton" | "completed-program";
  readonly cost: number;
  readonly count: number;
}

export type MapSynthesisResult =
  | {
      readonly kind: "synthesized";
      readonly program: Expression;
      readonly cost: number;
      readonly candidatesTested: number;
      readonly inferredExamples: readonly Example[];
      readonly trace: readonly SearchTraceEntry[];
    }
  | {
      readonly kind: "refuted" | "underconstrained" | "not-found";
      readonly reason: string;
      readonly trace: readonly SearchTraceEntry[];
    };

type FrontierItem =
  | {
      readonly stage: "skeleton";
      readonly expression: Expression;
    }
  | {
      readonly stage: "completed-program";
      readonly expression: Expression;
      readonly scalarBody: Expression;
    };

const INT_TO_INT = functionOf(INT, INT);
const INT_LIST = listOf(INT);
const INT_LIST_TO_INT_LIST = functionOf(INT_LIST, INT_LIST);
const FRONTIER_BATCH_SIZE = 64;

export const MAP_SKELETON: Expression = {
  kind: "lambda",
  parameter: "xs",
  parameterType: INT_LIST,
  body: {
    kind: "map",
    mapper: { kind: "hole", name: "f", expectedType: INT_TO_INT },
    list: { kind: "variable", name: "xs" },
  },
};

export function synthesizeMap(
  examples: readonly MapExample[],
  options: SearchOptions = {},
): MapSynthesisResult {
  validateSearchOptions(options);

  let frontier: Frontier<FrontierItem> = pushFrontier(
    emptyFrontier<FrontierItem>(),
    { stage: "skeleton", expression: MAP_SKELETON },
    expressionCost(MAP_SKELETON),
  );
  const trace: SearchTraceEntry[] = [];

  const skeletonEntry = popMinFrontier(frontier);
  if (skeletonEntry === undefined) {
    throw new Error("The initial synthesis frontier was unexpectedly empty.");
  }
  frontier = skeletonEntry.frontier;
  recordTrace(trace, skeletonEntry.item.stage, skeletonEntry.cost);

  const skeletonType = inferType(skeletonEntry.item.expression, []);
  if (
    skeletonType === undefined ||
    !typeEquals(skeletonType, INT_LIST_TO_INT_LIST)
  ) {
    throw new TypeError("The built-in map skeleton is ill-typed.");
  }

  const deduction = deduceMapExamples(examples);
  if (deduction.kind === "refuted") {
    return { kind: "refuted", reason: deduction.reason, trace };
  }

  if (deduction.examples.length === 0) {
    return {
      kind: "underconstrained",
      reason: "empty lists provide no examples for the map function hole",
      trace,
    };
  }

  let candidatesTested = 0;

  for (const bucket of enumerateExpressionsByCost(options)) {
    for (
      let start = 0;
      start < bucket.expressions.length;
      start += FRONTIER_BATCH_SIZE
    ) {
      const batch = bucket.expressions.slice(
        start,
        start + FRONTIER_BATCH_SIZE,
      );

      for (const scalarBody of batch) {
        const mapper: Expression = {
          kind: "lambda",
          parameter: "x",
          parameterType: INT,
          body: scalarBody,
        };
        const program = substituteHole(MAP_SKELETON, "f", mapper);
        const programType = inferType(program, []);
        if (
          programType === undefined ||
          !typeEquals(programType, INT_LIST_TO_INT_LIST)
        ) {
          throw new TypeError("Filling the map hole produced an ill-typed program.");
        }

        frontier = pushFrontier(
          frontier,
          { stage: "completed-program", expression: program, scalarBody },
          expressionCost(program),
        );
      }

      while (frontier.size > 0) {
        const completedEntry = popMinFrontier(frontier);
        if (completedEntry === undefined) {
          throw new Error("A nonempty frontier could not produce an item.");
        }
        frontier = completedEntry.frontier;
        recordTrace(
          trace,
          completedEntry.item.stage,
          completedEntry.cost,
        );

        if (completedEntry.item.stage !== "completed-program") {
          throw new Error("An unexpected open hypothesis remained in the frontier.");
        }

        candidatesTested += 1;
        const fitsInferredExamples = scalarMatches(
          completedEntry.item.scalarBody,
          deduction.examples,
        );
        const fitsTopLevelExamples =
          fitsInferredExamples &&
          examples.every((example) =>
            programMatches(completedEntry.item.expression, example),
          );

        if (fitsTopLevelExamples) {
          return {
            kind: "synthesized",
            program: completedEntry.item.expression,
            cost: completedEntry.cost,
            candidatesTested,
            inferredExamples: deduction.examples,
            trace,
          };
        }
      }
    }
  }

  return {
    kind: "not-found",
    reason: "no int -> int expression was found within the cost bound",
    trace,
  };
}

function recordTrace(
  trace: SearchTraceEntry[],
  stage: SearchTraceEntry["stage"],
  cost: number,
): void {
  const last = trace[trace.length - 1];
  if (last !== undefined && last.stage === stage && last.cost === cost) {
    trace[trace.length - 1] = { stage, cost, count: last.count + 1 };
    return;
  }
  trace.push({ stage, cost, count: 1 });
}

export function substituteHole(
  expression: Expression,
  holeName: string,
  replacement: Expression,
): Expression {
  return substituteHoleInContext(expression, holeName, replacement, []);
}

function substituteHoleInContext(
  expression: Expression,
  holeName: string,
  replacement: Expression,
  environment: TypeEnvironment,
): Expression {
  switch (expression.kind) {
    case "int":
    case "variable":
      return expression;
    case "hole": {
      if (expression.name !== holeName) {
        return expression;
      }

      const replacementType = inferType(replacement, environment);
      if (
        replacementType === undefined ||
        !typeEquals(replacementType, expression.expectedType)
      ) {
        throw new TypeError(`Replacement for ?${holeName} has the wrong type.`);
      }
      return replacement;
    }
    case "list":
      return {
        ...expression,
        elements: expression.elements.map((element) =>
          substituteHoleInContext(
            element,
            holeName,
            replacement,
            environment,
          ),
        ),
      };
    case "binary":
      return {
        ...expression,
        left: substituteHoleInContext(
          expression.left,
          holeName,
          replacement,
          environment,
        ),
        right: substituteHoleInContext(
          expression.right,
          holeName,
          replacement,
          environment,
        ),
      };
    case "lambda":
      return {
        ...expression,
        body: substituteHoleInContext(
          expression.body,
          holeName,
          replacement,
          [
            ...environment,
            {
              name: expression.parameter,
              type: expression.parameterType,
            },
          ],
        ),
      };
    case "map":
      return {
        ...expression,
        mapper: substituteHoleInContext(
          expression.mapper,
          holeName,
          replacement,
          environment,
        ),
        list: substituteHoleInContext(
          expression.list,
          holeName,
          replacement,
          environment,
        ),
      };
  }
}

function programMatches(program: Expression, example: MapExample): boolean {
  if (program.kind !== "lambda") {
    return false;
  }

  const inputLiteral: Expression = {
    kind: "list",
    elementType: INT,
    elements: example.input.map((value) => ({ kind: "int", value })),
  };
  const closedBody = substituteVariable(
    program.body,
    program.parameter,
    inputLiteral,
  );

  try {
    const actual = expectIntList(evaluateExpression(closedBody));
    return arraysEqual(actual, example.output);
  } catch {
    return false;
  }
}

function scalarMatches(
  expression: Expression,
  examples: readonly Example[],
): boolean {
  try {
    return examples.every(
      ({ input, output }) => evaluateScalar(expression, input) === output,
    );
  } catch (error) {
    if (error instanceof RangeError) {
      return false;
    }
    throw error;
  }
}

function substituteVariable(
  expression: Expression,
  variableName: string,
  replacement: Expression,
): Expression {
  switch (expression.kind) {
    case "int":
    case "hole":
      return expression;
    case "variable":
      return expression.name === variableName ? replacement : expression;
    case "list":
      return {
        ...expression,
        elements: expression.elements.map((element) =>
          substituteVariable(element, variableName, replacement),
        ),
      };
    case "binary":
      return {
        ...expression,
        left: substituteVariable(expression.left, variableName, replacement),
        right: substituteVariable(expression.right, variableName, replacement),
      };
    case "lambda":
      return expression.parameter === variableName
        ? expression
        : {
            ...expression,
            body: substituteVariable(
              expression.body,
              variableName,
              replacement,
            ),
          };
    case "map":
      return {
        ...expression,
        mapper: substituteVariable(
          expression.mapper,
          variableName,
          replacement,
        ),
        list: substituteVariable(expression.list, variableName, replacement),
      };
  }
}

function arraysEqual(
  left: readonly number[],
  right: readonly number[],
): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}
