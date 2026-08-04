import {
  primitiveLiteral,
  primitiveValueEquals,
  renderPrimitiveValue,
  typeEquals,
  type Expression,
  type PrimitiveType,
  type PrimitiveValue,
} from "../ast.js";
import {
  deduceFilterExamples,
  deduceFoldExamples,
  deduceMapExamples,
  type FoldExample,
  type MapExample,
  type PredicateExample,
  type ScalarExample,
} from "../deduction/index.js";
import {
  evaluateExpression,
  expectBool,
} from "../evaluation/index.js";
import {
  DEFAULT_CONSTANTS,
  DEFAULT_MAX_COST,
  DEFAULT_STRING_CONSTANTS,
  enumerateExpressionsByCost,
  type SearchOptions,
} from "../enumeration/index.js";
import { inferType } from "../typecheck.js";
import { substituteHole } from "./expressions.js";
import type {
  BodyBucketCache,
  AnyFamilyPlan,
  CompletedCandidate,
  FamilyConstraintResult,
  FamilyPlan,
  FamilyPreparation,
  FoldConstraint,
  IOExample,
  ViableFamily,
} from "./types.js";

export function prepareFamily(
  plan: AnyFamilyPlan,
  examples: readonly IOExample[],
  options: SearchOptions,
): FamilyPreparation {
  const maxCost = options.maxCost ?? DEFAULT_MAX_COST;
  const constants = options.constants ?? DEFAULT_CONSTANTS;
  const stringConstants =
    options.stringConstants ?? DEFAULT_STRING_CONSTANTS;

  switch (plan.family) {
    case "map": {
      const outputType = plan.bodyType;
      const deduction = deduceMapExamples(
        toListExamples(examples),
        plan.inputElementType,
        outputType,
      );
      if (deduction.kind === "refuted") {
        return { kind: "refuted", reason: deduction.reason };
      }
      return {
        kind: "viable",
        viable: {
          family: "map",
          plan,
          scalarExamples: deduction.examples,
          buckets: makeBodyBuckets(options, {
            maxCost,
            variables: [
              { name: "x", type: plan.inputElementType.kind },
            ],
            targetType: outputType.kind,
          }),
        },
      };
    }
    case "filter": {
      const deduction = deduceFilterExamples(
        toListExamples(examples),
        plan.inputElementType,
      );
      if (deduction.kind === "refuted") {
        return { kind: "refuted", reason: deduction.reason };
      }
      return {
        kind: "viable",
        viable: {
          family: "filter",
          plan,
          predicateExamples: deduction.examples,
          buckets: makeBodyBuckets(options, {
            maxCost,
            variables: [
              { name: "x", type: plan.inputElementType.kind },
            ],
            targetType: "bool",
          }),
        },
      };
    }
    case "fold": {
      const accumulatorType = plan.bodyType;
      const deduction = deduceFoldExamples(
        toFoldExamples(examples),
        plan.inputElementType,
        accumulatorType,
      );
      if (deduction.kind === "refuted") {
        return { kind: "refuted", reason: deduction.reason };
      }
      const foldConstraint: FoldConstraint = {
        init: deduction.init,
        steps: deduction.steps,
      };
      return {
        kind: "viable",
        viable: {
          family: "fold",
          plan,
          foldConstraint,
          initCandidates: foldInitCandidates(
            foldConstraint,
            accumulatorType,
            constants,
            stringConstants,
          ),
          buckets: makeBodyBuckets(options, {
            maxCost,
            variables: [
              { name: "acc", type: accumulatorType.kind },
              { name: "x", type: plan.inputElementType.kind },
            ],
            targetType: accumulatorType.kind,
          }),
        },
      };
    }
  }
}

function foldInitCandidates(
  constraint: FoldConstraint,
  accumulatorType: PrimitiveType,
  constants: readonly number[],
  stringConstants: readonly string[],
): readonly PrimitiveValue[] {
  if (constraint.init !== undefined) {
    return [constraint.init];
  }
  switch (accumulatorType.kind) {
    case "int":
      return [...new Set(constants)];
    case "bool":
      return [true, false];
    case "string":
      return [...new Set(stringConstants)];
  }
}

function makeBodyBuckets(
  base: SearchOptions,
  override: SearchOptions,
): BodyBucketCache {
  return {
    generator: enumerateExpressionsByCost({
      maxCost: override.maxCost ?? DEFAULT_MAX_COST,
      constants: base.constants ?? DEFAULT_CONSTANTS,
      variables: override.variables ?? [],
      targetType: override.targetType ?? "int",
      ...(base.stringConstants === undefined
        ? {}
        : { stringConstants: base.stringConstants }),
    }),
    byCost: [],
    exhausted: false,
  };
}

export function bodiesAtCost(
  cache: BodyBucketCache,
  cost: number,
): readonly Expression[] {
  while (cache.byCost.length <= cost && !cache.exhausted) {
    const next = cache.generator.next();
    if (next.done === true) {
      cache.exhausted = true;
      break;
    }
    if (next.value.cost !== cache.byCost.length) {
      throw new Error("Body buckets arrived out of cost order.");
    }
    cache.byCost.push(next.value.expressions);
  }
  return cache.byCost[cost] ?? [];
}

export function completeFamilyBody(
  viable: ViableFamily,
  body: Expression,
): readonly CompletedCandidate[] {
  const plan = viable.plan;
  switch (viable.family) {
    case "map": {
      const mapper: Expression = {
        kind: "lambda",
        parameter: "x",
        parameterType: plan.inputElementType,
        body,
      };
      return [
        completedCandidate(
          plan,
          substituteHole(plan.skeleton, "f", mapper),
          body,
          undefined,
        ),
      ];
    }
    case "filter": {
      const predicate: Expression = {
        kind: "lambda",
        parameter: "x",
        parameterType: plan.inputElementType,
        body,
      };
      return [
        completedCandidate(
          plan,
          substituteHole(plan.skeleton, "p", predicate),
          body,
          undefined,
        ),
      ];
    }
    case "fold": {
      const reducer: Expression = {
        kind: "lambda",
        parameter: "acc",
        parameterType: plan.bodyType,
        body: {
          kind: "lambda",
          parameter: "x",
          parameterType: plan.inputElementType,
          body,
        },
      };
      const withReducer = substituteHole(plan.skeleton, "f", reducer);
      return viable.initCandidates.map((init) =>
        completedCandidate(
          plan,
          substituteHole(withReducer, "init", primitiveLiteral(init)),
          body,
          init,
        ),
      );
    }
  }
}

function completedCandidate(
  plan: FamilyPlan,
  program: Expression,
  body: Expression,
  init: PrimitiveValue | undefined,
): CompletedCandidate {
  const programType = inferType(program, []);
  if (
    programType === undefined ||
    !typeEquals(programType, plan.programType)
  ) {
    throw new TypeError(
      `Filling the ${plan.family} skeleton produced an ill-typed program.`,
    );
  }
  return { family: plan.family, program, body, init };
}

export function checkFamilyConstraint(
  candidate: CompletedCandidate,
  viable: ViableFamily,
): FamilyConstraintResult {
  switch (viable.family) {
    case "map":
      return scalarMatches(
        candidate.body,
        viable.scalarExamples,
        viable.plan.inputElementType,
      );
    case "filter":
      return predicateMatches(
        candidate.body,
        viable.predicateExamples,
        viable.plan.inputElementType,
      );
    case "fold":
      return foldMatches(candidate, viable);
  }
}

function scalarMatches(
  expression: Expression,
  examples: readonly ScalarExample<PrimitiveValue, PrimitiveValue>[],
  inputType: PrimitiveType,
): FamilyConstraintResult {
  for (const { input, output } of examples) {
    const label = `inferred sub-example ?f(${renderPrimitiveValue(input)})`;
    try {
      const actual = evaluatePrimitive(expression, [
        { name: "x", type: inputType, value: input },
      ]);
      if (!primitiveValueEquals(actual, output)) {
        return failed(
          `${label}: expected ${renderPrimitiveValue(output)}, got ${renderPrimitiveValue(actual)}`,
        );
      }
    } catch (error) {
      if (error instanceof RangeError) {
        return evaluationFailed(label, output, error);
      }
      throw error;
    }
  }
  return satisfied();
}

function predicateMatches(
  expression: Expression,
  examples: readonly PredicateExample<PrimitiveValue>[],
  inputType: PrimitiveType,
): FamilyConstraintResult {
  for (const { input, output } of examples) {
    const label = `inferred sub-example ?p(${renderPrimitiveValue(input)})`;
    try {
      const actual = expectBool(
        evaluateExpression(expression, [
          { name: "x", type: inputType, value: input },
        ]),
      );
      if (actual !== output) {
        return failed(`${label}: expected ${output}, got ${actual}`);
      }
    } catch (error) {
      if (error instanceof RangeError) {
        return evaluationFailed(label, output, error);
      }
      throw error;
    }
  }
  return satisfied();
}

function foldMatches(
  candidate: CompletedCandidate,
  viable: Extract<ViableFamily, { readonly family: "fold" }>,
): FamilyConstraintResult {
  const constraint = viable.foldConstraint;
  if (
    constraint.init !== undefined &&
    (candidate.init === undefined ||
      !primitiveValueEquals(candidate.init, constraint.init))
  ) {
    return failed(
      `inferred init constraint ?init: expected ${renderPrimitiveValue(constraint.init)}, got ${candidate.init === undefined ? "undefined" : renderPrimitiveValue(candidate.init)}`,
    );
  }

  for (const { accumulator, element, output } of constraint.steps) {
    const label = `inferred reducer sub-example ?f(${renderPrimitiveValue(accumulator)}, ${renderPrimitiveValue(element)})`;
    try {
      const actual = evaluatePrimitive(candidate.body, [
        { name: "acc", type: viable.plan.bodyType, value: accumulator },
        {
          name: "x",
          type: viable.plan.inputElementType,
          value: element,
        },
      ]);
      if (!primitiveValueEquals(actual, output)) {
        return failed(
          `${label}: expected ${renderPrimitiveValue(output)}, got ${renderPrimitiveValue(actual)}`,
        );
      }
    } catch (error) {
      if (error instanceof RangeError) {
        return evaluationFailed(label, output, error);
      }
      throw error;
    }
  }
  return satisfied();
}

function evaluatePrimitive(
  expression: Expression,
  bindings: readonly {
    readonly name: string;
    readonly type: PrimitiveType;
    readonly value: PrimitiveValue;
  }[],
): PrimitiveValue {
  const value = evaluateExpression(expression, bindings);
  if (
    typeof value !== "number" &&
    typeof value !== "boolean" &&
    typeof value !== "string"
  ) {
    throw new TypeError("Expected a primitive synthesis value.");
  }
  return value;
}

function satisfied(): FamilyConstraintResult {
  return { kind: "satisfied" };
}

function failed(reason: string): FamilyConstraintResult {
  return { kind: "failed", reason };
}

function evaluationFailed(
  label: string,
  expected: PrimitiveValue,
  error: RangeError,
): FamilyConstraintResult {
  return failed(
    `${label}: expected ${renderPrimitiveValue(expected)}; evaluation error: ${error.message}`,
  );
}

function toListExamples(
  examples: readonly IOExample[],
): readonly MapExample<PrimitiveValue, PrimitiveValue>[] {
  return examples.map(({ input, output }) => {
    if (isPrimitiveOutput(output)) {
      throw new Error("A list-output family received a scalar example.");
    }
    return { input, output };
  });
}

function toFoldExamples(
  examples: readonly IOExample[],
): readonly FoldExample<PrimitiveValue, PrimitiveValue>[] {
  return examples.map(({ input, output }) => {
    if (!isPrimitiveOutput(output)) {
      throw new Error("The fold family received a list example.");
    }
    return { input, output };
  });
}

function isPrimitiveOutput(
  output: IOExample["output"],
): output is PrimitiveValue {
  return (
    typeof output === "number" ||
    typeof output === "boolean" ||
    typeof output === "string"
  );
}
