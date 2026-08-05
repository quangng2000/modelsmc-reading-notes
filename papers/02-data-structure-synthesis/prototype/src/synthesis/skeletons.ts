import {
  BOOL,
  INT,
  functionOf,
  listOf,
  typeEquals,
  type Expression,
  type ObjectType,
  type PrimitiveType,
} from "../ast.js";
import { inferType } from "../typecheck.js";
import type { FamilyName, FamilyPlan } from "./types.js";

// Completion overheads: total program cost = overhead + hole-body cost.
// Every primitive literal costs 1, so these values are type-independent.
const MAP_COMPLETION_OVERHEAD = 3;
const FILTER_COMPLETION_OVERHEAD = 3;
const FOLD_COMPLETION_OVERHEAD = 5;

export function makeMapPlan(
  inputElementType: PrimitiveType,
  outputElementType: PrimitiveType,
): FamilyPlan<"map"> {
  const inputList = listOf(inputElementType);
  const outputList = listOf(outputElementType);
  const skeleton: Expression = {
    kind: "lambda",
    parameter: "xs",
    parameterType: inputList,
    body: {
      kind: "map",
      mapper: {
        kind: "hole",
        name: "f",
        expectedType: functionOf(inputElementType, outputElementType),
      },
      list: { kind: "variable", name: "xs" },
    },
  };
  return {
    family: "map",
    skeleton,
    programType: functionOf(inputList, outputList),
    inputElementType,
    bodyType: outputElementType,
    completionOverhead: MAP_COMPLETION_OVERHEAD,
  };
}

export function makeFilterPlan(
  elementType: PrimitiveType,
): FamilyPlan<"filter"> {
  const listType = listOf(elementType);
  const skeleton: Expression = {
    kind: "lambda",
    parameter: "xs",
    parameterType: listType,
    body: {
      kind: "filter",
      predicate: {
        kind: "hole",
        name: "p",
        expectedType: functionOf(elementType, BOOL),
      },
      list: { kind: "variable", name: "xs" },
    },
  };
  return {
    family: "filter",
    skeleton,
    programType: functionOf(listType, listType),
    inputElementType: elementType,
    bodyType: BOOL,
    completionOverhead: FILTER_COMPLETION_OVERHEAD,
  };
}

export function makeFoldPlan(
  elementType: PrimitiveType,
  accumulatorType: PrimitiveType,
): FamilyPlan<"fold"> {
  const inputList = listOf(elementType);
  const skeleton: Expression = {
    kind: "lambda",
    parameter: "xs",
    parameterType: inputList,
    body: {
      kind: "fold",
      reducer: {
        kind: "hole",
        name: "f",
        expectedType: functionOf(
          accumulatorType,
          functionOf(elementType, accumulatorType),
        ),
      },
      initial: {
        kind: "hole",
        name: "init",
        expectedType: accumulatorType,
      },
      list: { kind: "variable", name: "xs" },
    },
  };
  return {
    family: "fold",
    skeleton,
    programType: functionOf(inputList, accumulatorType),
    inputElementType: elementType,
    bodyType: accumulatorType,
    completionOverhead: FOLD_COMPLETION_OVERHEAD,
  };
}

const MAP_PLAN = makeMapPlan(INT, INT);
const FILTER_PLAN = makeFilterPlan(INT);
const FOLD_PLAN = makeFoldPlan(INT, INT);

// Legacy constants remain stable for callers and the original integer tests.
export const MAP_SKELETON: Expression = Object.freeze(MAP_PLAN.skeleton);
export const FILTER_SKELETON: Expression = Object.freeze(
  FILTER_PLAN.skeleton,
);
export const FOLD_SKELETON: Expression = Object.freeze(FOLD_PLAN.skeleton);

export function completionOverhead(family: FamilyName): number {
  return defaultPlanFor(family).completionOverhead;
}

export function skeletonFor(family: FamilyName): Expression {
  return defaultPlanFor(family).skeleton;
}

export function programTypeFor(family: FamilyName): ObjectType {
  return defaultPlanFor(family).programType;
}

export function assertSkeletonWellTyped(family: FamilyName): void {
  assertPlanWellTyped(defaultPlanFor(family));
}

export function assertPlanWellTyped(plan: FamilyPlan): void {
  const skeletonType = inferType(plan.skeleton, []);
  if (
    skeletonType === undefined ||
    !typeEquals(skeletonType, plan.programType)
  ) {
    throw new TypeError(`The built-in ${plan.family} skeleton is ill-typed.`);
  }
}

function defaultPlanFor(family: FamilyName): FamilyPlan {
  switch (family) {
    case "map":
      return MAP_PLAN;
    case "filter":
      return FILTER_PLAN;
    case "fold":
      return FOLD_PLAN;
  }
}
