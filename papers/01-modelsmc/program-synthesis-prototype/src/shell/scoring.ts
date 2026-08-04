import {
  acceptProgram,
  evaluate,
  expressionCost,
  inferType,
  matchesExample,
  valueMatchesType,
  type BoolList,
  type IntList,
  type Program,
  type RuntimeValue,
  type StaticType,
} from "../core/language.verify.js";

export interface ScoringOptions {
  readonly inputType: StaticType;
  readonly outputType: StaticType;
  readonly examples: readonly { readonly input: RuntimeValue; readonly output: RuntimeValue }[];
  readonly lossScale: number;
  readonly costScale: number;
  readonly lossCap: number;
  readonly maxCost: number;
}

export interface ExampleEvaluation {
  readonly input: RuntimeValue;
  readonly expected: RuntimeValue;
  readonly predicted: RuntimeValue;
  readonly exact: boolean;
  readonly loss: number;
}

export interface ValidScore {
  readonly kind: "Scored";
  readonly inferredType: StaticType;
  readonly evaluations: readonly ExampleEvaluation[];
  readonly totalLoss: number;
  readonly exactMatches: number;
  readonly cost: number;
  readonly logTarget: number;
  readonly exactProgram: boolean;
}

export interface RejectedScore {
  readonly kind: "Rejected";
  readonly reason: string;
  readonly inferredType?: StaticType;
  readonly cost?: number;
}

export type ProgramScore = ValidScore | RejectedScore;

export class CoreInvariantError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CoreInvariantError";
  }
}

function cappedIntegerDistance(left: bigint, right: bigint, cap: number): number {
  const raw = left >= right ? left - right : right - left;
  const bigintCap = BigInt(cap);
  return raw > bigintCap ? cap : Number(raw);
}

function cappedAdd(left: number, right: number, cap: number): number {
  return left >= cap - right ? cap : left + right;
}

function intListLoss(predicted: IntList, expected: IntList, cap: number): number {
  let left = predicted;
  let right = expected;
  let loss = 0;
  while (left.kind === "IntCons" && right.kind === "IntCons") {
    loss = cappedAdd(loss, cappedIntegerDistance(left.head, right.head, cap), cap);
    if (loss === cap) return cap;
    left = left.tail;
    right = right.tail;
  }
  while (left.kind === "IntCons") {
    loss = cappedAdd(loss, 1, cap);
    if (loss === cap) return cap;
    left = left.tail;
  }
  while (right.kind === "IntCons") {
    loss = cappedAdd(loss, 1, cap);
    if (loss === cap) return cap;
    right = right.tail;
  }
  return loss;
}

function boolListLoss(predicted: BoolList, expected: BoolList, cap: number): number {
  let left = predicted;
  let right = expected;
  let loss = 0;
  while (left.kind === "BoolCons" && right.kind === "BoolCons") {
    loss = cappedAdd(loss, left.head === right.head ? 0 : 1, cap);
    if (loss === cap) return cap;
    left = left.tail;
    right = right.tail;
  }
  while (left.kind === "BoolCons") {
    loss = cappedAdd(loss, 1, cap);
    if (loss === cap) return cap;
    left = left.tail;
  }
  while (right.kind === "BoolCons") {
    loss = cappedAdd(loss, 1, cap);
    if (loss === cap) return cap;
    right = right.tail;
  }
  return loss;
}

function softLoss(predicted: RuntimeValue, expected: RuntimeValue, cap: number): number {
  if (predicted.kind === "IntValue" && expected.kind === "IntValue") {
    return cappedIntegerDistance(predicted.intValue, expected.intValue, cap);
  }
  if (predicted.kind === "BoolValue" && expected.kind === "BoolValue") {
    return predicted.boolValue === expected.boolValue ? 0 : 1;
  }
  if (predicted.kind === "IntListValue" && expected.kind === "IntListValue") {
    return intListLoss(predicted.intListValue, expected.intListValue, cap);
  }
  if (predicted.kind === "BoolListValue" && expected.kind === "BoolListValue") {
    return boolListLoss(predicted.boolListValue, expected.boolListValue, cap);
  }
  throw new CoreInvariantError("verified evaluation returned a value with the wrong output type");
}

export function scoreProgram(program: Program, options: ScoringOptions): ProgramScore {
  const inferred = inferType(program, options.inputType);
  if (inferred.kind === "TypeError") {
    return { kind: "Rejected", reason: "the verified type checker rejected the AST" };
  }
  if (inferred.inferred !== options.outputType) {
    return {
      kind: "Rejected",
      reason: `output type mismatch: inferred ${inferred.inferred}, expected ${options.outputType}`,
      inferredType: inferred.inferred,
    };
  }

  const exactCost = expressionCost(program);
  if (exactCost > BigInt(options.maxCost)) {
    return {
      kind: "Rejected",
      reason: `expression cost ${exactCost.toString()} exceeds maximum ${options.maxCost}`,
      inferredType: inferred.inferred,
    };
  }
  // This conversion is exact because maxCost is validated as a safe integer and
  // the comparison above rejects every larger structural cost.
  const cost = Number(exactCost);

  const evaluations: ExampleEvaluation[] = [];
  let totalLoss = 0;
  let exactMatches = 0;
  for (const example of options.examples) {
    if (!valueMatchesType(example.input, options.inputType)) {
      throw new CoreInvariantError("an example input crossed the shell/core boundary with the wrong type");
    }
    const evaluated = evaluate(program, example.input);
    if (evaluated.kind === "EvalError") {
      throw new CoreInvariantError(
        "a program accepted by the verified type checker failed during evaluation",
      );
    }
    if (!valueMatchesType(evaluated.output, options.outputType)) {
      throw new CoreInvariantError(
        "a program accepted by the verified type checker returned the wrong output type",
      );
    }
    const exact = matchesExample(program, example);
    const loss = softLoss(evaluated.output, example.output, options.lossCap);
    totalLoss += loss;
    if (exact) exactMatches += 1;
    evaluations.push({
      input: example.input,
      expected: example.output,
      predicted: evaluated.output,
      exact,
      loss,
    });
  }

  const logTarget = -options.lossScale * totalLoss - options.costScale * cost;
  return {
    kind: "Scored",
    inferredType: inferred.inferred,
    evaluations,
    totalLoss,
    exactMatches,
    cost,
    logTarget,
    exactProgram: acceptProgram(program, [...options.examples]),
  };
}
