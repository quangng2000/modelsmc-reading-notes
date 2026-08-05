import type { Expr, Program, StaticType } from "../../core/language.verify.js";
import type { RandomSource } from "../engine/random.js";
import {
  countPrograms,
  lnBigInt,
  listElementType,
  logPriorNormalizer,
  type CostTable,
  type ExprScope,
  type ProgramCountTables,
} from "./count.js";

/**
 * Sampling from the normalized Occam prior
 *   p(m) = exp(-beta * cost(m)) / Z
 * over the finite accepted-program space, by backward sampling on the counting
 * DP tables: first the cost, then the family and the subtree splits
 * proportional to their exact BigInt counts, then uniform choices at the
 * leaves. Given the cost, every choice is exact (BigInt weights, rejection
 * sampling); the cost marginal itself is a float categorical, so the realized
 * density matches the closed form logPriorProbability up to float rounding of
 * order 2^-32 per category — negligible against Monte Carlo noise at any
 * feasible sample size, but not bit-exact.
 */

/** Uniform BigInt in [0, bound) by rejection sampling on 32-bit blocks. */
export function randomBigIntBelow(rng: RandomSource, bound: bigint): bigint {
  if (bound <= 0n) throw new RangeError("bound must be positive");
  const bits = bound.toString(2).length;
  const blocks = Math.ceil(bits / 32);
  for (;;) {
    let candidate = 0n;
    for (let block = 0; block < blocks; block += 1) {
      candidate = (candidate << 32n) | BigInt(Math.floor(rng.next() * 4_294_967_296));
    }
    candidate &= (1n << BigInt(bits)) - 1n;
    if (candidate < bound) return candidate;
  }
}

interface WeightedChoice<T> {
  readonly weight: bigint;
  readonly build: () => T;
}

function pickWeighted<T>(rng: RandomSource, choices: readonly WeightedChoice<T>[]): T {
  let total = 0n;
  for (const choice of choices) total += choice.weight;
  if (total === 0n) throw new Error("cannot sample from an empty space");
  let target = randomBigIntBelow(rng, total);
  for (const choice of choices) {
    if (target < choice.weight) return choice.build();
    target -= choice.weight;
  }
  throw new Error("unreachable: weighted pick fell through");
}

function sampleExpression(
  type: StaticType,
  cost: number,
  scope: ExprScope,
  table: CostTable,
  constants: readonly bigint[],
  rng: RandomSource,
): Expr {
  if (cost === 1) {
    const leaves: WeightedChoice<Expr>[] = [];
    if (scope.input === type) leaves.push({ weight: 1n, build: () => ({ kind: "Input" }) });
    if (scope.item === type) leaves.push({ weight: 1n, build: () => ({ kind: "Item" }) });
    if (scope.accumulator === type) {
      leaves.push({ weight: 1n, build: () => ({ kind: "Accumulator" }) });
    }
    if (type === "IntType") {
      leaves.push({
        weight: BigInt(constants.length),
        build: () => ({
          kind: "IntLiteral",
          intValue: constants[Math.floor(rng.next() * constants.length)]!,
        }),
      });
    } else if (type === "BoolType") {
      leaves.push({ weight: 2n, build: () => ({ kind: "BoolLiteral", boolValue: rng.next() < 0.5 }) });
    } else if (type === "IntListType") {
      leaves.push({ weight: 1n, build: () => ({ kind: "EmptyIntList" }) });
    } else {
      leaves.push({ weight: 1n, build: () => ({ kind: "EmptyBoolList" }) });
    }
    return pickWeighted(rng, leaves);
  }

  const inner = cost - 1;
  const choices: WeightedChoice<Expr>[] = [];
  const recurse = (childType: StaticType, childCost: number): Expr =>
    sampleExpression(childType, childCost, scope, table, constants, rng);

  if (type === "BoolType" && table.BoolType[inner]! > 0n) {
    choices.push({
      weight: table.BoolType[inner]!,
      build: () => ({ kind: "Not", operand: recurse("BoolType", inner) }),
    });
  }
  const binaryProductions: readonly {
    readonly kinds: readonly string[];
    readonly left: StaticType;
    readonly right: StaticType;
    readonly result: StaticType;
    readonly prepend: boolean;
  }[] = [
    { kinds: ["Add", "Subtract", "Multiply"], left: "IntType", right: "IntType", result: "IntType", prepend: false },
    { kinds: ["LessThan", "EqualInt"], left: "IntType", right: "IntType", result: "BoolType", prepend: false },
    { kinds: ["And"], left: "BoolType", right: "BoolType", result: "BoolType", prepend: false },
    { kinds: ["PrependInt"], left: "IntType", right: "IntListType", result: "IntListType", prepend: true },
    { kinds: ["PrependBool"], left: "BoolType", right: "BoolListType", result: "BoolListType", prepend: true },
  ];
  for (const production of binaryProductions) {
    if (production.result !== type) continue;
    for (let leftCost = 1; leftCost <= inner - 1; leftCost += 1) {
      const ways =
        table[production.left][leftCost]! * table[production.right][inner - leftCost]!;
      if (ways === 0n) continue;
      const split = leftCost;
      choices.push({
        weight: ways * BigInt(production.kinds.length),
        build: () => {
          const kind = production.kinds[Math.floor(rng.next() * production.kinds.length)]!;
          const left = recurse(production.left, split);
          const right = recurse(production.right, inner - split);
          return production.prepend
            ? ({ kind, head: left, tail: right } as Expr)
            : ({ kind, left, right } as Expr);
        },
      });
    }
  }
  for (let conditionCost = 1; conditionCost <= inner - 2; conditionCost += 1) {
    for (let thenCost = 1; thenCost <= inner - conditionCost - 1; thenCost += 1) {
      const elseCost = inner - conditionCost - thenCost;
      const ways =
        table.BoolType[conditionCost]! * table[type][thenCost]! * table[type][elseCost]!;
      if (ways === 0n) continue;
      const conditionSplit = conditionCost;
      const thenSplit = thenCost;
      choices.push({
        weight: ways,
        build: () => ({
          kind: "IfThenElse",
          condition: recurse("BoolType", conditionSplit),
          thenExpr: recurse(type, thenSplit),
          elseExpr: recurse(type, inner - conditionSplit - thenSplit),
        }),
      });
    }
  }
  return pickWeighted(rng, choices);
}

export interface PriorSampler {
  /** Draw one program; its exact log prior probability is -beta*cost - logZ. */
  sample(): { readonly program: Program; readonly cost: number; readonly logPrior: number };
  readonly logZ: number;
  readonly tables: ProgramCountTables;
}

export function createPriorSampler(options: {
  readonly inputType: StaticType;
  readonly outputType: StaticType;
  readonly costCap: number;
  readonly constants: readonly bigint[];
  readonly beta: number;
  readonly rng: RandomSource;
}): PriorSampler {
  const { inputType, outputType, costCap, constants, beta, rng } = options;
  const tables = countPrograms(inputType, outputType, costCap, constants.length);
  const logZ = logPriorNormalizer(tables.total, beta);

  // Cost marginal p(c) ∝ N(c) * exp(-beta c): small enough for float categorical.
  const costWeights: number[] = new Array(costCap + 1).fill(0);
  for (let cost = 1; cost <= costCap; cost += 1) {
    const count = tables.total[cost]!;
    if (count > 0n) costWeights[cost] = Math.exp(lnBigInt(count) - beta * cost - logZ);
  }

  const elementType = listElementType(inputType);

  function sampleProgramOfCost(cost: number): Program {
    const families: WeightedChoice<Program>[] = [];
    if (tables.expression[cost]! > 0n) {
      families.push({
        weight: tables.expression[cost]!,
        build: () => ({
          kind: "ExpressionProgram",
          body: sampleExpression(outputType, cost, { input: inputType }, tables.outerBody, constants, rng),
        }),
      });
    }
    if (tables.map[cost]! > 0n && tables.mapper !== undefined) {
      const outputElementType = listElementType(outputType)!;
      families.push({
        weight: tables.map[cost]!,
        build: () => ({
          kind: "MapProgram",
          mapper: sampleExpression(outputElementType, cost - 2, { item: elementType! }, tables.mapper!, constants, rng),
        }),
      });
    }
    if (tables.fold[cost]! > 0n && tables.foldInitial !== undefined && tables.foldReducer !== undefined) {
      const budget = cost - 3;
      for (let initialCost = 1; initialCost <= budget - 1; initialCost += 1) {
        const ways =
          tables.foldInitial[outputType][initialCost]! *
          tables.foldReducer[outputType][budget - initialCost]!;
        if (ways === 0n) continue;
        const initialSplit = initialCost;
        families.push({
          weight: ways,
          build: () => ({
            kind: "FoldRightProgram",
            initial: sampleExpression(outputType, initialSplit, {}, tables.foldInitial!, constants, rng),
            reducer: sampleExpression(
              outputType,
              budget - initialSplit,
              { item: elementType!, accumulator: outputType },
              tables.foldReducer!,
              constants,
              rng,
            ),
          }),
        });
      }
    }
    return pickWeighted(rng, families);
  }

  return {
    logZ,
    tables,
    sample() {
      let draw = rng.next();
      let cost = costCap;
      for (let candidate = 1; candidate <= costCap; candidate += 1) {
        draw -= costWeights[candidate]!;
        if (draw <= 0) {
          cost = candidate;
          break;
        }
      }
      const program = sampleProgramOfCost(cost);
      return { program, cost, logPrior: -beta * cost - logZ };
    },
  };
}
