import type { StaticType } from "../../core/language.verify.js";

/**
 * Exact counting of well-typed programs by structural cost.
 *
 * Proposition 2 of the results paper (machine-checked cost monotonicity plus
 * the cost cap) makes the accepted program space finite; this module computes
 * that space's size exactly, per cost, with a dynamic program over the typed
 * grammar. The counts give the exact normalizing constant of the Occam prior
 *   p(m) = exp(-beta * cost(m)) / Z,   Z = sum_c N(c) * exp(-beta * c),
 * and support exact uniform-within-cost sampling (see sample.ts).
 *
 * The support counted here is scoring acceptance: the program type-checks
 * against the task signature (core inferType) and cost(m) <= costCap. The
 * decoder's additional depth/node caps are not applied; at the cost caps used
 * in the experiments they are not binding for the enumerated range, and the
 * enumeration tests cross-check every counted program against the verified
 * type checker.
 */

export const ALL_TYPES: readonly StaticType[] = [
  "IntType",
  "BoolType",
  "IntListType",
  "BoolListType",
];

/** Variables in scope for an expression position. */
export interface ExprScope {
  readonly input?: StaticType;
  readonly item?: StaticType;
  readonly accumulator?: StaticType;
}

export type CostTable = Record<StaticType, bigint[]>;

function zeroTable(costCap: number): CostTable {
  const make = () => new Array<bigint>(costCap + 1).fill(0n);
  return {
    IntType: make(),
    BoolType: make(),
    IntListType: make(),
    BoolListType: make(),
  };
}

function convolve2(left: readonly bigint[], right: readonly bigint[], total: number): bigint {
  let sum = 0n;
  for (let a = 1; a <= total - 1; a += 1) {
    if (left[a]! === 0n) continue;
    sum += left[a]! * right[total - a]!;
  }
  return sum;
}

function convolve3(
  first: readonly bigint[],
  second: readonly bigint[],
  third: readonly bigint[],
  total: number,
): bigint {
  let sum = 0n;
  for (let a = 1; a <= total - 2; a += 1) {
    if (first[a]! === 0n) continue;
    for (let b = 1; b <= total - a - 1; b += 1) {
      if (second[b]! === 0n) continue;
      sum += first[a]! * second[b]! * third[total - a - b]!;
    }
  }
  return sum;
}

/**
 * counts[t][c] = number of distinct expressions of inferred type t and
 * structural cost exactly c, under the given scope and literal catalog.
 */
export function countExpressions(
  scope: ExprScope,
  costCap: number,
  integerConstantCount: number,
): CostTable {
  if (!Number.isSafeInteger(costCap) || costCap < 0) {
    throw new RangeError("costCap must be a nonnegative safe integer");
  }
  const counts = zeroTable(costCap);
  if (costCap === 0) return counts;

  // Cost-1 leaves.
  counts.IntType[1] = BigInt(integerConstantCount);
  counts.BoolType[1] = 2n;
  counts.IntListType[1] = 1n; // EmptyIntList
  counts.BoolListType[1] = 1n; // EmptyBoolList
  for (const binding of [scope.input, scope.item, scope.accumulator]) {
    if (binding !== undefined) counts[binding][1] = counts[binding][1]! + 1n;
  }

  for (let cost = 2; cost <= costCap; cost += 1) {
    const inner = cost - 1;
    // Not : Bool -> Bool
    counts.BoolType[cost] = counts.BoolType[cost]! + counts.BoolType[inner]!;
    // Add | Subtract | Multiply : Int x Int -> Int
    counts.IntType[cost] =
      counts.IntType[cost]! + 3n * convolve2(counts.IntType, counts.IntType, inner);
    // LessThan | EqualInt : Int x Int -> Bool
    counts.BoolType[cost] =
      counts.BoolType[cost]! + 2n * convolve2(counts.IntType, counts.IntType, inner);
    // And : Bool x Bool -> Bool
    counts.BoolType[cost] =
      counts.BoolType[cost]! + convolve2(counts.BoolType, counts.BoolType, inner);
    // PrependInt : Int x List<Int> -> List<Int>
    counts.IntListType[cost] =
      counts.IntListType[cost]! + convolve2(counts.IntType, counts.IntListType, inner);
    // PrependBool : Bool x List<Bool> -> List<Bool>
    counts.BoolListType[cost] =
      counts.BoolListType[cost]! + convolve2(counts.BoolType, counts.BoolListType, inner);
    // IfThenElse : Bool x t x t -> t
    for (const resultType of ALL_TYPES) {
      counts[resultType][cost] =
        counts[resultType][cost]! +
        convolve3(counts.BoolType, counts[resultType], counts[resultType], inner);
    }
  }
  return counts;
}

export function listElementType(type: StaticType): StaticType | undefined {
  if (type === "IntListType") return "IntType";
  if (type === "BoolListType") return "BoolType";
  return undefined;
}

export interface ProgramCountTables {
  /** N(c): accepted programs of cost exactly c, all families combined. */
  readonly total: bigint[];
  readonly expression: bigint[];
  readonly map: bigint[];
  readonly fold: bigint[];
  /** Expression tables reused by the sampler/enumerator. */
  readonly outerBody: CostTable;
  readonly mapper: CostTable | undefined;
  readonly foldInitial: CostTable | undefined;
  readonly foldReducer: CostTable | undefined;
}

/**
 * Per-cost counts of accepted programs for a task signature.
 *
 * - ExpressionProgram(body): body typed outputType in scope {input}.
 *   cost = bodyCost.
 * - MapProgram(mapper): input a list of E, output a list of F; mapper typed F
 *   in scope {item: E}. cost = 2 + mapperCost.
 * - FoldRightProgram(initial, reducer): input a list of E; initial typed
 *   outputType in the closed scope; reducer typed outputType in scope
 *   {item: E, accumulator: outputType}. cost = 3 + initialCost + reducerCost.
 */
export function countPrograms(
  inputType: StaticType,
  outputType: StaticType,
  costCap: number,
  integerConstantCount: number,
): ProgramCountTables {
  const expressionCounts = new Array<bigint>(costCap + 1).fill(0n);
  const mapCounts = new Array<bigint>(costCap + 1).fill(0n);
  const foldCounts = new Array<bigint>(costCap + 1).fill(0n);

  const outerBody = countExpressions({ input: inputType }, costCap, integerConstantCount);
  for (let cost = 1; cost <= costCap; cost += 1) {
    expressionCounts[cost] = outerBody[outputType][cost]!;
  }

  const elementType = listElementType(inputType);
  const outputElementType = listElementType(outputType);

  let mapper: CostTable | undefined;
  if (elementType !== undefined && outputElementType !== undefined && costCap >= 3) {
    mapper = countExpressions({ item: elementType }, costCap - 2, integerConstantCount);
    for (let cost = 3; cost <= costCap; cost += 1) {
      mapCounts[cost] = mapper[outputElementType][cost - 2]!;
    }
  }

  let foldInitial: CostTable | undefined;
  let foldReducer: CostTable | undefined;
  if (elementType !== undefined && costCap >= 5) {
    foldInitial = countExpressions({}, costCap - 4, integerConstantCount);
    foldReducer = countExpressions(
      { item: elementType, accumulator: outputType },
      costCap - 4,
      integerConstantCount,
    );
    for (let cost = 5; cost <= costCap; cost += 1) {
      let sum = 0n;
      const budget = cost - 3;
      for (let initialCost = 1; initialCost <= budget - 1; initialCost += 1) {
        const initialWays = foldInitial[outputType][initialCost]!;
        if (initialWays === 0n) continue;
        sum += initialWays * foldReducer[outputType][budget - initialCost]!;
      }
      foldCounts[cost] = sum;
    }
  }

  const total = new Array<bigint>(costCap + 1).fill(0n);
  for (let cost = 0; cost <= costCap; cost += 1) {
    total[cost] = expressionCounts[cost]! + mapCounts[cost]! + foldCounts[cost]!;
  }
  return {
    total,
    expression: expressionCounts,
    map: mapCounts,
    fold: foldCounts,
    outerBody,
    mapper,
    foldInitial,
    foldReducer,
  };
}

/** Natural log of a positive BigInt, accurate to ~1e-15 relative error. */
export function lnBigInt(value: bigint): number {
  if (value <= 0n) throw new RangeError("lnBigInt requires a positive value");
  const digits = value.toString();
  if (digits.length <= 15) return Math.log(Number(value));
  const mantissa = Number(digits.slice(0, 15));
  return Math.log(mantissa) + (digits.length - 15) * Math.LN10;
}

export function logSumExp(terms: readonly number[]): number {
  let max = -Infinity;
  for (const term of terms) if (term > max) max = term;
  if (max === -Infinity) return -Infinity;
  let sum = 0;
  for (const term of terms) sum += Math.exp(term - max);
  return max + Math.log(sum);
}

/**
 * log Z of the Occam prior p(m) proportional to exp(-beta * cost(m)) over the
 * finite accepted-program space with cost <= costCap.
 */
export function logPriorNormalizer(totalCounts: readonly bigint[], beta: number): number {
  const terms: number[] = [];
  for (let cost = 1; cost < totalCounts.length; cost += 1) {
    const count = totalCounts[cost]!;
    if (count === 0n) continue;
    terms.push(lnBigInt(count) - beta * cost);
  }
  return logSumExp(terms);
}

/** log prior probability of a single accepted program of the given cost. */
export function logPriorProbability(cost: number, beta: number, logZ: number): number {
  return -beta * cost - logZ;
}
