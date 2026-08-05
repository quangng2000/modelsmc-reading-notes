import type {
  BoolList,
  IntList,
  RuntimeValue,
} from "../../core/language.verify.js";
import { CoreInvariantError } from "./errors.js";

function cappedIntegerDistance(left: bigint, right: bigint, cap: number): number {
  const raw = left >= right ? left - right : right - left;
  const bigintCap = BigInt(cap);
  return raw > bigintCap ? cap : Number(raw);
}

function intItems(list: IntList): bigint[] {
  const result: bigint[] = [];
  let remaining = list;
  while (remaining.kind === "IntCons") {
    result.push(remaining.head);
    remaining = remaining.tail;
  }
  return result;
}

function boolItems(list: BoolList): boolean[] {
  const result: boolean[] = [];
  let remaining = list;
  while (remaining.kind === "BoolCons") {
    result.push(remaining.head);
    remaining = remaining.tail;
  }
  return result;
}

/**
 * Bounded sequence edit distance. Insertions and deletions cost one, so a
 * shifted-but-useful list is not punished as if every following item changed.
 */
function editDistance<T>(
  predicted: readonly T[],
  expected: readonly T[],
  substitutionCost: (left: T, right: T) => number,
  cap: number,
): number {
  let previous = Array.from(
    { length: expected.length + 1 },
    (_unused, index) => Math.min(index, cap),
  );
  for (let leftIndex = 1; leftIndex <= predicted.length; leftIndex += 1) {
    const current = new Array<number>(expected.length + 1);
    current[0] = Math.min(leftIndex, cap);
    for (let rightIndex = 1; rightIndex <= expected.length; rightIndex += 1) {
      const deletion = previous[rightIndex]! + 1;
      const insertion = current[rightIndex - 1]! + 1;
      const substitution =
        previous[rightIndex - 1]! +
        substitutionCost(predicted[leftIndex - 1]!, expected[rightIndex - 1]!);
      current[rightIndex] = Math.min(deletion, insertion, substitution, cap);
    }
    previous = current;
  }
  return previous[expected.length]!;
}

export function softLoss(
  predicted: RuntimeValue,
  expected: RuntimeValue,
  cap: number,
): number {
  if (predicted.kind === "IntValue" && expected.kind === "IntValue") {
    return cappedIntegerDistance(predicted.intValue, expected.intValue, cap);
  }
  if (predicted.kind === "BoolValue" && expected.kind === "BoolValue") {
    return predicted.boolValue === expected.boolValue ? 0 : 1;
  }
  if (predicted.kind === "IntListValue" && expected.kind === "IntListValue") {
    return editDistance(
      intItems(predicted.intListValue),
      intItems(expected.intListValue),
      (left, right) => Math.min(cappedIntegerDistance(left, right, cap), 2),
      cap,
    );
  }
  if (predicted.kind === "BoolListValue" && expected.kind === "BoolListValue") {
    return editDistance(
      boolItems(predicted.boolListValue),
      boolItems(expected.boolListValue),
      (left, right) => (left === right ? 0 : 1),
      cap,
    );
  }
  throw new CoreInvariantError("verified evaluation returned a value with the wrong output type");
}
