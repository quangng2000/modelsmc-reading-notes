import type { BoolList, IntList, RuntimeValue } from "../../core/language.verify.js";
import { CoreInvariantError } from "./errors.js";

/**
 * Calibrated observation model: a proper conditional distribution
 * p(observed | predicted) for every output type, replacing the edit-distance
 * surrogate when a true likelihood is wanted.
 *
 * - Int: two-sided geometric on the integers, centred at the prediction.
 *     p(y | yhat) = (1-rho)/(1+rho) * rho^{|y-yhat|}   (sums to 1 over Z)
 * - Bool: epsilon-flip channel.
 * - Lists: a pair-HMM insertion/deletion/substitution channel. At every
 *   source position the action set {delete pd, insert pi, emit-match 1-pd-pi}
 *   sums to one; past the last source element the actions are {insert pi,
 *   stop 1-pi}. Every action's emission is itself a normalized distribution,
 *   so the channel is a proper distribution over finite output lists by
 *   construction. The likelihood sums over all alignments via the forward
 *   algorithm in log space.
 */
export interface NoiseModel {
  /** Two-sided geometric decay for scalar-int and matched-element noise, in (0,1). */
  readonly rhoMatch: number;
  /** Two-sided geometric decay (around 0) for inserted int elements, in (0,1). */
  readonly rhoInsert: number;
  /** Flip probability for Bool outputs and matched Bool elements, in (0,1). */
  readonly epsFlip: number;
  /** Pair-HMM action probabilities; pDelete + pInsert < 1. */
  readonly pDelete: number;
  readonly pInsert: number;
}

export const DEFAULT_NOISE: NoiseModel = {
  rhoMatch: 0.25,
  rhoInsert: 0.5,
  epsFlip: 0.05,
  pDelete: 0.05,
  pInsert: 0.05,
};

export function validateNoiseModel(noise: NoiseModel): void {
  const inOpenUnit = (value: number) => Number.isFinite(value) && value > 0 && value < 1;
  if (!inOpenUnit(noise.rhoMatch) || !inOpenUnit(noise.rhoInsert) || !inOpenUnit(noise.epsFlip)) {
    throw new RangeError("noise decay/flip parameters must lie strictly inside (0, 1)");
  }
  if (!inOpenUnit(noise.pDelete) || !inOpenUnit(noise.pInsert) || noise.pDelete + noise.pInsert >= 1) {
    throw new RangeError("pDelete and pInsert must be in (0,1) with pDelete + pInsert < 1");
  }
}

function absDifferenceAsNumber(left: bigint, right: bigint): number {
  const difference = left >= right ? left - right : right - left;
  return Number(difference); // saturates to Infinity beyond ~1e308; log-lik then -Infinity
}

/** log p(observed | predicted) under the two-sided geometric on Z. */
export function logIntEmission(predicted: bigint, observed: bigint, rho: number): number {
  return Math.log((1 - rho) / (1 + rho)) + absDifferenceAsNumber(observed, predicted) * Math.log(rho);
}

/** log q(value) for an inserted int element: two-sided geometric around 0. */
export function logIntInsertion(value: bigint, rho: number): number {
  return logIntEmission(0n, value, rho);
}

export function logBoolEmission(predicted: boolean, observed: boolean, eps: number): number {
  return predicted === observed ? Math.log(1 - eps) : Math.log(eps);
}

export function logBoolInsertion(): number {
  return Math.log(0.5);
}

function intListToArray(list: IntList): bigint[] {
  const items: bigint[] = [];
  let tail = list;
  while (tail.kind === "IntCons") {
    items.push(tail.head);
    tail = tail.tail;
  }
  return items;
}

function boolListToArray(list: BoolList): boolean[] {
  const items: boolean[] = [];
  let tail = list;
  while (tail.kind === "BoolCons") {
    items.push(tail.head);
    tail = tail.tail;
  }
  return items;
}

function logSumExp2(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const max = a > b ? a : b;
  return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
}

/**
 * Pair-HMM forward likelihood, generic over element emissions.
 * source = predicted elements, observed = observed elements.
 */
function logListChannel<T>(
  source: readonly T[],
  observed: readonly T[],
  noise: NoiseModel,
  logMatch: (sourceElement: T, observedElement: T) => number,
  logInsert: (observedElement: T) => number,
): number {
  const lnDelete = Math.log(noise.pDelete);
  const lnInsert = Math.log(noise.pInsert);
  const lnMatch = Math.log(1 - noise.pDelete - noise.pInsert);
  const lnStop = Math.log(1 - noise.pInsert);
  const m = source.length;
  const n = observed.length;

  let previous = new Array<number>(n + 1).fill(-Infinity);
  previous[0] = 0;
  for (let j = 1; j <= n; j += 1) {
    previous[j] = previous[j - 1]! + lnInsert + logInsert(observed[j - 1]!);
  }
  for (let i = 1; i <= m; i += 1) {
    const current = new Array<number>(n + 1).fill(-Infinity);
    current[0] = previous[0]! + lnDelete;
    for (let j = 1; j <= n; j += 1) {
      const viaDelete = previous[j]! + lnDelete;
      const viaInsert = current[j - 1]! + lnInsert + logInsert(observed[j - 1]!);
      const viaMatch = previous[j - 1]! + lnMatch + logMatch(source[i - 1]!, observed[j - 1]!);
      current[j] = logSumExp2(logSumExp2(viaDelete, viaInsert), viaMatch);
    }
    previous = current;
  }
  return previous[n]! + lnStop;
}

export function logIntListEmission(predicted: IntList, observed: IntList, noise: NoiseModel): number {
  return logListChannel(
    intListToArray(predicted),
    intListToArray(observed),
    noise,
    (sourceElement, observedElement) => logIntEmission(sourceElement, observedElement, noise.rhoMatch),
    (observedElement) => logIntInsertion(observedElement, noise.rhoInsert),
  );
}

export function logBoolListEmission(predicted: BoolList, observed: BoolList, noise: NoiseModel): number {
  return logListChannel(
    boolListToArray(predicted),
    boolListToArray(observed),
    noise,
    (sourceElement, observedElement) => logBoolEmission(sourceElement, observedElement, noise.epsFlip),
    () => logBoolInsertion(),
  );
}

/** log p(observed | predicted) dispatched on the (already type-checked) value kind. */
export function logExampleLikelihood(
  predicted: RuntimeValue,
  observed: RuntimeValue,
  noise: NoiseModel,
): number {
  if (predicted.kind === "IntValue" && observed.kind === "IntValue") {
    return logIntEmission(predicted.intValue, observed.intValue, noise.rhoMatch);
  }
  if (predicted.kind === "BoolValue" && observed.kind === "BoolValue") {
    return logBoolEmission(predicted.boolValue, observed.boolValue, noise.epsFlip);
  }
  if (predicted.kind === "IntListValue" && observed.kind === "IntListValue") {
    return logIntListEmission(predicted.intListValue, observed.intListValue, noise);
  }
  if (predicted.kind === "BoolListValue" && observed.kind === "BoolListValue") {
    return logBoolListEmission(predicted.boolListValue, observed.boolListValue, noise);
  }
  throw new CoreInvariantError("calibrated likelihood received mismatched value kinds");
}
