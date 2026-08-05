import assert from "node:assert/strict";
import test from "node:test";

import { boolCons, boolNil, intCons, intNil, type BoolList } from "../src/core/language.verify.js";
import {
  DEFAULT_NOISE,
  logBoolEmission,
  logBoolListEmission,
  logIntEmission,
  logIntListEmission,
  validateNoiseModel,
  type NoiseModel,
} from "../src/shell/scoring/emission.js";

const NOISE: NoiseModel = { ...DEFAULT_NOISE };

function listOfBools(items: readonly boolean[]): BoolList {
  let list = boolNil();
  for (let index = items.length - 1; index >= 0; index -= 1) list = boolCons(items[index]!, list);
  return list;
}

test("the default noise model is valid", () => {
  validateNoiseModel(NOISE);
});

test("the two-sided geometric int emission is a probability distribution on Z", () => {
  const rho = NOISE.rhoMatch;
  const predicted = 3n;
  const window = 400;
  let mass = 0;
  for (let k = -window; k <= window; k += 1) {
    mass += Math.exp(logIntEmission(predicted, predicted + BigInt(k), rho));
  }
  // Analytic tail beyond the window on each side: c * rho^{window+1} / (1-rho).
  const c = (1 - rho) / (1 + rho);
  const tail = (2 * c * Math.pow(rho, window + 1)) / (1 - rho);
  assert.ok(Math.abs(mass + tail - 1) < 1e-12, `total mass ${mass + tail}`);
  // Symmetry.
  assert.equal(
    logIntEmission(predicted, predicted + 7n, rho),
    logIntEmission(predicted, predicted - 7n, rho),
  );
});

test("the Bool emission sums to one", () => {
  const total =
    Math.exp(logBoolEmission(true, true, NOISE.epsFlip)) +
    Math.exp(logBoolEmission(true, false, NOISE.epsFlip));
  assert.ok(Math.abs(total - 1) < 1e-14);
});

/**
 * Independent length-marginal chain: because every emission distribution sums
 * to one, marginalizing values out of the pair-HMM leaves a pure action chain
 * over (source position, emitted count). P(|y| = j) from that chain must equal
 * the exhaustive sum of exp(logLikelihood) over every Bool sequence of length j.
 */
function lengthMarginal(sourceLength: number, maxLength: number, noise: NoiseModel): number[] {
  const pd = noise.pDelete;
  const pi = noise.pInsert;
  const pm = 1 - pd - pi;
  // reach[i][j]: probability of consuming i source items having emitted j.
  const reach: number[][] = Array.from({ length: sourceLength + 1 }, () =>
    new Array<number>(maxLength + 1).fill(0),
  );
  reach[0]![0] = 1;
  for (let i = 0; i <= sourceLength; i += 1) {
    for (let j = 0; j <= maxLength; j += 1) {
      const mass = reach[i]![j]!;
      if (mass === 0) continue;
      if (i < sourceLength) {
        reach[i + 1]![j]! += mass * pd;
        if (j < maxLength) reach[i + 1]![j + 1]! += mass * pm;
      }
      if (j < maxLength) reach[i]![j + 1]! += mass * pi;
    }
  }
  const marginal = new Array<number>(maxLength + 1).fill(0);
  for (let j = 0; j <= maxLength; j += 1) marginal[j] = reach[sourceLength]![j]! * (1 - pi);
  return marginal;
}

test("the Bool-list channel matches its independent length-marginal chain exactly", () => {
  const source = [true, false];
  const sourceList = listOfBools(source);
  const maxLength = 5;
  const marginal = lengthMarginal(source.length, maxLength, NOISE);
  for (let length = 0; length <= maxLength; length += 1) {
    let exhaustive = 0;
    for (let mask = 0; mask < 1 << length; mask += 1) {
      const observed: boolean[] = [];
      for (let bit = 0; bit < length; bit += 1) observed.push(((mask >> bit) & 1) === 1);
      exhaustive += Math.exp(logBoolListEmission(sourceList, listOfBools(observed), NOISE));
    }
    assert.ok(
      Math.abs(exhaustive - marginal[length]!) < 1e-12,
      `length ${length}: exhaustive ${exhaustive} vs chain ${marginal[length]}`,
    );
  }
  // The chain itself must be (numerically) complete once lengths are unbounded.
  const longMarginal = lengthMarginal(source.length, 200, NOISE);
  const total = longMarginal.reduce((sum, p) => sum + p, 0);
  assert.ok(Math.abs(total - 1) < 1e-9, `chain total ${total}`);
});

test("the int-list channel matches closed forms on trivial cases", () => {
  const pd = NOISE.pDelete;
  const pi = NOISE.pInsert;
  const cInsert = (1 - NOISE.rhoInsert) / (1 + NOISE.rhoInsert);

  // Empty -> empty: immediately stop.
  assert.ok(Math.abs(logIntListEmission(intNil(), intNil(), NOISE) - Math.log(1 - pi)) < 1e-12);
  // [a] -> empty: delete then stop.
  assert.ok(
    Math.abs(logIntListEmission(intCons(5n, intNil()), intNil(), NOISE) - Math.log(pd * (1 - pi))) <
      1e-12,
  );
  // Empty -> [0]: one insertion of value 0 (probability c) then stop.
  const insertZero = Math.log(pi * cInsert * (1 - pi));
  assert.ok(
    Math.abs(logIntListEmission(intNil(), intCons(0n, intNil()), NOISE) - insertZero) < 1e-12,
  );
});

test("an exact copy is likelier than any single corruption", () => {
  const source = intCons(1n, intCons(2n, intNil()));
  const copy = logIntListEmission(source, intCons(1n, intCons(2n, intNil())), NOISE);
  const substituted = logIntListEmission(source, intCons(1n, intCons(3n, intNil())), NOISE);
  const dropped = logIntListEmission(source, intCons(1n, intNil()), NOISE);
  const inserted = logIntListEmission(source, intCons(1n, intCons(0n, intCons(2n, intNil()))), NOISE);
  assert.ok(copy > substituted && copy > dropped && copy > inserted);
});

test("degenerate parameters are rejected", () => {
  assert.throws(() => validateNoiseModel({ ...NOISE, rhoMatch: 1 }));
  assert.throws(() => validateNoiseModel({ ...NOISE, pDelete: 0.6, pInsert: 0.5 }));
  assert.throws(() => validateNoiseModel({ ...NOISE, epsFlip: 0 }));
});
