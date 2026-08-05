import assert from "node:assert/strict";
import test from "node:test";

import {
  inferType,
  expressionCost,
  type Expr,
  type Program,
} from "../src/core/language.verify.js";
import {
  countExpressions,
  countPrograms,
  enumeratePrograms,
  lnBigInt,
  logPriorNormalizer,
  createPriorSampler,
} from "../src/shell/prior/index.js";
import { SeededRandom } from "../src/shell/engine/random.js";

const CONSTANTS: readonly bigint[] = [0n, 1n];

/**
 * Raw, completely untyped AST enumeration: every constructor with every child
 * combination at each cost. Filtering these through the verified core's
 * inferType and comparing counts against the typed DP is the completeness
 * check — the DP must count exactly the programs the core accepts.
 */
function rawExpressions(costCap: number): Expr[][] {
  const byCost: Expr[][] = [[], []];
  const leaves: Expr[] = [
    { kind: "Input" },
    { kind: "Item" },
    { kind: "Accumulator" },
    ...CONSTANTS.map((value): Expr => ({ kind: "IntLiteral", intValue: value })),
    { kind: "BoolLiteral", boolValue: false },
    { kind: "BoolLiteral", boolValue: true },
    { kind: "EmptyIntList" },
    { kind: "EmptyBoolList" },
  ];
  byCost[1] = leaves;
  for (let cost = 2; cost <= costCap; cost += 1) {
    const bucket: Expr[] = [];
    for (const operand of byCost[cost - 1]!) bucket.push({ kind: "Not", operand });
    for (let leftCost = 1; leftCost <= cost - 2; leftCost += 1) {
      for (const left of byCost[leftCost]!) {
        for (const right of byCost[cost - 1 - leftCost]!) {
          for (const kind of ["Add", "Subtract", "Multiply", "LessThan", "EqualInt", "And"] as const) {
            bucket.push({ kind, left, right });
          }
          bucket.push({ kind: "PrependInt", head: left, tail: right });
          bucket.push({ kind: "PrependBool", head: left, tail: right });
        }
      }
    }
    for (let conditionCost = 1; conditionCost <= cost - 3; conditionCost += 1) {
      for (const condition of byCost[conditionCost]!) {
        for (let thenCost = 1; thenCost <= cost - 2 - conditionCost; thenCost += 1) {
          for (const thenExpr of byCost[thenCost]!) {
            for (const elseExpr of byCost[cost - 1 - conditionCost - thenCost]!) {
              bucket.push({ kind: "IfThenElse", condition, thenExpr, elseExpr });
            }
          }
        }
      }
    }
    byCost[cost] = bucket;
  }
  return byCost;
}

test("expression-family DP counts match raw enumeration filtered by the verified type checker", () => {
  const costCap = 5;
  const raw = rawExpressions(costCap);
  const tables = countPrograms("IntListType", "IntListType", costCap, CONSTANTS.length);
  for (let cost = 1; cost <= costCap; cost += 1) {
    let accepted = 0n;
    for (const body of raw[cost]!) {
      const program: Program = { kind: "ExpressionProgram", body };
      const inferred = inferType(program, "IntListType");
      if (inferred.kind === "TypeOk" && inferred.inferred === "IntListType") accepted += 1n;
    }
    assert.equal(
      tables.expression[cost],
      accepted,
      `expression family mismatch at cost ${cost}`,
    );
  }
});

test("map-family DP counts match raw enumeration filtered by the verified type checker", () => {
  const mapperCap = 3;
  const raw = rawExpressions(mapperCap);
  const tables = countPrograms("IntListType", "IntListType", mapperCap + 2, CONSTANTS.length);
  for (let mapperCost = 1; mapperCost <= mapperCap; mapperCost += 1) {
    let accepted = 0n;
    for (const mapper of raw[mapperCost]!) {
      const program: Program = { kind: "MapProgram", mapper };
      const inferred = inferType(program, "IntListType");
      if (inferred.kind === "TypeOk" && inferred.inferred === "IntListType") accepted += 1n;
    }
    assert.equal(
      tables.map[mapperCost + 2],
      accepted,
      `map family mismatch at mapper cost ${mapperCost}`,
    );
  }
});

test("fold-family DP factors match raw enumeration per (initial, reducer) split", () => {
  const raw = rawExpressions(3);
  const tables = countPrograms("IntListType", "IntListType", 9, CONSTANTS.length);
  assert.ok(tables.foldInitial && tables.foldReducer);
  for (let initialCost = 1; initialCost <= 2; initialCost += 1) {
    for (let reducerCost = 1; reducerCost <= 3; reducerCost += 1) {
      let accepted = 0n;
      for (const initial of raw[initialCost]!) {
        for (const reducer of raw[reducerCost]!) {
          const program: Program = { kind: "FoldRightProgram", initial, reducer };
          const inferred = inferType(program, "IntListType");
          if (inferred.kind === "TypeOk" && inferred.inferred === "IntListType") accepted += 1n;
        }
      }
      assert.equal(
        tables.foldInitial!.IntListType[initialCost]! *
          tables.foldReducer!.IntListType[reducerCost]!,
        accepted,
        `fold split mismatch at (${initialCost}, ${reducerCost})`,
      );
    }
  }
});

test("typed enumeration is sound, complete against the DP, and cost-faithful", () => {
  const costCap = 8;
  const tables = countPrograms("IntListType", "IntListType", costCap, CONSTANTS.length);
  const seenCounts = new Array<bigint>(costCap + 1).fill(0n);
  for (const { program, cost } of enumeratePrograms("IntListType", "IntListType", costCap, CONSTANTS)) {
    const inferred = inferType(program, "IntListType");
    assert.equal(inferred.kind, "TypeOk", "enumerated program rejected by the verified checker");
    if (inferred.kind === "TypeOk") assert.equal(inferred.inferred, "IntListType");
    assert.equal(Number(expressionCost(program)), cost, "enumerator cost disagrees with verified cost");
    seenCounts[cost] = seenCounts[cost]! + 1n;
  }
  for (let cost = 1; cost <= costCap; cost += 1) {
    assert.equal(seenCounts[cost], tables.total[cost], `enumeration count mismatch at cost ${cost}`);
  }
});

test("scalar-output signatures exclude the map family but keep folds", () => {
  const tables = countPrograms("IntListType", "IntType", 7, CONSTANTS.length);
  for (let cost = 1; cost <= 7; cost += 1) assert.equal(tables.map[cost], 0n);
  assert.ok(tables.fold[5]! > 0n, "foldr to a scalar accumulator must be counted");
  const enumerated = [...enumeratePrograms("IntListType", "IntType", 7, CONSTANTS)];
  const foldCount = enumerated.filter((entry) => entry.program.kind === "FoldRightProgram").length;
  assert.equal(BigInt(foldCount), tables.fold.reduce((sum, n) => sum + n, 0n));
});

test("lnBigInt is accurate for small and astronomically large values", () => {
  assert.ok(Math.abs(lnBigInt(123456789n) - Math.log(123456789)) < 1e-12);
  assert.ok(Math.abs(lnBigInt(10n ** 30n) - 30 * Math.LN10) < 1e-9);
  assert.ok(Math.abs(lnBigInt(7n * 10n ** 40n) - (Math.log(7) + 40 * Math.LN10)) < 1e-9);
});

test("the prior sampler draws type-correct programs at the exact advertised density", () => {
  const costCap = 7;
  const beta = 0.02;
  const sampler = createPriorSampler({
    inputType: "IntListType",
    outputType: "IntListType",
    costCap,
    constants: CONSTANTS,
    beta,
    rng: new SeededRandom(20260805),
  });
  const tables = sampler.tables;
  const logZ = logPriorNormalizer(tables.total, beta);
  assert.ok(Math.abs(sampler.logZ - logZ) < 1e-12);

  const draws = 4000;
  const costHistogram = new Map<number, number>();
  let identityCount = 0;
  for (let draw = 0; draw < draws; draw += 1) {
    const { program, cost, logPrior } = sampler.sample();
    const inferred = inferType(program, "IntListType");
    assert.equal(inferred.kind, "TypeOk");
    assert.equal(Number(expressionCost(program)), cost);
    assert.ok(Math.abs(logPrior - (-beta * cost - logZ)) < 1e-12);
    costHistogram.set(cost, (costHistogram.get(cost) ?? 0) + 1);
    if (program.kind === "ExpressionProgram" && program.body.kind === "Input") identityCount += 1;
  }

  // Cost-marginal check: each observed frequency within 5 sigma of exact p(c).
  for (let cost = 1; cost <= costCap; cost += 1) {
    const count = tables.total[cost]!;
    if (count === 0n) continue;
    const probability = Math.exp(lnBigInt(count) - beta * cost - logZ);
    const observed = (costHistogram.get(cost) ?? 0) / draws;
    const sigma = Math.sqrt((probability * (1 - probability)) / draws);
    assert.ok(
      Math.abs(observed - probability) < 5 * sigma + 1e-9,
      `cost ${cost}: observed ${observed}, expected ${probability}`,
    );
  }

  // Point-mass check on a single program: the identity has prior e^{-beta}/Z.
  const identityProbability = Math.exp(-beta * 1 - logZ);
  const identitySigma = Math.sqrt((identityProbability * (1 - identityProbability)) / draws);
  assert.ok(
    Math.abs(identityCount / draws - identityProbability) < 5 * identitySigma + 1e-9,
    `identity: observed ${identityCount / draws}, expected ${identityProbability}`,
  );
});

test("countExpressions honours scope bindings exactly", () => {
  // Closed scope at cost 1: only literals and empty lists.
  const closed = countExpressions({}, 1, CONSTANTS.length);
  assert.equal(closed.IntType[1], BigInt(CONSTANTS.length));
  assert.equal(closed.BoolType[1], 2n);
  assert.equal(closed.IntListType[1], 1n);
  // Adding a binding adds exactly one leaf of its type.
  const withItem = countExpressions({ item: "IntType" }, 1, CONSTANTS.length);
  assert.equal(withItem.IntType[1], BigInt(CONSTANTS.length) + 1n);
  const withBoth = countExpressions({ item: "IntType", accumulator: "IntType" }, 1, CONSTANTS.length);
  assert.equal(withBoth.IntType[1], BigInt(CONSTANTS.length) + 2n);
});
