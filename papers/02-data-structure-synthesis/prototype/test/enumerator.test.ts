import assert from "node:assert/strict";
import test from "node:test";

import { renderExpression, type Expression } from "../src/ast.js";
import { expressionCost } from "../src/cost.js";
import {
  enumerateBestFirst,
  enumerateExpressionsByCost,
  evaluateScalar,
  synthesize,
} from "../src/enumeration/index.js";

test("enumerates every expression in its exact cost bucket", () => {
  const buckets = [...enumerateExpressionsByCost({ maxCost: 3 })];

  assert.deepEqual(
    buckets.map(({ cost }) => cost),
    [0, 1, 2, 3],
  );

  for (const bucket of buckets) {
    for (const expression of bucket.expressions) {
      assert.equal(expressionCost(expression), bucket.cost);
    }
  }

  const costTwo = buckets[2];
  const costThree = buckets[3];
  assert.ok(costTwo);
  assert.ok(costThree);

  const renderedAtTwo = costTwo.expressions.map(renderExpression);
  const renderedAtThree = costThree.expressions.map(renderExpression);
  const renderedAtOne = buckets[1]?.expressions.map(renderExpression);
  assert.ok(renderedAtOne);
  assert.deepEqual(renderedAtOne, ["-1", "0", "1", "2", "(x + x)", "(x - x)"]);

  const expectedAtTwo = [
    ...["+", "-"].flatMap((operator) => [
      ...renderedAtOne.map((right) => `(x ${operator} ${right})`),
      ...renderedAtOne.map((left) => `(${left} ${operator} x)`),
    ]),
    "(x * x)",
    "(x % x)",
  ];
  assert.deepEqual([...renderedAtTwo].sort(), expectedAtTwo.sort());
  assert.deepEqual(
    buckets.slice(0, 3).map(({ expressions }) => expressions.length),
    [1, 6, 26],
  );
  assert.ok(renderedAtThree.includes("(1 + 1)"));
});

test("the bucketed frontier yields candidates in nondecreasing cost", () => {
  const costs = [...enumerateBestFirst({ maxCost: 3 })].map(
    ({ cost }) => cost,
  );

  assert.ok(costs.length > 0);
  for (let index = 1; index < costs.length; index += 1) {
    const prior = costs[index - 1];
    const current = costs[index];
    assert.ok(prior !== undefined && current !== undefined);
    assert.ok(prior <= current);
  }
});

test("returns the first minimum-cost scalar program", () => {
  const result = synthesize(
    [
      { input: 1, output: 3 },
      { input: 2, output: 4 },
    ],
    { maxCost: 4 },
  );

  assert.ok(result);
  assert.equal(renderExpression(result.expression), "(x + 2)");
  assert.equal(result.cost, 2);
  assert.equal(evaluateScalar(result.expression, 8), 10);
});

test("finds identity first and rejects conflicting examples", () => {
  const identity = synthesize(
    [
      { input: -2, output: -2 },
      { input: 7, output: 7 },
    ],
    { maxCost: 4 },
  );
  assert.ok(identity);
  assert.equal(renderExpression(identity.expression), "x");
  assert.equal(identity.candidatesTested, 1);

  assert.equal(
    synthesize(
      [
        { input: 1, output: 2 },
        { input: 1, output: 3 },
      ],
      { maxCost: 3 },
    ),
    undefined,
  );
});

test("does not accept a candidate through unsafe-integer rounding", () => {
  const result = synthesize(
    [
      { input: Number.MAX_SAFE_INTEGER, output: -1 },
      { input: Number.MAX_SAFE_INTEGER - 1, output: -2 },
    ],
    { maxCost: 3 },
  );
  assert.equal(result, undefined);

  const overflowing: Expression = {
    kind: "binary",
    operator: "+",
    left: { kind: "variable", name: "x" },
    right: { kind: "int", value: 2 },
  };
  assert.throws(
    () => evaluateScalar(overflowing, Number.MAX_SAFE_INTEGER),
    RangeError,
  );
});
