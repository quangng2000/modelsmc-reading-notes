import assert from "node:assert/strict";
import test from "node:test";

import { BOOL, INT, renderExpression } from "../src/ast.js";
import { expressionCost } from "../src/cost.js";
import { enumerateExpressionsByCost } from "../src/enumeration/index.js";
import { inferType } from "../src/typecheck.js";

test("enumerates deduped, well-typed bool candidates in exact-cost buckets", () => {
  const buckets = [
    ...enumerateExpressionsByCost({ maxCost: 2, targetType: "bool" }),
  ];

  assert.deepEqual(
    buckets.map(({ cost }) => cost),
    [0, 1, 2],
  );
  assert.deepEqual(buckets[0]?.expressions, []);

  // A comparison costs 1 + its operands, so comparisons of two cost-0
  // variables land in the cost-1 bucket alongside the bool literals. (The
  // design sketch placed (x < x) at cost 2, which contradicts its own cost
  // table; the cost table wins.)
  assert.deepEqual(
    buckets[1]?.expressions.map(renderExpression),
    ["true", "false", "(x < x)", "(x <= x)", "(x == x)"],
  );

  const renderedAtTwo = buckets[2]?.expressions.map(renderExpression);
  assert.ok(renderedAtTwo);
  assert.ok(renderedAtTwo.includes("(0 < x)"));
  assert.ok(renderedAtTwo.includes("(x == 2)"));
  assert.ok(renderedAtTwo.includes("!(true)"));

  for (const bucket of buckets) {
    const rendered = bucket.expressions.map(renderExpression);
    assert.equal(new Set(rendered).size, rendered.length);

    for (const expression of bucket.expressions) {
      assert.equal(expressionCost(expression), bucket.cost);
      assert.deepEqual(inferType(expression, [{ name: "x", type: INT }]), BOOL);
    }
  }
});

test("enumerates int candidates over two variables including (acc + x)", () => {
  const buckets = [
    ...enumerateExpressionsByCost({
      maxCost: 1,
      variables: ["acc", "x"],
      targetType: "int",
    }),
  ];

  assert.deepEqual(
    buckets[0]?.expressions.map(renderExpression),
    ["acc", "x"],
  );

  // (acc + x) costs 1: the operator costs 1 and both variables cost 0, so it
  // sits in the cost-1 bucket.
  const renderedAtOne = buckets[1]?.expressions.map(renderExpression);
  assert.ok(renderedAtOne);
  assert.ok(renderedAtOne.includes("(acc + x)"));
});
