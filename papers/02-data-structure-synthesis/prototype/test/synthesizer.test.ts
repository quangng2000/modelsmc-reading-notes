import assert from "node:assert/strict";
import test from "node:test";

import { renderExpression } from "../src/ast.js";
import { synthesizeMap } from "../src/synthesis/index.js";

test("synthesizes a completed map program end to end", () => {
  const result = synthesizeMap(
    [
      { input: [1, 2], output: [3, 4] },
      { input: [3], output: [5] },
    ],
    { maxCost: 4 },
  );

  assert.equal(result.kind, "synthesized");
  if (result.kind !== "synthesized") {
    return;
  }

  assert.equal(
    renderExpression(result.program),
    "(xs: list<int>) => map((x: int) => (x + 2), xs)",
  );
  assert.equal(result.cost, 5);
  assert.deepEqual(result.inferredExamples, [
    { input: 1, output: 3 },
    { input: 2, output: 4 },
    { input: 3, output: 5 },
  ]);
  assert.equal(result.candidatesTested, 11);
  assert.deepEqual(result.trace, [
    { stage: "skeleton", cost: 2, count: 1 },
    { stage: "completed-program", cost: 3, count: 1 },
    { stage: "completed-program", cost: 4, count: 6 },
    { stage: "completed-program", cost: 5, count: 4 },
  ]);
});

test("propagates deduction refutation and empty-example ambiguity", () => {
  const refuted = synthesizeMap(
    [{ input: [1, 1], output: [2, 3] }],
    { maxCost: 4 },
  );
  assert.equal(refuted.kind, "refuted");

  const underconstrained = synthesizeMap(
    [{ input: [], output: [] }],
    { maxCost: 4 },
  );
  assert.equal(underconstrained.kind, "underconstrained");
});

test("validates options eagerly and reports a bounded miss", () => {
  assert.throws(
    () =>
      synthesizeMap([{ input: [], output: [] }], {
        maxCost: -1,
      }),
    /maxCost/,
  );

  const notFound = synthesizeMap(
    [{ input: [1], output: [2] }],
    { maxCost: 0, constants: [] },
  );
  assert.equal(notFound.kind, "not-found");
});
