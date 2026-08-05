import assert from "node:assert/strict";
import test from "node:test";

import { deduceMapExamples } from "../src/deduction/index.js";

test("infers scalar examples for the map function hole", () => {
  const result = deduceMapExamples([
    { input: [1, 2], output: [3, 4] },
    { input: [2, 3], output: [4, 5] },
  ]);

  assert.deepEqual(result, {
    kind: "inferred",
    examples: [
      { input: 1, output: 3 },
      { input: 2, output: 4 },
      { input: 3, output: 5 },
    ],
  });
});

test("refutes map when a repeated input needs two outputs", () => {
  const result = deduceMapExamples([
    { input: [1, 1], output: [2, 3] },
  ]);

  assert.equal(result.kind, "refuted");
  if (result.kind === "refuted") {
    assert.match(result.reason, /cannot send 1 to both 2 and 3/);
  }
});

test("refutes map when list lengths differ", () => {
  const result = deduceMapExamples([{ input: [1, 2], output: [3] }]);

  assert.deepEqual(result, {
    kind: "refuted",
    reason: "map preserves list length",
  });
});

test("finds conflicts across examples and rejects unsafe integers", () => {
  const conflict = deduceMapExamples([
    { input: [1], output: [2] },
    { input: [1], output: [3] },
  ]);
  assert.equal(conflict.kind, "refuted");

  assert.throws(
    () =>
      deduceMapExamples([
        { input: [Number.MAX_SAFE_INTEGER + 1], output: [0] },
      ]),
    /safe integers/,
  );
});
