import assert from "node:assert/strict";
import test from "node:test";

import { deduceFilterExamples } from "../src/deduction/index.js";

test("infers predicate examples for kept and dropped values", () => {
  const result = deduceFilterExamples([
    { input: [1, 2, 3, 4], output: [2, 4] },
  ]);

  assert.deepEqual(result, {
    kind: "inferred",
    examples: [
      { input: 1, output: false },
      { input: 2, output: true },
      { input: 3, output: false },
      { input: 4, output: true },
    ],
  });
});

test("deduplicates consistent predicate examples across examples", () => {
  const result = deduceFilterExamples([
    { input: [1, 2], output: [2] },
    { input: [2, 3], output: [2] },
  ]);

  assert.deepEqual(result, {
    kind: "inferred",
    examples: [
      { input: 1, output: false },
      { input: 2, output: true },
      { input: 3, output: false },
    ],
  });
});

test("accepts an empty output by dropping every value", () => {
  const result = deduceFilterExamples([{ input: [5, 6], output: [] }]);

  assert.deepEqual(result, {
    kind: "inferred",
    examples: [
      { input: 5, output: false },
      { input: 6, output: false },
    ],
  });
});

test("refutes filter when the output introduces elements", () => {
  const introduced = deduceFilterExamples([{ input: [1], output: [2] }]);
  assert.equal(introduced.kind, "refuted");
  if (introduced.kind === "refuted") {
    assert.match(introduced.reason, /cannot introduce 2/);
  }

  const duplicated = deduceFilterExamples([{ input: [1], output: [1, 1] }]);
  assert.equal(duplicated.kind, "refuted");
  if (duplicated.kind === "refuted") {
    assert.match(duplicated.reason, /cannot introduce 1/);
  }
});

test("refutes filter when only some copies of a value survive", () => {
  const result = deduceFilterExamples([{ input: [1, 2, 1], output: [1] }]);

  assert.equal(result.kind, "refuted");
  if (result.kind === "refuted") {
    assert.match(result.reason, /all or none of the 2 copies of 1, not 1/);
  }
});

test("refutes filter when the output reorders surviving elements", () => {
  const result = deduceFilterExamples([{ input: [1, 2], output: [2, 1] }]);

  assert.deepEqual(result, {
    kind: "refuted",
    reason: "filter preserves element order",
  });
});

test("finds conflicts across examples and rejects unsafe integers", () => {
  const conflict = deduceFilterExamples([
    { input: [1], output: [1] },
    { input: [1], output: [] },
  ]);
  assert.equal(conflict.kind, "refuted");
  if (conflict.kind === "refuted") {
    assert.match(conflict.reason, /both keep and drop 1/);
  }

  assert.throws(
    () =>
      deduceFilterExamples([
        { input: [Number.MAX_SAFE_INTEGER + 1], output: [] },
      ]),
    /safe integers/,
  );
});
