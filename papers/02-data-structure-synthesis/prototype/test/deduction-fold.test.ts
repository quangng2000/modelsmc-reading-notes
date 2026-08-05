import assert from "node:assert/strict";
import test from "node:test";

import { deduceFoldExamples } from "../src/deduction/index.js";

test("infers init from an empty-list example", () => {
  const result = deduceFoldExamples([{ input: [], output: 5 }]);

  assert.deepEqual(result, { kind: "inferred", init: 5, steps: [] });
});

test("refutes two examples with the same input but different outputs", () => {
  const result = deduceFoldExamples([
    { input: [1, 2], output: 3 },
    { input: [1, 2], output: 4 },
  ]);

  assert.equal(result.kind, "refuted");
  if (result.kind === "refuted") {
    assert.match(result.reason, /cannot send \[1, 2\] to both 3 and 4/);
  }
});

test("peels one-element extensions into reducer steps", () => {
  const result = deduceFoldExamples([
    { input: [1, 2], output: 3 },
    { input: [1, 2, 3], output: 6 },
  ]);

  assert.deepEqual(result, {
    kind: "inferred",
    init: undefined,
    steps: [{ accumulator: 3, element: 3, output: 6 }],
  });
});

test("derives the single-element step from a known init", () => {
  const result = deduceFoldExamples([
    { input: [], output: 0 },
    { input: [7], output: 7 },
  ]);

  assert.deepEqual(result, {
    kind: "inferred",
    init: 0,
    steps: [{ accumulator: 0, element: 7, output: 7 }],
  });
});

test("refutes conflicting steps and dedupes consistent ones", () => {
  const conflict = deduceFoldExamples([
    { input: [1], output: 5 },
    { input: [1, 7], output: 9 },
    { input: [2], output: 5 },
    { input: [2, 7], output: 8 },
  ]);
  assert.equal(conflict.kind, "refuted");
  if (conflict.kind === "refuted") {
    assert.match(conflict.reason, /fold step cannot send \(5, 7\) to both 9 and 8/);
  }

  // The same (accumulator, element) pair peeled from two different example
  // pairs collapses to one step when the outputs agree.
  const deduped = deduceFoldExamples([
    { input: [1], output: 5 },
    { input: [1, 7], output: 9 },
    { input: [2], output: 5 },
    { input: [2, 7], output: 9 },
  ]);
  assert.deepEqual(deduped, {
    kind: "inferred",
    init: undefined,
    steps: [{ accumulator: 5, element: 7, output: 9 }],
  });
});

test("throws on unsafe integers in inputs or outputs", () => {
  assert.throws(
    () =>
      deduceFoldExamples([
        { input: [Number.MAX_SAFE_INTEGER + 1], output: 0 },
      ]),
    /safe integers/,
  );
  assert.throws(
    () =>
      deduceFoldExamples([
        { input: [1], output: Number.MAX_SAFE_INTEGER + 1 },
      ]),
    /safe integers/,
  );
});

test("stays inferred with unknown init and possibly empty steps", () => {
  const result = deduceFoldExamples([{ input: [1, 2, 3], output: 6 }]);

  assert.deepEqual(result, { kind: "inferred", init: undefined, steps: [] });
});
