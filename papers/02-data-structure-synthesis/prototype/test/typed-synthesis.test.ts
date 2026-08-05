import assert from "node:assert/strict";
import test from "node:test";

import {
  BOOL,
  INT,
  STRING,
  listOf,
  renderExpression,
} from "../src/ast.js";
import { synthesizeProgram } from "../src/synthesis/index.js";

test("synthesizes a string-to-bool map from an explicit signature", () => {
  const outcome = synthesizeProgram(
    [
      {
        input: ["yes", "no", "yes"],
        output: [true, false, true],
      },
      { input: ["no", "yes"], output: [false, true] },
    ],
    {
      inputType: listOf(STRING),
      outputType: listOf(BOOL),
      stringConstants: ["yes"],
      maxCost: 3,
    },
  );

  assert.equal(outcome.kind, "synthesized");
  if (outcome.kind !== "synthesized") {
    return;
  }
  assert.equal(outcome.family, "map");
  assert.equal(
    renderExpression(outcome.program),
    '(xs: list<string>) => map((x: string) => (x == "yes"), xs)',
  );
});

test("synthesizes string length with a cross-type map", () => {
  const outcome = synthesizeProgram(
    [
      { input: ["", "cat", "λ"], output: [0, 3, 1] },
      { input: ["hello"], output: [5] },
    ],
    {
      inputType: listOf(STRING),
      outputType: listOf(INT),
      maxCost: 2,
    },
  );

  assert.equal(outcome.kind, "synthesized");
  if (outcome.kind === "synthesized") {
    assert.equal(
      renderExpression(outcome.program),
      "(xs: list<string>) => map((x: string) => length(x), xs)",
    );
  }
});

test("synthesizes a typed string filter", () => {
  const outcome = synthesizeProgram(
    [
      { input: ["yes", "no"], output: ["yes"] },
      { input: ["no", "yes", "yes"], output: ["yes", "yes"] },
    ],
    {
      inputType: listOf(STRING),
      outputType: listOf(STRING),
      stringConstants: ["yes"],
      maxCost: 3,
    },
  );

  assert.equal(outcome.kind, "synthesized");
  if (outcome.kind === "synthesized") {
    assert.equal(outcome.family, "filter");
    assert.equal(
      renderExpression(outcome.program),
      '(xs: list<string>) => filter((x: string) => (x == "yes"), xs)',
    );
  }
});

test("finds the minimum-cost composed predicate at its exact bound", () => {
  const outcome = synthesizeProgram(
    [{ input: ["a", "b", "c"], output: [false, false, true] }],
    {
      inputType: listOf(STRING),
      outputType: listOf(BOOL),
      stringConstants: ["a", "b"],
      maxCost: 5,
    },
  );

  assert.equal(outcome.kind, "synthesized");
  if (outcome.kind === "synthesized") {
    assert.equal(outcome.cost, 8);
    assert.equal(
      renderExpression(outcome.program),
      '(xs: list<string>) => map((x: string) => ((x == "a") == (x == "b")), xs)',
    );
  }
});

test("synthesizes bool disjunction and string concatenation folds", () => {
  const any = synthesizeProgram(
    [
      { input: [], output: false },
      { input: [false], output: false },
      { input: [false, true], output: true },
      { input: [true, false], output: true },
    ],
    {
      inputType: listOf(BOOL),
      outputType: BOOL,
      maxCost: 2,
    },
  );
  assert.equal(any.kind, "synthesized");
  if (any.kind === "synthesized") {
    assert.equal(
      renderExpression(any.program),
      "(xs: list<bool>) => foldl((acc: bool) => (x: bool) => (acc || x), false, xs)",
    );
  }

  const concatenate = synthesizeProgram(
    [
      { input: [], output: "" },
      { input: ["a"], output: "a" },
      { input: ["a", "b"], output: "ab" },
      { input: ["x", "y", "z"], output: "xyz" },
    ],
    {
      inputType: listOf(STRING),
      outputType: STRING,
      maxCost: 2,
    },
  );
  assert.equal(concatenate.kind, "synthesized");
  if (concatenate.kind === "synthesized") {
    assert.equal(
      renderExpression(concatenate.program),
      '(xs: list<string>) => foldl((acc: string) => (x: string) => (acc ++ x), "", xs)',
    );
  }
});

test("requires complete compatible explicit signatures", () => {
  assert.throws(
    () =>
      synthesizeProgram([{ input: ["x"], output: [true] }], {
        inputType: listOf(STRING),
      }),
    /provided together/,
  );
  assert.throws(
    () =>
      synthesizeProgram([{ input: ["x"], output: [1] }], {
        inputType: listOf(STRING),
        outputType: listOf(BOOL),
      }),
    /must match bool/,
  );
});

test("rejects runtime values that only resemble declared list types", () => {
  const signature = {
    inputType: listOf(STRING),
    outputType: listOf(BOOL),
  };

  for (const input of [7, "yes"]) {
    assert.throws(
      () =>
        synthesizeProgram([{ input, output: [true] } as any], signature),
      /example 1 input must match list<string>/,
    );
  }

  assert.throws(
    () =>
      synthesizeProgram(
        [{ input: ["yes"], output: "true" } as any],
        signature,
      ),
    /example 1 output must match list<bool>/,
  );

  assert.throws(
    () =>
      synthesizeProgram(
        [{ input: ["yes"], output: [true] } as any],
        { inputType: listOf(STRING), outputType: BOOL },
      ),
    /example 1 output must match bool/,
  );
});
