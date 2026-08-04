import assert from "node:assert/strict";
import test from "node:test";

import { BOOL, STRING } from "../src/ast.js";
import {
  deduceFilterExamples,
  deduceFoldExamples,
  deduceMapExamples,
  type FoldDeductionResult,
  type MapDeductionResult,
  type PredicateExample,
  type ScalarExample,
} from "../src/deduction/index.js";
import {
  synthesizeMap,
  type MapSynthesisResult,
} from "../src/synthesis/index.js";

function expectType<T>(_value: T): void {}

test("legacy deduction APIs retain numeric result types", () => {
  const map = deduceMapExamples([{ input: [1], output: [2] }]);
  expectType<MapDeductionResult>(map);
  if (map.kind === "inferred") {
    expectType<readonly ScalarExample[]>(map.examples);
    expectType<number>(map.examples[0]?.input ?? 0);
    expectType<number>(map.examples[0]?.output ?? 0);
  }

  const filter = deduceFilterExamples([{ input: [1, 2], output: [2] }]);
  if (filter.kind === "inferred") {
    expectType<readonly PredicateExample[]>(filter.examples);
    expectType<number>(filter.examples[0]?.input ?? 0);
  }

  const fold = deduceFoldExamples([{ input: [], output: 0 }]);
  expectType<FoldDeductionResult>(fold);
  if (fold.kind === "inferred") {
    expectType<number | undefined>(fold.init);
    expectType<number | undefined>(fold.steps[0]?.element);
    expectType<number | undefined>(fold.steps[0]?.accumulator);
  }
});

test("typed deduction APIs retain explicit primitive result types", () => {
  const map = deduceMapExamples<string, boolean>(
    [{ input: ["yes", "no"], output: [true, false] }],
    STRING,
    BOOL,
  );
  if (map.kind === "inferred") {
    expectType<readonly ScalarExample<string, boolean>[]>(map.examples);
    expectType<string>(map.examples[0]?.input ?? "");
    expectType<boolean>(map.examples[0]?.output ?? false);
  }

  const filter = deduceFilterExamples<string>(
    [{ input: ["yes", "no"], output: ["yes"] }],
    STRING,
  );
  if (filter.kind === "inferred") {
    expectType<readonly PredicateExample<string>[]>(filter.examples);
    expectType<string>(filter.examples[0]?.input ?? "");
  }

  const fold = deduceFoldExamples<boolean, string>(
    [{ input: [], output: "" }],
    BOOL,
    STRING,
  );
  if (fold.kind === "inferred") {
    expectType<string | undefined>(fold.init);
    expectType<boolean | undefined>(fold.steps[0]?.element);
    expectType<string | undefined>(fold.steps[0]?.accumulator);
  }

  assert.equal(map.kind, "inferred");
  assert.equal(filter.kind, "inferred");
  assert.equal(fold.kind, "inferred");
});

test("legacy synthesizeMap exposes numeric inferred examples", () => {
  const result = synthesizeMap([{ input: [1], output: [2] }], {
    maxCost: 2,
  });
  expectType<MapSynthesisResult>(result);
  if (result.kind === "synthesized") {
    expectType<readonly ScalarExample[]>(result.inferredExamples);
    expectType<number>(result.inferredExamples[0]?.input ?? 0);
    expectType<number>(result.inferredExamples[0]?.output ?? 0);
  }
});
