import assert from "node:assert/strict";
import test from "node:test";

import { INT, listOf, renderExpression, type Expression } from "../src/ast.js";
import {
  evaluateExpression,
  expectInt,
} from "../src/evaluation/index.js";
import {
  synthesizeProgram,
  type SynthesisOutcome,
} from "../src/synthesis/index.js";

const SUM_EXAMPLES = [
  { input: [], output: 0 },
  { input: [1], output: 1 },
  { input: [1, 2], output: 3 },
  { input: [1, 2, 3], output: 6 },
] as const;

function runProgram(program: Expression, input: readonly number[]): number {
  assert.equal(program.kind, "lambda");
  if (program.kind !== "lambda") {
    throw new Error("unreachable: asserted above");
  }
  return expectInt(
    evaluateExpression(program.body, [
      { name: program.parameter, type: listOf(INT), value: input },
    ]),
  );
}

function assertSearchInvariants(outcome: SynthesisOutcome): void {
  if (outcome.kind !== "synthesized") {
    throw new Error("expected a synthesized outcome");
  }
  assert.ok(outcome.candidatesTested > 0);

  const popCosts = outcome.trace
    .filter(({ stage }) => stage === "completed-program")
    .map(({ cost }) => cost);
  assert.ok(popCosts.length > 0);
  for (let index = 1; index < popCosts.length; index += 1) {
    const prior = popCosts[index - 1];
    const current = popCosts[index];
    assert.ok(prior !== undefined && current !== undefined);
    assert.ok(prior <= current);
  }
}

test("synthesizes a filter program and reports the map refutation", () => {
  const outcome = synthesizeProgram(
    [{ input: [1, -2, 3, 0], output: [1, 3] }],
    { maxCost: 4 },
  );

  assert.equal(outcome.kind, "synthesized");
  if (outcome.kind !== "synthesized") {
    return;
  }

  assert.equal(outcome.family, "filter");
  assert.equal(
    renderExpression(outcome.program),
    "(xs: list<int>) => filter((x: int) => (0 < x), xs)",
  );
  assert.equal(outcome.cost, 5);
  assert.deepEqual(outcome.familyReports, [
    { family: "map", status: "refuted", reason: "map preserves list length" },
    { family: "filter", status: "viable", reason: undefined },
  ]);
  assertSearchInvariants(outcome);
});

test("synthesizes a sum fold with deduced init 0", () => {
  const outcome = synthesizeProgram([...SUM_EXAMPLES], { maxCost: 4 });

  assert.equal(outcome.kind, "synthesized");
  if (outcome.kind !== "synthesized") {
    return;
  }

  assert.equal(outcome.family, "fold");
  assert.equal(
    renderExpression(outcome.program),
    "(xs: list<int>) => foldl((acc: int) => (x: int) => (acc + x), 0, xs)",
  );
  // Total cost 6 = 2 (skeleton) + 2 (curried lambdas) + 1 (init literal)
  // + 1 (body (acc + x): "+" costs 1, variables cost 0). DESIGN.md's test
  // sketch claims 7 by pricing the body at 2, which contradicts its own
  // cost table.
  assert.equal(outcome.cost, 6);
  assert.deepEqual(outcome.familyReports, [
    { family: "fold", status: "viable", reason: undefined },
  ]);
  assertSearchInvariants(outcome);
});

test("map identity beats filter-true across families", () => {
  const outcome = synthesizeProgram(
    [
      { input: [1, 2], output: [1, 2] },
      { input: [5], output: [5] },
    ],
    { maxCost: 4 },
  );

  assert.equal(outcome.kind, "synthesized");
  if (outcome.kind !== "synthesized") {
    return;
  }

  // Map identity totals 3 (overhead 3 + body x at 0); filter-true would
  // total 4 (overhead 3 + literal true at 1), so map wins the shared frontier.
  assert.equal(outcome.family, "map");
  assert.equal(
    renderExpression(outcome.program),
    "(xs: list<int>) => map((x: int) => x, xs)",
  );
  assert.equal(outcome.cost, 3);
});

test("reproduces the map regression through the family engine", () => {
  const outcome = synthesizeProgram(
    [
      { input: [1, 2], output: [3, 4] },
      { input: [3], output: [5] },
    ],
    { maxCost: 4 },
  );

  assert.equal(outcome.kind, "synthesized");
  if (outcome.kind !== "synthesized") {
    return;
  }

  assert.equal(outcome.family, "map");
  assert.equal(
    renderExpression(outcome.program),
    "(xs: list<int>) => map((x: int) => (x + 2), xs)",
  );
  assert.equal(outcome.cost, 5);
});

test("throws when examples mix list and scalar outputs", () => {
  assert.throws(
    () =>
      synthesizeProgram([
        { input: [1], output: [1] },
        { input: [2], output: 2 },
      ]),
    /mix list and scalar outputs/,
  );
});

test("folds without an empty-list example by enumerating init constants", () => {
  const examples = [
    { input: [1], output: 2 },
    { input: [1, 2], output: 4 },
  ] as const;
  const outcome = synthesizeProgram([...examples], { maxCost: 4 });

  assert.equal(outcome.kind, "synthesized");
  if (outcome.kind !== "synthesized") {
    return;
  }

  assert.equal(outcome.family, "fold");
  for (const example of examples) {
    assert.equal(runProgram(outcome.program, example.input), example.output);
  }
  // Hand-computed lower bound: the fold overhead alone is 5 (skeleton 2,
  // curried lambdas 2, init literal 1), so any fold program costs >= 5.
  assert.ok(outcome.cost >= 5);
  assertSearchInvariants(outcome);
});
