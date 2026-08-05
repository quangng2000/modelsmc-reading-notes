import assert from "node:assert/strict";
import test from "node:test";

import {
  acceptProgram,
  type Program,
} from "../src/core/language.verify.js";
import { CatalogProposer } from "../src/shell/catalog/index.js";
import { deriveSynthesisTrace } from "../src/shell/deduction/index.js";
import { SynthesisEngine } from "../src/shell/engine/index.js";
import { renderProgram } from "../src/shell/ast/index.js";
import { scoreProgram } from "../src/shell/scoring/index.js";
import {
  affineConfig,
  affineExamples,
  listConfig,
  twicePlusOne,
} from "./support/fixtures.js";

test("scoring rejects ill-typed ASTs and exactly scores a matching program", () => {
  const config = affineConfig();
  const options = {
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    lossScale: config.lossScale,
    costScale: config.costScale,
    lossCap: config.lossCap,
    maxCost: config.maxCost,
  };
  const illTyped: Program = {
    kind: "ExpressionProgram",
    body: {
      kind: "Add",
      left: { kind: "BoolLiteral", boolValue: true },
      right: { kind: "Input" },
    },
  };

  const rejected = scoreProgram(illTyped, options);
  if (rejected.kind !== "Rejected") assert.fail("ill-typed AST was unexpectedly scored");
  assert.match(rejected.reason, /verified type checker rejected/);

  const exact = scoreProgram(twicePlusOne, options);
  if (exact.kind !== "Scored") assert.fail(`exact AST was rejected: ${exact.reason}`);
  assert.equal(exact.inferredType, "IntType");
  assert.equal(exact.totalLoss, 0);
  assert.equal(exact.exactMatches, affineExamples.length);
  assert.equal(exact.cost, 5);
  assert.equal(exact.logTarget, -0.75);
  assert.equal(exact.exactProgram, true);
  assert.ok(exact.evaluations.every((evaluation) => evaluation.exact));
});

test("scores exact and near-miss list programs with sequence edit loss", () => {
  const config = listConfig("map successor", "List<Int>", [
    { input: [], output: [] },
    { input: ["1", "2"], output: ["2", "3"] },
  ]);
  const increment: Program = {
    kind: "MapProgram",
    mapper: {
      kind: "Add",
      left: { kind: "Item" },
      right: { kind: "IntLiteral", intValue: 1n },
    },
  };
  const identity: Program = {
    kind: "MapProgram",
    mapper: { kind: "Item" },
  };
  const options = {
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    lossScale: config.lossScale,
    costScale: config.costScale,
    lossCap: config.lossCap,
    maxCost: config.maxCost,
  };

  const exact = scoreProgram(increment, options);
  const near = scoreProgram(identity, options);
  assert.equal(exact.kind, "Scored");
  assert.equal(near.kind, "Scored");
  if (exact.kind === "Scored" && near.kind === "Scored") {
    assert.equal(exact.totalLoss, 0);
    assert.equal(exact.exactProgram, true);
    assert.equal(near.totalLoss, 2);
    assert.equal(near.exactProgram, false);
  }

  const shiftedConfig = listConfig("extra prefix", "List<Int>", [
    { input: [], output: ["1", "2"] },
  ]);
  const extraPrefix: Program = {
    kind: "ExpressionProgram",
    body: {
      kind: "PrependInt",
      head: { kind: "IntLiteral", intValue: 9n },
      tail: {
        kind: "PrependInt",
        head: { kind: "IntLiteral", intValue: 1n },
        tail: {
          kind: "PrependInt",
          head: { kind: "IntLiteral", intValue: 2n },
          tail: { kind: "EmptyIntList" },
        },
      },
    },
  };
  const shifted = scoreProgram(extraPrefix, {
    inputType: shiftedConfig.inputType,
    outputType: shiftedConfig.outputType,
    examples: shiftedConfig.examples,
    lossScale: shiftedConfig.lossScale,
    costScale: shiftedConfig.costScale,
    lossCap: shiftedConfig.lossCap,
    maxCost: shiftedConfig.maxCost,
  });
  assert.equal(shifted.kind, "Scored");
  if (shifted.kind === "Scored") assert.equal(shifted.totalLoss, 1);
});

test("deduction explains map refutation and foldr suffix subexamples", () => {
  const contradictoryMap = listConfig("contradictory map", "List<Int>", [
    { input: ["1", "1"], output: ["2", "3"] },
  ]);
  const mapEvents = deriveSynthesisTrace(
    contradictoryMap.inputType,
    contradictoryMap.outputType,
    contradictoryMap.examples,
  );
  assert.ok(
    mapEvents.some(
      (event) =>
        event.kind === "family.refuted" &&
        /map: refuted.*contradictory/.test(event.message),
    ),
  );

  const fold = listConfig("foldr sum", "Int", [
    { input: [], output: "0" },
    { input: ["3"], output: "3" },
    { input: ["2", "3"], output: "5" },
    { input: ["1", "2", "3"], output: "6" },
  ]);
  const foldEvents = deriveSynthesisTrace(fold.inputType, fold.outputType, fold.examples);
  const reducer = foldEvents.find(
    (event) => event.kind === "deduction.inferred" && event.data.hole === "reducer",
  );
  assert.ok(reducer !== undefined);
  assert.match(reducer.message, /\(3, 0\) -> 3/);
  assert.match(reducer.message, /\(2, 3\) -> 5/);
  assert.match(reducer.message, /\(1, 5\) -> 6/);
});

test("the catalog SMC deterministically synthesizes map and foldr programs", async () => {
  const mapConfig = listConfig("map successor", "List<Int>", [
    { input: [], output: [] },
    { input: ["1"], output: ["2"] },
    { input: ["1", "2"], output: ["2", "3"] },
    { input: ["-2", "0", "3"], output: ["-1", "1", "4"] },
  ]);
  const foldConfig = listConfig("foldr sum", "Int", [
    { input: [], output: "0" },
    { input: ["3"], output: "3" },
    { input: ["2", "3"], output: "5" },
    { input: ["1", "2", "3"], output: "6" },
  ]);

  const map = await new SynthesisEngine({
    config: mapConfig,
    proposer: new CatalogProposer(),
  }).run();
  const fold = await new SynthesisEngine({
    config: foldConfig,
    proposer: new CatalogProposer(),
  }).run();

  assert.equal(map.exact, true);
  assert.equal(map.best.expression.kind, "MapProgram");
  assert.equal(acceptProgram(map.best.expression, [...mapConfig.examples]), true);
  assert.match(renderProgram(map.best.expression, mapConfig.inputType), /map/);
  assert.equal(fold.exact, true);
  assert.equal(fold.best.expression.kind, "FoldRightProgram");
  assert.equal(acceptProgram(fold.best.expression, [...foldConfig.examples]), true);
  assert.match(renderProgram(fold.best.expression, foldConfig.inputType), /foldr/);
});
