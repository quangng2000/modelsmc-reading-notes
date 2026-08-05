import assert from "node:assert/strict";
import test from "node:test";

import {
  ConfigurationError,
  parseExperimentConfig,
} from "../src/shell/config/index.js";
import {
  AstDecodeError,
  decodeExpr,
  decodeProgram,
  renderExpr,
  renderProgram,
} from "../src/shell/ast/index.js";
import {
  affineConfig,
  affineExamples,
  listConfig,
} from "./support/fixtures.js";

test("parses a homogeneous config and rejects mixed example signatures", () => {
  const config = affineConfig();

  assert.equal(config.inputType, "IntType");
  assert.equal(config.outputType, "IntType");
  assert.deepEqual(config.integerConstants, [0n, 1n, 2n, -1n, -2n]);
  assert.deepEqual(config.examples, affineExamples);

  assert.throws(
    () =>
      parseExperimentConfig(
        JSON.stringify({
          examples: [
            { input: "0", output: "1" },
            { input: true, output: "2" },
          ],
        }),
      ),
    (error: unknown) =>
      error instanceof ConfigurationError &&
      /share one input type and one output type/.test(error.message),
  );
});

test("decodes only bounded ASTs with allowed integer constants", () => {
  const limits = {
    integerConstants: [-1n, 0n, 1n, 2n],
    maxDepth: 2,
    maxNodes: 3,
  };
  const decoded = decodeExpr(
    {
      kind: "Add",
      left: { kind: "Input" },
      right: { kind: "IntLiteral", intValue: "1" },
    },
    limits,
  );

  assert.deepEqual(decoded, {
    kind: "Add",
    left: { kind: "Input" },
    right: { kind: "IntLiteral", intValue: 1n },
  });
  assert.throws(
    () =>
      decodeExpr(
        {
          kind: "Not",
          operand: { kind: "Not", operand: { kind: "Input" } },
        },
        limits,
      ),
    (error: unknown) =>
      error instanceof AstDecodeError && /maximum AST depth 2/.test(error.message),
  );
  assert.throws(
    () =>
      decodeExpr(
        {
          kind: "Add",
          left: { kind: "Input" },
          right: { kind: "IntLiteral", intValue: "1" },
        },
        { ...limits, maxNodes: 2 },
      ),
    (error: unknown) =>
      error instanceof AstDecodeError && /maximum node count 2/.test(error.message),
  );
  assert.throws(
    () => decodeExpr({ kind: "IntLiteral", intValue: "3" }, limits),
    (error: unknown) =>
      error instanceof AstDecodeError &&
      /not in the allowed constant catalog/.test(error.message),
  );
});

test("parses explicitly typed list examples and rejects ambiguous or mixed lists", () => {
  const config = listConfig("map successor", "List<Int>", [
    { input: [], output: [] },
    { input: ["1"], output: ["2"] },
    { input: ["-2", "0", "3"], output: ["-1", "1", "4"] },
  ]);

  assert.equal(config.inputType, "IntListType");
  assert.equal(config.outputType, "IntListType");
  assert.equal(
    renderProgram({ kind: "MapProgram", mapper: { kind: "Item" } }, config.inputType),
    "λxs: List<Int>. map (λitem: Int. item) xs",
  );

  assert.throws(
    () => parseExperimentConfig(JSON.stringify({ examples: [{ input: [], output: [] }] })),
    (error: unknown) =>
      error instanceof ConfigurationError && /signature must explicitly declare/.test(error.message),
  );
  assert.throws(
    () =>
      parseExperimentConfig(
        JSON.stringify({
          signature: { input: "List<Int>", output: "List<Int>" },
          examples: [{ input: ["1", true], output: ["2"] }],
        }),
      ),
    (error: unknown) =>
      error instanceof ConfigurationError && /decimal integer string/.test(error.message),
  );
});

test("decodes bounded map and foldr program wrappers", () => {
  const limits = {
    integerConstants: [-1n, 0n, 1n, 2n],
    maxDepth: 5,
    maxNodes: 8,
  };
  const map = decodeProgram(
    {
      kind: "MapProgram",
      mapper: {
        kind: "Add",
        left: { kind: "Item" },
        right: { kind: "IntLiteral", intValue: "1" },
      },
    },
    limits,
  );
  const fold = decodeProgram(
    {
      kind: "FoldRightProgram",
      initial: { kind: "IntLiteral", intValue: "0" },
      reducer: {
        kind: "Add",
        left: { kind: "Item" },
        right: { kind: "Accumulator" },
      },
    },
    limits,
  );

  assert.equal(renderExpr(map), "map (λitem. (item + 1)) xs");
  assert.equal(renderExpr(fold), "foldr (λitem. λacc. (item + acc)) 0 xs");
  assert.throws(
    () => decodeProgram({ kind: "Item" }, limits),
    (error: unknown) =>
      error instanceof AstDecodeError && /complete program wrapper/.test(error.message),
  );
});
