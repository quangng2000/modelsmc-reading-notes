import assert from "node:assert/strict";
import test from "node:test";

import {
  acceptProgram,
  intValue,
  type Example,
  type Expr,
  type Program,
} from "../src/core/language.verify.js";
import { CatalogProposer } from "../src/shell/catalog-proposer.js";
import {
  ConfigurationError,
  parseExperimentConfig,
  type ExperimentConfig,
} from "../src/shell/config.js";
import { AstDecodeError, decodeExpr, decodeProgram } from "../src/shell/decode.js";
import { deriveSynthesisTrace } from "../src/shell/deduction.js";
import { SynthesisEngine } from "../src/shell/engine.js";
import {
  effectiveSampleSize,
  normalizeLogWeights,
} from "../src/shell/numerics.js";
import type {
  ProposalContext,
  ProposalResult,
  Proposer,
} from "../src/shell/proposal.js";
import { OllamaProposer } from "../src/shell/ollama-proposer.js";
import { SeededRandom } from "../src/shell/random.js";
import { renderExpr, renderProgram } from "../src/shell/render.js";
import { systematicResample } from "../src/shell/resampling.js";
import { scoreProgram } from "../src/shell/scoring.js";

const twicePlusOne: Program = {
  kind: "ExpressionProgram",
  body: {
    kind: "Add",
    left: {
      kind: "Multiply",
      left: { kind: "IntLiteral", intValue: 2n },
      right: { kind: "Input" },
    },
    right: { kind: "IntLiteral", intValue: 1n },
  },
};

const affineExamples: Example[] = [
  { input: intValue(-2n), output: intValue(-3n) },
  { input: intValue(-1n), output: intValue(-1n) },
  { input: intValue(0n), output: intValue(1n) },
  { input: intValue(1n), output: intValue(3n) },
  { input: intValue(2n), output: intValue(5n) },
];

function affineConfig(overrides: Record<string, unknown> = {}): ExperimentConfig {
  return parseExperimentConfig(
    JSON.stringify({
      name: "test affine synthesis",
      examples: [
        { input: "-2", output: "-3" },
        { input: "-1", output: "-1" },
        { input: "0", output: "1" },
        { input: "1", output: "3" },
        { input: "2", output: "5" },
      ],
      integerConstants: ["0", "1", "2", "-1", "-2"],
      particles: 8,
      iterations: 6,
      cloneProbability: 0.25,
      essThreshold: 0.6,
      seed: 7,
      lossScale: 2,
      costScale: 0.15,
      lossCap: 1_000,
      maxCost: 12,
      maxDepth: 8,
      maxNodes: 63,
      ...overrides,
    }),
  );
}

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
    () =>
      decodeExpr(
        { kind: "IntLiteral", intValue: "3" },
        limits,
      ),
    (error: unknown) =>
      error instanceof AstDecodeError && /not in the allowed constant catalog/.test(error.message),
  );
});

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

test("normalizes log weights and computes ESS and systematic ancestors deterministically", () => {
  const normalized = normalizeLogWeights([Math.log(1), Math.log(2), Math.log(1)]);

  assert.equal(normalized.usedUniformFallback, false);
  assert.deepEqual(normalized.weights, [0.25, 0.5, 0.25]);
  assert.ok(Math.abs(effectiveSampleSize(normalized.weights) - 8 / 3) < 1e-12);
  assert.deepEqual(
    systematicResample(normalized.weights, { next: () => 0 }),
    [0, 1, 1],
  );

  const first = systematicResample(normalized.weights, new SeededRandom(41));
  const second = systematicResample(normalized.weights, new SeededRandom(41));
  assert.deepEqual(first, second);
});

test("the seeded catalog engine deterministically finds an exact affine program", async () => {
  const config = affineConfig();
  const first = await new SynthesisEngine({
    config,
    proposer: new CatalogProposer(),
  }).run();
  const second = await new SynthesisEngine({
    config,
    proposer: new CatalogProposer(),
  }).run();

  assert.equal(first.exact, true);
  assert.equal(first.best.score.totalLoss, 0);
  assert.equal(acceptProgram(first.best.expression, [...config.examples]), true);
  assert.equal(renderExpr(first.best.expression), "(1 + (2 * x))");
  assert.equal(renderProgram(first.best.expression, config.inputType), "λx: Int. (1 + (2 * x))");
  assert.equal(renderExpr(second.best.expression), renderExpr(first.best.expression));
  assert.equal(second.proposalCalls, first.proposalCalls);
  assert.deepEqual(second.iterations, first.iterations);
});

class ThrowingProposer implements Proposer {
  readonly name = "throwing-test-proposer";

  async propose(_context: ProposalContext): Promise<ProposalResult> {
    throw new Error("deliberate proposal failure");
  }
}

class IllTypedProposer implements Proposer {
  readonly name = "ill-typed-test-proposer";

  async propose(_context: ProposalContext): Promise<ProposalResult> {
    return {
      expression: {
        kind: "ExpressionProgram",
        body: { kind: "BoolLiteral", boolValue: true },
      },
      rationale: "deliberately wrong output type",
      source: "catalog",
    };
  }
}

class ExactAffineProposer implements Proposer {
  readonly name = "exact-affine-test-proposer";

  async propose(_context: ProposalContext): Promise<ProposalResult> {
    return {
      expression: twicePlusOne,
      rationale: "return the exact affine program",
      source: "catalog",
    };
  }
}

test("falls back to ancestors when a proposer throws", async () => {
  const config = affineConfig({
    particles: 3,
    iterations: 1,
    cloneProbability: 0,
  });
  const result = await new SynthesisEngine({
    config,
    proposer: new ThrowingProposer(),
  }).run();

  assert.equal(result.proposalCalls, config.particles);
  assert.ok(result.particles.every((particle) => particle.origin === "fallback"));
  assert.ok(
    result.particles.every((particle) =>
      particle.rationale.includes("proposal failed: deliberate proposal failure"),
    ),
  );
  assert.ok(result.particles.every((particle) => particle.parentId !== null));
});

test("falls back to ancestors when a proposer returns an ill-typed AST", async () => {
  const config = affineConfig({
    particles: 3,
    iterations: 1,
    cloneProbability: 0,
  });
  const result = await new SynthesisEngine({
    config,
    proposer: new IllTypedProposer(),
  }).run();

  assert.equal(result.proposalCalls, config.particles);
  assert.ok(result.particles.every((particle) => particle.origin === "fallback"));
  assert.ok(
    result.particles.every((particle) =>
      particle.rationale.includes("proposal rejected: output type mismatch"),
    ),
  );
  assert.ok(result.particles.every((particle) => particle.parentId !== null));
});

test("the champion archive retains an exact discovery even under an adverse soft objective", async () => {
  const config = affineConfig({
    particles: 1,
    iterations: 1,
    cloneProbability: 0,
    lossScale: 0.0001,
    costScale: 100,
  });
  const result = await new SynthesisEngine({
    config,
    proposer: new ExactAffineProposer(),
  }).run();

  assert.equal(result.exact, true);
  assert.deepEqual(result.best.expression, twicePlusOne);
  assert.ok(result.best.score.logTarget < -400);
});

function listConfig(
  name: string,
  output: "List<Int>" | "Int",
  examples: readonly { readonly input: readonly string[]; readonly output: readonly string[] | string }[],
): ExperimentConfig {
  return parseExperimentConfig(
    JSON.stringify({
      name,
      signature: { input: "List<Int>", output },
      examples,
      integerConstants: ["-2", "-1", "0", "1", "2", "3"],
      particles: 8,
      iterations: 7,
      cloneProbability: 0.25,
      essThreshold: 0.6,
      seed: 7,
      lossScale: 2,
      costScale: 0.15,
      lossCap: 1000,
      maxCost: 20,
      maxDepth: 10,
      maxNodes: 127,
    }),
  );
}

test("parses explicitly typed list examples and rejects ambiguous or mixed lists", () => {
  const config = listConfig("map successor", "List<Int>", [
    { input: [], output: [] },
    { input: ["1"], output: ["2"] },
    { input: ["-2", "0", "3"], output: ["-1", "1", "4"] },
  ]);

  assert.equal(config.inputType, "IntListType");
  assert.equal(config.outputType, "IntListType");
  assert.equal(renderProgram({
    kind: "MapProgram",
    mapper: { kind: "Item" },
  }, config.inputType), "λxs: List<Int>. map (λitem: Int. item) xs");

  assert.throws(
    () => parseExperimentConfig(JSON.stringify({ examples: [{ input: [], output: [] }] })),
    (error: unknown) =>
      error instanceof ConfigurationError && /signature must explicitly declare/.test(error.message),
  );
  assert.throws(
    () =>
      parseExperimentConfig(JSON.stringify({
        signature: { input: "List<Int>", output: "List<Int>" },
        examples: [{ input: ["1", true], output: ["2"] }],
      })),
    (error: unknown) => error instanceof ConfigurationError && /decimal integer string/.test(error.message),
  );
});

test("decodes bounded map and foldr program wrappers", () => {
  const limits = {
    integerConstants: [-1n, 0n, 1n, 2n],
    maxDepth: 5,
    maxNodes: 8,
  };
  const map = decodeProgram({
    kind: "MapProgram",
    mapper: {
      kind: "Add",
      left: { kind: "Item" },
      right: { kind: "IntLiteral", intValue: "1" },
    },
  }, limits);
  const fold = decodeProgram({
    kind: "FoldRightProgram",
    initial: { kind: "IntLiteral", intValue: "0" },
    reducer: {
      kind: "Add",
      left: { kind: "Item" },
      right: { kind: "Accumulator" },
    },
  }, limits);

  assert.equal(renderExpr(map), "map (λitem. (item + 1)) xs");
  assert.equal(renderExpr(fold), "foldr (λitem. λacc. (item + acc)) 0 xs");
  assert.throws(
    () => decodeProgram({ kind: "Item" }, limits),
    (error: unknown) =>
      error instanceof AstDecodeError && /complete program wrapper/.test(error.message),
  );
});

test("scores exact and near-miss list programs with element and length loss", () => {
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
  assert.ok(mapEvents.some((event) =>
    event.kind === "family.refuted" && /map: refuted.*contradictory/.test(event.message)
  ));

  const fold = listConfig("foldr sum", "Int", [
    { input: [], output: "0" },
    { input: ["3"], output: "3" },
    { input: ["2", "3"], output: "5" },
    { input: ["1", "2", "3"], output: "6" },
  ]);
  const foldEvents = deriveSynthesisTrace(fold.inputType, fold.outputType, fold.examples);
  const reducer = foldEvents.find((event) =>
    event.kind === "deduction.inferred" && event.data.hole === "reducer"
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

test("the Ollama adapter requests and decodes a complete bounded Program AST", async () => {
  const config = listConfig("map successor", "List<Int>", [
    { input: [], output: [] },
    { input: ["1"], output: ["2"] },
  ]);
  const ancestor: Program = {
    kind: "ExpressionProgram",
    body: { kind: "Input" },
  };
  const ancestorScore = scoreProgram(ancestor, {
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    lossScale: config.lossScale,
    costScale: config.costScale,
    lossCap: config.lossCap,
    maxCost: config.maxCost,
  });
  if (ancestorScore.kind !== "Scored") assert.fail(ancestorScore.reason);

  let requestBody = "";
  const requester: typeof fetch = async (_input, init) => {
    requestBody = String(init?.body ?? "");
    return new Response(JSON.stringify({
      choices: [{
        finish_reason: "stop",
        message: {
          content: JSON.stringify({
            expression: {
              kind: "MapProgram",
              mapper: {
                kind: "Add",
                left: { kind: "Item" },
                right: { kind: "IntLiteral", intValue: "1" },
              },
            },
            rationale: "increment each scoped item",
          }),
        },
      }],
    }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };
  const proposer = new OllamaProposer({ requester, timeoutMs: 1000 });
  const proposal = await proposer.propose({
    requestIndex: 0,
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    integerConstants: config.integerConstants,
    maxDepth: config.maxDepth,
    maxNodes: config.maxNodes,
    maxCost: config.maxCost,
    ancestor,
    ancestorScore,
  });

  assert.equal(proposal.expression.kind, "MapProgram");
  assert.equal(proposal.rationale, "increment each scoped item");
  assert.match(requestBody, /typed_program_proposal/);
  assert.match(requestBody, /FoldRightProgram/);
  assert.match(requestBody, /must not reference the outer Input/);
});
