import assert from "node:assert/strict";
import test from "node:test";

import {
  acceptProgram,
  type Program,
} from "../src/core/language.verify.js";
import { CatalogProposer } from "../src/shell/catalog/index.js";
import {
  parseExperimentConfig,
  type ExperimentConfig,
} from "../src/shell/config/index.js";
import { SynthesisEngine } from "../src/shell/engine/index.js";
import type {
  ProposalContext,
  ProposalResult,
  Proposer,
} from "../src/shell/proposal/index.js";
import { renderExpr, renderProgram } from "../src/shell/ast/index.js";
import { affineConfig, twicePlusOne } from "./support/fixtures.js";

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
  assert.equal(
    renderProgram(first.best.expression, config.inputType),
    "λx: Int. (1 + (2 * x))",
  );
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

const squareAllItems: Program = {
  kind: "FoldRightProgram",
  initial: { kind: "EmptyIntList" },
  reducer: {
    kind: "PrependInt",
    head: {
      kind: "Multiply",
      left: { kind: "Item" },
      right: { kind: "Item" },
    },
    tail: { kind: "Accumulator" },
  },
};

const squareAboveLowerBound: Program = {
  kind: "FoldRightProgram",
  initial: { kind: "EmptyIntList" },
  reducer: {
    kind: "IfThenElse",
    condition: {
      kind: "LessThan",
      left: { kind: "IntLiteral", intValue: -2n },
      right: { kind: "Item" },
    },
    thenExpr: {
      kind: "PrependInt",
      head: {
        kind: "Multiply",
        left: { kind: "Item" },
        right: { kind: "Item" },
      },
      tail: { kind: "Accumulator" },
    },
    elseExpr: { kind: "Accumulator" },
  },
};

const boundedSquare: Program = {
  kind: "FoldRightProgram",
  initial: { kind: "EmptyIntList" },
  reducer: {
    kind: "IfThenElse",
    condition: {
      kind: "And",
      left: {
        kind: "LessThan",
        left: { kind: "IntLiteral", intValue: -2n },
        right: { kind: "Item" },
      },
      right: {
        kind: "LessThan",
        left: { kind: "Item" },
        right: { kind: "IntLiteral", intValue: 3n },
      },
    },
    thenExpr: {
      kind: "PrependInt",
      head: {
        kind: "Multiply",
        left: { kind: "Item" },
        right: { kind: "Item" },
      },
      tail: { kind: "Accumulator" },
    },
    elseExpr: { kind: "Accumulator" },
  },
};

function boundedSquareConfig(overrides: Record<string, unknown>): ExperimentConfig {
  return parseExperimentConfig(
    JSON.stringify({
      name: "hidden bounded list transformation",
      signature: { input: "List<Int>", output: "List<Int>" },
      examples: [
        { input: [], output: [] },
        { input: ["-3"], output: [] },
        { input: ["-1"], output: ["1"] },
        { input: ["0"], output: ["0"] },
        { input: ["1"], output: ["1"] },
        { input: ["2"], output: ["4"] },
        { input: ["4"], output: [] },
        { input: ["3", "4"], output: [] },
        { input: ["2", "3", "4"], output: ["4"] },
        { input: ["1", "2", "3", "4"], output: ["1", "4"] },
        {
          input: ["-1", "0", "1", "2", "3", "4"],
          output: ["1", "0", "1", "4"],
        },
        {
          input: ["-3", "-2", "-1", "0", "1", "2", "3", "4"],
          output: ["1", "0", "1", "4"],
        },
      ],
      integerConstants: ["-3", "-2", "-1", "0", "1", "2", "3", "4"],
      particles: 2,
      iterations: 2,
      cloneProbability: 0,
      essThreshold: 1,
      seed: 17,
      lossScale: 0.75,
      costScale: 0.1,
      lossCap: 1000,
      maxCost: 24,
      maxDepth: 12,
      maxNodes: 191,
      ...overrides,
    }),
  );
}

class TwoStepBoundedSquareProposer implements Proposer {
  readonly name = "two-step-bounded-square-test-proposer";

  async propose(context: ProposalContext): Promise<ProposalResult> {
    if (renderExpr(context.ancestor) === renderExpr(squareAboveLowerBound)) {
      return {
        expression: boundedSquare,
        rationale: "add the missing upper bound after feedback exposes extra high values",
        source: "catalog",
      };
    }
    return {
      expression: (context.slot ?? 0) % 2 === 0 ? squareAboveLowerBound : squareAllItems,
      rationale: "first-stage partial bounded transformation",
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

test("iterative particle feedback can solve a task that equal-budget one-shot proposals cannot", async () => {
  const proposer = new TwoStepBoundedSquareProposer();
  const oneShot = await new SynthesisEngine({
    config: boundedSquareConfig({ particles: 4, iterations: 1 }),
    proposer,
  }).run();
  const iterative = await new SynthesisEngine({
    config: boundedSquareConfig({ particles: 2, iterations: 2 }),
    proposer,
  }).run();

  assert.equal(oneShot.proposalCalls, 4);
  assert.equal(iterative.proposalCalls, 4);
  assert.equal(oneShot.exact, false);
  assert.equal(iterative.exact, true);
  assert.equal(iterative.firstExactIteration, 2);
  assert.equal(iterative.firstExactProposalCall, 3);
  assert.deepEqual(iterative.best.expression, boundedSquare);
  assert.equal(iterative.championLineage.length, 3);
  const lineageLosses = iterative.championLineage.map(
    (particle) => particle.score.totalLoss,
  );
  assert.ok(
    lineageLosses[0]! > lineageLosses[1]!,
    `expected first refinement to reduce loss, got ${lineageLosses.join(" -> ")}`,
  );
  assert.ok(
    lineageLosses[1]! > lineageLosses[2]!,
    `expected second refinement to reduce loss, got ${lineageLosses.join(" -> ")}`,
  );
});
