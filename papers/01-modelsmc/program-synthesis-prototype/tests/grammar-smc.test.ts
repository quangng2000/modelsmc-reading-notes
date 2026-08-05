import assert from "node:assert/strict";
import test from "node:test";

import { parseExperimentConfig } from "../src/shell/config/index.js";
import { NullTraceSink } from "../src/shell/engine/index.js";
import {
  buildGrammarSpace,
  exactGrammarTarget,
  runGrammarSmc,
} from "../src/shell/grammar-smc/index.js";

function incrementConfig(particles = 1_024) {
  return parseExperimentConfig(
    JSON.stringify({
      name: "finite grammar increment control",
      signature: { input: "Int", output: "Int" },
      examples: [
        { input: "0", output: "1" },
        { input: "1", output: "2" },
        { input: "2", output: "3" },
      ],
      integerConstants: ["0", "1"],
      particles,
      iterations: 6,
      essThreshold: 0.8,
      seed: 17,
      lossScale: 2,
      costScale: 0.15,
      maxCost: 10,
    }),
  );
}

test("bounded enumeration produces only scored typed programs and includes x + 1", () => {
  const config = incrementConfig(32);
  const space = buildGrammarSpace(config, 3, 10_000);
  assert.equal(space.states.length, 30);
  assert.ok(space.states.every((state) => state.score.inferredType === "IntType"));
  assert.ok(space.states.every((state) => state.score.cost <= 3));
  assert.ok(
    space.states.some(
      (state) =>
        state.program.kind === "ExpressionProgram" &&
        state.program.body.kind === "Add" &&
        state.score.exactProgram,
    ),
  );
  assert.ok(
    Math.abs(space.states.reduce((sum, state) => sum + state.priorProbability, 0) - 1) < 1e-12,
  );
});

test("exact enumeration normalizes the tempered target", () => {
  const config = incrementConfig(32);
  const space = buildGrammarSpace(config, 3, 10_000);
  const priorTarget = exactGrammarTarget(space, 0, config.lossScale);
  assert.ok(Math.abs(priorTarget.logNormalizingConstant) < 1e-12);
  for (let index = 0; index < space.states.length; index += 1) {
    assert.ok(Math.abs(priorTarget.probabilities[index]! - space.states[index]!.priorProbability) < 1e-12);
  }

  const target = exactGrammarTarget(space, 1, config.lossScale);
  assert.ok(Math.abs(target.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.ok(target.logNormalizingConstant <= 0);
  assert.ok(target.exactProgramMass > 0.9);
  assert.ok(target.meanLoss < 0.1);
});

test("the enumeration safety limit fails before an accidental grammar explosion", () => {
  assert.throws(
    () => buildGrammarSpace(incrementConfig(32), 3, 1),
    /grammar enumeration exceeded --grammar-limit 1/,
  );
});

test("grammar SMC is deterministic by seed and tracks the exact finite target", () => {
  const config = incrementConfig();
  const options = {
    config,
    maxCost: 3,
    generationLimit: 10_000,
    betaMax: 1,
    movesPerStage: 1,
    trace: new NullTraceSink(),
  };
  const first = runGrammarSmc(options);
  const second = runGrammarSmc(options);

  assert.deepEqual(first.particles, second.particles);
  assert.deepEqual(first.stages, second.stages);
  assert.equal(first.best.score.exactProgram, true);
  assert.ok(Math.abs(first.exactProgramMassEstimate - first.reference.exactProgramMass) < 0.03);
  assert.ok(Math.abs(first.meanLossEstimate - first.reference.meanLoss) < 0.03);
  assert.ok(Math.abs(first.logNormalizingConstantError) < 0.25);
  assert.ok(first.totalVariationDistance < 0.05);
});
