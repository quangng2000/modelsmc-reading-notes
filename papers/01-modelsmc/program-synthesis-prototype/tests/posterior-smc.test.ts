import assert from "node:assert/strict";
import test from "node:test";

import { intValue } from "../src/core/language.verify.js";
import { jsonStringify, programToJsonValue, renderProgram } from "../src/shell/ast/render.js";
import { SeededRandom } from "../src/shell/engine/random.js";
import { enumeratePrograms, logSumExp } from "../src/shell/prior/index.js";
import { scoreCalibrated } from "../src/shell/scoring/calibrated.js";
import { DEFAULT_NOISE } from "../src/shell/scoring/emission.js";
import { evaluateDraw, type ProposalTask, type RawDraw } from "../src/shell/posterior/llm-proposal.js";
import { linearGammas, runCalibratedSmc } from "../src/shell/posterior/smc.js";

/**
 * End-to-end mathematical validation of the calibrated tempered SMC on a task
 * whose posterior is exactly enumerable, with the LLM island driven by a MOCK
 * proposal of known distribution over the FULL support (plus deliberate
 * garbage to exercise acceptance-conditioning). Both islands' weighted
 * populations must converge to the exact posterior.
 */

const TASK: ProposalTask = {
  inputType: "IntType",
  outputType: "IntType",
  examples: [
    { input: intValue(1n), output: intValue(2n) },
    { input: intValue(2n), output: intValue(3n) },
    { input: intValue(-3n), output: intValue(-2n) },
  ],
  integerConstants: [0n, 1n],
  maxDepth: 12,
  maxNodes: 191,
  costCap: 3,
  beta: 0.15,
  noise: DEFAULT_NOISE,
};

interface ExactAtom {
  readonly rendered: string;
  readonly canonical: string;
  readonly probability: number;
  readonly logLikelihood: number;
}

function exactPosterior(): ExactAtom[] {
  const scored: { rendered: string; canonical: string; logPost: number; logLik: number }[] = [];
  for (const { program } of enumeratePrograms(TASK.inputType, TASK.outputType, TASK.costCap, TASK.integerConstants)) {
    const score = scoreCalibrated(program, {
      inputType: TASK.inputType,
      outputType: TASK.outputType,
      examples: TASK.examples,
      beta: TASK.beta,
      noise: TASK.noise,
    });
    if ("rejected" in score) throw new Error(score.rejected);
    scored.push({
      rendered: renderProgram(program, TASK.inputType),
      canonical: jsonStringify(programToJsonValue(program)),
      logPost: score.logPosteriorUnnormalized,
      logLik: score.logLikelihood,
    });
  }
  const logZ = logSumExp(scored.map((entry) => entry.logPost));
  return scored.map((entry) => ({
    rendered: entry.rendered,
    canonical: entry.canonical,
    probability: Math.exp(entry.logPost - logZ),
    logLikelihood: entry.logLik,
  }));
}

function estimateByRendering(
  particles: readonly { proposal: { program: unknown } }[],
  weights: readonly number[],
): Map<string, number> {
  const masses = new Map<string, number>();
  particles.forEach((particle, index) => {
    const rendered = renderProgram(particle.proposal.program as never, TASK.inputType);
    masses.set(rendered, (masses.get(rendered) ?? 0) + weights[index]!);
  });
  return masses;
}

function totalVariation(estimate: Map<string, number>, atoms: readonly ExactAtom[]): number {
  let tv = 0;
  for (const atom of atoms) tv += Math.abs((estimate.get(atom.rendered) ?? 0) - atom.probability);
  return tv / 2;
}

test("the support is small and contains exact solvers", () => {
  const atoms = exactPosterior();
  assert.ok(atoms.length >= 20 && atoms.length <= 60, `support size ${atoms.length}`);
  const solverMass = atoms
    .filter((atom) => atom.logLikelihood === Math.max(...atoms.map((entry) => entry.logLikelihood)))
    .reduce((sum, atom) => sum + atom.probability, 0);
  assert.ok(solverMass > 0.5, `top-likelihood mass ${solverMass}`);
});

test("prior-island tempered SMC converges to the exact posterior", async () => {
  const atoms = exactPosterior();
  const result = await runCalibratedSmc({
    task: TASK,
    particleCount: 600,
    gammas: linearGammas(5),
    moves: "prior",
    sweepsPerStage: 2,
    essThresholdRatio: 0.5,
    rng: new SeededRandom(11),
  });
  assert.equal(result.llmDraws, 0);
  const tv = totalVariation(estimateByRendering(result.particles, result.normalizedWeights), atoms);
  assert.ok(tv < 0.12, `prior island TV ${tv}`);
});

test("llm-island tempered SMC with a known mock proposal converges to the exact posterior", async () => {
  const atoms = exactPosterior();
  // Mock proposal: full support, deliberately skewed AWAY from the posterior
  // (mass proportional to 1/(rank+2)) plus 20% garbage text.
  const skewed = [...atoms].sort((left, right) => left.probability - right.probability);
  const rawWeights = skewed.map((_atom, index) => 1 / (index + 2));
  const totalWeight = rawWeights.reduce((sum, weight) => sum + weight, 0);
  const mockProbs = rawWeights.map((weight) => (0.8 * weight) / totalWeight);
  const garbageProb = 0.2;
  const mockRng = new SeededRandom(1234);
  const drawer = async (): Promise<RawDraw> => {
    let draw = mockRng.next();
    if (draw < garbageProb) return { text: "not json at all", sumLogProb: Math.log(garbageProb), doneReason: "stop" };
    draw -= garbageProb;
    for (let index = 0; index < skewed.length; index += 1) {
      draw -= mockProbs[index]!;
      if (draw <= 0) {
        return {
          text: skewed[index]!.canonical,
          sumLogProb: Math.log(mockProbs[index]!),
          doneReason: "stop",
        };
      }
    }
    return { text: skewed[skewed.length - 1]!.canonical, sumLogProb: Math.log(mockProbs[skewed.length - 1]!), doneReason: "stop" };
  };

  // Sanity: the mock draws must decode and be accepted.
  const probe = await drawer();
  if (probe.text !== "not json at all") {
    const outcome = evaluateDraw(probe, TASK);
    assert.ok(!("rejected" in outcome), "mock canonical text must be accepted");
  }

  const result = await runCalibratedSmc({
    task: TASK,
    particleCount: 400,
    gammas: linearGammas(4),
    moves: "llm",
    sweepsPerStage: 2,
    essThresholdRatio: 0.5,
    rng: new SeededRandom(77),
    drawer,
  });
  assert.ok(result.llmDraws > 400, "llm island must consume draws");
  const estimate = estimateByRendering(result.particles, result.normalizedWeights);
  const tv = totalVariation(estimate, atoms);
  assert.ok(tv < 0.12, `llm island TV ${tv}`);

  // The estimator must CORRECT the deliberate skew: the top posterior atom is
  // among the LEAST proposed, yet its estimated mass must approach truth.
  const top = [...atoms].sort((left, right) => right.probability - left.probability)[0]!;
  const estimated = estimate.get(top.rendered) ?? 0;
  assert.ok(
    Math.abs(estimated - top.probability) < 0.1,
    `top atom: estimated ${estimated} vs true ${top.probability}`,
  );
});

test("gamma schedules are validated", () => {
  assert.deepEqual(linearGammas(4), [0.25, 0.5, 0.75, 1]);
  assert.throws(() => linearGammas(0));
});
