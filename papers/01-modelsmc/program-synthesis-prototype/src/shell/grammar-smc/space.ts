import { jsonStringify } from "../ast/index.js";
import type { ExperimentConfig } from "../config/index.js";
import { normalizeLogWeights } from "../engine/numerics.js";
import { scoreProgram } from "../scoring/index.js";
import { enumeratePrograms } from "./enumerate.js";
import { logSumExp } from "./math.js";
import type { GrammarSpace, GrammarState, GrammarTarget } from "./types.js";

export function buildGrammarSpace(
  config: ExperimentConfig,
  maxCost: number,
  generationLimit: number,
): GrammarSpace {
  const programs = enumeratePrograms(config, maxCost, generationLimit);
  const scored = programs.map((program) => {
    const score = scoreProgram(program, { ...config, maxCost });
    if (score.kind === "Rejected") {
      throw new Error(`enumerator produced a rejected program: ${score.reason}`);
    }
    return { program, score, logPriorWeight: -config.costScale * score.cost };
  });
  const priorLogNormalizer = logSumExp(scored.map((state) => state.logPriorWeight));
  const states: GrammarState[] = scored.map((state, index) => ({
    index,
    key: jsonStringify(state.program),
    program: state.program,
    score: state.score,
    logPriorProbability: state.logPriorWeight - priorLogNormalizer,
    priorProbability: Math.exp(state.logPriorWeight - priorLogNormalizer),
  }));
  return { states, maxCost, priorLogNormalizer };
}

export function exactGrammarTarget(
  space: GrammarSpace,
  beta: number,
  lossScale: number,
): GrammarTarget {
  if (!Number.isFinite(beta) || beta < 0) throw new Error("beta must be finite and nonnegative");
  const logMasses = space.states.map(
    (state) => state.logPriorProbability - beta * lossScale * state.score.totalLoss,
  );
  const logNormalizingConstant = logSumExp(logMasses);
  const probabilities = normalizeLogWeights(logMasses).weights;
  const exactProgramMass = probabilities.reduce(
    (sum, probability, index) => sum + (space.states[index]!.score.exactProgram ? probability : 0),
    0,
  );
  const meanLoss = probabilities.reduce(
    (sum, probability, index) => sum + probability * space.states[index]!.score.totalLoss,
    0,
  );
  return { beta, probabilities, logNormalizingConstant, exactProgramMass, meanLoss };
}
