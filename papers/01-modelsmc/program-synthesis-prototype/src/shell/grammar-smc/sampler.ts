import { effectiveSampleSize, normalizeLogWeights } from "../engine/numerics.js";
import { SeededRandom } from "../engine/random.js";
import { systematicResample } from "../engine/resampling.js";
import { cumulativeProbabilities, logSumExp, sampleCategorical, totalVariation } from "./math.js";
import { buildGrammarSpace, exactGrammarTarget } from "./space.js";
import type {
  GrammarSmcOptions,
  GrammarSmcParticle,
  GrammarSmcResult,
  GrammarSmcStage,
} from "./types.js";

export function runGrammarSmc(options: GrammarSmcOptions): GrammarSmcResult {
  const { config } = options;
  if (!Number.isFinite(options.betaMax) || options.betaMax <= 0) {
    throw new Error("beta max must be finite and positive");
  }
  if (!Number.isSafeInteger(options.movesPerStage) || options.movesPerStage < 0) {
    throw new Error("moves per stage must be a nonnegative integer");
  }

  const space = buildGrammarSpace(config, options.maxCost, options.generationLimit);
  const reference = exactGrammarTarget(space, options.betaMax, config.lossScale);
  const priorCumulative = cumulativeProbabilities(space.states.map((state) => state.priorProbability));
  const random = new SeededRandom(config.seed);
  let stateIndices = Array.from({ length: config.particles }, () =>
    sampleCategorical(priorCumulative, random),
  );
  let weights = stateIndices.map(() => 1 / config.particles);
  let betaPrevious = 0;
  let logNormalizingConstantEstimate = 0;
  const stages: GrammarSmcStage[] = [];

  options.trace.emit(
    "grammar-space",
    `exact grammar contains ${space.states.length} programs through cost ${options.maxCost}; initial particles sampled from normalized Occam prior`,
    { states: space.states.length, maxCost: options.maxCost },
  );

  for (let stage = 1; stage <= config.iterations; stage += 1) {
    const beta = (options.betaMax * stage) / config.iterations;
    const deltaBeta = beta - betaPrevious;
    const incrementalLogWeights = stateIndices.map(
      (stateIndex, particleIndex) =>
        Math.log(weights[particleIndex]!) -
        deltaBeta * config.lossScale * space.states[stateIndex]!.score.totalLoss,
    );
    const logIncrement = logSumExp(incrementalLogWeights);
    logNormalizingConstantEstimate += logIncrement;
    weights = normalizeLogWeights(incrementalLogWeights).weights;

    const ess = effectiveSampleSize(weights);
    const resampled = ess / config.particles < config.essThreshold;
    if (resampled) {
      const ancestors = systematicResample(weights, random);
      stateIndices = ancestors.map((ancestor) => stateIndices[ancestor]!);
      weights = stateIndices.map(() => 1 / config.particles);
    }

    let acceptedMoves = 0;
    const attemptedMoves = config.particles * options.movesPerStage;
    for (let move = 0; move < options.movesPerStage; move += 1) {
      stateIndices = stateIndices.map((currentIndex) => {
        const proposedIndex = sampleCategorical(priorCumulative, random);
        const currentLoss = space.states[currentIndex]!.score.totalLoss;
        const proposedLoss = space.states[proposedIndex]!.score.totalLoss;
        // Independent MH with q = p0. The known grammar prior cancels exactly.
        const logAcceptance = -beta * config.lossScale * (proposedLoss - currentLoss);
        if (Math.log(random.next()) < Math.min(0, logAcceptance)) {
          acceptedMoves += 1;
          return proposedIndex;
        }
        return currentIndex;
      });
    }

    const summary: GrammarSmcStage = {
      stage,
      beta,
      effectiveSampleSize: ess,
      resampled,
      acceptedMoves,
      attemptedMoves,
      logNormalizingConstantEstimate,
    };
    stages.push(summary);
    options.trace.emit(
      "grammar-stage",
      `stage ${stage}/${config.iterations}: beta=${beta.toFixed(4)} ESS=${ess.toFixed(2)}/${config.particles}; resampled=${resampled}; MH accepted=${acceptedMoves}/${attemptedMoves}; logZ=${logNormalizingConstantEstimate.toFixed(6)}`,
      { ...summary },
    );
    betaPrevious = beta;
  }

  const empirical = space.states.map(() => 0);
  for (let particleIndex = 0; particleIndex < stateIndices.length; particleIndex += 1) {
    empirical[stateIndices[particleIndex]!]! += weights[particleIndex]!;
  }
  const particles: GrammarSmcParticle[] = stateIndices.map((stateIndex, index) => ({
    stateIndex,
    weight: weights[index]!,
  }));
  const exactProgramMassEstimate = empirical.reduce(
    (sum, probability, index) => sum + (space.states[index]!.score.exactProgram ? probability : 0),
    0,
  );
  const meanLossEstimate = empirical.reduce(
    (sum, probability, index) => sum + probability * space.states[index]!.score.totalLoss,
    0,
  );
  const sampledStates = space.states.filter((_state, index) => empirical[index]! > 0);
  sampledStates.sort(
    (left, right) => {
      const leftLogTarget =
        left.logPriorProbability - options.betaMax * config.lossScale * left.score.totalLoss;
      const rightLogTarget =
        right.logPriorProbability - options.betaMax * config.lossScale * right.score.totalLoss;
      return rightLogTarget - leftLogTarget || left.key.localeCompare(right.key);
    },
  );

  return {
    space,
    reference,
    particles,
    stages,
    best: sampledStates[0]!,
    exactProgramMassEstimate,
    meanLossEstimate,
    logNormalizingConstantEstimate,
    logNormalizingConstantError:
      logNormalizingConstantEstimate - reference.logNormalizingConstant,
    totalVariationDistance: totalVariation(empirical, reference.probabilities),
  };
}
