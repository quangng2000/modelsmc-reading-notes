/**
 * Calibrated importance sampling with the exact prior as the proposal.
 *
 * Because createPriorSampler draws from the normalized Occam prior with a
 * known density, self-normalized importance weights reduce to the calibrated
 * likelihood: w_i = exp(logLik(m_i)). The weighted population is then a
 * consistent estimator of the TRUE posterior — the first fully calibrated
 * sampler in this prototype — and is validated here against the exact
 * enumerated posterior (total-variation distance, exact-solve mass, ESS).
 *
 * Read the outputs accordingly: at feasible sample sizes the ESS fraction is
 * small and PER-PROGRAM estimates are low-count (some true atoms unsampled);
 * the semantic functionals (exact-solve mass) are the converged summaries.
 * The point of this baseline is to quantify that inefficiency exactly — it is
 * the formal case for a better proposal with an importance correction.
 *
 * Usage:
 *   npx tsx experiments/prior-is.ts <config.json> \
 *     [--cost-cap N] [--beta X] [--samples 1000,10000,100000] [--seed S] [--top K]
 */
import { renderProgram } from "../src/shell/ast/render.js";
import { SeededRandom } from "../src/shell/engine/random.js";
import { createPriorSampler } from "../src/shell/prior/index.js";
import { scoreCalibrated } from "../src/shell/scoring/calibrated.js";
import { computeExactPosterior, formatProbability, parseHarnessArgs } from "./posterior-lib.js";

const args = parseHarnessArgs(process.argv.slice(2));

console.log(`[prior-is] computing exact posterior for ground truth...`);
const exact = computeExactPosterior(args);
console.log(
  `[prior-is] support ${exact.programCount} programs; exact-solve mass ${formatProbability(exact.exactSolveMass)}; entropy ${exact.entropy.toFixed(3)} nats`,
);

const rng = new SeededRandom(args.seed);
const sampler = createPriorSampler({
  inputType: args.config.inputType,
  outputType: args.config.outputType,
  costCap: args.costCap,
  constants: args.config.integerConstants,
  beta: args.beta,
  rng,
});

console.log(
  `[prior-is] proposal = exact prior sampler (logZ ${sampler.logZ.toFixed(4)}); weights = calibrated likelihood`,
);
console.log("samples      ESS        ESS%   exact-mass-est  exact-mass-true  TV-distance");

const exactSolvers = new Set(
  exact.entries.filter((entry) => entry.exact).map((entry) => entry.rendered),
);
const maxSamples = Math.max(...args.sampleSizes);
const weightedMass = new Map<string, number>();
let weightSum = 0;
let weightSquareSum = 0;
let exactMassWeighted = 0;
let drawn = 0;

for (const checkpoint of [...args.sampleSizes].sort((a, b) => a - b)) {
  while (drawn < checkpoint && drawn < maxSamples) {
    const { program } = sampler.sample();
    const score = scoreCalibrated(program, {
      inputType: args.config.inputType,
      outputType: args.config.outputType,
      examples: args.config.examples,
      beta: args.beta,
      noise: args.noise,
    });
    if ("rejected" in score) throw new Error(`prior sample rejected: ${score.rejected}`);
    // Self-normalized IS with proposal = prior: weight ∝ likelihood.
    const weight = Math.exp(score.logLikelihood);
    const rendered = renderProgram(program, args.config.inputType);
    weightedMass.set(rendered, (weightedMass.get(rendered) ?? 0) + weight);
    weightSum += weight;
    weightSquareSum += weight * weight;
    const truth = exact.probabilityByRendering.get(rendered);
    if (truth === undefined) {
      throw new Error(`sampled program missing from exact support: ${rendered}`);
    }
    if (exactSolvers.has(rendered)) exactMassWeighted += weight;
    drawn += 1;
  }

  const effectiveSampleSize = weightSum === 0 ? 0 : (weightSum * weightSum) / weightSquareSum;
  let totalVariation = 0;
  for (const [rendered, truth] of exact.probabilityByRendering) {
    const estimate = (weightedMass.get(rendered) ?? 0) / (weightSum === 0 ? 1 : weightSum);
    totalVariation += Math.abs(estimate - truth);
  }
  totalVariation /= 2;
  console.log(
    `${String(drawn).padStart(7)}  ${effectiveSampleSize.toFixed(1).padStart(9)}  ${((100 * effectiveSampleSize) / drawn).toFixed(2).padStart(5)}%  ` +
      `${formatProbability(weightSum === 0 ? 0 : exactMassWeighted / weightSum).padStart(14)}  ` +
      `${formatProbability(exact.exactSolveMass).padStart(15)}  ${totalVariation.toFixed(6)}`,
  );
}

console.log(`[prior-is] top ${args.top} exact-posterior programs vs IS estimates:`);
for (const entry of exact.entries.slice(0, args.top)) {
  const estimate = (weightedMass.get(entry.rendered) ?? 0) / (weightSum === 0 ? 1 : weightSum);
  console.log(
    `  true=${formatProbability(entry.probability).padStart(12)}  est=${formatProbability(estimate).padStart(12)}  ${entry.exact ? "EXACT" : "     "}  ${entry.rendered}`,
  );
}
