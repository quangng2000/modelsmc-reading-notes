/**
 * Exact calibrated posterior over the finite accepted-program space.
 *
 * Enumerates every program the verifier accepts up to the cost cap, scores it
 * with the normalized Occam prior exp(-beta*cost)/Z and the calibrated noise
 * likelihood, and prints the exact normalized posterior. This is ground truth
 * for every sampler in the calibrated pipeline.
 *
 * Usage:
 *   npx tsx experiments/exact-posterior.ts <config.json> \
 *     [--cost-cap N] [--beta X] [--top K] [--max-programs M]
 */
import { computeExactPosterior, formatProbability, parseHarnessArgs } from "./posterior-lib.js";

const args = parseHarnessArgs(process.argv.slice(2));
const startedAt = Date.now();
const posterior = computeExactPosterior(args);

console.log(`[exact] task: ${args.config.name}`);
console.log(
  `[exact] support: ${posterior.programCount} accepted programs at cost cap ${args.costCap} (beta=${args.beta})`,
);
console.log(`[exact] log Z_prior = ${posterior.logZPrior.toFixed(6)}`);
console.log(`[exact] log Z_posterior (unnormalized-prior convention) = ${posterior.logZPosterior.toFixed(6)}`);
console.log(`[exact] posterior entropy = ${posterior.entropy.toFixed(4)} nats`);
console.log(`[exact] posterior mass on exact-solving programs = ${formatProbability(posterior.exactSolveMass)}`);
console.log(`[exact] top ${args.top} programs by posterior probability:`);
for (const entry of posterior.entries.slice(0, args.top)) {
  const marker = entry.exact ? "EXACT" : "     ";
  console.log(
    `  ${formatProbability(entry.probability).padStart(12)}  ${marker}  cost=${String(entry.cost).padStart(2)}  logLik=${entry.logLikelihood.toFixed(3).padStart(10)}  ${entry.rendered}`,
  );
}
const topMass = posterior.entries
  .slice(0, args.top)
  .reduce((sum, entry) => sum + entry.probability, 0);
console.log(`[exact] mass of top ${args.top}: ${formatProbability(topMass)}`);
console.log(`[exact] computed in ${((Date.now() - startedAt) / 1000).toFixed(1)}s`);
