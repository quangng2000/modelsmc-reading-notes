import { readFileSync } from "node:fs";
import { acceptProgram, type Program } from "../src/core/language.verify.js";
import { renderProgram } from "../src/shell/ast/render.js";
import { parseExperimentConfig, type ExperimentConfig } from "../src/shell/config/index.js";
import {
  countPrograms,
  enumeratePrograms,
  lnBigInt,
  logPriorNormalizer,
  logSumExp,
} from "../src/shell/prior/index.js";
import { scoreCalibrated } from "../src/shell/scoring/calibrated.js";
import { DEFAULT_NOISE, validateNoiseModel, type NoiseModel } from "../src/shell/scoring/emission.js";

/**
 * Shared machinery for the calibrated-posterior experiments: exact enumeration
 * of the finite accepted-program space, exact normalized posterior, and the
 * common CLI plumbing.
 */

export interface PosteriorEntry {
  readonly program: Program;
  readonly rendered: string;
  readonly cost: number;
  readonly logLikelihood: number;
  /** Normalized posterior probability. */
  readonly probability: number;
  readonly exact: boolean;
}

export interface ExactPosterior {
  readonly entries: readonly PosteriorEntry[]; // sorted by descending probability
  readonly logZPrior: number;
  readonly logZPosterior: number; // log sum over programs of exp(-beta*cost + logLik), prior unnormalized
  readonly programCount: number;
  readonly exactSolveMass: number;
  readonly entropy: number;
  /** rendered program -> probability, over the full support. */
  readonly probabilityByRendering: ReadonlyMap<string, number>;
}

export function computeExactPosterior(options: {
  readonly config: ExperimentConfig;
  readonly costCap: number;
  readonly beta: number;
  readonly noise: NoiseModel;
  readonly maxPrograms: number;
}): ExactPosterior {
  const { config, costCap, beta, noise, maxPrograms } = options;
  validateNoiseModel(noise);
  const tables = countPrograms(
    config.inputType,
    config.outputType,
    costCap,
    config.integerConstants.length,
  );
  const totalCount = tables.total.reduce((sum, count) => sum + count, 0n);
  if (totalCount > BigInt(maxPrograms)) {
    throw new Error(
      `accepted-program space has ${totalCount} programs at cost cap ${costCap}; ` +
        `raise --cost-cap guard (${maxPrograms}) deliberately or lower the cap`,
    );
  }
  const logZPrior = logPriorNormalizer(tables.total, beta);

  const scored: { program: Program; cost: number; logLikelihood: number; logPosterior: number }[] = [];
  const seenPerCost = new Array<bigint>(costCap + 1).fill(0n);
  for (const { program, cost } of enumeratePrograms(
    config.inputType,
    config.outputType,
    costCap,
    config.integerConstants,
  )) {
    const score = scoreCalibrated(program, {
      inputType: config.inputType,
      outputType: config.outputType,
      examples: config.examples,
      beta,
      noise,
    });
    if ("rejected" in score) {
      throw new Error(`enumerated program rejected by scoring: ${score.rejected}`);
    }
    seenPerCost[cost] = seenPerCost[cost]! + 1n;
    scored.push({
      program,
      cost,
      logLikelihood: score.logLikelihood,
      logPosterior: score.logPosteriorUnnormalized,
    });
  }
  for (let cost = 1; cost <= costCap; cost += 1) {
    if (seenPerCost[cost]! !== tables.total[cost]!) {
      throw new Error(
        `enumeration/count mismatch at cost ${cost}: ${seenPerCost[cost]} vs ${tables.total[cost]}`,
      );
    }
  }

  const logZPosterior = logSumExp(scored.map((entry) => entry.logPosterior));
  let exactSolveMass = 0;
  let entropy = 0;
  const probabilityByRendering = new Map<string, number>();
  const entries: PosteriorEntry[] = scored.map((entry) => {
    const probability = Math.exp(entry.logPosterior - logZPosterior);
    const exact = acceptProgram(entry.program, [...config.examples]);
    if (exact) exactSolveMass += probability;
    if (probability > 0) entropy -= probability * Math.log(probability);
    const rendered = renderProgram(entry.program, config.inputType);
    probabilityByRendering.set(rendered, (probabilityByRendering.get(rendered) ?? 0) + probability);
    return {
      program: entry.program,
      rendered,
      cost: entry.cost,
      logLikelihood: entry.logLikelihood,
      probability,
      exact,
    };
  });
  entries.sort((left, right) => right.probability - left.probability);
  return {
    entries,
    logZPrior,
    logZPosterior,
    programCount: scored.length,
    exactSolveMass,
    entropy,
    probabilityByRendering,
  };
}

export interface HarnessArgs {
  readonly config: ExperimentConfig;
  readonly configPath: string;
  readonly costCap: number;
  readonly beta: number;
  readonly noise: NoiseModel;
  readonly top: number;
  readonly maxPrograms: number;
  readonly seed: number;
  readonly sampleSizes: readonly number[];
}

export function parseHarnessArgs(argv: readonly string[]): HarnessArgs {
  const positional = argv.filter((argument) => !argument.startsWith("--"));
  const configPath = positional[0];
  if (configPath === undefined) throw new Error("usage: <config.json> [--cost-cap N] [--beta X] ...");
  const flag = (name: string): string | undefined => {
    const index = argv.indexOf(name);
    if (index === -1) return undefined;
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) throw new Error(`${name} requires a value`);
    return value;
  };
  const config = parseExperimentConfig(readFileSync(configPath, "utf8"));
  const costCap = Number(flag("--cost-cap") ?? config.maxCost);
  const beta = Number(flag("--beta") ?? config.costScale);
  if (!Number.isSafeInteger(costCap) || costCap < 1) throw new Error("--cost-cap must be a positive integer");
  if (!Number.isFinite(beta) || beta < 0) throw new Error("--beta must be a nonnegative number");
  const top = Number(flag("--top") ?? 10);
  const maxPrograms = Number(flag("--max-programs") ?? 2_000_000);
  const seed = Number(flag("--seed") ?? 17);
  const sampleSizes = (flag("--samples") ?? "1000,10000,100000")
    .split(",")
    .map((token) => Number(token.trim()));
  if (sampleSizes.some((size) => !Number.isSafeInteger(size) || size < 1)) {
    throw new Error("--samples must be a comma-separated list of positive integers");
  }
  return { config, configPath, costCap, beta, noise: DEFAULT_NOISE, top, maxPrograms, seed, sampleSizes };
}

export function formatProbability(probability: number): string {
  if (probability >= 0.0001) return probability.toFixed(6);
  return probability.toExponential(3);
}
