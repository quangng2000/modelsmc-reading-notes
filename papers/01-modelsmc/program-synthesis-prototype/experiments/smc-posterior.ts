/**
 * Calibrated tempered SMC over programs, validated against the exact posterior.
 *
 * Runs one or both exactly-computable islands:
 *   prior island — MH rejuvenation from the exact prior sampler (no LLM)
 *   llm island   — MH rejuvenation from the LLM with cached proposal densities
 * and reports each island's weighted posterior estimate vs the enumerated
 * ground truth (TV distance, exact-solve mass), plus their ensemble average.
 *
 * Usage:
 *   npm run smc-posterior -- <config.json> --cost-cap N [--model gpt-oss:20b]
 *     [--particles 32] [--stages 5] [--sweeps 1] [--islands prior,llm]
 *     [--seed 17] [--ollama-url http://localhost:11434]
 */
import { acceptProgram } from "../src/core/language.verify.js";
import { renderProgram } from "../src/shell/ast/render.js";
import { deriveSynthesisTrace } from "../src/shell/deduction/index.js";
import { SeededRandom } from "../src/shell/engine/random.js";
import { createOllamaDrawer, proposalPrompt, type ProposalTask } from "../src/shell/posterior/llm-proposal.js";
import { createSidecarClient } from "../src/shell/posterior/sidecar-client.js";
import { linearGammas, runCalibratedSmc, type CalibratedSmcResult } from "../src/shell/posterior/smc.js";
import { computeExactPosterior, formatProbability, parseHarnessArgs } from "./posterior-lib.js";

const argv = process.argv.slice(2);
const args = parseHarnessArgs(argv);
const flag = (name: string, fallback: string): string => {
  const index = argv.indexOf(name);
  return index === -1 ? fallback : argv[index + 1]!;
};
const model = flag("--model", "gpt-oss:20b");
const particles = Number(flag("--particles", "32"));
const stages = Number(flag("--stages", "5"));
const sweeps = Number(flag("--sweeps", "1"));
const islands = flag("--islands", "prior,llm").split(",").map((token) => token.trim());
const baseUrl = flag("--ollama-url", "http://localhost:11434");

const task: ProposalTask = {
  inputType: args.config.inputType,
  outputType: args.config.outputType,
  examples: args.config.examples,
  integerConstants: args.config.integerConstants,
  maxDepth: args.config.maxDepth,
  maxNodes: args.config.maxNodes,
  costCap: args.costCap,
  beta: args.beta,
  noise: args.noise,
};

console.log(`[smc] task: ${args.config.name}`);
console.log(`[smc] signature: ${task.inputType} -> ${task.outputType}; ${task.examples.length} examples`);
for (const event of deriveSynthesisTrace(task.inputType, task.outputType, task.examples)) {
  console.log(`[deduction] ${event.message}`);
}

console.log(`[smc] computing exact posterior for ground truth (cap ${args.costCap})...`);
const exact = computeExactPosterior(args);
console.log(
  `[smc] support ${exact.programCount}; exact-solve mass ${formatProbability(exact.exactSolveMass)}; entropy ${exact.entropy.toFixed(3)} nats`,
);
const exactSolvers = new Set(exact.entries.filter((entry) => entry.exact).map((entry) => entry.rendered));

function summarize(label: string, result: CalibratedSmcResult): Map<string, number> {
  const masses = new Map<string, number>();
  result.particles.forEach((particle, index) => {
    const rendered = renderProgram(particle.proposal.program, task.inputType);
    masses.set(rendered, (masses.get(rendered) ?? 0) + result.normalizedWeights[index]!);
  });
  let tv = 0;
  for (const [rendered, truth] of exact.probabilityByRendering) {
    tv += Math.abs((masses.get(rendered) ?? 0) - truth);
  }
  tv /= 2;
  let solverMass = 0;
  for (const [rendered, mass] of masses) if (exactSolvers.has(rendered)) solverMass += mass;
  const unique = masses.size;
  console.log(
    `[smc:${label}] TV=${tv.toFixed(6)}  exact-mass est=${formatProbability(solverMass)} (truth ${formatProbability(exact.exactSolveMass)})  unique=${unique}  llmDraws=${result.llmDraws}`,
  );
  return masses;
}

const gammas = linearGammas(stages);
console.log(`[smc] particles=${particles} stages=${stages} sweeps=${sweeps} gammas=[${gammas.map((g) => g.toFixed(2)).join(", ")}]`);

const estimates = new Map<string, Map<string, number>>();
const rejectionCounts = new Map<string, number>();

const sidecarUrl = flag("--sidecar-url", "http://127.0.0.1:8765");
const genTokens = Number(flag("--gen-tokens", "200"));
const proposalTemp = Number(flag("--proposal-temp", "1"));
const finalPopulations = new Map<string, CalibratedSmcResult>();

for (const island of islands) {
  if (island !== "prior" && island !== "llm" && island !== "llm-feedback") {
    throw new Error(`unknown island: ${island}`);
  }
  console.log(`\n[smc] === ${island} island ===`);
  const result = await runCalibratedSmc({
    task,
    particleCount: particles,
    gammas,
    moves: island,
    sweepsPerStage: sweeps,
    essThresholdRatio: 0.5,
    rng: new SeededRandom(args.seed + (island === "llm" ? 1000 : island === "llm-feedback" ? 2000 : 0)),
    drawer:
      island === "llm"
        ? createOllamaDrawer({ model, baseUrl, prompt: proposalPrompt(task) })
        : undefined,
    sidecar: island === "llm-feedback" ? createSidecarClient(sidecarUrl) : undefined,
    generateMaxTokens: genTokens,
    proposalTemperature: proposalTemp,
    onReject: (reason) => {
      const total = [...rejectionCounts.values()].reduce((sum, count) => sum + count, 0);
      rejectionCounts.set(reason, (rejectionCounts.get(reason) ?? 0) + 1);
      if (total < 12) console.log(`  [${island}] draw rejected: ${reason}`);
      else if (total % 25 === 0) console.log(`  [${island}] ${total} draws rejected so far`);
    },
    trace: (message) => console.log(`  [${island}] ${message}`),
  });
  finalPopulations.set(island, result);
  estimates.set(island, summarize(island, result));
}

if (estimates.size === 2) {
  const combined = new Map<string, number>();
  for (const masses of estimates.values()) {
    for (const [rendered, mass] of masses) combined.set(rendered, (combined.get(rendered) ?? 0) + mass / 2);
  }
  let tv = 0;
  for (const [rendered, truth] of exact.probabilityByRendering) {
    tv += Math.abs((combined.get(rendered) ?? 0) - truth);
  }
  tv /= 2;
  let solverMass = 0;
  for (const [rendered, mass] of combined) if (exactSolvers.has(rendered)) solverMass += mass;
  console.log(
    `\n[smc:ensemble] TV=${tv.toFixed(6)}  exact-mass est=${formatProbability(solverMass)} (truth ${formatProbability(exact.exactSolveMass)})`,
  );
}

if (rejectionCounts.size > 0) {
  console.log("\n[smc] llm draw rejections:");
  for (const [reason, count] of [...rejectionCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6)) {
    console.log(`  ${String(count).padStart(4)}x ${reason}`);
  }
}

console.log(`\n[smc] top ${args.top} exact-posterior programs vs island estimates:`);
for (const entry of exact.entries.slice(0, args.top)) {
  const parts = [...estimates.entries()]
    .map(([island, masses]) => `${island}=${formatProbability(masses.get(entry.rendered) ?? 0)}`)
    .join("  ");
  console.log(
    `  true=${formatProbability(entry.probability).padStart(12)}  ${parts}  ${entry.exact ? "EXACT" : "     "}  ${entry.rendered.slice(0, 80)}`,
  );
}

// The user-facing verdict: each island's best program (MAP estimate).
for (const [island, result] of finalPopulations) {
  const masses = estimates.get(island)!;
  const byProgram = new Map<string, (typeof result.particles)[number]>();
  result.particles.forEach((particle) => {
    const rendered = renderProgram(particle.proposal.program, task.inputType);
    if (!byProgram.has(rendered)) byProgram.set(rendered, particle);
  });
  const [bestRendered, bestMass] = [...masses.entries()].sort((a, b) => b[1] - a[1])[0]!;
  const best = byProgram.get(bestRendered)!;
  const exactOnAll = acceptProgram(best.proposal.program, [...task.examples]);
  console.log(`\n[result:${island}] best program (highest estimated posterior mass):`);
  console.log(`[result:${island}] program: ${bestRendered}`);
  console.log(
    `[result:${island}] cost=${best.proposal.cost} logLik=${best.proposal.logLikelihood.toFixed(3)} ` +
      `estimated-mass=${formatProbability(bestMass)} true-mass=${formatProbability(exact.probabilityByRendering.get(bestRendered) ?? 0)}`,
  );
  console.log(`[result:${island}] exact on every example (verified): ${exactOnAll}`);
}
