/**
 * Annotate a synthesis JSONL trace with the calibrated quantities.
 *
 * Replays a --log-file trace and prints, in chronological order: the family
 * deduction story, every scored particle with its search loss ALONGSIDE the
 * calibrated log prior (normalized via the exact DP logZ), the true log
 * likelihood, and the unnormalized log posterior — plus rejections/failures
 * and a final ranking by log posterior.
 *
 * Usage:
 *   npm run annotate -- <trace.jsonl> <config.json> [--beta X]
 */
import { readFileSync } from "node:fs";
import { decodeProgram } from "../src/shell/ast/decode.js";
import { renderProgram } from "../src/shell/ast/render.js";
import { parseExperimentConfig } from "../src/shell/config/index.js";
import { countPrograms, logPriorNormalizer } from "../src/shell/prior/index.js";
import { scoreCalibrated } from "../src/shell/scoring/calibrated.js";
import { DEFAULT_NOISE } from "../src/shell/scoring/emission.js";

const argv = process.argv.slice(2);
const positional = argv.filter((argument) => !argument.startsWith("--"));
const [tracePath, configPath] = positional;
if (tracePath === undefined || configPath === undefined) {
  throw new Error("usage: <trace.jsonl> <config.json> [--beta X]");
}
const betaIndex = argv.indexOf("--beta");
const config = parseExperimentConfig(readFileSync(configPath, "utf8"));
const beta = betaIndex === -1 ? config.costScale : Number(argv[betaIndex + 1]);

const tables = countPrograms(
  config.inputType,
  config.outputType,
  config.maxCost,
  config.integerConstants.length,
);
const logZ = logPriorNormalizer(tables.total, beta);
console.log(
  `[annotate] normalized Occam prior over cost <= ${config.maxCost}: log Z = ${logZ.toFixed(4)} (beta = ${beta})`,
);
console.log(
  "[annotate] columns: logPrior = -beta*cost - logZ (normalized); logLik = calibrated channel; logPost = logPrior + logLik\n",
);

interface Row {
  readonly label: string;
  readonly family: string;
  readonly cost: number;
  readonly loss: number | undefined;
  readonly logPrior: number;
  readonly logLikelihood: number;
  readonly logPosterior: number;
  readonly exact: boolean;
  readonly rendered: string;
}
const rows: Row[] = [];

function annotate(expressionJson: unknown, label: string, loss: number | undefined, exact: boolean): void {
  const program = decodeProgram(expressionJson, {
    integerConstants: config.integerConstants,
    maxDepth: config.maxDepth,
    maxNodes: config.maxNodes,
  });
  const score = scoreCalibrated(program, {
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    beta,
    noise: DEFAULT_NOISE,
  });
  if ("rejected" in score) {
    console.log(`  ${label}  [not scoreable: ${score.rejected}]`);
    return;
  }
  const logPrior = score.logPriorUnnormalized - logZ;
  const rendered = renderProgram(program, config.inputType);
  const row: Row = {
    label,
    family: program.kind.replace("Program", ""),
    cost: score.cost,
    loss,
    logPrior,
    logLikelihood: score.logLikelihood,
    logPosterior: logPrior + score.logLikelihood,
    exact,
    rendered,
  };
  rows.push(row);
  console.log(
    `  ${label.padEnd(26)} ${row.family.padEnd(10)} cost=${String(row.cost).padStart(3)}  ` +
      `loss=${loss === undefined ? "   ?" : String(loss).padStart(4)}  ` +
      `logPrior=${logPrior.toFixed(2).padStart(8)}  logLik=${score.logLikelihood.toFixed(2).padStart(9)}  ` +
      `logPost=${row.logPosterior.toFixed(2).padStart(9)}${exact ? "  EXACT" : ""}`,
  );
  console.log(`      ${rendered.length > 110 ? `${rendered.slice(0, 107)}...` : rendered}`);
}

for (const line of readFileSync(tracePath, "utf8").split("\n")) {
  if (line.trim() === "") continue;
  const event = JSON.parse(line) as Record<string, unknown>;
  const kind = String(event.kind ?? "");
  const message = String(event.message ?? "");

  if (message.startsWith("family ") || message.startsWith("deduction for") || kind === "search.started") {
    console.log(`[deduction] ${message}`);
    continue;
  }
  if (kind === "particle.scored") {
    annotate(
      event.expression,
      `particle ${event.particleId} (${String(event.origin)})`,
      typeof event.totalLoss === "number" ? event.totalLoss : undefined,
      event.exactProgram === true,
    );
    continue;
  }
  if (kind === "proposal.rejected" || kind === "proposal.failed") {
    console.log(`  [${kind}] ${String(event.reason ?? message).slice(0, 140)}`);
    continue;
  }
  if (kind === "resampling.applied" || message.includes("ESS")) {
    console.log(`[smc] ${message.slice(0, 160)}`);
    continue;
  }
  if (kind === "run.completed") {
    console.log(`\n[result] ${message.slice(0, 200)}`);
  }
}

rows.sort((left, right) => right.logPosterior - left.logPosterior);
const seen = new Set<string>();
console.log("\n[annotate] distinct programs encountered, ranked by calibrated log posterior:");
for (const row of rows) {
  if (seen.has(row.rendered)) continue;
  seen.add(row.rendered);
  if (seen.size > 10) break;
  console.log(
    `  ${row.logPosterior.toFixed(2).padStart(9)}  ${row.exact ? "EXACT" : "     "}  cost=${String(row.cost).padStart(3)}  ${row.rendered.slice(0, 100)}`,
  );
}
