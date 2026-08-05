/**
 * Print the accepted-program-space growth curve for a task signature — how
 * many programs the verifier admits at each cost, and where exact enumeration
 * stops being feasible.
 *
 * Usage: npx tsx experiments/space-size.ts <config.json> [--cost-cap N]
 */
import { readFileSync } from "node:fs";
import { parseExperimentConfig } from "../src/shell/config/index.js";
import { countPrograms, lnBigInt, logPriorNormalizer } from "../src/shell/prior/index.js";

const argv = process.argv.slice(2);
const configPath = argv.find((argument) => !argument.startsWith("--"));
if (configPath === undefined) throw new Error("usage: <config.json> [--cost-cap N]");
const capIndex = argv.indexOf("--cost-cap");
const config = parseExperimentConfig(readFileSync(configPath, "utf8"));
const costCap = capIndex === -1 ? config.maxCost : Number(argv[capIndex + 1]);

const tables = countPrograms(
  config.inputType,
  config.outputType,
  costCap,
  config.integerConstants.length,
);
console.log(`[space] task: ${config.name}`);
console.log(`[space] signature: ${config.inputType} -> ${config.outputType}; ${config.integerConstants.length} constants; cap ${costCap}`);
console.log("cost      N(c)          cumulative   ~enumerable?");
let cumulative = 0n;
for (let cost = 1; cost <= costCap; cost += 1) {
  cumulative += tables.total[cost]!;
  const feasible = cumulative <= 2_000_000n ? "yes" : "";
  const logN = tables.total[cost]! > 0n ? `10^${(lnBigInt(tables.total[cost]!) / Math.LN10).toFixed(1)}` : "0";
  console.log(
    `${String(cost).padStart(4)}  ${logN.padStart(9)}  ${(`10^${cumulative > 0n ? (lnBigInt(cumulative) / Math.LN10).toFixed(1) : "0"}`).padStart(11)}   ${feasible}`,
  );
}
console.log(`[space] log Z_prior (beta=${config.costScale}) = ${logPriorNormalizer(tables.total, config.costScale).toFixed(4)}`);
