/**
 * Statistics for the foldr-bounded-square matrix.
 *
 * Reads experiments/results/summary.json, writes experiments/results/stats.json,
 * and prints a per-cell table plus within-proposer Fisher exact tests
 * (one-shot vs iterative success).
 *
 * Usage: npx tsx experiments/analyze.ts
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const OUT_DIR = resolve(import.meta.dirname, "results");

interface RunResult {
  readonly runId: string;
  readonly proposer: string;
  readonly arm: string;
  readonly seed: number;
  readonly exact: boolean | null;
  readonly bestLoss: number | null;
  readonly bestCost: number | null;
  readonly firstExactProposalCall: number | null;
  readonly acceptedProposals: number;
  readonly rejectedProposals: number;
  readonly failedProposals: number;
  readonly failureReasons: readonly string[];
  readonly lineageLosses: readonly number[];
  readonly durationMs: number;
  readonly error: string | null;
}

interface CellStats {
  readonly proposer: string;
  readonly arm: string;
  readonly n: number;
  readonly exactCount: number;
  readonly successRate: number;
  readonly wilsonLow: number;
  readonly wilsonHigh: number;
  readonly meanBestLoss: number;
  readonly medianBestLoss: number;
  readonly bestLosses: readonly number[];
  readonly firstExactCalls: readonly number[];
  readonly rejectedTotal: number;
  readonly failedTotal: number;
  readonly meanDurationS: number;
}

function wilson(successes: number, n: number): { low: number; high: number } {
  if (n === 0) return { low: 0, high: 1 };
  const z = 1.959964; // 95%
  const p = successes / n;
  const denominator = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denominator;
  const margin = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denominator;
  return { low: Math.max(0, center - margin), high: Math.min(1, center + margin) };
}

/** Two-sided Fisher exact test for a 2x2 table [[a,b],[c,d]] via hypergeometric enumeration. */
function fisherExact(a: number, b: number, c: number, d: number): number {
  const logFactorialCache: number[] = [0];
  function logFactorial(k: number): number {
    for (let index = logFactorialCache.length; index <= k; index += 1) {
      logFactorialCache[index] = logFactorialCache[index - 1]! + Math.log(index);
    }
    return logFactorialCache[k]!;
  }
  const row1 = a + b;
  const row2 = c + d;
  const col1 = a + c;
  const total = a + b + c + d;
  function logProbability(x: number): number {
    return (
      logFactorial(row1) +
      logFactorial(row2) +
      logFactorial(col1) +
      logFactorial(total - col1) -
      logFactorial(total) -
      logFactorial(x) -
      logFactorial(row1 - x) -
      logFactorial(col1 - x) -
      logFactorial(row2 - col1 + x)
    );
  }
  const observed = logProbability(a);
  let pValue = 0;
  const lowest = Math.max(0, col1 - row2);
  const highest = Math.min(row1, col1);
  for (let x = lowest; x <= highest; x += 1) {
    const candidate = logProbability(x);
    if (candidate <= observed + 1e-9) pValue += Math.exp(candidate);
  }
  return Math.min(1, pValue);
}

/**
 * Exact two-sided permutation test on the difference of mean final loss
 * between two arms (enumerates every partition; C(20,10) = 184,756 at n=m=10).
 */
function permutationTest(groupA: readonly number[], groupB: readonly number[]): number {
  const pooled = [...groupA, ...groupB];
  const sizeA = groupA.length;
  const total = pooled.length;
  const sumA = groupA.reduce((sum, value) => sum + value, 0);
  const sumAll = pooled.reduce((sum, value) => sum + value, 0);
  const observed = Math.abs(sumA / sizeA - (sumAll - sumA) / (total - sizeA));
  let extreme = 0;
  let count = 0;
  const chosen: number[] = [];
  function recurse(next: number, partialSum: number): void {
    if (chosen.length === sizeA) {
      count += 1;
      const difference = Math.abs(partialSum / sizeA - (sumAll - partialSum) / (total - sizeA));
      if (difference >= observed - 1e-9) extreme += 1;
      return;
    }
    if (total - next < sizeA - chosen.length) return;
    chosen.push(next);
    recurse(next + 1, partialSum + pooled[next]!);
    chosen.pop();
    recurse(next + 1, partialSum);
  }
  recurse(0, 0);
  return extreme / count;
}

function median(values: readonly number[]): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

const summary = JSON.parse(readFileSync(resolve(OUT_DIR, "summary.json"), "utf8")) as {
  runs: RunResult[];
};
const validRuns = summary.runs.filter((run) => run.error === null && run.exact !== null);
const erroredRuns = summary.runs.filter((run) => run.error !== null);

const cells = new Map<string, RunResult[]>();
for (const run of validRuns) {
  const key = `${run.proposer}|${run.arm}`;
  cells.set(key, [...(cells.get(key) ?? []), run]);
}

const stats: CellStats[] = [...cells.entries()].map(([key, runs]) => {
  const [proposer, arm] = key.split("|") as [string, string];
  const exactCount = runs.filter((run) => run.exact === true).length;
  const bestLosses = runs.map((run) => run.bestLoss ?? NaN);
  const interval = wilson(exactCount, runs.length);
  return {
    proposer,
    arm,
    n: runs.length,
    exactCount,
    successRate: exactCount / runs.length,
    wilsonLow: interval.low,
    wilsonHigh: interval.high,
    meanBestLoss: bestLosses.reduce((sum, loss) => sum + loss, 0) / runs.length,
    medianBestLoss: median(bestLosses),
    bestLosses,
    firstExactCalls: runs
      .map((run) => run.firstExactProposalCall)
      .filter((call): call is number => call !== null),
    rejectedTotal: runs.reduce((sum, run) => sum + run.rejectedProposals, 0),
    failedTotal: runs.reduce((sum, run) => sum + run.failedProposals, 0),
    meanDurationS: runs.reduce((sum, run) => sum + run.durationMs, 0) / runs.length / 1000,
  };
});

const proposerOrder = ["claude-sonnet-5", "claude-haiku-4-5", "qwen3-coder-30b", "catalog"];
stats.sort(
  (left, right) =>
    proposerOrder.indexOf(left.proposer) - proposerOrder.indexOf(right.proposer) ||
    left.arm.localeCompare(right.arm),
);

const fisher: Record<string, { table: number[][]; p: number }> = {};
const lossPermutation: Record<
  string,
  { oneShotMean: number; iterativeMean: number; p: number }
> = {};
for (const proposer of proposerOrder) {
  const oneShot = stats.find((cell) => cell.proposer === proposer && cell.arm === "one-shot");
  const iterative = stats.find((cell) => cell.proposer === proposer && cell.arm === "iterative");
  if (!oneShot || !iterative) continue;
  const table = [
    [iterative.exactCount, iterative.n - iterative.exactCount],
    [oneShot.exactCount, oneShot.n - oneShot.exactCount],
  ];
  fisher[proposer] = {
    table,
    p: fisherExact(table[0]![0]!, table[0]![1]!, table[1]![0]!, table[1]![1]!),
  };
  lossPermutation[proposer] = {
    oneShotMean: oneShot.meanBestLoss,
    iterativeMean: iterative.meanBestLoss,
    p: permutationTest(iterative.bestLosses, oneShot.bestLosses),
  };
}

writeFileSync(
  resolve(OUT_DIR, "stats.json"),
  JSON.stringify(
    { cells: stats, fisher, lossPermutation, erroredRuns: erroredRuns.map((run) => run.runId) },
    null,
    2,
  ),
);

console.log("proposer            arm        n  exact  rate    95% CI          mean-loss  med  rej  fail  first-exact-calls");
for (const cell of stats) {
  console.log(
    `${cell.proposer.padEnd(19)} ${cell.arm.padEnd(10)} ${String(cell.n).padStart(2)}  ` +
      `${String(cell.exactCount).padStart(4)}  ${(cell.successRate * 100).toFixed(0).padStart(3)}%  ` +
      `[${(cell.wilsonLow * 100).toFixed(0).padStart(3)}%,${(cell.wilsonHigh * 100).toFixed(0).padStart(4)}%]  ` +
      `${cell.meanBestLoss.toFixed(1).padStart(8)}  ${String(cell.medianBestLoss).padStart(3)}  ` +
      `${String(cell.rejectedTotal).padStart(3)}  ${String(cell.failedTotal).padStart(4)}  ` +
      `[${cell.firstExactCalls.join(", ")}]`,
  );
}
console.log("\nFisher exact (iterative vs one-shot success, two-sided):");
for (const [proposer, test] of Object.entries(fisher)) {
  console.log(`  ${proposer.padEnd(19)} ${JSON.stringify(test.table)}  p=${test.p.toFixed(4)}`);
}
console.log("\nExact permutation test on mean final loss (iterative vs one-shot, two-sided):");
for (const [proposer, test] of Object.entries(lossPermutation)) {
  console.log(
    `  ${proposer.padEnd(19)} iterative=${test.iterativeMean.toFixed(1)} one-shot=${test.oneShotMean.toFixed(1)}  p=${test.p.toFixed(4)}`,
  );
}
if (erroredRuns.length > 0) {
  console.log(`\nExcluded ${erroredRuns.length} errored runs: ${erroredRuns.map((run) => run.runId).join(", ")}`);
}
const failureReasonCounts = new Map<string, number>();
for (const run of validRuns) {
  for (const reason of run.failureReasons) {
    const shortReason = reason.slice(0, 80);
    failureReasonCounts.set(shortReason, (failureReasonCounts.get(shortReason) ?? 0) + 1);
  }
}
if (failureReasonCounts.size > 0) {
  console.log("\nProposal failure reasons (within otherwise-valid runs):");
  for (const [reason, count] of failureReasonCounts) console.log(`  ${count}x ${reason}`);
}
