/**
 * Experiment matrix runner for the foldr-bounded-square study.
 *
 * Runs every (proposer x search-arm x seed) cell as a child synthesis process,
 * parses the JSONL trace, and writes experiments/results/summary.json.
 *
 * Usage:
 *   ANTHROPIC_API_KEY=... npx tsx experiments/run-matrix.ts [--skip-ollama] [--seeds N] [--ollama-seeds N] [--resume]
 *
 * --resume skips any cell whose JSONL trace already contains a run.completed
 * event and re-parses it instead, so an interrupted matrix continues where it stopped.
 */
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const TASK = "examples/foldr-bounded-square.json";
const LOG_DIR = resolve(ROOT, "runs/experiments");
const OUT_DIR = resolve(ROOT, "experiments/results");

interface Arm {
  readonly arm: "one-shot" | "iterative";
  readonly flags: readonly string[];
}
const ARMS: readonly Arm[] = [
  { arm: "one-shot", flags: ["--particles", "4", "--iterations", "1", "--alpha", "0"] },
  {
    arm: "iterative",
    flags: ["--particles", "2", "--iterations", "2", "--alpha", "0", "--ess-threshold", "1"],
  },
];

interface ProposerSpec {
  readonly proposer: string;
  readonly flags: readonly string[];
  readonly seeds: number;
  readonly concurrency: number;
  readonly timeoutMs: number;
}

function parseIntFlag(name: string, fallback: number): number {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  const value = Number(process.argv[index + 1]);
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`);
  return value;
}
const CLOUD_SEEDS = parseIntFlag("--seeds", 10);
const OLLAMA_SEEDS = parseIntFlag("--ollama-seeds", 5);
const SKIP_OLLAMA = process.argv.includes("--skip-ollama");

const PROPOSERS: readonly ProposerSpec[] = [
  {
    proposer: "catalog",
    flags: ["--proposal", "catalog"],
    seeds: CLOUD_SEEDS,
    concurrency: 4,
    timeoutMs: 120_000,
  },
  {
    proposer: "claude-sonnet-5",
    flags: ["--proposal", "anthropic", "--model", "claude-sonnet-5"],
    seeds: CLOUD_SEEDS,
    concurrency: 4,
    timeoutMs: 120_000,
  },
  {
    proposer: "claude-haiku-4-5",
    flags: ["--proposal", "anthropic", "--model", "claude-haiku-4-5"],
    seeds: CLOUD_SEEDS,
    concurrency: 4,
    timeoutMs: 120_000,
  },
  ...(SKIP_OLLAMA
    ? []
    : [
        {
          proposer: "qwen3-coder-30b",
          flags: ["--proposal", "ollama", "--model", "qwen3-coder:30b-a3b-q8_0"],
          seeds: OLLAMA_SEEDS,
          concurrency: 1, // local GPU: one run at a time
          timeoutMs: 420_000,
        } satisfies ProposerSpec,
      ]),
];

interface RunResult {
  readonly runId: string;
  readonly proposer: string;
  readonly arm: string;
  readonly seed: number;
  readonly exitCode: number | null;
  readonly durationMs: number;
  readonly exact: boolean | null;
  readonly bestLoss: number | null;
  readonly bestCost: number | null;
  readonly firstExactProposalCall: number | null;
  readonly proposalCalls: number | null;
  readonly acceptedProposals: number;
  readonly rejectedProposals: number;
  readonly failedProposals: number;
  readonly failureReasons: readonly string[];
  readonly lineageLosses: readonly number[];
  readonly error: string | null;
}

const RESUME = process.argv.includes("--resume");

function hasCompletedTrace(logFile: string): boolean {
  if (!existsSync(logFile)) return false;
  try {
    return readFileSync(logFile, "utf8").includes('"kind":"run.completed"');
  } catch {
    return false;
  }
}

function runOne(spec: ProposerSpec, arm: Arm, seed: number): Promise<RunResult> {
  const runId = `${spec.proposer}_${arm.arm}_seed${seed}`;
  const logFile = resolve(LOG_DIR, `${runId}.jsonl`);
  if (RESUME && hasCompletedTrace(logFile)) {
    return parseTrace(runId, spec, arm, seed, logFile, 0, 0, "");
  }
  const args = [
    "tsx",
    "src/shell/cli.ts",
    TASK,
    ...spec.flags,
    ...arm.flags,
    "--seed",
    String(seed),
    "--max-tokens",
    "2048",
    "--timeout-ms",
    String(spec.timeoutMs),
    "--log-file",
    logFile,
  ];
  const startedAt = Date.now();
  return new Promise((resolvePromise) => {
    const child = spawn("npx", args, { cwd: ROOT, env: process.env, stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += String(chunk);
    });
    child.stdout.on("data", () => {});
    child.on("close", (exitCode) => {
      void parseTrace(runId, spec, arm, seed, logFile, exitCode, Date.now() - startedAt, stderr).then(
        resolvePromise,
      );
    });
  });
}

async function parseTrace(
  runId: string,
  spec: ProposerSpec,
  arm: Arm,
  seed: number,
  logFile: string,
  exitCode: number | null,
  durationMs: number,
  stderr: string,
): Promise<RunResult> {
  let exact: boolean | null = null;
  let bestLoss: number | null = null;
  let bestCost: number | null = null;
  let firstExactProposalCall: number | null = null;
  let proposalCalls: number | null = null;
  let acceptedProposals = 0;
  let rejectedProposals = 0;
  let failedProposals = 0;
  const failureReasons: string[] = [];
  let lineageLosses: number[] = [];
  let error: string | null = exitCode === 0 ? null : stderr.trim().slice(0, 400) || `exit code ${exitCode}`;
  try {
    const lines = readFileSync(logFile, "utf8").split("\n").filter((line) => line.trim() !== "");
    for (const line of lines) {
      const event = JSON.parse(line) as Record<string, unknown>;
      if (event.kind === "proposal.accepted") acceptedProposals += 1;
      if (event.kind === "proposal.rejected") rejectedProposals += 1;
      if (event.kind === "proposal.failed") {
        failedProposals += 1;
        if (typeof event.reason === "string") failureReasons.push(event.reason.slice(0, 200));
      }
      if (event.kind === "run.completed") {
        exact = typeof event.exact === "boolean" ? event.exact : null;
        bestLoss = typeof event.bestLoss === "number" ? event.bestLoss : null;
        bestCost = typeof event.bestCost === "number" ? event.bestCost : null;
        firstExactProposalCall =
          typeof event.firstExactProposalCall === "number" ? event.firstExactProposalCall : null;
        proposalCalls = typeof event.proposalCalls === "number" ? event.proposalCalls : null;
        if (Array.isArray(event.championLineage)) {
          lineageLosses = event.championLineage
            .map((step) => (step as { loss?: unknown }).loss)
            .filter((loss): loss is number => typeof loss === "number");
        }
      }
    }
    if (exact === null && error === null) error = "run finished without a run.completed event";
  } catch (readError) {
    error = `could not read trace: ${readError instanceof Error ? readError.message : String(readError)}`;
  }
  return {
    runId,
    proposer: spec.proposer,
    arm: arm.arm,
    seed,
    exitCode,
    durationMs,
    exact,
    bestLoss,
    bestCost,
    firstExactProposalCall,
    proposalCalls,
    acceptedProposals,
    rejectedProposals,
    failedProposals,
    failureReasons,
    lineageLosses,
    error,
  };
}

async function runPool(spec: ProposerSpec): Promise<RunResult[]> {
  const cells: { arm: Arm; seed: number }[] = [];
  for (const arm of ARMS) {
    for (let seed = 1; seed <= spec.seeds; seed += 1) cells.push({ arm, seed });
  }
  const results: RunResult[] = [];
  let cursor = 0;
  async function worker(): Promise<void> {
    while (cursor < cells.length) {
      const cell = cells[cursor]!;
      cursor += 1;
      const result = await runOne(spec, cell.arm, cell.seed);
      results.push(result);
      const status = result.error !== null ? `ERROR ${result.error}` : result.exact ? "EXACT" : `loss=${result.bestLoss}`;
      console.log(
        `[matrix] ${result.runId}: ${status} (first-exact call=${result.firstExactProposalCall ?? "-"}, ` +
          `accepted=${result.acceptedProposals} rejected=${result.rejectedProposals} failed=${result.failedProposals}, ` +
          `${Math.round(result.durationMs / 1000)}s)`,
      );
    }
  }
  await Promise.all(Array.from({ length: Math.min(spec.concurrency, cells.length) }, worker));
  return results;
}

async function main(): Promise<void> {
  if (PROPOSERS.some((spec) => spec.flags.includes("anthropic")) && !process.env.ANTHROPIC_API_KEY) {
    // The anthropic cells would all fail without a key; stop early with a clear message.
    console.error("[matrix] ANTHROPIC_API_KEY is not set");
    process.exitCode = 2;
    return;
  }
  mkdirSync(LOG_DIR, { recursive: true });
  mkdirSync(OUT_DIR, { recursive: true });
  const startedAt = new Date().toISOString();
  // Proposer pools run concurrently with each other; each pool bounds its own concurrency.
  const perProposer = await Promise.all(PROPOSERS.map((spec) => runPool(spec)));
  const runs = perProposer.flat();
  const summaryPath = resolve(OUT_DIR, "summary.json");
  const existing = existsSync(summaryPath)
    ? (JSON.parse(readFileSync(summaryPath, "utf8")) as { runs?: RunResult[] })
    : {};
  const previous = (existing.runs ?? []).filter(
    (run) => !runs.some((fresh) => fresh.runId === run.runId),
  );
  writeFileSync(
    summaryPath,
    JSON.stringify(
      {
        task: TASK,
        startedAt,
        finishedAt: new Date().toISOString(),
        budgetProposalCalls: 4,
        runs: [...previous, ...runs].sort((a, b) => a.runId.localeCompare(b.runId)),
      },
      null,
      2,
    ),
  );
  console.log(`[matrix] wrote ${runs.length} fresh runs (${previous.length} kept) to ${summaryPath}`);
}

await main();
