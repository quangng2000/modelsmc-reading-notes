#!/usr/bin/env node

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { CatalogProposer } from "./catalog-proposer.js";
import {
  ConfigurationError,
  parseExperimentConfig,
  withConfigOverrides,
  type ConfigOverrides,
} from "./config.js";
import { SynthesisEngine } from "./engine.js";
import { OllamaProposer } from "./ollama-proposer.js";
import type { Proposer } from "./proposal.js";
import { formatNumber } from "./numerics.js";
import { jsonStringify, renderProgram, renderType } from "./render.js";
import { TraceSink } from "./trace.js";

interface CliOptions {
  readonly configPath: string;
  readonly proposal: "catalog" | "ollama";
  readonly trace: boolean;
  readonly logFile?: string;
  readonly model: string;
  readonly ollamaUrl: string;
  readonly timeoutMs: number;
  readonly temperature: number;
  readonly reasoningEffort: "low" | "medium" | "high";
  readonly overrides: ConfigOverrides;
}

const USAGE = `Usage:
  npm run synthesize -- <config.json> [options]

Options:
  --proposal <catalog|ollama>  Proposal backend (default: catalog)
  --model <name>               Ollama model (default: gpt-oss:20b)
  --ollama-url <url>           OpenAI-compatible base URL
  --timeout-ms <integer>       Ollama timeout (default: 180000)
  --temperature <number>       Ollama temperature (default: 0)
  --reasoning-effort <level>   low, medium, or high (default: low)
  --particles <integer>        Override particle count
  --iterations <integer>       Override iteration count
  --alpha <number>             Override clone probability
  --ess-threshold <number>     Override relative ESS threshold
  --seed <integer>             Override deterministic seed
  --trace                      Print every SMC step
  --log-file <path>            Write deterministic JSONL trace
  --help                       Show this help
`;

class CliArgumentError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CliArgumentError";
  }
}

function requireValue(args: readonly string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new CliArgumentError(`${flag} requires a value`);
  }
  return value;
}

function finiteNumber(value: string, flag: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new CliArgumentError(`${flag} must be a finite number`);
  return parsed;
}

function safeInteger(value: string, flag: string): number {
  const parsed = finiteNumber(value, flag);
  if (!Number.isSafeInteger(parsed)) throw new CliArgumentError(`${flag} must be a safe integer`);
  return parsed;
}

export function parseCliArgs(args: readonly string[]): CliOptions | "help" {
  let configPath: string | undefined;
  let proposal: "catalog" | "ollama" = "catalog";
  let trace = false;
  let logFile: string | undefined;
  let model = "gpt-oss:20b";
  let ollamaUrl = "http://localhost:11434/v1";
  let timeoutMs = 180_000;
  let temperature = 0;
  let reasoningEffort: "low" | "medium" | "high" = "low";
  const overrides: {
    particles?: number;
    iterations?: number;
    cloneProbability?: number;
    essThreshold?: number;
    seed?: number;
  } = {};

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]!;
    if (argument === "--help" || argument === "-h") return "help";
    if (argument === "--trace") {
      trace = true;
      continue;
    }
    if (!argument.startsWith("--")) {
      if (configPath !== undefined) {
        throw new CliArgumentError(`unexpected positional argument: ${argument}`);
      }
      configPath = argument;
      continue;
    }

    const value = requireValue(args, index, argument);
    index += 1;
    if (argument === "--proposal") {
      if (value !== "catalog" && value !== "ollama") {
        throw new CliArgumentError("--proposal must be catalog or ollama");
      }
      proposal = value;
    } else if (argument === "--model") {
      model = value;
    } else if (argument === "--ollama-url") {
      ollamaUrl = value;
    } else if (argument === "--timeout-ms") {
      timeoutMs = safeInteger(value, argument);
      if (timeoutMs <= 0) throw new CliArgumentError("--timeout-ms must be positive");
    } else if (argument === "--temperature") {
      temperature = finiteNumber(value, argument);
      if (temperature < 0) throw new CliArgumentError("--temperature must be nonnegative");
    } else if (argument === "--reasoning-effort") {
      if (value !== "low" && value !== "medium" && value !== "high") {
        throw new CliArgumentError("--reasoning-effort must be low, medium, or high");
      }
      reasoningEffort = value;
    } else if (argument === "--log-file") {
      logFile = value;
    } else if (argument === "--particles") {
      overrides.particles = safeInteger(value, argument);
    } else if (argument === "--iterations") {
      overrides.iterations = safeInteger(value, argument);
    } else if (argument === "--alpha") {
      overrides.cloneProbability = finiteNumber(value, argument);
    } else if (argument === "--ess-threshold") {
      overrides.essThreshold = finiteNumber(value, argument);
    } else if (argument === "--seed") {
      overrides.seed = safeInteger(value, argument);
    } else {
      throw new CliArgumentError(`unknown option: ${argument}`);
    }
  }

  if (configPath === undefined) throw new CliArgumentError("a configuration path is required");
  return {
    configPath,
    proposal,
    trace,
    ...(logFile === undefined ? {} : { logFile }),
    model,
    ollamaUrl,
    timeoutMs,
    temperature,
    reasoningEffort,
    overrides,
  };
}

function createProposer(options: CliOptions): Proposer {
  if (options.proposal === "catalog") return new CatalogProposer();
  return new OllamaProposer({
    model: options.model,
    baseUrl: options.ollamaUrl,
    timeoutMs: options.timeoutMs,
    temperature: options.temperature,
    reasoningEffort: options.reasoningEffort,
  });
}

export async function main(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  try {
    const options = parseCliArgs(args);
    if (options === "help") {
      console.log(USAGE);
      return 0;
    }
    const raw = readFileSync(options.configPath, "utf8");
    const config = withConfigOverrides(parseExperimentConfig(raw), options.overrides);
    const trace = new TraceSink({
      enabled: options.trace,
      ...(options.logFile === undefined ? {} : { logFile: options.logFile }),
    });
    const result = await new SynthesisEngine({
      config,
      proposer: createProposer(options),
      trace,
    }).run();

    console.log(`[result] task: ${config.name}`);
    console.log(
      `[result] signature: ${renderType(config.inputType)} -> ${renderType(config.outputType)}`,
    );
    console.log(`[result] program: ${renderProgram(result.best.expression, config.inputType)}`);
    console.log(`[result] body AST: ${jsonStringify(result.best.expression)}`);
    console.log(
      `[result] loss=${formatNumber(result.best.score.totalLoss)} cost=${result.best.score.cost} log-target=${formatNumber(result.best.score.logTarget)} weight=${formatNumber(result.best.weight)}`,
    );
    console.log(`[result] exact on every example: ${result.exact}`);
    console.log(`[result] proposal calls: ${result.proposalCalls}`);
    if (trace.logFile !== undefined) console.log(`[result] JSONL trace: ${trace.logFile}`);
    return 0;
  } catch (error) {
    if (error instanceof CliArgumentError || error instanceof ConfigurationError) {
      console.error(`[error] ${error.message}`);
      console.error("Run with --help for usage.");
      return 2;
    }
    const detail = error instanceof Error ? error.stack ?? error.message : String(error);
    console.error(`[error] ${detail}`);
    return 1;
  }
}

const entryPath = process.argv[1];
if (entryPath !== undefined && import.meta.url === pathToFileURL(entryPath).href) {
  process.exitCode = await main();
}
