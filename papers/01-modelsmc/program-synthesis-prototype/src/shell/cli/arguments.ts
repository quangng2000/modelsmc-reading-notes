import type { ConfigOverrides } from "../config/index.js";

export interface CliOptions {
  readonly configPath: string;
  readonly proposal: "catalog" | "ollama" | "anthropic" | "grammar-smc";
  readonly trace: boolean;
  readonly logFile?: string;
  readonly model: string;
  readonly ollamaUrl: string;
  readonly anthropicUrl: string;
  readonly timeoutMs: number;
  readonly temperature?: number;
  readonly maxTokens: number;
  readonly reasoningEffort?: "low" | "medium" | "high";
  readonly grammarMaxCost: number;
  readonly grammarLimit: number;
  readonly betaMax: number;
  readonly movesPerStage: number;
  readonly overrides: ConfigOverrides;
}

export const USAGE = `Usage:
  npm run synthesize -- <config.json> [options]

Options:
  --proposal <catalog|ollama|anthropic|grammar-smc>
                               Search backend (default: catalog)
  --model <name>               Model id (default: gpt-oss:20b for ollama, claude-sonnet-5 for anthropic)
  --ollama-url <url>           OpenAI-compatible base URL for the ollama backend
  --anthropic-url <url>        Anthropic API base URL (default: https://api.anthropic.com)
  --timeout-ms <integer>       Request timeout (default: 180000)
  --temperature <number>       Sampling temperature (ollama default 0; omitted unless set, as the Claude 5 family rejects it)
  --max-tokens <integer>       Response token limit (default: 2048)
  --reasoning-effort <level>   Send low, medium, or high reasoning (ollama backend only)
  --grammar-max-cost <integer> Exact grammar cost bound (default: 5)
  --grammar-limit <integer>    Enumeration safety limit (default: 100000)
  --beta-max <number>          Final inverse temperature (default: 1)
  --moves-per-stage <integer>  Independent MH moves per stage (default: 1)
  --particles <integer>        Override particle count
  --iterations <integer>       Override iteration count
  --alpha <number>             Override clone probability (LLM/catalog only)
  --ess-threshold <number>     Override relative ESS threshold
  --seed <integer>             Override deterministic seed
  --trace                      Print every SMC step
  --log-file <path>            Write deterministic JSONL trace
  --help                       Show this help
`;

export class CliArgumentError extends Error {
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
  let proposal: "catalog" | "ollama" | "anthropic" | "grammar-smc" = "catalog";
  let trace = false;
  let logFile: string | undefined;
  let model: string | undefined;
  let ollamaUrl = "http://localhost:11434/v1";
  let anthropicUrl = "https://api.anthropic.com";
  let timeoutMs = 180_000;
  let temperature: number | undefined;
  let maxTokens = 2_048;
  let reasoningEffort: "low" | "medium" | "high" | undefined;
  let grammarMaxCost = 5;
  let grammarLimit = 100_000;
  let betaMax = 1;
  let movesPerStage = 1;
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
      if (
        value !== "catalog" &&
        value !== "ollama" &&
        value !== "anthropic" &&
        value !== "grammar-smc"
      ) {
        throw new CliArgumentError(
          "--proposal must be catalog, ollama, anthropic, or grammar-smc",
        );
      }
      proposal = value;
    } else if (argument === "--model") {
      model = value;
    } else if (argument === "--ollama-url") {
      ollamaUrl = value;
    } else if (argument === "--anthropic-url") {
      anthropicUrl = value;
    } else if (argument === "--timeout-ms") {
      timeoutMs = safeInteger(value, argument);
      if (timeoutMs <= 0) throw new CliArgumentError("--timeout-ms must be positive");
    } else if (argument === "--temperature") {
      temperature = finiteNumber(value, argument);
      if (temperature < 0) throw new CliArgumentError("--temperature must be nonnegative");
    } else if (argument === "--max-tokens") {
      maxTokens = safeInteger(value, argument);
      if (maxTokens <= 0) throw new CliArgumentError("--max-tokens must be positive");
    } else if (argument === "--reasoning-effort") {
      if (value !== "low" && value !== "medium" && value !== "high") {
        throw new CliArgumentError("--reasoning-effort must be low, medium, or high");
      }
      reasoningEffort = value;
    } else if (argument === "--grammar-max-cost") {
      grammarMaxCost = safeInteger(value, argument);
      if (grammarMaxCost <= 0) throw new CliArgumentError("--grammar-max-cost must be positive");
    } else if (argument === "--grammar-limit") {
      grammarLimit = safeInteger(value, argument);
      if (grammarLimit <= 0) throw new CliArgumentError("--grammar-limit must be positive");
    } else if (argument === "--beta-max") {
      betaMax = finiteNumber(value, argument);
      if (betaMax <= 0) throw new CliArgumentError("--beta-max must be positive");
    } else if (argument === "--moves-per-stage") {
      movesPerStage = safeInteger(value, argument);
      if (movesPerStage < 0) {
        throw new CliArgumentError("--moves-per-stage must be nonnegative");
      }
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
  const resolvedModel = model ?? (proposal === "anthropic" ? "claude-sonnet-5" : "gpt-oss:20b");
  return {
    configPath,
    proposal,
    trace,
    ...(logFile === undefined ? {} : { logFile }),
    model: resolvedModel,
    ollamaUrl,
    anthropicUrl,
    timeoutMs,
    ...(temperature === undefined ? {} : { temperature }),
    maxTokens,
    ...(reasoningEffort === undefined ? {} : { reasoningEffort }),
    grammarMaxCost,
    grammarLimit,
    betaMax,
    movesPerStage,
    overrides,
  };
}
