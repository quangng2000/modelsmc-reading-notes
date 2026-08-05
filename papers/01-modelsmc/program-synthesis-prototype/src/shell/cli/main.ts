import { readFileSync } from "node:fs";
import {
  ConfigurationError,
  parseExperimentConfig,
  withConfigOverrides,
} from "../config/index.js";
import { jsonStringify, renderProgram, renderType } from "../ast/index.js";
import { formatNumber, SynthesisEngine, TraceSink } from "../engine/index.js";
import { CliArgumentError, parseCliArgs, USAGE } from "./arguments.js";
import { createProposer } from "./proposer.js";

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
