import type { ExperimentConfig } from "../config/index.js";
import { formatNumber, type TraceSink } from "../engine/index.js";
import { runGrammarSmc } from "../grammar-smc/index.js";
import { jsonStringify, renderProgram } from "../ast/index.js";
import type { CliOptions } from "./arguments.js";

export function runGrammarControl(
  config: ExperimentConfig,
  options: CliOptions,
  trace: TraceSink,
): void {
  const result = runGrammarSmc({
    config,
    maxCost: options.grammarMaxCost,
    generationLimit: options.grammarLimit,
    betaMax: options.betaMax,
    movesPerStage: options.movesPerStage,
    trace,
  });
  const exactPrograms = result.space.states.filter((state) => state.score.exactProgram).length;

  console.log("[result] mode: calibrated finite-grammar SMC control (no LLM)");
  console.log(
    `[result] grammar states=${result.space.states.length} exact-programs=${exactPrograms} max-cost=${result.space.maxCost}`,
  );
  console.log(`[result] sampled best: ${renderProgram(result.best.program, config.inputType)}`);
  console.log(`[result] body AST: ${jsonStringify(result.best.program)}`);
  console.log(
    `[result] sampled best loss=${formatNumber(result.best.score.totalLoss)} cost=${result.best.score.cost} exact=${result.best.score.exactProgram}`,
  );
  console.log(
    `[result] exact-program mass: particles=${formatNumber(result.exactProgramMassEstimate)} enumeration=${formatNumber(result.reference.exactProgramMass)}`,
  );
  console.log(
    `[result] mean loss: particles=${formatNumber(result.meanLossEstimate)} enumeration=${formatNumber(result.reference.meanLoss)}`,
  );
  console.log(
    `[result] log(Z_beta/Z_0): particles=${formatNumber(result.logNormalizingConstantEstimate)} enumeration=${formatNumber(result.reference.logNormalizingConstant)} error=${formatNumber(result.logNormalizingConstantError)}`,
  );
  console.log(`[result] total-variation distance: ${formatNumber(result.totalVariationDistance)}`);
}
