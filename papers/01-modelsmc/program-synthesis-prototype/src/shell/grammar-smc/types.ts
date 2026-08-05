import type { Program } from "../../core/language.verify.js";
import type { ExperimentConfig } from "../config/index.js";
import type { ValidScore } from "../scoring/index.js";
import type { TraceSink } from "../engine/index.js";

export interface GrammarState {
  readonly index: number;
  readonly key: string;
  readonly program: Program;
  readonly score: ValidScore;
  readonly logPriorProbability: number;
  readonly priorProbability: number;
}

export interface GrammarSpace {
  readonly states: readonly GrammarState[];
  readonly maxCost: number;
  readonly priorLogNormalizer: number;
}

export interface GrammarTarget {
  readonly beta: number;
  readonly probabilities: readonly number[];
  /** log(Z_beta / Z_0); the grammar prior is normalized, so Z_0 = 1. */
  readonly logNormalizingConstant: number;
  readonly exactProgramMass: number;
  readonly meanLoss: number;
}

export interface GrammarSmcOptions {
  readonly config: ExperimentConfig;
  readonly maxCost: number;
  readonly generationLimit: number;
  readonly betaMax: number;
  readonly movesPerStage: number;
  readonly trace: TraceSink;
}

export interface GrammarSmcStage {
  readonly stage: number;
  readonly beta: number;
  readonly effectiveSampleSize: number;
  readonly resampled: boolean;
  readonly acceptedMoves: number;
  readonly attemptedMoves: number;
  readonly logNormalizingConstantEstimate: number;
}

export interface GrammarSmcParticle {
  readonly stateIndex: number;
  readonly weight: number;
}

export interface GrammarSmcResult {
  readonly space: GrammarSpace;
  readonly reference: GrammarTarget;
  readonly particles: readonly GrammarSmcParticle[];
  readonly stages: readonly GrammarSmcStage[];
  readonly best: GrammarState;
  readonly exactProgramMassEstimate: number;
  readonly meanLossEstimate: number;
  readonly logNormalizingConstantEstimate: number;
  readonly logNormalizingConstantError: number;
  readonly totalVariationDistance: number;
}
