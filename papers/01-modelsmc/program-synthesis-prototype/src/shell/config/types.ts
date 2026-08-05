import type { Example, StaticType } from "../../core/language.verify.js";

export interface ExperimentConfig {
  readonly name: string;
  readonly examples: readonly Example[];
  readonly inputType: StaticType;
  readonly outputType: StaticType;
  readonly integerConstants: readonly bigint[];
  readonly particles: number;
  readonly iterations: number;
  readonly cloneProbability: number;
  readonly essThreshold: number;
  readonly seed: number;
  readonly lossScale: number;
  readonly costScale: number;
  readonly lossCap: number;
  readonly maxCost: number;
  readonly maxDepth: number;
  readonly maxNodes: number;
}

export interface ConfigOverrides {
  readonly particles?: number;
  readonly iterations?: number;
  readonly cloneProbability?: number;
  readonly essThreshold?: number;
  readonly seed?: number;
}
