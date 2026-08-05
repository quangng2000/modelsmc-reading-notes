import type { RuntimeValue, StaticType } from "../../core/language.verify.js";

export interface ScoringOptions {
  readonly inputType: StaticType;
  readonly outputType: StaticType;
  readonly examples: readonly { readonly input: RuntimeValue; readonly output: RuntimeValue }[];
  readonly lossScale: number;
  readonly costScale: number;
  readonly lossCap: number;
  readonly maxCost: number;
}

export interface ExampleEvaluation {
  readonly input: RuntimeValue;
  readonly expected: RuntimeValue;
  readonly predicted: RuntimeValue;
  readonly exact: boolean;
  readonly loss: number;
}

export interface ValidScore {
  readonly kind: "Scored";
  readonly inferredType: StaticType;
  readonly evaluations: readonly ExampleEvaluation[];
  readonly totalLoss: number;
  readonly exactMatches: number;
  readonly cost: number;
  readonly logTarget: number;
  readonly exactProgram: boolean;
}

export interface RejectedScore {
  readonly kind: "Rejected";
  readonly reason: string;
  readonly inferredType?: StaticType;
  readonly cost?: number;
}

export type ProgramScore = ValidScore | RejectedScore;
