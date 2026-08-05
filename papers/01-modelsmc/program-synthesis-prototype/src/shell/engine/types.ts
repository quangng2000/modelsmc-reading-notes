import type { Program } from "../../core/language.verify.js";
import type { ValidScore } from "../scoring/index.js";

export interface Particle {
  readonly id: number;
  readonly parentId: number | null;
  readonly expression: Program;
  readonly score: ValidScore;
  readonly weight: number;
  readonly origin: "initial" | "clone" | "catalog" | "ollama" | "fallback";
  readonly rationale: string;
}

export interface IterationSummary {
  readonly iteration: number;
  readonly essBefore: number;
  readonly relativeEssBefore: number;
  readonly resampled: boolean;
  readonly ancestors: readonly number[];
  readonly bestParticleId: number;
  readonly bestExpression: Program;
  readonly bestWeight: number;
}

export interface SynthesisResult {
  readonly particles: readonly Particle[];
  readonly iterations: readonly IterationSummary[];
  readonly best: Particle;
  readonly exact: boolean;
  readonly proposalCalls: number;
  readonly firstExactIteration: number | null;
  readonly firstExactProposalCall: number | null;
  readonly championLineage: readonly Particle[];
}
