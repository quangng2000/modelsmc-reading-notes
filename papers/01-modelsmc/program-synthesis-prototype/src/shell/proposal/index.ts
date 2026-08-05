import type { Program, RuntimeValue, StaticType } from "../../core/language.verify.js";
import type { ValidScore } from "../scoring/index.js";

export interface ProposalContext {
  readonly requestIndex: number;
  readonly iteration?: number;
  readonly slot?: number;
  readonly avoidPrograms?: readonly Program[];
  readonly inputType: StaticType;
  readonly outputType: StaticType;
  readonly examples: readonly { readonly input: RuntimeValue; readonly output: RuntimeValue }[];
  readonly integerConstants: readonly bigint[];
  readonly maxDepth: number;
  readonly maxNodes: number;
  readonly maxCost: number;
  readonly ancestor: Program;
  readonly ancestorScore: ValidScore;
  readonly ancestorFeedback?: string;
}

export interface ProposalResult {
  readonly expression: Program;
  readonly rationale: string;
  readonly source: "catalog" | "ollama" | "anthropic";
}

export interface Proposer {
  readonly name: string;
  propose(context: ProposalContext): Promise<ProposalResult>;
}

export class ProposalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ProposalError";
  }
}
