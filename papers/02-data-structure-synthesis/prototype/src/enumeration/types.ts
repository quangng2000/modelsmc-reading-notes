import type {
  Expression,
  ObjectType,
  PrimitiveType,
  PrimitiveValue,
} from "../ast.js";

export type PrimitiveTypeName = PrimitiveType["kind"];

export interface TypedVariableOption {
  readonly name: string;
  readonly type: PrimitiveTypeName;
}

// String names are the legacy form and continue to denote int variables.
export type VariableOption = string | TypedVariableOption;

export interface TypedEvaluationBinding {
  readonly name: string;
  readonly type: PrimitiveTypeName | ObjectType;
  readonly value: PrimitiveValue;
}

export interface LegacyEvaluationBinding {
  readonly name: string;
  readonly value: number;
}

export type EvaluationBinding =
  | LegacyEvaluationBinding
  | TypedEvaluationBinding;

export interface Example {
  readonly input: number;
  readonly output: number;
}

export interface CostBucket {
  readonly cost: number;
  readonly expressions: readonly Expression[];
}

export interface BestFirstCandidate {
  readonly expression: Expression;
  readonly cost: number;
}

export interface SearchOptions {
  readonly maxCost?: number;
  readonly constants?: readonly number[];
  readonly stringConstants?: readonly string[];
  readonly variables?: readonly VariableOption[];
  readonly targetType?: PrimitiveTypeName;
}

export interface SynthesisResult {
  readonly expression: Expression;
  readonly cost: number;
  readonly candidatesTested: number;
}
