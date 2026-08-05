import type {
  Expression,
  ObjectType,
  PrimitiveType,
  PrimitiveValue,
} from "../ast.js";
import type {
  FoldStepExample,
  PredicateExample,
  ScalarExample,
} from "../deduction/index.js";
import type {
  CostBucket,
  SearchOptions,
} from "../enumeration/index.js";

export interface SearchTraceEntry {
  readonly stage: "skeleton" | "completed-program";
  readonly cost: number;
  readonly count: number;
}

export type MapSynthesisResult<
  Input extends PrimitiveValue = number,
  Output extends PrimitiveValue = number,
> =
  | {
      readonly kind: "synthesized";
      readonly program: Expression;
      readonly cost: number;
      readonly candidatesTested: number;
      readonly inferredExamples: readonly ScalarExample<Input, Output>[];
      readonly trace: readonly SearchTraceEntry[];
    }
  | {
      readonly kind: "refuted" | "underconstrained" | "not-found";
      readonly reason: string;
      readonly trace: readonly SearchTraceEntry[];
    };

export type OutputValue = readonly PrimitiveValue[] | PrimitiveValue;

export interface IOExample {
  readonly input: readonly PrimitiveValue[];
  readonly output: OutputValue;
}

export interface SynthesisSignature {
  readonly inputType: { readonly kind: "list"; readonly element: PrimitiveType };
  readonly outputType:
    | PrimitiveType
    | { readonly kind: "list"; readonly element: PrimitiveType };
}

export type FamilyName = "map" | "filter" | "fold";

export type FamilyDeduction =
  | {
      readonly kind: "map";
      readonly inputType: PrimitiveType;
      readonly outputType: PrimitiveType;
      readonly examples: readonly ScalarExample<PrimitiveValue, PrimitiveValue>[];
    }
  | {
      readonly kind: "filter";
      readonly elementType: PrimitiveType;
      readonly examples: readonly PredicateExample<PrimitiveValue>[];
    }
  | {
      readonly kind: "fold";
      readonly elementType: PrimitiveType;
      readonly accumulatorType: PrimitiveType;
      readonly init: PrimitiveValue | undefined;
      readonly initCandidates: readonly PrimitiveValue[];
      readonly steps: readonly FoldStepExample<PrimitiveValue, PrimitiveValue>[];
    };

export type CandidateTestResult =
  | {
      readonly disposition: "rejected-by-deduction";
      readonly reason: string;
    }
  | {
      readonly disposition: "rejected-by-examples" | "accepted";
      readonly reason?: never;
    };

type CandidateTestedEvent = {
  readonly type: "candidate-tested";
  readonly number: number;
  readonly family: FamilyName;
  readonly program: Expression;
  readonly cost: number;
} & CandidateTestResult;

export type SynthesisEvent =
  | {
      readonly type: "search-start";
      readonly exampleCount: number;
      readonly families: readonly FamilyName[];
    }
  | {
      readonly type: "family-refuted";
      readonly family: FamilyName;
      readonly reason: string;
    }
  | {
      readonly type: "family-viable";
      readonly family: FamilyName;
      readonly skeleton: Expression;
      readonly cost: number;
      readonly deduction: FamilyDeduction;
    }
  | CandidateTestedEvent
  | {
      readonly type: "search-finished";
      readonly outcome: SynthesisOutcome["kind"];
      readonly candidatesTested: number;
    };

export interface SynthesisOptions extends SearchOptions {
  readonly inputType?: ObjectType;
  readonly outputType?: ObjectType;
  readonly onEvent?: (event: SynthesisEvent) => void;
}

export interface FamilyReport {
  readonly family: FamilyName;
  readonly status: "viable" | "refuted";
  readonly reason: string | undefined;
}

export type SynthesisOutcome =
  | {
      readonly kind: "synthesized";
      readonly family: FamilyName;
      readonly program: Expression;
      readonly cost: number;
      readonly candidatesTested: number;
      readonly familyReports: readonly FamilyReport[];
      readonly trace: readonly SearchTraceEntry[];
    }
  | {
      readonly kind: "refuted" | "not-found";
      readonly reason: string;
      readonly familyReports: readonly FamilyReport[];
      readonly trace: readonly SearchTraceEntry[];
    };

export interface FoldConstraint {
  readonly init: PrimitiveValue | undefined;
  readonly steps: readonly FoldStepExample<PrimitiveValue, PrimitiveValue>[];
}

export type FamilyConstraintResult =
  | { readonly kind: "satisfied" }
  | { readonly kind: "failed"; readonly reason: string };

// A body-bucket cache holds one enumerator generator per family and pulls
// buckets on demand, so each exact-cost bucket is generated at most once even
// though the search loop revisits families at every total-cost level.
export interface BodyBucketCache {
  readonly generator: Generator<CostBucket>;
  readonly byCost: (readonly Expression[])[];
  exhausted: boolean;
}

export interface FamilyPlan<F extends FamilyName = FamilyName> {
  readonly family: F;
  readonly skeleton: Expression;
  readonly programType: ObjectType;
  readonly inputElementType: PrimitiveType;
  readonly bodyType: PrimitiveType;
  readonly completionOverhead: number;
}

export type AnyFamilyPlan =
  | FamilyPlan<"map">
  | FamilyPlan<"filter">
  | FamilyPlan<"fold">;

export type ViableFamily =
  | {
      readonly family: "map";
      readonly plan: FamilyPlan<"map">;
      readonly scalarExamples: readonly ScalarExample<PrimitiveValue, PrimitiveValue>[];
      readonly buckets: BodyBucketCache;
    }
  | {
      readonly family: "filter";
      readonly plan: FamilyPlan<"filter">;
      readonly predicateExamples: readonly PredicateExample<PrimitiveValue>[];
      readonly buckets: BodyBucketCache;
    }
  | {
      readonly family: "fold";
      readonly plan: FamilyPlan<"fold">;
      readonly foldConstraint: FoldConstraint;
      readonly initCandidates: readonly PrimitiveValue[];
      readonly buckets: BodyBucketCache;
    };

export type FamilyPreparation =
  | { readonly kind: "viable"; readonly viable: ViableFamily }
  | { readonly kind: "refuted"; readonly reason: string };

export interface CompletedCandidate {
  readonly family: FamilyName;
  readonly program: Expression;
  readonly body: Expression;
  readonly init: PrimitiveValue | undefined;
}
