import type {
  CandidateTestResult,
  CompletedCandidate,
  FamilyDeduction,
  SynthesisEvent,
  SynthesisOptions,
  ViableFamily,
} from "./types.js";

export function emitCandidateTested(
  options: SynthesisOptions,
  candidate: CompletedCandidate,
  cost: number,
  number: number,
  result: CandidateTestResult,
): void {
  emitEvent(options, {
    type: "candidate-tested",
    number,
    family: candidate.family,
    program: candidate.program,
    cost,
    ...result,
  });
}

export function summarizeDeduction(viable: ViableFamily): FamilyDeduction {
  switch (viable.family) {
    case "map":
      return {
        kind: "map",
        inputType: viable.plan.inputElementType,
        outputType: viable.plan.bodyType,
        examples: viable.scalarExamples,
      };
    case "filter":
      return {
        kind: "filter",
        elementType: viable.plan.inputElementType,
        examples: viable.predicateExamples,
      };
    case "fold":
      return {
        kind: "fold",
        elementType: viable.plan.inputElementType,
        accumulatorType: viable.plan.bodyType,
        init: viable.foldConstraint.init,
        initCandidates: viable.initCandidates,
        steps: viable.foldConstraint.steps,
      };
  }
}

export function emitEvent(
  options: SynthesisOptions,
  event: SynthesisEvent,
): void {
  options.onEvent?.(event);
}
