import { INT, listOf, typeEquals } from "../ast.js";
import { expressionCost } from "../cost.js";
import {
  deduceMapExamples,
  type MapExample,
} from "../deduction/index.js";
import {
  DEFAULT_MAX_COST,
  validateSearchOptions,
} from "../enumeration/index.js";
import {
  emptyFrontier,
  popMinFrontier,
  pushFrontier,
  type Frontier,
} from "../frontier.js";
import {
  bodiesAtCost,
  checkFamilyConstraint,
  completeFamilyBody,
  prepareFamily,
} from "./families.js";
import {
  emitCandidateTested,
  emitEvent,
  summarizeDeduction,
} from "./events.js";
import { resolveSignature } from "./signature.js";
import {
  MAP_SKELETON,
  assertPlanWellTyped,
  makeFilterPlan,
  makeFoldPlan,
  makeMapPlan,
} from "./skeletons.js";
import type {
  AnyFamilyPlan,
  CompletedCandidate,
  FamilyReport,
  IOExample,
  MapSynthesisResult,
  SearchTraceEntry,
  SynthesisOptions,
  SynthesisOutcome,
  SynthesisSignature,
  ViableFamily,
} from "./types.js";
import { programMatches } from "./validation.js";

export function synthesizeProgram(
  examples: readonly IOExample[],
  options: SynthesisOptions = {},
): SynthesisOutcome {
  if (examples.length === 0) {
    throw new Error("At least one input-output example is required.");
  }

  const signature = resolveSignature(examples, options);
  return searchFamilies(
    examples,
    options,
    signature,
    plansForSignature(signature),
  );
}

// Compatibility wrapper for the original list<int> -> list<int> map-only
// API. The generalized engine still performs the actual search.
export function synthesizeMap(
  examples: readonly MapExample[],
  options: SynthesisOptions = {},
): MapSynthesisResult {
  validateSearchOptions(options);

  const skeletonTrace: readonly SearchTraceEntry[] = [
    { stage: "skeleton", cost: expressionCost(MAP_SKELETON), count: 1 },
  ];

  const deduction = deduceMapExamples(examples, INT, INT);
  if (deduction.kind === "refuted") {
    return { kind: "refuted", reason: deduction.reason, trace: skeletonTrace };
  }
  if (deduction.examples.length === 0) {
    return {
      kind: "underconstrained",
      reason: "empty lists provide no examples for the map function hole",
      trace: skeletonTrace,
    };
  }

  const signature: SynthesisSignature = {
    inputType: listOf(INT),
    outputType: listOf(INT),
  };
  const outcome = searchFamilies(
    examples,
    options,
    signature,
    [makeMapPlan(INT, INT)],
  );
  switch (outcome.kind) {
    case "synthesized":
      return {
        kind: "synthesized",
        program: outcome.program,
        cost: outcome.cost,
        candidatesTested: outcome.candidatesTested,
        inferredExamples: deduction.examples,
        trace: outcome.trace,
      };
    case "refuted":
      return { kind: "refuted", reason: outcome.reason, trace: outcome.trace };
    case "not-found":
      return {
        kind: "not-found",
        reason: "no int -> int expression was found within the cost bound",
        trace: outcome.trace,
      };
  }
}

function plansForSignature(
  signature: SynthesisSignature,
): readonly AnyFamilyPlan[] {
  const inputElement = signature.inputType.element;
  const output = signature.outputType;
  if (output.kind === "list") {
    const plans: AnyFamilyPlan[] = [
      makeMapPlan(inputElement, output.element),
    ];
    if (typeEquals(inputElement, output.element)) {
      plans.push(makeFilterPlan(inputElement));
    }
    return plans;
  }
  return [makeFoldPlan(inputElement, output)];
}

function searchFamilies(
  examples: readonly IOExample[],
  options: SynthesisOptions,
  signature: SynthesisSignature,
  plans: readonly AnyFamilyPlan[],
): SynthesisOutcome {
  validateSearchOptions(options);

  emitEvent(options, {
    type: "search-start",
    exampleCount: examples.length,
    families: plans.map(({ family }) => family),
  });

  const maxCost = options.maxCost ?? DEFAULT_MAX_COST;
  const trace: SearchTraceEntry[] = [];
  const familyReports: FamilyReport[] = [];
  const viableFamilies: ViableFamily[] = [];

  for (const plan of plans) {
    const preparation = prepareFamily(plan, examples, options);
    if (preparation.kind === "refuted") {
      familyReports.push({
        family: plan.family,
        status: "refuted",
        reason: preparation.reason,
      });
      emitEvent(options, {
        type: "family-refuted",
        family: plan.family,
        reason: preparation.reason,
      });
      continue;
    }

    assertPlanWellTyped(plan);
    familyReports.push({
      family: plan.family,
      status: "viable",
      reason: undefined,
    });
    viableFamilies.push(preparation.viable);
    emitEvent(options, {
      type: "family-viable",
      family: plan.family,
      skeleton: plan.skeleton,
      cost: expressionCost(plan.skeleton),
      deduction: summarizeDeduction(preparation.viable),
    });
  }

  if (viableFamilies.length === 0) {
    const outcome: SynthesisOutcome = {
      kind: "refuted",
      reason: familyReports
        .map((report) => `${report.family}: ${report.reason ?? "refuted"}`)
        .join("; "),
      familyReports,
      trace,
    };
    emitEvent(options, {
      type: "search-finished",
      outcome: outcome.kind,
      candidatesTested: 0,
    });
    return outcome;
  }

  for (const viable of viableFamilies) {
    recordTrace(
      trace,
      "skeleton",
      expressionCost(viable.plan.skeleton),
    );
  }

  let frontier: Frontier<CompletedCandidate> = emptyFrontier();
  let candidatesTested = 0;
  let lastPoppedCost = 0;
  const maxCompletionOverhead = Math.max(
    ...plans.map(({ completionOverhead }) => completionOverhead),
  );

  // The outer loop preserves global nondecreasing total program cost across
  // all type-compatible family plans.
  for (
    let totalCost = 0;
    totalCost <= maxCost + maxCompletionOverhead;
    totalCost += 1
  ) {
    for (const viable of viableFamilies) {
      const bodyBudget =
        totalCost - viable.plan.completionOverhead;
      if (bodyBudget < 0 || bodyBudget > maxCost) {
        continue;
      }

      for (const body of bodiesAtCost(viable.buckets, bodyBudget)) {
        for (const candidate of completeFamilyBody(viable, body)) {
          if (expressionCost(candidate.program) !== totalCost) {
            throw new Error(
              "A completed program was scheduled at the wrong total cost.",
            );
          }
          frontier = pushFrontier(frontier, candidate, totalCost);
        }
      }
    }

    while (frontier.size > 0) {
      const popped = popMinFrontier(frontier);
      if (popped === undefined) {
        throw new Error("A nonempty frontier could not produce an item.");
      }
      frontier = popped.frontier;

      if (popped.cost > totalCost || popped.cost < lastPoppedCost) {
        throw new Error("Frontier pops must be nondecreasing in cost.");
      }
      lastPoppedCost = popped.cost;
      recordTrace(trace, "completed-program", popped.cost);
      candidatesTested += 1;

      const viable = viableFamilies.find(
        (entry) => entry.family === popped.item.family,
      );
      if (viable === undefined) {
        throw new Error("A candidate was popped for a non-viable family.");
      }

      const constraintResult = checkFamilyConstraint(popped.item, viable);
      if (constraintResult.kind === "failed") {
        emitCandidateTested(
          options,
          popped.item,
          popped.cost,
          candidatesTested,
          {
            disposition: "rejected-by-deduction",
            reason: constraintResult.reason,
          },
        );
        continue;
      }
      if (
        !examples.every((example) =>
          programMatches(popped.item.program, example, signature),
        )
      ) {
        emitCandidateTested(
          options,
          popped.item,
          popped.cost,
          candidatesTested,
          { disposition: "rejected-by-examples" },
        );
        continue;
      }

      emitCandidateTested(
        options,
        popped.item,
        popped.cost,
        candidatesTested,
        { disposition: "accepted" },
      );
      const outcome: SynthesisOutcome = {
        kind: "synthesized",
        family: popped.item.family,
        program: popped.item.program,
        cost: popped.cost,
        candidatesTested,
        familyReports,
        trace,
      };
      emitEvent(options, {
        type: "search-finished",
        outcome: outcome.kind,
        candidatesTested,
      });
      return outcome;
    }
  }

  const outcome: SynthesisOutcome = {
    kind: "not-found",
    reason: "no program within the cost bound satisfies every example",
    familyReports,
    trace,
  };
  emitEvent(options, {
    type: "search-finished",
    outcome: outcome.kind,
    candidatesTested,
  });
  return outcome;
}

function recordTrace(
  trace: SearchTraceEntry[],
  stage: SearchTraceEntry["stage"],
  cost: number,
): void {
  const last = trace[trace.length - 1];
  if (last !== undefined && last.stage === stage && last.cost === cost) {
    trace[trace.length - 1] = { stage, cost, count: last.count + 1 };
    return;
  }
  trace.push({ stage, cost, count: 1 });
}
