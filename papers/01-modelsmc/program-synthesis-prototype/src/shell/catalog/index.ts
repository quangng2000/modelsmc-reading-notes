import type { StaticType } from "../../core/language.verify.js";
import type { ProposalContext, ProposalResult, Proposer } from "../proposal/index.js";
import { buildCatalog } from "./build.js";

export class CatalogProposer implements Proposer {
  readonly name = "catalog";

  async propose(context: ProposalContext): Promise<ProposalResult> {
    const candidates = buildCatalog(context);
    if (candidates.length === 0) throw new Error("the configured grammar produced no candidates");
    const expression = candidates[context.requestIndex % candidates.length]!;
    return {
      expression,
      rationale: `offline catalog candidate ${context.requestIndex % candidates.length + 1}/${candidates.length}`,
      source: "catalog",
    };
  }
}

export function inferredCatalogSize(
  inputType: StaticType,
  outputType: StaticType,
  constants: readonly bigint[],
  maxCost: number,
): number {
  const placeholderScore = {
    kind: "Scored" as const,
    inferredType: outputType,
    evaluations: [],
    totalLoss: 0,
    exactMatches: 0,
    cost: 1,
    logTarget: 0,
    exactProgram: false,
  };
  return buildCatalog({
    requestIndex: 0,
    inputType,
    outputType,
    examples: [],
    integerConstants: constants,
    maxDepth: 1,
    maxNodes: 1,
    maxCost,
    ancestor: { kind: "ExpressionProgram", body: { kind: "Input" } },
    ancestorScore: placeholderScore,
  }).length;
}
