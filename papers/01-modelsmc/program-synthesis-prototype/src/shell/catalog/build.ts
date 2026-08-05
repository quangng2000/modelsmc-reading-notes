import {
  expressionCost,
  inferType,
  type Program,
} from "../../core/language.verify.js";
import { renderProgram } from "../ast/render.js";
import type { ProposalContext } from "../proposal/index.js";
import { listElementType } from "./expressions.js";
import { buildListCatalog, buildScalarToListCatalog } from "./lists.js";
import { buildScalarCatalog } from "./scalar.js";

export function buildCatalog(context: ProposalContext): Program[] {
  const inputIsList = listElementType(context.inputType) !== null;
  const outputIsList = listElementType(context.outputType) !== null;
  const candidates = inputIsList
    ? buildListCatalog(context)
    : outputIsList
      ? buildScalarToListCatalog(context)
      : buildScalarCatalog(context).map((body): Program => ({
        kind: "ExpressionProgram",
        body,
      }));
  const seen = new Set<string>();
  return candidates.filter((candidate) => {
    const type = inferType(candidate, context.inputType);
    if (type.kind === "TypeError" || type.inferred !== context.outputType) return false;
    if (expressionCost(candidate) > BigInt(context.maxCost)) return false;
    const rendered = renderProgram(candidate, context.inputType);
    if (seen.has(rendered)) return false;
    seen.add(rendered);
    return true;
  });
}
