import {
  expressionUsesInput,
  type Program,
} from "../../core/language.verify.js";
import { decodeProgram } from "../ast/decode.js";
import { ProposalError, type ProposalContext, type ProposalResult } from "./index.js";
import { isRecord } from "./schema.js";

function assertCombinatorScope(program: Program): void {
  if (program.kind === "MapProgram" && expressionUsesInput(program.mapper)) {
    throw new ProposalError(
      "MapProgram.mapper must not reference the outer Input; use Item for the current element",
    );
  }
  if (
    program.kind === "FoldRightProgram" &&
    (expressionUsesInput(program.initial) || expressionUsesInput(program.reducer))
  ) {
    throw new ProposalError(
      "FoldRightProgram initial/reducer must not reference the outer Input",
    );
  }
}

// Some providers do not strictly enforce a recursive tool/response schema and
// return the nested `expression` AST as a JSON-encoded string rather than an
// object. Accept either form here; a non-string value passes through unchanged,
// so the strict-schema path is unaffected.
function normalizeExpression(raw: unknown): unknown {
  if (typeof raw !== "string") return raw;
  try {
    return JSON.parse(raw) as unknown;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new ProposalError(
      `model response expression is a string but not valid JSON: ${detail}`,
    );
  }
}

export function decodeEnvelope(
  value: unknown,
  context: ProposalContext,
  source: ProposalResult["source"],
): ProposalResult {
  if (!isRecord(value)) throw new ProposalError("model response must be a JSON object");
  const keys = Object.keys(value).sort();
  if (keys.length !== 2 || keys[0] !== "expression" || keys[1] !== "rationale") {
    throw new ProposalError("model response must contain exactly expression and rationale");
  }
  if (typeof value.rationale !== "string") {
    throw new ProposalError("model response rationale must be a string");
  }
  const rationale = value.rationale.trim().slice(0, 2_000);
  const expression = decodeProgram(normalizeExpression(value.expression), {
    integerConstants: context.integerConstants,
    maxDepth: context.maxDepth,
    maxNodes: context.maxNodes,
  });
  assertCombinatorScope(expression);
  return { expression, rationale, source };
}
