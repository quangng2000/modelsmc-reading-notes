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

function describeShape(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "an array";
  return `a ${typeof value}`;
}

// Forced tool calls do not strictly enforce the input schema server-side, so
// live runs produce benign envelope variants alongside well-formed expressions:
// `rationale` omitted (or null) after a large expression tree, and extra keys
// carrying only null. Tolerate exactly those variants; an extra key holding a
// real value still fails, and the expression itself always goes through the
// bounded AST decoder. Rejections enumerate the received keys so that trace
// and summary failureReasons capture the actual envelope shape.
export function decodeEnvelope(
  value: unknown,
  context: ProposalContext,
  source: ProposalResult["source"],
): ProposalResult {
  if (!isRecord(value)) {
    throw new ProposalError(
      `model response must be a JSON object; received ${describeShape(value)}`,
    );
  }
  const keys = Object.keys(value).sort();
  const extraKeys = keys.filter(
    (key) => key !== "expression" && key !== "rationale" && value[key] !== null,
  );
  if (value.expression === null || value.expression === undefined || extraKeys.length > 0) {
    const received = keys
      .map((key) => (value[key] === null ? `${key} (null)` : key))
      .join(", ");
    throw new ProposalError(
      `model response must contain expression and rationale; received: ${received === "" ? "(no keys)" : received}`,
    );
  }
  const rawRationale = value.rationale ?? "";
  if (typeof rawRationale !== "string") {
    throw new ProposalError(
      `model response rationale must be a string; received ${describeShape(rawRationale)}`,
    );
  }
  const rationale = rawRationale.trim().slice(0, 2_000);
  const expression = decodeProgram(normalizeExpression(value.expression), {
    integerConstants: context.integerConstants,
    maxDepth: context.maxDepth,
    maxNodes: context.maxNodes,
  });
  assertCombinatorScope(expression);
  return { expression, rationale, source };
}
