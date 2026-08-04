import type { Expr, Program } from "../core/language.verify.js";

export class AstDecodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AstDecodeError";
  }
}

export interface AstDecodeLimits {
  readonly integerConstants: readonly bigint[];
  readonly maxDepth: number;
  readonly maxNodes: number;
}

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assertExactKeys(record: JsonRecord, expected: readonly string[], path: string): void {
  const actual = Object.keys(record).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new AstDecodeError(
      `${path} must contain exactly: ${wanted.join(", ")}; received: ${actual.join(", ") || "(none)"}`,
    );
  }
}

function parseLiteral(value: unknown, allowed: readonly bigint[], path: string): bigint {
  if (typeof value !== "string" || !/^-?(?:0|[1-9][0-9]*)$/.test(value)) {
    throw new AstDecodeError(`${path} must be a decimal integer string`);
  }
  let parsed: bigint;
  try {
    parsed = BigInt(value);
  } catch {
    throw new AstDecodeError(`${path} is not a valid integer`);
  }
  if (!allowed.some((candidate) => candidate === parsed)) {
    throw new AstDecodeError(
      `${path}=${parsed.toString()} is not in the allowed constant catalog`,
    );
  }
  return parsed;
}

interface Decoder {
  readonly expression: (value: unknown) => Expr;
  readonly program: (value: unknown) => Program;
}

function createDecoder(limits: AstDecodeLimits): Decoder {
  let nodeCount = 0;
  const active = new WeakSet<object>();

  function visitExpr(candidate: unknown, depth: number, path: string): Expr {
    if (depth > limits.maxDepth) {
      throw new AstDecodeError(`${path} exceeds maximum AST depth ${limits.maxDepth}`);
    }
    if (!isRecord(candidate)) throw new AstDecodeError(`${path} must be an AST object`);
    if (active.has(candidate)) throw new AstDecodeError(`${path} contains a cycle`);
    active.add(candidate);
    nodeCount += 1;
    if (nodeCount > limits.maxNodes) {
      throw new AstDecodeError(`AST exceeds maximum node count ${limits.maxNodes}`);
    }

    try {
      const kind = candidate.kind;
      if (typeof kind !== "string") throw new AstDecodeError(`${path}.kind must be a string`);

      if (kind === "Input") {
        assertExactKeys(candidate, ["kind"], path);
        return { kind: "Input" };
      }
      if (kind === "Item") {
        assertExactKeys(candidate, ["kind"], path);
        return { kind: "Item" };
      }
      if (kind === "Accumulator") {
        assertExactKeys(candidate, ["kind"], path);
        return { kind: "Accumulator" };
      }
      if (kind === "IntLiteral") {
        assertExactKeys(candidate, ["kind", "intValue"], path);
        return {
          kind: "IntLiteral",
          intValue: parseLiteral(candidate.intValue, limits.integerConstants, `${path}.intValue`),
        };
      }
      if (kind === "BoolLiteral") {
        assertExactKeys(candidate, ["kind", "boolValue"], path);
        if (typeof candidate.boolValue !== "boolean") {
          throw new AstDecodeError(`${path}.boolValue must be a Boolean`);
        }
        return { kind: "BoolLiteral", boolValue: candidate.boolValue };
      }
      if (kind === "EmptyIntList") {
        assertExactKeys(candidate, ["kind"], path);
        return { kind: "EmptyIntList" };
      }
      if (kind === "EmptyBoolList") {
        assertExactKeys(candidate, ["kind"], path);
        return { kind: "EmptyBoolList" };
      }
      if (kind === "PrependInt") {
        assertExactKeys(candidate, ["kind", "head", "tail"], path);
        return {
          kind: "PrependInt",
          head: visitExpr(candidate.head, depth + 1, `${path}.head`),
          tail: visitExpr(candidate.tail, depth + 1, `${path}.tail`),
        };
      }
      if (kind === "PrependBool") {
        assertExactKeys(candidate, ["kind", "head", "tail"], path);
        return {
          kind: "PrependBool",
          head: visitExpr(candidate.head, depth + 1, `${path}.head`),
          tail: visitExpr(candidate.tail, depth + 1, `${path}.tail`),
        };
      }
      if (kind === "Not") {
        assertExactKeys(candidate, ["kind", "operand"], path);
        return {
          kind: "Not",
          operand: visitExpr(candidate.operand, depth + 1, `${path}.operand`),
        };
      }
      if (kind === "IfThenElse") {
        assertExactKeys(candidate, ["kind", "condition", "thenExpr", "elseExpr"], path);
        return {
          kind: "IfThenElse",
          condition: visitExpr(candidate.condition, depth + 1, `${path}.condition`),
          thenExpr: visitExpr(candidate.thenExpr, depth + 1, `${path}.thenExpr`),
          elseExpr: visitExpr(candidate.elseExpr, depth + 1, `${path}.elseExpr`),
        };
      }
      if (
        kind === "Add" ||
        kind === "Subtract" ||
        kind === "Multiply" ||
        kind === "LessThan" ||
        kind === "EqualInt" ||
        kind === "And"
      ) {
        assertExactKeys(candidate, ["kind", "left", "right"], path);
        const left = visitExpr(candidate.left, depth + 1, `${path}.left`);
        const right = visitExpr(candidate.right, depth + 1, `${path}.right`);
        if (kind === "Add") return { kind, left, right };
        if (kind === "Subtract") return { kind, left, right };
        if (kind === "Multiply") return { kind, left, right };
        if (kind === "LessThan") return { kind, left, right };
        if (kind === "EqualInt") return { kind, left, right };
        return { kind, left, right };
      }

      throw new AstDecodeError(`${path}.kind=${JSON.stringify(kind)} is not in the grammar`);
    } finally {
      active.delete(candidate);
    }
  }

  function visitProgram(candidate: unknown, depth: number, path: string): Program {
    if (depth > limits.maxDepth) {
      throw new AstDecodeError(`${path} exceeds maximum AST depth ${limits.maxDepth}`);
    }
    if (!isRecord(candidate)) throw new AstDecodeError(`${path} must be an AST object`);
    if (active.has(candidate)) throw new AstDecodeError(`${path} contains a cycle`);
    active.add(candidate);
    nodeCount += 1;
    if (nodeCount > limits.maxNodes) {
      throw new AstDecodeError(`AST exceeds maximum node count ${limits.maxNodes}`);
    }

    try {
      const kind = candidate.kind;
      if (typeof kind !== "string") throw new AstDecodeError(`${path}.kind must be a string`);
      if (kind === "ExpressionProgram") {
        assertExactKeys(candidate, ["kind", "body"], path);
        return {
          kind: "ExpressionProgram",
          body: visitExpr(candidate.body, depth + 1, `${path}.body`),
        };
      }
      if (kind === "MapProgram") {
        assertExactKeys(candidate, ["kind", "mapper"], path);
        return {
          kind: "MapProgram",
          mapper: visitExpr(candidate.mapper, depth + 1, `${path}.mapper`),
        };
      }
      if (kind === "FoldRightProgram") {
        assertExactKeys(candidate, ["kind", "initial", "reducer"], path);
        return {
          kind: "FoldRightProgram",
          initial: visitExpr(candidate.initial, depth + 1, `${path}.initial`),
          reducer: visitExpr(candidate.reducer, depth + 1, `${path}.reducer`),
        };
      }
      throw new AstDecodeError(
        `${path}.kind=${JSON.stringify(kind)} is not a complete program wrapper`,
      );
    } finally {
      active.delete(candidate);
    }
  }

  return {
    expression: (value) => visitExpr(value, 1, "expression"),
    program: (value) => visitProgram(value, 1, "program"),
  };
}

export function decodeExpr(value: unknown, limits: AstDecodeLimits): Expr {
  return createDecoder(limits).expression(value);
}

export function decodeProgram(value: unknown, limits: AstDecodeLimits): Program {
  return createDecoder(limits).program(value);
}
