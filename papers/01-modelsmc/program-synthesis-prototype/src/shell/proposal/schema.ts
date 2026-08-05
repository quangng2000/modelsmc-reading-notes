export type JsonRecord = Record<string, unknown>;

export function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function stripJsonFence(content: string): string {
  const trimmed = content.trim();
  if (!trimmed.startsWith("```")) return trimmed;
  return trimmed.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
}

export function responseSchema(constants: readonly bigint[]): JsonRecord {
  const binaryTags = ["Add", "Subtract", "Multiply", "LessThan", "EqualInt", "And"];
  return {
    type: "object",
    additionalProperties: false,
    required: ["expression", "rationale"],
    properties: {
      expression: { $ref: "#/$defs/program" },
      rationale: { type: "string" },
    },
    $defs: {
      program: {
        oneOf: [
          {
            type: "object",
            additionalProperties: false,
            required: ["kind", "body"],
            properties: {
              kind: { const: "ExpressionProgram" },
              body: { $ref: "#/$defs/expression" },
            },
          },
          {
            type: "object",
            additionalProperties: false,
            required: ["kind", "mapper"],
            properties: {
              kind: { const: "MapProgram" },
              mapper: { $ref: "#/$defs/expression" },
            },
          },
          {
            type: "object",
            additionalProperties: false,
            required: ["kind", "initial", "reducer"],
            properties: {
              kind: { const: "FoldRightProgram" },
              initial: { $ref: "#/$defs/expression" },
              reducer: { $ref: "#/$defs/expression" },
            },
          },
        ],
      },
      expression: {
        oneOf: [
          {
            type: "object",
            additionalProperties: false,
            required: ["kind"],
            properties: { kind: { const: "Input" } },
          },
          {
            type: "object",
            additionalProperties: false,
            required: ["kind"],
            properties: { kind: { const: "Item" } },
          },
          {
            type: "object",
            additionalProperties: false,
            required: ["kind"],
            properties: { kind: { const: "Accumulator" } },
          },
          {
            type: "object",
            additionalProperties: false,
            required: ["kind", "intValue"],
            properties: {
              kind: { const: "IntLiteral" },
              intValue: { type: "string", enum: constants.map((value) => value.toString()) },
            },
          },
          {
            type: "object",
            additionalProperties: false,
            required: ["kind", "boolValue"],
            properties: {
              kind: { const: "BoolLiteral" },
              boolValue: { type: "boolean" },
            },
          },
          {
            type: "object",
            additionalProperties: false,
            required: ["kind"],
            properties: { kind: { enum: ["EmptyIntList", "EmptyBoolList"] } },
          },
          {
            type: "object",
            additionalProperties: false,
            required: ["kind", "head", "tail"],
            properties: {
              kind: { enum: ["PrependInt", "PrependBool"] },
              head: { $ref: "#/$defs/expression" },
              tail: { $ref: "#/$defs/expression" },
            },
          },
          {
            type: "object",
            additionalProperties: false,
            required: ["kind", "operand"],
            properties: {
              kind: { const: "Not" },
              operand: { $ref: "#/$defs/expression" },
            },
          },
          {
            type: "object",
            additionalProperties: false,
            required: ["kind", "left", "right"],
            properties: {
              kind: { enum: binaryTags },
              left: { $ref: "#/$defs/expression" },
              right: { $ref: "#/$defs/expression" },
            },
          },
          {
            type: "object",
            additionalProperties: false,
            required: ["kind", "condition", "thenExpr", "elseExpr"],
            properties: {
              kind: { const: "IfThenElse" },
              condition: { $ref: "#/$defs/expression" },
              thenExpr: { $ref: "#/$defs/expression" },
              elseExpr: { $ref: "#/$defs/expression" },
            },
          },
        ],
      },
    },
  };
}
