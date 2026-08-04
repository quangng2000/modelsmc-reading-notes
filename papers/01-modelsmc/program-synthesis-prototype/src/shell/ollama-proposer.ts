import {
  expressionUsesInput,
  type Program,
} from "../core/language.verify.js";
import { decodeProgram } from "./decode.js";
import { ProposalError, type ProposalContext, type ProposalResult, type Proposer } from "./proposal.js";
import {
  jsonStringify,
  programToJsonValue,
  renderProgram,
  renderType,
  renderValue,
} from "./render.js";

export interface OllamaProposerOptions {
  readonly model?: string;
  readonly baseUrl?: string;
  readonly timeoutMs?: number;
  readonly temperature?: number;
  readonly reasoningEffort?: "low" | "medium" | "high";
  readonly requester?: typeof fetch;
}

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stripJsonFence(content: string): string {
  const trimmed = content.trim();
  if (!trimmed.startsWith("```")) return trimmed;
  return trimmed.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
}

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

function responseSchema(constants: readonly bigint[]): JsonRecord {
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

function promptFor(context: ProposalContext): string {
  const examples = context.examples.map((example, index) => ({
    index: index + 1,
    input: renderValue(example.input),
    output: renderValue(example.output),
    ancestorPrediction: renderValue(context.ancestorScore.evaluations[index]!.predicted),
    loss: context.ancestorScore.evaluations[index]!.loss,
  }));
  return [
    `Synthesize a ${renderType(context.inputType)} -> ${renderType(context.outputType)} program.`,
    "Return one complete JSON Program AST, never source code.",
    `Allowed integer constants: ${context.integerConstants.map((value) => value.toString()).join(", ")}.`,
    "The root must be ExpressionProgram{body}, MapProgram{mapper}, or FoldRightProgram{initial,reducer}.",
    "Allowed expression nodes: Input, Item, Accumulator, IntLiteral, BoolLiteral, EmptyIntList, EmptyBoolList, PrependInt, PrependBool, Add, Subtract, Multiply, LessThan, EqualInt, Not, And, IfThenElse.",
    "Item is bound only in a MapProgram mapper or FoldRightProgram reducer. Accumulator is bound only in a FoldRightProgram reducer. MapProgram and FoldRightProgram scoped expressions must not reference the outer Input.",
    "MapProgram requires a list input and preserves list length. FoldRightProgram traverses a list from right to left; its reducer must return the same type as its initial value.",
    `Maximum cost: ${context.maxCost}; maximum depth: ${context.maxDepth}; maximum nodes: ${context.maxNodes}.`,
    `Current ancestor: ${renderProgram(context.ancestor, context.inputType)}`,
    `Current ancestor JSON: ${jsonStringify(programToJsonValue(context.ancestor))}`,
    `Current loss: ${context.ancestorScore.totalLoss}; cost: ${context.ancestorScore.cost}.`,
    `Examples and current errors: ${jsonStringify(examples)}`,
    "Prefer a small well-typed expression that reduces the observed errors.",
  ].join("\n");
}

function decodeEnvelope(value: unknown, context: ProposalContext): ProposalResult {
  if (!isRecord(value)) throw new ProposalError("model response must be a JSON object");
  const keys = Object.keys(value).sort();
  if (keys.length !== 2 || keys[0] !== "expression" || keys[1] !== "rationale") {
    throw new ProposalError("model response must contain exactly expression and rationale");
  }
  if (typeof value.rationale !== "string") {
    throw new ProposalError("model response rationale must be a string");
  }
  const rationale = value.rationale.trim().slice(0, 2_000);
  const expression = decodeProgram(value.expression, {
    integerConstants: context.integerConstants,
    maxDepth: context.maxDepth,
    maxNodes: context.maxNodes,
  });
  assertCombinatorScope(expression);
  return { expression, rationale, source: "ollama" };
}

export class OllamaProposer implements Proposer {
  readonly name = "ollama";
  private readonly model: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly temperature: number;
  private readonly reasoningEffort: "low" | "medium" | "high";
  private readonly requester: typeof fetch;

  constructor(options: OllamaProposerOptions = {}) {
    this.model = options.model ?? "gpt-oss:20b";
    this.baseUrl = (options.baseUrl ?? "http://localhost:11434/v1").replace(/\/$/, "");
    this.timeoutMs = options.timeoutMs ?? 180_000;
    this.temperature = options.temperature ?? 0;
    this.reasoningEffort = options.reasoningEffort ?? "low";
    this.requester = options.requester ?? fetch;
  }

  async propose(context: ProposalContext): Promise<ProposalResult> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.requester(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          model: this.model,
          temperature: this.temperature,
          max_tokens: 768,
          reasoning_effort: this.reasoningEffort,
          messages: [
            {
              role: "system",
              content:
                "You propose typed AST data for a small program synthesizer. Follow the JSON schema exactly. Do not emit executable code.",
            },
            { role: "user", content: promptFor(context) },
          ],
          response_format: {
            type: "json_schema",
            json_schema: {
              name: "typed_program_proposal",
              strict: true,
              schema: responseSchema(context.integerConstants),
            },
          },
        }),
      });

      if (!response.ok) {
        const detail = (await response.text()).slice(0, 500);
        throw new ProposalError(`Ollama returned HTTP ${response.status}: ${detail}`);
      }
      const outer = (await response.json()) as unknown;
      if (!isRecord(outer) || !Array.isArray(outer.choices) || outer.choices.length !== 1) {
        throw new ProposalError("Ollama response must contain exactly one choice");
      }
      const choice = outer.choices[0];
      if (!isRecord(choice) || choice.finish_reason !== "stop" || !isRecord(choice.message)) {
        throw new ProposalError("Ollama response did not finish with a complete message");
      }
      // Deliberately access only content. A provider-specific message.reasoning field is never read.
      const content = choice.message.content;
      if (typeof content !== "string" || content.trim() === "") {
        throw new ProposalError("Ollama returned empty message content");
      }
      let inner: unknown;
      try {
        inner = JSON.parse(stripJsonFence(content)) as unknown;
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        throw new ProposalError(`Ollama content is not valid JSON: ${detail}`);
      }
      return decodeEnvelope(inner, context);
    } catch (error) {
      if (error instanceof ProposalError) throw error;
      if (error instanceof Error && error.name === "AbortError") {
        throw new ProposalError(`Ollama request timed out after ${this.timeoutMs} ms`);
      }
      const detail = error instanceof Error ? error.message : String(error);
      throw new ProposalError(`Ollama request failed: ${detail}`);
    } finally {
      clearTimeout(timeout);
    }
  }
}
