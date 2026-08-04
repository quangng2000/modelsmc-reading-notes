import {
  expressionUsesInput,
  type Program,
} from "../../core/language.verify.js";
import { decodeProgram } from "../ast/decode.js";
import {
  ProposalError,
  type ProposalContext,
  type ProposalResult,
  type Proposer,
} from "../proposal/index.js";
import { promptFor } from "./prompt.js";
import { isRecord, responseSchema, stripJsonFence } from "./schema.js";

export interface OllamaProposerOptions {
  readonly model?: string;
  readonly baseUrl?: string;
  readonly timeoutMs?: number;
  readonly temperature?: number;
  readonly maxTokens?: number;
  readonly reasoningEffort?: "low" | "medium" | "high";
  readonly requester?: typeof fetch;
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
  private readonly maxTokens: number;
  private readonly reasoningEffort: "low" | "medium" | "high";
  private readonly requester: typeof fetch;

  constructor(options: OllamaProposerOptions = {}) {
    this.model = options.model ?? "gpt-oss:20b";
    this.baseUrl = (options.baseUrl ?? "http://localhost:11434/v1").replace(/\/$/, "");
    this.timeoutMs = options.timeoutMs ?? 180_000;
    this.temperature = options.temperature ?? 0;
    this.maxTokens = options.maxTokens ?? 2_048;
    if (!Number.isSafeInteger(this.maxTokens) || this.maxTokens <= 0) {
      throw new RangeError("maxTokens must be a positive safe integer");
    }
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
          max_tokens: this.maxTokens,
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
