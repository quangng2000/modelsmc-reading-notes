import {
  ProposalError,
  type ProposalContext,
  type ProposalResult,
  type Proposer,
} from "../proposal/index.js";
import { decodeEnvelope } from "../proposal/decode.js";
import { promptFor } from "../proposal/prompt.js";
import { isRecord, responseSchema } from "../proposal/schema.js";

export interface AnthropicProposerOptions {
  readonly apiKey: string;
  readonly model?: string;
  readonly baseUrl?: string;
  readonly timeoutMs?: number;
  readonly temperature?: number;
  readonly maxTokens?: number;
  readonly anthropicVersion?: string;
  readonly requester?: typeof fetch;
}

// A forced tool call is the reliable way to obtain a strict, schema-constrained
// JSON envelope from the Messages API: the model must return exactly this tool's
// input, and decodeEnvelope re-validates it against the verified boundary decoder.
const PROPOSAL_TOOL_NAME = "emit_typed_program_proposal";

export class AnthropicProposer implements Proposer {
  readonly name = "anthropic";
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly temperature: number | undefined;
  private readonly maxTokens: number;
  private readonly anthropicVersion: string;
  private readonly requester: typeof fetch;

  constructor(options: AnthropicProposerOptions) {
    if (typeof options.apiKey !== "string" || options.apiKey.trim() === "") {
      throw new ProposalError("Anthropic API key is required; set ANTHROPIC_API_KEY");
    }
    this.apiKey = options.apiKey;
    this.model = options.model ?? "claude-sonnet-5";
    this.baseUrl = (options.baseUrl ?? "https://api.anthropic.com").replace(/\/$/, "");
    this.timeoutMs = options.timeoutMs ?? 180_000;
    // The Claude 5 family rejects an explicit temperature, so it is opt-in and
    // omitted by default rather than defaulted to 0 like the Ollama backend.
    this.temperature = options.temperature;
    this.maxTokens = options.maxTokens ?? 2_048;
    if (!Number.isSafeInteger(this.maxTokens) || this.maxTokens <= 0) {
      throw new RangeError("maxTokens must be a positive safe integer");
    }
    this.anthropicVersion = options.anthropicVersion ?? "2023-06-01";
    this.requester = options.requester ?? fetch;
  }

  async propose(context: ProposalContext): Promise<ProposalResult> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.requester(`${this.baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": this.apiKey,
          "anthropic-version": this.anthropicVersion,
        },
        signal: controller.signal,
        body: JSON.stringify({
          model: this.model,
          max_tokens: this.maxTokens,
          ...(this.temperature === undefined ? {} : { temperature: this.temperature }),
          system:
            "You propose typed AST data for a small program synthesizer. Call the emit_typed_program_proposal tool with data that follows its input schema exactly. Do not emit executable code.",
          tools: [
            {
              name: PROPOSAL_TOOL_NAME,
              description:
                "Emit one complete, bounded typed Program AST proposal and its rationale.",
              input_schema: responseSchema(context.integerConstants),
            },
          ],
          tool_choice: { type: "tool", name: PROPOSAL_TOOL_NAME },
          messages: [{ role: "user", content: promptFor(context) }],
        }),
      });

      if (!response.ok) {
        const detail = (await response.text()).slice(0, 500);
        throw new ProposalError(`Anthropic returned HTTP ${response.status}: ${detail}`);
      }
      const outer = (await response.json()) as unknown;
      if (!isRecord(outer) || !Array.isArray(outer.content)) {
        throw new ProposalError("Anthropic response must contain a content array");
      }
      if (outer.stop_reason !== "tool_use") {
        throw new ProposalError(
          `Anthropic did not stop on a tool call (stop_reason=${String(outer.stop_reason)})`,
        );
      }
      const proposals = outer.content.filter(
        (block): block is { type: string; name: string; input: unknown } =>
          isRecord(block) && block.type === "tool_use" && block.name === PROPOSAL_TOOL_NAME,
      );
      if (proposals.length !== 1) {
        throw new ProposalError(
          "Anthropic response must contain exactly one proposal tool call",
        );
      }
      return decodeEnvelope(proposals[0]!.input, context, "anthropic");
    } catch (error) {
      if (error instanceof ProposalError) throw error;
      if (error instanceof Error && error.name === "AbortError") {
        throw new ProposalError(`Anthropic request timed out after ${this.timeoutMs} ms`);
      }
      const detail = error instanceof Error ? error.message : String(error);
      throw new ProposalError(`Anthropic request failed: ${detail}`);
    } finally {
      clearTimeout(timeout);
    }
  }
}
