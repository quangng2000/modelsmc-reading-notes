import { AnthropicProposer } from "../anthropic/index.js";
import { CatalogProposer } from "../catalog/index.js";
import { OllamaProposer } from "../ollama/index.js";
import { ProposalError, type Proposer } from "../proposal/index.js";
import type { CliOptions } from "./arguments.js";

export function createProposer(options: CliOptions): Proposer {
  if (options.proposal === "catalog") return new CatalogProposer();
  if (options.proposal === "anthropic") {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (apiKey === undefined || apiKey.trim() === "") {
      throw new ProposalError(
        "ANTHROPIC_API_KEY is not set; export it before using --proposal anthropic",
      );
    }
    return new AnthropicProposer({
      apiKey,
      model: options.model,
      baseUrl: options.anthropicUrl,
      timeoutMs: options.timeoutMs,
      ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
      maxTokens: options.maxTokens,
    });
  }
  return new OllamaProposer({
    model: options.model,
    baseUrl: options.ollamaUrl,
    timeoutMs: options.timeoutMs,
    ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
    maxTokens: options.maxTokens,
    ...(options.reasoningEffort === undefined
      ? {}
      : { reasoningEffort: options.reasoningEffort }),
  });
}
