import { CatalogProposer } from "../catalog/index.js";
import { OllamaProposer } from "../ollama/index.js";
import type { Proposer } from "../proposal/index.js";
import type { CliOptions } from "./arguments.js";

export function createProposer(options: CliOptions): Proposer {
  if (options.proposal === "catalog") return new CatalogProposer();
  return new OllamaProposer({
    model: options.model,
    baseUrl: options.ollamaUrl,
    timeoutMs: options.timeoutMs,
    temperature: options.temperature,
    maxTokens: options.maxTokens,
    reasoningEffort: options.reasoningEffort,
  });
}
