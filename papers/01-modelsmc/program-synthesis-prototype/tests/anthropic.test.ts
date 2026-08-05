import assert from "node:assert/strict";
import test from "node:test";

import type { Program } from "../src/core/language.verify.js";
import { AnthropicProposer } from "../src/shell/anthropic/index.js";
import { scoreProgram } from "../src/shell/scoring/index.js";
import { listConfig } from "./support/fixtures.js";

test("the Anthropic adapter forces a tool call and decodes the returned AST", async () => {
  const config = listConfig("map successor", "List<Int>", [
    { input: [], output: [] },
    { input: ["1"], output: ["2"] },
  ]);
  const ancestor: Program = { kind: "ExpressionProgram", body: { kind: "Input" } };
  const ancestorScore = scoreProgram(ancestor, {
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    lossScale: config.lossScale,
    costScale: config.costScale,
    lossCap: config.lossCap,
    maxCost: config.maxCost,
  });
  if (ancestorScore.kind !== "Scored") assert.fail(ancestorScore.reason);

  let requestUrl = "";
  let requestHeaders: Record<string, string> = {};
  let requestBody = "";
  const requester: typeof fetch = async (input, init) => {
    requestUrl = String(input);
    requestHeaders = (init?.headers ?? {}) as Record<string, string>;
    requestBody = String(init?.body ?? "");
    return new Response(
      JSON.stringify({
        stop_reason: "tool_use",
        content: [
          { type: "text", text: "increment each scoped item" },
          {
            type: "tool_use",
            name: "emit_typed_program_proposal",
            input: {
              expression: {
                kind: "MapProgram",
                mapper: {
                  kind: "Add",
                  left: { kind: "Item" },
                  right: { kind: "IntLiteral", intValue: "1" },
                },
              },
              rationale: "increment each scoped item",
            },
          },
        ],
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  };

  const proposal = await new AnthropicProposer({
    apiKey: "sk-test-key",
    requester,
    timeoutMs: 1000,
  }).propose({
    requestIndex: 0,
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    integerConstants: config.integerConstants,
    maxDepth: config.maxDepth,
    maxNodes: config.maxNodes,
    maxCost: config.maxCost,
    ancestor,
    ancestorScore,
    ancestorFeedback: "proposal rejected: expression cost 33 exceeds maximum 30",
  });

  assert.equal(proposal.expression.kind, "MapProgram");
  assert.equal(proposal.rationale, "increment each scoped item");
  assert.equal(proposal.source, "anthropic");
  assert.match(requestUrl, /\/v1\/messages$/);
  assert.equal(requestHeaders["x-api-key"], "sk-test-key");
  assert.equal(requestHeaders["anthropic-version"], "2023-06-01");
  const parsed = JSON.parse(requestBody) as {
    model: string;
    tool_choice: { type: string; name: string };
  };
  assert.equal(parsed.model, "claude-sonnet-5");
  assert.deepEqual(parsed.tool_choice, { type: "tool", name: "emit_typed_program_proposal" });
  assert.match(requestBody, /must not reference the outer Input/);
  assert.equal(
    Object.hasOwn(parsed, "temperature"),
    false,
    "temperature must be omitted unless explicitly set, as the Claude 5 family rejects it",
  );
});

test("the Anthropic adapter accepts a JSON-encoded expression string", async () => {
  const config = listConfig("map successor", "List<Int>", [
    { input: [], output: [] },
    { input: ["1"], output: ["2"] },
  ]);
  const ancestor: Program = { kind: "ExpressionProgram", body: { kind: "Input" } };
  const ancestorScore = scoreProgram(ancestor, {
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    lossScale: config.lossScale,
    costScale: config.costScale,
    lossCap: config.lossCap,
    maxCost: config.maxCost,
  });
  if (ancestorScore.kind !== "Scored") assert.fail(ancestorScore.reason);

  // Anthropic's lenient recursive-schema handling can return the nested AST as a
  // JSON-encoded string; the decode boundary must still accept it.
  const requester: typeof fetch = async () =>
    new Response(
      JSON.stringify({
        stop_reason: "tool_use",
        content: [
          {
            type: "tool_use",
            name: "emit_typed_program_proposal",
            input: {
              expression: JSON.stringify({
                kind: "MapProgram",
                mapper: {
                  kind: "Add",
                  left: { kind: "Item" },
                  right: { kind: "IntLiteral", intValue: "1" },
                },
              }),
              rationale: "increment each scoped item",
            },
          },
        ],
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );

  const proposal = await new AnthropicProposer({
    apiKey: "sk-test-key",
    requester,
    timeoutMs: 1000,
  }).propose({
    requestIndex: 0,
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    integerConstants: config.integerConstants,
    maxDepth: config.maxDepth,
    maxNodes: config.maxNodes,
    maxCost: config.maxCost,
    ancestor,
    ancestorScore,
    ancestorFeedback: "initial seed",
  });

  assert.equal(proposal.expression.kind, "MapProgram");
  assert.equal(proposal.source, "anthropic");
});

test("the Anthropic adapter rejects a response that did not stop on a tool call", async () => {
  const config = listConfig("identity", "List<Int>", [{ input: [], output: [] }]);
  const ancestor: Program = { kind: "ExpressionProgram", body: { kind: "Input" } };
  const ancestorScore = scoreProgram(ancestor, {
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    lossScale: config.lossScale,
    costScale: config.costScale,
    lossCap: config.lossCap,
    maxCost: config.maxCost,
  });
  if (ancestorScore.kind !== "Scored") assert.fail(ancestorScore.reason);

  const requester: typeof fetch = async () =>
    new Response(
      JSON.stringify({ stop_reason: "end_turn", content: [{ type: "text", text: "no tool" }] }),
      { status: 200, headers: { "content-type": "application/json" } },
    );

  await assert.rejects(
    () =>
      new AnthropicProposer({ apiKey: "sk-test-key", requester, timeoutMs: 1000 }).propose({
        requestIndex: 0,
        inputType: config.inputType,
        outputType: config.outputType,
        examples: config.examples,
        integerConstants: config.integerConstants,
        maxDepth: config.maxDepth,
        maxNodes: config.maxNodes,
        maxCost: config.maxCost,
        ancestor,
        ancestorScore,
        ancestorFeedback: "initial seed",
      }),
    /did not stop on a tool call/,
  );
});

test("the Anthropic adapter requires an API key", () => {
  assert.throws(
    () => new AnthropicProposer({ apiKey: "  " }),
    /Anthropic API key is required/,
  );
});
